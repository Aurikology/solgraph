import json

import httpx
import pytest
from pydantic import TypeAdapter, ValidationError

from solgraph.graph import MemoryGraph, _lit, counterparties, trace_origin
from solgraph.ingest import ingest_wallet
from solgraph.models import Address, SOL_MINT
from solgraph.monitor import diff
from solgraph.rpc import RpcError, SolanaRpc, parse_transaction

from .conftest import ALICE, BOB, FUNDER, MINT, MINTER, TXS, make_handler


# --- validation -------------------------------------------------------------------------------

@pytest.mark.parametrize("bad", ["", "hello", "0" * 44, "O" * 44, "A" * 31, "A" * 45,
                                 "AAAA' }) DETACH DELETE n //AAAAAAAAAAAAAAAAAA", "x$$; DROP TABLE"])
def test_bad_addresses_are_rejected(bad):
    with pytest.raises(ValidationError):
        TypeAdapter(Address).validate_python(bad)


@pytest.mark.parametrize("bad", ["abc'", "SOL' OR 1=1", "$$", "A" * 31, "x" * 91])
def test_cypher_literal_refuses_anything_but_base58(bad):
    with pytest.raises(ValueError):
        _lit(bad)


def test_cypher_literal_quotes_valid_values():
    assert _lit(ALICE) == f"'{ALICE}'"
    assert _lit(SOL_MINT) == "'SOL'"


# --- parsing ----------------------------------------------------------------------------------

def test_sol_transfer_excludes_the_fee():
    sig = "sig1" + "1" * 60
    tx = parse_transaction(sig, TXS[sig])
    assert len(tx.transfers) == 1
    t = tx.transfers[0]
    assert (t.source, t.destination, t.mint, t.amount) == (FUNDER, ALICE, SOL_MINT, 1.5)
    assert tx.fee_sol == 0.000005 and tx.success and tx.signers == [FUNDER]


def test_token_transfer_is_owner_to_owner_not_token_account():
    sig = "sig4" + "4" * 60
    tx = parse_transaction(sig, TXS[sig])
    assert [(t.source, t.destination, t.mint, t.amount) for t in tx.transfers] == [(ALICE, BOB, MINT, 25.0)]


def test_swap_yields_one_transfer_per_asset():
    sig = "swap" + "9" * 60
    tx = {"slot": 1, "blockTime": 1, "transaction": {"message": {"accountKeys": [
            {"pubkey": ALICE, "signer": True}, {"pubkey": BOB, "signer": False}]}},
          "meta": {"err": None, "fee": 5000,
                   "preBalances": [2_000_000_000, 0], "postBalances": [1_000_000_000 - 5000 + 1_000_000_000 - 1_000_000_000, 1_000_000_000],
                   "preTokenBalances": [{"accountIndex": 1, "mint": MINT, "owner": BOB, "uiTokenAmount": {"uiAmount": 50.0}}],
                   "postTokenBalances": [{"accountIndex": 1, "mint": MINT, "owner": BOB, "uiTokenAmount": {"uiAmount": 0.0}},
                                         {"accountIndex": 2, "mint": MINT, "owner": ALICE, "uiTokenAmount": {"uiAmount": 50.0}}]}}
    got = {(t.source, t.destination, t.mint, t.amount) for t in parse_transaction(sig, tx).transfers}
    assert got == {(ALICE, BOB, SOL_MINT, 1.0), (BOB, ALICE, MINT, 50.0)}


def test_failed_transaction_is_marked():
    sig = "sig2" + "2" * 60
    raw = json.loads(json.dumps(TXS[sig]))
    raw["meta"]["err"] = {"InstructionError": [0, "Custom"]}
    raw["meta"]["postBalances"] = [b for b in raw["meta"]["preBalances"]]
    raw["meta"]["postBalances"][0] -= 5000
    tx = parse_transaction(sig, raw)
    assert not tx.success and tx.transfers == []


# --- rpc --------------------------------------------------------------------------------------

def test_rpc_retries_rate_limits():
    rpc = SolanaRpc("http://fake", backoff=0,
                    client=httpx.Client(transport=httpx.MockTransport(make_handler(fail_first=2))))
    assert rpc.balances(ALICE).sol == 1.1


def test_rpc_gives_up_with_an_actionable_message():
    rpc = SolanaRpc("http://fake", retries=1, backoff=0,
                    client=httpx.Client(transport=httpx.MockTransport(make_handler(fail_first=9))))
    with pytest.raises(RpcError, match="rate-limit"):
        rpc.balances(ALICE)


def test_rpc_surfaces_node_errors(rpc):
    with pytest.raises(RpcError, match="Method not found"):
        rpc.call("sendTransaction", ["x"])


def test_rpc_never_sends_transactions():
    import inspect

    import solgraph.rpc as m
    import re
    used = set(re.findall(r'self\.call\(\s*"(\w+)"', inspect.getsource(m)))
    assert used == {"getBalance", "getTokenAccountsByOwner", "getSignaturesForAddress", "getTransaction"}


# --- graph ------------------------------------------------------------------------------------

def _loaded(rpc):
    g = MemoryGraph()
    for w in (ALICE, BOB, FUNDER, MINTER):
        ingest_wallet(rpc, g, w)
    return g


def test_ingest_is_idempotent(rpc):
    g = _loaded(rpc)
    assert g.stats() == {"wallets": 4, "tokens": 2, "transactions": 4, "transfers": 4}


def test_trace_origin_follows_hops(rpc):
    g = _loaded(rpc)
    edges = trace_origin(g, BOB, SOL_MINT)
    assert [(e.source, e.destination) for e in edges] == [(ALICE, BOB), (FUNDER, ALICE)]
    assert [(e.source, e.destination) for e in trace_origin(g, BOB, SOL_MINT, depth=1)] == [(ALICE, BOB)]
    assert [(e.source, e.destination) for e in trace_origin(g, BOB, MINT)] == [(ALICE, BOB), (MINTER, ALICE)]


def test_counterparties_totals(rpc):
    rows = {(r["wallet"], r["mint"]): r for r in counterparties(_loaded(rpc), ALICE)}
    assert rows[(BOB, SOL_MINT)]["sent"] == 0.4
    assert rows[(FUNDER, SOL_MINT)]["received"] == 1.5
    assert rows[(MINTER, MINT)]["received"] == 100.0


# --- monitor ----------------------------------------------------------------------------------

def test_diff_reports_only_changes():
    assert diff({"SOL": 1.0, MINT: 5.0}, {"SOL": 1.0, MINT: 7.5, "X": 1}) == [
        {"mint": MINT, "before": 5.0, "after": 7.5, "change": 2.5},
        {"mint": "X", "before": 0.0, "after": 1, "change": 1}]
