"""A fake Solana RPC built on httpx.MockTransport, and three wallets with a small history:

    FUNDER --1.5 SOL--> ALICE --0.4 SOL--> BOB
    MINTER --100 USDC-like--> ALICE --25--> BOB
"""

from __future__ import annotations

import json

import httpx
import pytest

from solgraph.graph import MemoryGraph
from solgraph.monitor import Monitor
from solgraph.rpc import SolanaRpc
from solgraph.tools import Toolbox

ALICE = "A1iceWa11etXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"[:44]
BOB = "BobWa11etYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYY"[:44]
FUNDER = "FunderZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ"[:44]
MINTER = "MinterWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW"[:44]
MINT = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
SYS = "11111111111111111111111111111111"

USERS = {ALICE, BOB, FUNDER, MINTER}  # System Program-owned in the fake chain

FEE = 5000


def sol_tx(sig, src, dst, lamports, slot, t):
    return {
        "slot": slot, "blockTime": t,
        "transaction": {"message": {"accountKeys": [
            {"pubkey": src, "signer": True}, {"pubkey": dst, "signer": False},
            {"pubkey": SYS, "signer": False}]}},
        "meta": {"err": None, "fee": FEE,
                 "preBalances": [5_000_000_000, 1_000_000_000, 1],
                 "postBalances": [5_000_000_000 - lamports - FEE, 1_000_000_000 + lamports, 1],
                 "preTokenBalances": [], "postTokenBalances": []},
    }


def token_tx(sig, src, dst, amount, slot, t, src_before=100.0, dst_before=0.0):
    def tb(idx, owner, ui):
        return {"accountIndex": idx, "mint": MINT, "owner": owner,
                "uiTokenAmount": {"uiAmount": ui, "decimals": 6, "amount": str(int(ui * 1e6))}}
    return {
        "slot": slot, "blockTime": t,
        "transaction": {"message": {"accountKeys": [
            {"pubkey": src, "signer": True}, {"pubkey": "TokAcct1" + "1" * 36, "signer": False},
            {"pubkey": "TokAcct2" + "2" * 36, "signer": False}]}},
        "meta": {"err": None, "fee": FEE,
                 "preBalances": [2_000_000_000, 2_039_280, 2_039_280],
                 "postBalances": [2_000_000_000 - FEE, 2_039_280, 2_039_280],
                 "preTokenBalances": [tb(1, src, src_before), tb(2, dst, dst_before)],
                 "postTokenBalances": [tb(1, src, src_before - amount), tb(2, dst, dst_before + amount)]},
    }


TXS = {
    "sig1" + "1" * 60: sol_tx("s1", FUNDER, ALICE, 1_500_000_000, 100, 1_700_000_000),
    "sig2" + "2" * 60: sol_tx("s2", ALICE, BOB, 400_000_000, 200, 1_700_000_100),
    "sig3" + "3" * 60: token_tx("s3", MINTER, ALICE, 100.0, 150, 1_700_000_050),
    "sig4" + "4" * 60: token_tx("s4", ALICE, BOB, 25.0, 250, 1_700_000_150, src_before=100.0),
}

BY_WALLET = {
    ALICE: list(TXS),
    BOB: ["sig2" + "2" * 60, "sig4" + "4" * 60],
    FUNDER: ["sig1" + "1" * 60],
    MINTER: ["sig3" + "3" * 60],
}


def make_handler(calls: list | None = None, fail_first: int = 0):
    state = {"fails": fail_first}

    def handler(req: httpx.Request) -> httpx.Response:
        if state["fails"] > 0:
            state["fails"] -= 1
            return httpx.Response(429, json={})
        body = json.loads(req.content)
        m, p = body["method"], body["params"]
        if calls is not None:
            calls.append(m)
        if m == "getBalance":
            res = {"value": 1_100_000_000}
        elif m == "getTokenAccountsByOwner":
            res = {"value": [] if p[1]["programId"].startswith("Tokenz") else [
                {"account": {"data": {"parsed": {"info": {"mint": MINT, "tokenAmount": {
                    "uiAmount": 75.0, "decimals": 6}}}}}}]}
        elif m == "getSignaturesForAddress":
            res = [{"signature": s} for s in BY_WALLET.get(p[0], [])][: p[1]["limit"]]
        elif m == "getTransaction":
            res = TXS.get(p[0])
        elif m == "getMultipleAccounts":
            res = {"value": [{"owner": SYS, "executable": False, "lamports": 1}
                             if a in USERS else None for a in p[0]]}
        else:
            return httpx.Response(200, json={"jsonrpc": "2.0", "id": body["id"],
                                             "error": {"code": -32601, "message": "Method not found"}})
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": body["id"], "result": res})

    return handler


@pytest.fixture
def rpc():
    return SolanaRpc("http://fake", backoff=0,
                     client=httpx.Client(transport=httpx.MockTransport(make_handler())))


class FakeMarket:
    def token_info(self, query):
        return {"query": query, "symbol": "USDC", "price_usd": 1.0}


@pytest.fixture
def box(rpc, tmp_path):
    return Toolbox(rpc=rpc, graph=MemoryGraph(), market=FakeMarket(), monitor=Monitor(tmp_path))
