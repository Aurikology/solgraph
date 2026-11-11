"""Kraken portfolio: request signing, the read-only guard, and the numbers, against a fake Kraken."""

from __future__ import annotations

import base64
import json
from urllib.parse import parse_qs

import httpx
import pytest

from solgraph.kraken import (PRIVATE_READS, KrakenClient, KrakenError, Portfolio, sign,
                             split_asset, summarise_ledger)
from solgraph.tools import Toolbox, build_tools

SECRET = base64.b64encode(b"not-a-real-secret-" * 4).decode()


def L(refid, typ, asset, amount, fee="0", subtype=""):
    return {"refid": refid, "time": 1.7e9, "type": typ, "subtype": subtype, "aclass": "currency",
            "asset": asset, "amount": amount, "fee": fee, "balance": "0"}


LEDGER = [
    # bought 10 SOL for 2,000 CAD + 5 CAD fee
    L("T1", "trade", "SOL", "10"), L("T1", "trade", "ZCAD", "-2000", "5"),
    # bought 5 SOL with the Buy Crypto button for 1,200 CAD
    L("B1", "spend", "ZCAD", "-1200"), L("B1", "receive", "SOL", "5"),
    # sold 3 SOL for 900 CAD, 2 CAD fee
    L("T2", "trade", "SOL", "-3"), L("T2", "trade", "ZCAD", "900", "2"),
    # swapped SOL for BTC: no cash leg, so it does not move the average
    L("T3", "trade", "SOL", "-1"), L("T3", "trade", "XXBT", "0.002"),
    L("D1", "deposit", "SOL", "2"), L("W1", "withdrawal", "SOL", "-1", "0.01"),
    L("S1", "staking", "SOL.S", "0.25"),
    L("E1", "earn", "SOL.F", "0.05", subtype="reward"),
    L("X1", "transfer", "SOL.S", "4", subtype="stakingfromspot"),  # moving, not earning
    L("C1", "deposit", "ZCAD", "5000"),
]
BALANCE = {"SOL": "8.0", "SOL03.S": "4.25", "SOL.F": "0.05", "ZCAD": "100.5", "XXBT": "0.002",
           "DOT": "0.0000"}


class FakeKraken:
    """Records every request; answers Balance, Ledgers (in pages of `page`) and Ticker."""

    def __init__(self, ledger=LEDGER, page=50, errors=None):
        self.ledger, self.page, self.errors = ledger, page, list(errors or [])
        self.requests: list[httpx.Request] = []

    def __call__(self, req: httpx.Request) -> httpx.Response:
        self.requests.append(req)
        if self.errors:
            return httpx.Response(200, json={"error": [self.errors.pop(0)]})
        path = req.url.path
        if path == "/0/private/Balance":
            return httpx.Response(200, json={"error": [], "result": BALANCE})
        if path == "/0/private/Ledgers":
            ofs = int(parse_qs(req.content.decode()).get("ofs", ["0"])[0])
            chunk = self.ledger[ofs:ofs + self.page]
            return httpx.Response(200, json={"error": [], "result": {
                "ledger": {f"L{ofs + i}": e for i, e in enumerate(chunk)},
                "count": len(self.ledger)}})
        if path == "/0/public/Ticker":
            pair = req.url.params["pair"]
            prices = {"SOLCAD": "250.0", "XBTCAD": "140000.0"}
            if pair not in prices:
                return httpx.Response(200, json={"error": ["EQuery:Unknown asset pair"]})
            return httpx.Response(200, json={"error": [], "result": {
                pair: {"c": [prices[pair], "1"]}}})
        return httpx.Response(404, json={"error": ["EGeneral:Unknown method"]})


def portfolio(fake=None, **kw) -> tuple[Portfolio, FakeKraken]:
    fake = fake or FakeKraken()
    client = KrakenClient("pub-key", SECRET, client=httpx.Client(transport=httpx.MockTransport(fake)),
                          sleep=lambda s: None)
    return Portfolio(client, **kw), fake


def test_signature_matches_krakens_documented_example():
    secret = ("kQH5HW/8p1uGOVjbgWA7FunAmGO8lsSUXNsu3eow76sz84Q18fWxnyRzBHCd3pd5nE9qa99HAZtuZuj6"
              "F1huXg==")
    data = {"nonce": "1616492376594", "ordertype": "limit", "pair": "XBTUSD", "price": 37500,
            "type": "buy", "volume": 1.25}
    assert sign("/0/private/AddOrder", data, secret) == (
        "4/dpxb3iT4tp/ZCVEwSnEsLxx0bqyhLpdfOpc6fn7OR8+UClSV5n9E6aSS8MPtnRfp32bAb0nmbRn6H8ndwLUQ==")


def test_only_reads_are_allowed():
    assert PRIVATE_READS == {"Balance", "Ledgers"}
    p, fake = portfolio()
    for method in ("AddOrder", "Withdraw", "Stake", "CancelAll", "WalletTransfer"):
        with pytest.raises(PermissionError, match="read-only"):
            p.client.private(method)
    assert fake.requests == []  # refused before anything left the machine


def test_requests_are_signed_and_the_secret_never_leaves():
    p, fake = portfolio()
    p.client.private("Balance")
    req = fake.requests[0]
    assert req.method == "POST" and req.headers["API-Key"] == "pub-key"
    body = req.content.decode()
    assert req.headers["API-Sign"] == sign("/0/private/Balance", {"nonce": int(
        parse_qs(body)["nonce"][0])}, SECRET)
    assert SECRET not in body and SECRET not in str(req.headers)
    assert "pub-key" not in repr(p.client)


def test_nonces_always_increase():
    p, fake = portfolio()
    for _ in range(5):
        p.client.private("Balance")
    nonces = [int(parse_qs(r.content.decode())["nonce"][0]) for r in fake.requests]
    assert nonces == sorted(set(nonces))


def test_bad_secret_and_missing_key_are_explained():
    with pytest.raises(ValueError, match="base64"):
        KrakenClient("k", "not base64 !!")
    with pytest.raises(ValueError, match="KRAKEN_API_KEY"):
        KrakenClient("", SECRET)


@pytest.mark.parametrize("code,expected", [
    ("SOL", ("SOL", "spot")), ("SOL.S", ("SOL", "staked")), ("SOL03.S", ("SOL", "staked")),
    ("SOL.F", ("SOL", "earn (flexible)")), ("XXBT", ("BTC", "spot")), ("ZCAD", ("CAD", "spot")),
    ("USDC", ("USDC", "spot")), ("ETH2.S", ("ETH2", "staked")),
])
def test_asset_codes(code, expected):
    assert split_asset(code) == expected


def test_ledger_summary():
    s = summarise_ledger(LEDGER)["SOL"]
    assert s.bought == 15 and s.cost["CAD"] == 3205 and s.fees["CAD"] == 7
    assert s.sold == 3 and s.proceeds["CAD"] == 898
    assert s.deposited == 2 and s.withdrawn == pytest.approx(1.01)
    assert s.rewards == pytest.approx(0.30)  # staking + earn rewards, not the transfer


def test_portfolio_summary():
    p, _ = portfolio()
    out = p.summary()
    sol = next(a for a in out["assets"] if a["asset"] == "SOL")
    assert sol["amount"] == 12.3
    assert sol["where"] == {"spot": 8.0, "staked": 4.25, "earn (flexible)": 0.05}
    assert sol["price"] == 250.0 and sol["price_currency"] == "CAD" and sol["value"] == 3075.0
    assert sol["average_buy_price"] == {"CAD": round(3205 / 15, 4)}
    assert sol["change_since_average_buy_pct"] == round((250 / (3205 / 15) - 1) * 100, 1)
    assert sol["staking_rewards_earned"] == pytest.approx(0.3)
    assert "DOT" not in {a["asset"] for a in out["assets"]}  # zero balances are left out
    assert out["total_value_cad"] == round(3075.0 + 100.5 + 0.002 * 140000, 2)
    assert out["history_complete"] is True and out["ledger_entries_read"] == len(LEDGER)
    assert any("not tax advice" in n for n in out["notes"])


def test_asset_filter_and_cache():
    p, fake = portfolio()
    p.summary()
    n = len(fake.requests)
    only = p.summary("sol")
    assert [a["asset"] for a in only["assets"]] == ["SOL"]
    private = [r for r in fake.requests[n:] if "/private/" in r.url.path]
    assert private == []  # second call within the TTL reads the cache


def test_pagination_and_truncated_history():
    p, fake = portfolio(FakeKraken(page=4))
    assert p.summary()["ledger_entries_read"] == len(LEDGER)
    pages = [r for r in fake.requests if r.url.path.endswith("Ledgers")]
    assert len(pages) == -(-len(LEDGER) // 4)
    short, _ = portfolio(FakeKraken(page=4), max_pages=2)
    out = short.summary()
    assert out["history_complete"] is False and out["ledger_entries_read"] == 8
    assert any("most recent" in n for n in out["notes"])


def test_kraken_errors_come_with_a_fix():
    p, _ = portfolio(FakeKraken(errors=["EGeneral:Permission denied"]))
    with pytest.raises(KrakenError, match="Query Ledger Entries"):
        p.client.private("Balance")


def test_rate_limit_is_retried_once():
    p, fake = portfolio(FakeKraken(errors=["EAPI:Rate limit exceeded"]))
    assert p.client.private("Balance") == BALANCE
    assert len(fake.requests) == 2


def test_tool_without_and_with_kraken(box):
    tool = {t.name: t for t in build_tools(box)}["my_kraken_portfolio"]
    box.kraken_error = "Kraken is not connected. Create a read-only API key"
    assert "read-only API key" in json.loads(tool.invoke({}))["error"]
    box.kraken, _ = portfolio()
    out = json.loads(tool.invoke({"asset": "SOL"}))
    assert out["assets"][0]["asset"] == "SOL"
    assert "Invalid input" in json.loads(tool.invoke({"asset": "SOL; DROP"}))["error"]
    box.kraken, _ = portfolio(FakeKraken(errors=["EAPI:Invalid key"]))
    assert "KRAKEN_API_KEY" in json.loads(tool.invoke({}))["error"]


def test_open_kraken_reads_env(monkeypatch):
    from solgraph.app import open_kraken
    from solgraph.config import Settings

    s = Settings.load()
    monkeypatch.delenv("KRAKEN_API_KEY", raising=False)
    monkeypatch.delenv("KRAKEN_API_SECRET", raising=False)
    assert open_kraken(s)[0] is None and "not connected" in open_kraken(s)[1]
    monkeypatch.setenv("KRAKEN_API_KEY", "k")
    monkeypatch.setenv("KRAKEN_API_SECRET", "bad secret")
    assert "base64" in open_kraken(s)[1]
    monkeypatch.setenv("KRAKEN_API_SECRET", SECRET)
    pf, err = open_kraken(s)
    assert err is None and pf.quote == s.kraken_quote
