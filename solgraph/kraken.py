"""Your own Kraken account, read-only: balances, staking and what you paid.

Coins held on Kraken sit in Kraken's books, not in an on-chain wallet under your name, so the
blockchain tools cannot see them. This module asks Kraken's REST API instead.

Read-only is enforced twice:
1. The API key you create on Kraken should only have "Query Funds" and "Query Ledger Entries"
   ticked. Such a key cannot trade or withdraw, whatever code holds it.
2. This client refuses every private method except the two reads in `PRIVATE_READS`, before any
   network call. A test fails if that list ever grows a write.

Using the API is free: Kraken only charges fees on trades, and a read-only key cannot trade.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from urllib.parse import urlencode

import httpx

BASE_URL = "https://api.kraken.com"
PRIVATE_READS = frozenset({"Balance", "Ledgers"})
LEDGER_PAGE = 50  # Kraken returns at most 50 ledger entries per call

# Kraken's older 4-letter codes for assets that existed before 2019
LEGACY = {"XXBT": "BTC", "XBT": "BTC", "XETH": "ETH", "XXRP": "XRP", "XXDG": "DOGE",
          "XDG": "DOGE", "XLTC": "LTC", "XXLM": "XLM", "XETC": "ETC", "XZEC": "ZEC",
          "XXMR": "XMR", "ZUSD": "USD", "ZCAD": "CAD", "ZEUR": "EUR", "ZGBP": "GBP",
          "ZJPY": "JPY", "ZAUD": "AUD"}
FIAT = frozenset({"USD", "CAD", "EUR", "GBP", "JPY", "AUD", "CHF"})
CASH = FIAT | {"USDC", "USDT"}
PAIR_NAME = {"BTC": "XBT", "DOGE": "XDG"}  # names Kraken still uses in pair codes  # what a purchase can be paid in, for the average buy price
WHERE = {"": "spot", "S": "staked", "M": "opt-in rewards", "F": "earn (flexible)",
         "B": "earn (bonded)", "P": "parachain"}

HINTS = {
    "EAPI:Invalid key": "Kraken did not recognise the API key. Check KRAKEN_API_KEY in .env.",
    "EAPI:Invalid signature": "The signature was rejected. Check KRAKEN_API_SECRET in .env "
                              "(copy the private key exactly, including any '=' at the end).",
    "EGeneral:Permission denied": "The key is missing a permission. On Kraken, edit the key and "
                                  "tick 'Query Funds' and 'Query Ledger Entries' (nothing else).",
    "EAPI:Invalid nonce": "Kraken saw an out-of-order request. Wait a few seconds and try again; "
                          "don't use the same key in two programs at once.",
    "EAPI:Rate limit exceeded": "Kraken's rate limit was hit. Wait a minute and try again.",
}


class KrakenError(RuntimeError):
    pass


def sign(path: str, data: dict, secret_b64: str) -> str:
    """API-Sign: HMAC-SHA512 of (path + SHA256(nonce + postdata)), keyed with the decoded secret."""
    postdata = urlencode(data)
    digest = hashlib.sha256((str(data["nonce"]) + postdata).encode()).digest()
    mac = hmac.new(base64.b64decode(secret_b64), path.encode() + digest, hashlib.sha512)
    return base64.b64encode(mac.digest()).decode()


class KrakenClient:
    def __init__(self, api_key: str, api_secret: str, client: httpx.Client | None = None,
                 sleep=time.sleep):
        if not api_key or not api_secret:
            raise ValueError("Kraken needs both KRAKEN_API_KEY and KRAKEN_API_SECRET.")
        try:
            base64.b64decode(api_secret, validate=True)
        except (binascii.Error, ValueError):
            raise ValueError("KRAKEN_API_SECRET is not valid base64. Copy the private key from "
                             "Kraken exactly, including any '=' at the end.") from None
        self._key, self._secret = api_key, api_secret
        self._http = client or httpx.Client(timeout=20)
        self._sleep = sleep
        self._last_nonce = 0

    def __repr__(self) -> str:  # never show the key in logs or tracebacks
        return "KrakenClient(key=***)"

    def _nonce(self) -> int:
        self._last_nonce = max(int(time.time() * 1000), self._last_nonce + 1)
        return self._last_nonce

    def private(self, method: str, **params) -> dict:
        if method not in PRIVATE_READS:
            raise PermissionError(f"SolGraph is read-only: Kraken method {method!r} is not allowed.")
        path = f"/0/private/{method}"
        for attempt in range(2):
            data = {"nonce": self._nonce(), **params}
            headers = {"API-Key": self._key, "API-Sign": sign(path, data, self._secret),
                       "Content-Type": "application/x-www-form-urlencoded"}
            r = self._http.post(BASE_URL + path, content=urlencode(data), headers=headers)
            try:
                return self._result(r)
            except KrakenError as e:
                if attempt == 0 and "Rate limit" in str(e):
                    self._sleep(15)
                    continue
                raise
        raise AssertionError("unreachable")

    def public(self, method: str, **params) -> dict:
        return self._result(self._http.get(f"{BASE_URL}/0/public/{method}", params=params))

    @staticmethod
    def _result(r: httpx.Response) -> dict:
        if r.status_code >= 500:
            raise KrakenError(f"Kraken is having trouble (HTTP {r.status_code}). Try again later.")
        body = r.json()
        if body.get("error"):
            errs = [str(e) for e in body["error"]]
            hint = next((h for code, h in HINTS.items() if any(e.startswith(code) for e in errs)),
                        "")
            raise KrakenError(f"Kraken said: {', '.join(errs)}. {hint}".strip())
        return body["result"]


def split_asset(code: str) -> tuple[str, str]:
    """'SOL03.S' -> ('SOL', 'staked'); 'XXBT' -> ('BTC', 'spot'); 'SOL.F' -> ('SOL', 'earn (flexible)')."""
    name, _, suffix = code.partition(".")
    name = LEGACY.get(name, name)
    m = re.fullmatch(r"([A-Z]+?)\d{2}", name)
    if m and suffix:
        name = m.group(1)
    return name, WHERE.get(suffix, suffix.lower())


def _d(x) -> float:
    return float(x or 0)


@dataclass
class _Stats:
    bought: float = 0.0
    cost: dict = field(default_factory=lambda: defaultdict(float))       # by cash currency
    bought_by: dict = field(default_factory=lambda: defaultdict(float))  # qty bought, by currency
    sold: float = 0.0
    proceeds: dict = field(default_factory=lambda: defaultdict(float))
    rewards: float = 0.0
    bonuses: float = 0.0   # referral and promo rewards, free coins
    deposited: float = 0.0
    withdrawn: float = 0.0
    fees: dict = field(default_factory=lambda: defaultdict(float))       # by cash currency


def summarise_ledger(entries: list[dict]) -> dict[str, _Stats]:
    """Buys and sells (paired by refid), rewards, deposits and withdrawals, per asset."""
    stats: dict[str, _Stats] = defaultdict(_Stats)
    groups: dict[str, list[dict]] = defaultdict(list)
    for e in entries:
        asset, _ = split_asset(e["asset"])
        typ, sub, amt, fee = e["type"], e.get("subtype", ""), _d(e["amount"]), _d(e.get("fee"))
        if typ in ("trade", "spend", "receive"):
            groups[e["refid"]].append(e | {"_asset": asset})
        elif typ == "deposit" and asset not in CASH:
            stats[asset].deposited += amt - fee
        elif typ == "withdrawal" and asset not in CASH:
            stats[asset].withdrawn += -amt + fee
        elif (typ in ("staking", "reward") and amt > 0 and sub in ("", "reward")) or \
                (typ == "earn" and sub == "reward"):
            stats[asset].rewards += amt - fee
        elif typ == "reward" and amt > 0:  # e.g. subtype 'invitebonus' for a referral
            stats[asset].bonuses += amt - fee
    for legs in groups.values():
        crypto = [l for l in legs if l["_asset"] not in CASH]
        cash = [l for l in legs if l["_asset"] in CASH]
        if len(crypto) != 1 or len(cash) != 1:
            continue  # crypto-to-crypto swaps have no cash price to average
        c, m = crypto[0], cash[0]
        s, cur = stats[c["_asset"]], m["_asset"]
        ca, cf, ma, mf = _d(c["amount"]), _d(c.get("fee")), _d(m["amount"]), _d(m.get("fee"))
        if ca > 0 > ma:    # paid cash, got crypto
            s.bought += ca - cf
            s.bought_by[cur] += ca - cf
            s.cost[cur] += -ma + mf
            s.fees[cur] += mf
        elif ca < 0 < ma:  # sold crypto for cash
            s.sold += -ca + cf
            s.proceeds[cur] += ma - mf
            s.fees[cur] += mf
    return stats


@dataclass
class Portfolio:
    client: KrakenClient
    quote: str = "CAD"
    max_pages: int = 20   # 1,000 ledger entries
    ttl: float = 120.0
    _cache: tuple[float, dict, list, bool] | None = None

    def _fetch(self) -> tuple[dict, list[dict], bool]:
        if self._cache and time.monotonic() - self._cache[0] < self.ttl:
            return self._cache[1:]
        balances = self.client.private("Balance")
        entries: list[dict] = []
        complete = False
        for page in range(self.max_pages):
            if page >= 5:  # Kraken's call counter refills slowly; stay under it
                self.client._sleep(6)
            res = self.client.private("Ledgers", ofs=len(entries))
            batch = list(res.get("ledger", {}).values())
            entries += batch
            if not batch or len(entries) >= int(res.get("count", 0)):
                complete = True
                break
        self._cache = (time.monotonic(), balances, entries, complete)
        return balances, entries, complete

    def _price(self, asset: str) -> tuple[float | None, str | None]:
        if asset == self.quote:
            return 1.0, self.quote
        for cur in dict.fromkeys((self.quote, "USD")):
            if asset == cur:
                return 1.0, cur
            try:
                res = self.client.public("Ticker", pair=f"{PAIR_NAME.get(asset, asset)}{cur}")
                return float(next(iter(res.values()))["c"][0]), cur
            except (KrakenError, httpx.HTTPError, StopIteration, KeyError, ValueError):
                continue
        return None, None

    def summary(self, asset: str | None = None) -> dict:
        balances, entries, complete = self._fetch()
        held: dict[str, dict[str, float]] = defaultdict(dict)
        for code, amt in balances.items():
            name, where = split_asset(code)
            if _d(amt):
                held[name][where] = held[name].get(where, 0.0) + _d(amt)
        stats = summarise_ledger(entries)
        names = sorted(set(held) | {a for a, s in stats.items() if s.bought or s.rewards or s.bonuses})
        rows, total = [], 0.0  # the total is the whole account, even when one asset is asked for
        for name in names:
            qty = sum(held.get(name, {}).values())
            price, cur = self._price(name) if qty else (None, None)
            s = stats.get(name, _Stats())
            row = {"asset": name, "amount": round(qty, 8),
                   "where": {k: round(v, 8) for k, v in held.get(name, {}).items()},
                   "price": price, "price_currency": cur,
                   "value": round(qty * price, 2) if price is not None else None}
            if cur == self.quote and row["value"] is not None:
                total += row["value"]
            if s.bought:
                avg = {c: round(s.cost[c] / q, 4) for c, q in s.bought_by.items() if q}
                row["bought_on_kraken"] = round(s.bought, 8)
                row["average_buy_price"] = avg
                if any(s.fees.values()):  # Buy Crypto builds its fee into the price
                    row["fees_paid"] = {c: round(v, 2) for c, v in s.fees.items() if v}
                if price is not None and cur in avg and avg[cur]:
                    row["change_since_average_buy_pct"] = round((price / avg[cur] - 1) * 100, 1)
            if s.sold:
                row["sold_on_kraken"] = round(s.sold, 8)
                row["sale_proceeds"] = {c: round(v, 2) for c, v in s.proceeds.items()}
            if s.rewards:
                row["staking_rewards_earned"] = round(s.rewards, 8)
            if s.bonuses:
                row["bonuses_received"] = round(s.bonuses, 8)
            if s.deposited:
                row["deposited_from_outside"] = round(s.deposited, 8)
            if s.withdrawn:
                row["withdrawn_to_outside"] = round(s.withdrawn, 8)
            rows.append(row)
        return {
            "assets": [r for r in rows if not asset or r["asset"] == asset.upper()],
            f"total_value_{self.quote.lower()}": round(total, 2),
            "ledger_entries_read": len(entries),
            "history_complete": complete,
            "notes": [
                "Read-only. These coins are held by Kraken (custodial), not in an on-chain "
                "wallet, so they do not appear in the blockchain tools.",
                "Average buy price counts purchases made on Kraken only; coins deposited from "
                "elsewhere have no cost here. It is an estimate, not tax advice.",
            ] + ([] if complete else ["Only the most recent ledger entries were read, so totals "
                                      "from older history are missing."]),
        }
