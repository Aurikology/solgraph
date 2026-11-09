"""Token names and prices (Solana mainnet).

Uses Solana Agent Kit where its endpoints still work, and Jupiter's current APIs where they don't.
Checked 2026-11-09:

- kit `TokenDataManager.get_token_address_from_ticker` (DexScreener): works, used for tickers
- kit `TokenPriceFetcher.fetch_price` (api.jup.ag/price/v2): HTTP 404, replaced by price/v3
- kit `get_token_data_by_address` (tokens.jup.ag): host no longer resolves, replaced by tokens/v2

`SolanaAgentKit(...)` itself is never constructed. It requires a private key and logs it in
plain text at startup, and SolGraph holds no keys.
"""

from __future__ import annotations

import time

import httpx

from .models import BASE58

JUP_PRICE = "https://lite-api.jup.ag/price/v3"
JUP_SEARCH = "https://lite-api.jup.ag/tokens/v2/search"
WSOL = "So11111111111111111111111111111111111111112"


def _load_kit_token_data():
    """Import the kit's keyless token helper without running the kit's package __init__.

    solana-agent-kit-py 1.4.0 (the latest, unchanged on GitHub since 2025-02-05) cannot be
    imported: `solana_agent_kit/__init__` -> `agent` -> `tools/use_3land` -> `agent` is a
    circular import. Registering a bare package object first lets the leaf module load on its
    own, and it needs nothing from the broken chain.
    """
    import importlib.util
    import sys
    import types

    if "solana_agent_kit" not in sys.modules:
        spec = importlib.util.find_spec("solana_agent_kit")
        if spec is None or not spec.submodule_search_locations:
            return None
        pkg = types.ModuleType("solana_agent_kit")
        pkg.__path__ = list(spec.submodule_search_locations)
        sys.modules["solana_agent_kit"] = pkg
    from solana_agent_kit.tools.get_token_data import TokenDataManager

    return TokenDataManager


def _kit_ticker_lookup(ticker: str) -> str | None:
    try:
        tdm = _load_kit_token_data()
    except Exception:  # noqa: BLE001 - SolGraph falls back to Jupiter search
        return None
    return tdm.get_token_address_from_ticker(ticker) if tdm else None


class KitMarket:
    def __init__(self, client: httpx.Client | None = None, ticker_lookup=_kit_ticker_lookup,
                 backoff: float = 1.5):
        self.http = client or httpx.Client(timeout=15)
        self.backoff = backoff
        self.ticker_lookup = ticker_lookup

    def _get(self, url: str, **params):
        last = None
        for attempt in range(3):
            try:
                r = self.http.get(url, params=params)
                if r.status_code == 200:
                    return r.json()
                last = f"HTTP {r.status_code}"
            except httpx.TransportError as e:
                last = type(e).__name__
            if attempt < 2:
                time.sleep(self.backoff * 2**attempt)  # Jupiter's free tier rate-limits bursts
        raise RuntimeError(f"price service unavailable ({last})")

    def resolve(self, query: str) -> str | None:
        """Ticker or address -> mint. Verified tokens win.

        The kit's DexScreener lookup ranks pairs by reported FDV, which scam tokens inflate:
        on 2026-11-09 it returned counterfeit mints for both USDC and USDT. So Jupiter's
        verified list is asked first, and the kit is only a fallback for unlisted tickers.
        """
        q = query.strip()
        if q.upper() == "SOL":
            return WSOL
        if BASE58.fullmatch(q):
            return q
        hits = self._get(JUP_SEARCH, query=q)
        exact = [h for h in hits if h.get("symbol", "").lower() == q.lower()]
        verified = [h for h in exact if h.get("isVerified")]
        if verified:
            return verified[0]["id"]
        mint = self.ticker_lookup(q)
        if mint:
            return mint
        return (exact or [{}])[0].get("id")

    def token_info(self, query: str) -> dict:
        mint = self.resolve(query)
        if not mint:
            return {"error": f"No Solana token found for {query!r}."}
        meta = next((h for h in self._get(JUP_SEARCH, query=mint) if h.get("id") == mint), {})
        price = self._get(JUP_PRICE, ids=mint).get(mint) or {}
        return {
            "mint": mint,
            "name": meta.get("name"),
            "symbol": meta.get("symbol"),
            "verified": meta.get("isVerified"),
            "price_usd": price.get("usdPrice"),
            "price_change_24h_pct": price.get("priceChange24h"),
            "note": None if price else "No mainnet price: the token may be devnet-only or untraded.",
            "warning": None if meta.get("isVerified") else
                "Not on Jupiter's verified list. Tickers are not unique on Solana and copycat "
                "tokens are common; check the mint address before trusting this.",
        }
