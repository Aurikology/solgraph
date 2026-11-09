import httpx

from solgraph.market import JUP_PRICE, WSOL, KitMarket

USDC = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"


def fake_jupiter(req: httpx.Request) -> httpx.Response:
    url = str(req.url)
    if url.startswith(JUP_PRICE):
        ids = req.url.params["ids"]
        return httpx.Response(200, json={ids: {"usdPrice": 1.0001, "priceChange24h": 0.01}}
                              if ids in (USDC, WSOL) else {})
    q = req.url.params["query"]
    return httpx.Response(200, json=[{"id": USDC, "name": "USD Coin", "symbol": "USDC",
                                      "isVerified": True}] if q.upper() == "USDC" or q == USDC else [])


def market(lookup=lambda t: None):
    return KitMarket(httpx.Client(transport=httpx.MockTransport(fake_jupiter)), ticker_lookup=lookup,
                     backoff=0)


def test_rate_limit_is_retried():
    hits = []

    def flaky(req):
        hits.append(1)
        return httpx.Response(429) if len(hits) < 3 else fake_jupiter(req)

    m = KitMarket(httpx.Client(transport=httpx.MockTransport(flaky)), ticker_lookup=lambda t: None,
                  backoff=0)
    assert m.resolve("USDC") == USDC


def test_ticker_falls_back_to_jupiter_search():
    info = market().token_info("usdc")
    assert info["mint"] == USDC and info["symbol"] == "USDC" and info["price_usd"] == 1.0001


def test_verified_token_beats_the_kit_lookup():
    fake = "5vJmAfBjsS6wEHcgQGNm69P3fbyPmUU1dVfE5j1rraVZ"  # the counterfeit USDC the kit returned live
    info = market(lambda t: fake).token_info("USDC")
    assert info["mint"] == USDC and info["warning"] is None


def test_kit_lookup_is_the_fallback_and_is_flagged():
    seen = []
    m = market(lambda t: seen.append(t) or "1" * 32 + "BBBB")
    info = m.token_info("whatever")
    assert info["mint"] == "1" * 32 + "BBBB" and seen == ["whatever"]
    assert "verified" in info["warning"]


def test_sol_maps_to_wrapped_sol():
    assert market().resolve("sol") == WSOL


def test_unpriced_token_says_why():
    info = market().token_info("1" * 32 + "AAAA")
    assert info["price_usd"] is None and "devnet" in info["note"]


def test_unknown_ticker():
    assert "No Solana token" in market().token_info("NOTAREALTOKEN")["error"]
