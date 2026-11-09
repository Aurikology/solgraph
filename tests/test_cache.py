import os

import httpx
import pytest

from solgraph.cache import FOREVER, TTL, MemoryCache, RedisCache, SqliteCache, cache_key
from solgraph.ingest import ingest_wallet
from solgraph.graph import MemoryGraph
from solgraph.rpc import SolanaRpc

from .conftest import ALICE, make_handler


def cached_rpc(cache, calls):
    return SolanaRpc("http://fake", backoff=0, cache=cache,
                     client=httpx.Client(transport=httpx.MockTransport(make_handler(calls))))


def test_second_ingest_is_served_from_cache():
    calls: list = []
    rpc = cached_rpc(MemoryCache(), calls)
    ingest_wallet(rpc, MemoryGraph(), ALICE)
    first = len(calls)
    ingest_wallet(rpc, MemoryGraph(), ALICE)
    # getBalance, getTokenAccountsByOwner x2, getSignaturesForAddress, getTransaction x4,
    # getMultipleAccounts: 9 network calls the first time, none the second
    assert first == 9 and len(calls) == first
    assert (rpc.misses, rpc.hits) == (9, 9)


def test_ttls_match_what_can_change():
    assert TTL["getTransaction"] == FOREVER
    assert 0 < TTL["getBalance"] <= 60 and 0 < TTL["getSignaturesForAddress"] <= 60
    assert "sendTransaction" not in TTL


def test_short_ttl_expires_and_forever_does_not():
    now = [1000.0]
    c = MemoryCache(clock=lambda: now[0])
    c.set("bal", {"value": 1}, 15)
    c.set("tx", {"slot": 1}, FOREVER)
    now[0] += 16
    assert c.get("bal") is None and c.get("tx") == {"slot": 1}


def test_missing_transaction_is_not_cached():
    calls: list = []
    rpc = cached_rpc(MemoryCache(), calls)
    assert rpc.transaction("nope" * 16) is None
    assert rpc.transaction("nope" * 16) is None
    assert calls.count("getTransaction") == 2


def test_key_depends_on_network_and_params():
    a = cache_key("https://api.devnet.solana.com", "getBalance", [ALICE])
    assert a != cache_key("https://api.mainnet-beta.solana.com", "getBalance", [ALICE])
    assert a != cache_key("https://api.devnet.solana.com", "getBalance", [ALICE + "x"])


def test_sqlite_cache_survives_reopen(tmp_path):
    c = SqliteCache(tmp_path / "c.sqlite")
    c.set("k", {"value": [1, 2]}, FOREVER)
    c.set("gone", {"value": 1}, -1)  # already expired
    c2 = SqliteCache(tmp_path / "c.sqlite")
    assert c2.get("k") == {"value": [1, 2]} and c2.get("gone") is None


REDIS = os.environ.get("SOLGRAPH_TEST_REDIS_URL")


@pytest.mark.skipif(not REDIS, reason="set SOLGRAPH_TEST_REDIS_URL to a Redis server")
def test_redis_cache_round_trip():
    c = RedisCache(REDIS)
    key = cache_key("test", "getBalance", ["redis-test"])
    c.set(key, {"value": 42}, 15)
    assert c.get(key) == {"value": 42}
    assert 0 < c.r.ttl(key) <= 15
    c.set(key + "f", {"slot": 1}, FOREVER)
    assert c.r.ttl(key + "f") == -1  # no expiry
    c.r.delete(key, key + "f")
