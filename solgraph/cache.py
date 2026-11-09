"""A cache for RPC reads, so asking twice does not hit a rate-limited public endpoint twice.

What is cached, and for how long, depends on whether the answer can change:

- `getTransaction` (finalized, the RPC default) never changes, so it is kept for good.
- Account owners (`getMultipleAccounts`) almost never change: an hour.
- Balances and signature lists change with every new transaction: 15 seconds, enough to
  absorb the bursts one question causes without serving stale numbers to the next question.

Three stores share one tiny interface: `SqliteCache` (the default, a file in ~/.solgraph),
`RedisCache` (set SOLGRAPH_REDIS_URL, for a shared or long-running deployment) and
`MemoryCache` (tests). Values are the RPC's own JSON results.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Protocol

FOREVER = 0  # ttl meaning "never expires"
TTL = {
    "getTransaction": FOREVER,
    "getMultipleAccounts": 3600,
    "getBalance": 15,
    "getTokenAccountsByOwner": 15,
    "getSignaturesForAddress": 15,
}
DEFAULT_PATH = Path.home() / ".solgraph" / "rpc-cache.sqlite"


def cache_key(url: str, method: str, params: list) -> str:
    raw = json.dumps([url, method, params], sort_keys=True, separators=(",", ":"))
    return "solgraph:rpc:" + hashlib.sha256(raw.encode()).hexdigest()


class Cache(Protocol):
    def get(self, key: str): ...
    def set(self, key: str, value, ttl: int) -> None: ...


class MemoryCache:
    def __init__(self, clock=time.monotonic):
        self.clock = clock
        self.data: dict[str, tuple[float, object]] = {}

    def get(self, key):
        hit = self.data.get(key)
        if hit is None or (hit[0] and hit[0] < self.clock()):
            return None
        return hit[1]

    def set(self, key, value, ttl):
        self.data[key] = (self.clock() + ttl if ttl else 0, value)


class SqliteCache:
    def __init__(self, path: Path | str = DEFAULT_PATH):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        # one connection shared by FastAPI's worker threads, guarded by a lock
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.lock = threading.Lock()
        with self.lock:
            self.db.execute("CREATE TABLE IF NOT EXISTS rpc "
                            "(k TEXT PRIMARY KEY, expires REAL NOT NULL, v TEXT NOT NULL)")
            self.db.commit()

    def get(self, key):
        with self.lock:
            row = self.db.execute("SELECT expires, v FROM rpc WHERE k = ?", (key,)).fetchone()
        if row is None or (row[0] and row[0] < time.time()):
            return None
        return json.loads(row[1])

    def set(self, key, value, ttl):
        with self.lock:
            self.db.execute("INSERT OR REPLACE INTO rpc VALUES (?, ?, ?)",
                            (key, time.time() + ttl if ttl else 0, json.dumps(value)))
            self.db.commit()


class RedisCache:
    def __init__(self, url: str):
        import redis  # optional dependency: pip install "solgraph[redis]"

        self.r = redis.Redis.from_url(url, socket_connect_timeout=5)
        self.r.ping()  # fail at startup with a clear error, not on the first question

    def get(self, key):
        v = self.r.get(key)
        return None if v is None else json.loads(v)

    def set(self, key, value, ttl):
        self.r.set(key, json.dumps(value), ex=ttl or None)


def open_cache(redis_url: str | None, path: Path | str | None = DEFAULT_PATH) -> Cache | None:
    if redis_url:
        return RedisCache(redis_url)
    return SqliteCache(path) if path else None
