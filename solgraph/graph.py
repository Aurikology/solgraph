"""The relationship graph: wallets, tokens, transactions, and the transfers between wallets.

    (:Wallet {address, kind})  -[:TRANSFER {signature, mint, amount, block_time}]->  (:Wallet)
    (:Wallet)                  -[:HOLDS {amount}]->                            (:Token {mint})
    (:Wallet)                  -[:SIGNED]->                                          (:Transaction {signature, slot, ...})

Two backends share one interface. `AgeGraph` stores it in PostgreSQL with Apache AGE and
queries it in Cypher. `MemoryGraph` keeps it in Python dictionaries, so the app and the tests
run without a database. Multi-hop questions ("where did these tokens come from?") are
answered by `trace_origin`, which walks one-hop queries breadth-first, so both backends give
the same answer.

`kind` says whether an address is a person's wallet (`user`), a program, or an account a
program controls (`program_account`: pool vaults, PDAs, token accounts). See models.KINDS.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Protocol

from .models import KINDS, PROGRAM_KINDS, SOL_MINT, Transfer, TxRecord, WalletBalances

_SAFE = re.compile(r"^(?:SOL|[1-9A-HJ-NP-Za-km-z]{32,90})$")


@dataclass
class Edge:
    source: str
    destination: str
    mint: str
    amount: float
    signature: str
    block_time: int | None


class GraphStore(Protocol):
    def add_transaction(self, tx: TxRecord) -> None: ...
    def set_holdings(self, bal: WalletBalances) -> None: ...
    def incoming(self, address: str, mint: str | None = None) -> list[Edge]: ...
    def outgoing(self, address: str, mint: str | None = None) -> list[Edge]: ...
    def holdings(self, address: str) -> dict[str, float]: ...
    def stats(self) -> dict[str, int]: ...
    def set_kinds(self, kinds: dict[str, str]) -> None: ...
    def kinds(self, addresses) -> dict[str, str]: ...


# --- shared queries over either backend --------------------------------------------------------

def trace_origin(g: GraphStore, address: str, mint: str, depth: int = 3,
                 max_edges: int = 50) -> list[Edge]:
    """Breadth-first walk backwards along TRANSFER edges of one asset.

    The walk records an edge from a program or program-owned account but does not continue
    past it: everything that ever went into a pool vault is not where this wallet's tokens
    came from. Addresses of unknown kind are followed."""
    seen_sig: set[tuple[str, str, str]] = set()
    found: list[Edge] = []
    queue = deque([(address, 0)])
    visited = {address}
    while queue and len(found) < max_edges:
        node, d = queue.popleft()
        if d >= depth:
            continue
        for e in g.incoming(node, mint):
            key = (e.signature, e.source, e.destination)
            if key in seen_sig:
                continue
            seen_sig.add(key)
            found.append(e)
            if e.source not in visited:
                visited.add(e.source)
                if g.kinds([e.source]).get(e.source) not in PROGRAM_KINDS:
                    queue.append((e.source, d + 1))
    return found


def counterparties(g: GraphStore, address: str) -> list[dict]:
    """Every wallet this one has traded with, with totals in each direction per asset."""
    agg: dict[tuple[str, str], dict] = {}
    for e, direction, other in ([(e, "received", e.source) for e in g.incoming(address)]
                                + [(e, "sent", e.destination) for e in g.outgoing(address)]):
        row = agg.setdefault((other, e.mint), {"wallet": other, "mint": e.mint,
                                               "sent": 0.0, "received": 0.0, "transfers": 0})
        row[direction] += e.amount
        row["transfers"] += 1
    return sorted(agg.values(), key=lambda r: -r["transfers"])


# --- in-memory backend -------------------------------------------------------------------------

class MemoryGraph:
    def __init__(self):
        self.edges: dict[tuple[str, str, str, str], Edge] = {}
        self.txs: dict[str, TxRecord] = {}
        self._holds: dict[str, dict[str, float]] = {}
        self._in: dict[str, list[Edge]] = defaultdict(list)
        self._out: dict[str, list[Edge]] = defaultdict(list)
        self._kinds: dict[str, str] = {}

    def add_transaction(self, tx: TxRecord) -> None:
        self.txs[tx.signature] = tx
        for t in tx.transfers:
            key = (t.signature, t.source, t.destination, t.mint)
            if key in self.edges:
                continue
            e = Edge(t.source, t.destination, t.mint, t.amount, t.signature, t.block_time)
            self.edges[key] = e
            self._in[e.destination].append(e)
            self._out[e.source].append(e)

    def set_holdings(self, bal: WalletBalances) -> None:
        h = {SOL_MINT: bal.sol} | {t.mint: t.amount for t in bal.tokens}
        self._holds[bal.address] = h

    def incoming(self, address, mint=None):
        return [e for e in self._in.get(address, []) if mint is None or e.mint == mint]

    def outgoing(self, address, mint=None):
        return [e for e in self._out.get(address, []) if mint is None or e.mint == mint]

    def holdings(self, address):
        return dict(self._holds.get(address, {}))

    def set_kinds(self, kinds):
        self._kinds.update({a: _kind(k) for a, k in kinds.items()})

    def kinds(self, addresses):
        return {a: self._kinds.get(a, "unknown") for a in addresses}

    def stats(self):
        wallets = {a for k in self.edges for a in k[1:3]} | set(self._holds)
        mints = {k[3] for k in self.edges} | {m for h in self._holds.values() for m in h}
        return {"wallets": len(wallets), "tokens": len(mints), "transactions": len(self.txs),
                "transfers": len(self.edges)}


# --- Apache AGE backend ------------------------------------------------------------------------

def _lit(s: str) -> str:
    """Quote a value for Cypher. Only base58 identifiers and 'SOL' are ever accepted, so no
    string reaching a query can close the quote or the $$ block."""
    if not _SAFE.fullmatch(s):
        raise ValueError(f"refusing to put {s!r} into a Cypher query")
    return f"'{s}'"


def _kind(k: str) -> str:
    if k not in KINDS:
        raise ValueError(f"unknown account kind {k!r}")
    return k


def _num(x: float | int | None) -> str:
    return "null" if x is None else repr(float(x)) if isinstance(x, float) else str(int(x))


def _agtype(v: str | None):
    if v is None:
        return None
    v = re.sub(r"::(numeric|float|integer|string)$", "", v)
    return json.loads(v)


class AgeGraph:
    """Apache AGE on PostgreSQL. Needs the `age` extension (see docker-compose.yml)."""

    def __init__(self, dsn: str, graph: str = "solgraph", connect_timeout: int = 10):
        import psycopg  # imported here so the in-memory path needs no Postgres driver

        if not re.fullmatch(r"[a-z_][a-z0-9_]*", graph):
            raise ValueError("graph name must be lowercase letters, digits and _")
        self.graph = graph
        try:
            # connect_timeout: an unreachable host fails in seconds with this message instead
            # of hanging on the OS TCP timeout (about two minutes on Windows)
            self.conn = psycopg.connect(dsn, autocommit=True, connect_timeout=connect_timeout)
        except psycopg.OperationalError as e:
            raise ConnectionError(
                f"Could not connect to Postgres for the graph ({e.__class__.__name__}). Is the "
                "database running (`docker compose up -d`)? Use 127.0.0.1 rather than localhost "
                "in SOLGRAPH_PG_DSN: Docker publishes the port on IPv4 only, and localhost may "
                "try IPv6 first. Or unset SOLGRAPH_PG_DSN to use the in-memory graph.") from e
        with self.conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS age")
            cur.execute("LOAD 'age'")
            cur.execute('SET search_path = ag_catalog, "$user", public')
            cur.execute("SELECT count(*) FROM ag_catalog.ag_graph WHERE name = %s", (graph,))
            if cur.fetchone()[0] == 0:
                cur.execute("SELECT create_graph(%s)", (graph,))

    def _cypher(self, query: str, cols: tuple[str, ...] = ("r",)) -> list[tuple]:
        spec = ", ".join(f"{c} agtype" for c in cols)
        sql = f"SELECT * FROM cypher('{self.graph}', $$ {query} $$) AS ({spec})"
        with self.conn.cursor() as cur:
            cur.execute(sql)
            return [tuple(_agtype(v) for v in row) for row in cur.fetchall()] if cur.description else []

    def add_transaction(self, tx: TxRecord) -> None:
        sig = _lit(tx.signature)
        self._cypher(
            f"MERGE (t:Transaction {{signature: {sig}}}) "
            f"SET t.slot = {_num(tx.slot)}, t.block_time = {_num(tx.block_time)}, "
            f"t.fee_sol = {_num(tx.fee_sol)}, t.success = {str(tx.success).lower()} RETURN t.signature")
        for s in tx.signers:
            self._cypher(f"MATCH (t:Transaction {{signature: {sig}}}) "
                         f"MERGE (w:Wallet {{address: {_lit(s)}}}) MERGE (w)-[:SIGNED]->(t) RETURN w.address")
        for t in tx.transfers:
            self._add_transfer(t)

    def _add_transfer(self, t: Transfer) -> None:
        self._cypher(
            f"MERGE (a:Wallet {{address: {_lit(t.source)}}}) "
            f"MERGE (b:Wallet {{address: {_lit(t.destination)}}}) "
            f"MERGE (k:Token {{mint: {_lit(t.mint)}}}) "
            f"MERGE (a)-[e:TRANSFER {{signature: {_lit(t.signature)}, mint: {_lit(t.mint)}}}]->(b) "
            f"SET e.amount = {_num(t.amount)}, e.block_time = {_num(t.block_time)} RETURN e.amount")

    def set_holdings(self, bal: WalletBalances) -> None:
        w = _lit(bal.address)
        self._cypher(f"MATCH (w:Wallet {{address: {w}}})-[h:HOLDS]->(:Token) DELETE h RETURN 1")
        for mint, amt in [(SOL_MINT, bal.sol)] + [(t.mint, t.amount) for t in bal.tokens]:
            self._cypher(f"MERGE (w:Wallet {{address: {w}}}) MERGE (k:Token {{mint: {_lit(mint)}}}) "
                         f"MERGE (w)-[h:HOLDS]->(k) SET h.amount = {_num(amt)} RETURN h.amount")

    def _edges(self, pattern: str, mint: str | None) -> list[Edge]:
        where = f" WHERE e.mint = {_lit(mint)}" if mint else ""
        rows = self._cypher(f"MATCH {pattern}{where} RETURN a.address, b.address, e.mint, "
                            f"e.amount, e.signature, e.block_time",
                            ("src", "dst", "mint", "amount", "sig", "bt"))
        return [Edge(*r) for r in rows]

    def incoming(self, address, mint=None):
        return self._edges(f"(a:Wallet)-[e:TRANSFER]->(b:Wallet {{address: {_lit(address)}}})", mint)

    def outgoing(self, address, mint=None):
        return self._edges(f"(a:Wallet {{address: {_lit(address)}}})-[e:TRANSFER]->(b:Wallet)", mint)

    def holdings(self, address):
        rows = self._cypher(f"MATCH (w:Wallet {{address: {_lit(address)}}})-[h:HOLDS]->(k:Token) "
                            "RETURN k.mint, h.amount", ("mint", "amount"))
        return dict(rows)

    def set_kinds(self, kinds):
        for a, k in kinds.items():
            self._cypher(f"MERGE (w:Wallet {{address: {_lit(a)}}}) "
                         f"SET w.kind = '{_kind(k)}' RETURN w.kind")

    def kinds(self, addresses):
        addresses = list(addresses)
        if not addresses:
            return {}
        lst = ", ".join(_lit(a) for a in addresses)
        rows = dict(self._cypher(f"MATCH (w:Wallet) WHERE w.address IN [{lst}] "
                                 "RETURN w.address, w.kind", ("a", "k")))
        return {a: rows.get(a) or "unknown" for a in addresses}

    def stats(self):
        out = {}
        for label, key in (("Wallet", "wallets"), ("Token", "tokens"), ("Transaction", "transactions")):
            out[key] = self._cypher(f"MATCH (n:{label}) RETURN count(n)")[0][0]
        out["transfers"] = self._cypher("MATCH ()-[e:TRANSFER]->() RETURN count(e)")[0][0]
        return out


def open_graph(dsn: str | None, name: str = "solgraph") -> GraphStore:
    return AgeGraph(dsn, name) if dsn else MemoryGraph()
