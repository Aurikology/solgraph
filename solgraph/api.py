"""FastAPI: `uvicorn solgraph.api:app` then open http://127.0.0.1:8000/docs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field, ValidationError

from .agent import ask
from .app import App
from .graph import neighborhood
from .ingest import classify_unknown
from .models import Address, Mint

app = FastAPI(title="SolGraph", description="Read-only Solana explorer with an AI guide.")
_state: dict[str, App] = {}


def core() -> App:
    if "app" not in _state:
        _state["app"] = App.create()
    return _state["app"]


class Question(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


def _tool(name: str, **kw):
    out = json.loads(getattr(core().box, name)(**kw))
    if isinstance(out, dict) and "error" in out:
        raise HTTPException(502, out["error"])
    return out


_PAGE = Path(__file__).with_name("static") / "index.html"


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def page():
    """The graph explorer: type a wallet, load its history, see who it trades with."""
    return _PAGE.read_text(encoding="utf-8")


@app.get("/wallet/{address}/graph")
def graph(address: Address, depth: int = 2):
    """Nodes and transfer edges within `depth` hops, from the graph already loaded. Addresses
    whose kind is not known yet (loaded by an older version, or a failed lookup) are looked up
    first, so the drawing and the trace agree on what is a pool and what is a person."""
    a = core()
    depth = min(max(depth, 1), 3)
    g = neighborhood(a.graph, address, depth)
    if any(n["kind"] == "unknown" for n in g["nodes"]) and             classify_unknown(a.rpc, a.graph, [n["id"] for n in g["nodes"]]):
        g = neighborhood(a.graph, address, depth)  # kinds changed, so the program stop may too
    return g


@app.get("/health")
def health():
    a = core()
    cache = type(a.rpc.cache).__name__ if a.rpc.cache else None
    docs = a.box.docs
    return {"network": a.settings.network, "graph": type(a.graph).__name__, "cache": cache,
            "cache_hits": a.rpc.hits, "cache_misses": a.rpc.misses, **a.graph.stats(),
            "docs_store": type(docs.store).__name__ if docs else None,
            "docs_chunks": docs.count() if docs else 0,
            "kraken": a.box.kraken is not None}


@app.get("/kb/search")
def kb_search(q: str = Query(min_length=3, max_length=300),
              mode: Literal["hybrid", "vector", "keyword"] = "hybrid", k: int = 5):
    """Search the Solana docs index directly: hybrid (default), vector-only or keyword-only."""
    docs = core().box.docs
    if docs is None:
        raise HTTPException(503, core().box.docs_error or "Docs index not configured")
    hits = docs.search(q, min(max(k, 1), 10), mode)
    return {"query": q, "mode": mode, "results": [vars(h) for h in hits]}


@app.get("/wallet/{address}/balances")
def balances(address: Address):
    return _tool("wallet_balances", address=address)


@app.post("/wallet/{address}/ingest")
def ingest(address: Address, limit: int = 10):
    return _tool("recent_transactions", address=address, limit=min(max(limit, 1), 25))


@app.get("/wallet/{address}/counterparties")
def counterparties(address: Address):
    return _tool("wallet_counterparties", address=address)


@app.get("/wallet/{address}/origin/{mint}")
def origin(address: Address, mint: Mint, depth: int = 3):
    return _tool("trace_token_origin", address=address, mint=mint, depth=min(max(depth, 1), 5))


@app.post("/ask")
def ask_route(q: Question):
    try:
        return {"answer": ask(core().agent, q.question)}
    except ValidationError as e:
        raise HTTPException(422, str(e)) from e
