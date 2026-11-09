"""FastAPI: `uvicorn solgraph.api:app` then open http://127.0.0.1:8000/docs."""

from __future__ import annotations

import json

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ValidationError

from .agent import ask
from .app import App
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


@app.get("/health")
def health():
    a = core()
    return {"network": a.settings.network, "graph": type(a.graph).__name__, **a.graph.stats()}


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
