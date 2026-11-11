"""Wiring shared by the CLI and the API: settings -> RPC, graph, market, agent."""

from __future__ import annotations

import os
from dataclasses import dataclass

from .cache import open_cache
from .config import Settings
from .graph import GraphStore, open_graph
from .kraken import KrakenClient, Portfolio
from .market import KitMarket
from .monitor import Monitor
from .rag import DocsIndex, make_embedder, open_docstore
from .rpc import SolanaRpc
from .tools import Toolbox


def open_docs(s: Settings) -> tuple[DocsIndex | None, str | None]:
    """The docs index lives beside the graph (pgvector in the same Postgres) or in a local
    file. Embeddings come from Gemini (GOOGLE_API_KEY) or a local model; Anthropic has none."""
    key = (os.environ.get("GOOGLE_API_KEY") or "").strip() or None
    try:
        embedder = make_embedder(s.embed_model, key, s.embed_per_minute)
        return DocsIndex(open_docstore(s.pg_dsn), embedder), None
    except ImportError:
        return None, (f"Docs search with {s.embed_model} needs fastembed: "
                      'pip install "solgraph[local]", or set GOOGLE_API_KEY to use Gemini.')
    except (ConnectionError, ValueError) as e:
        return None, str(e)


def open_kraken(s: Settings) -> tuple[Portfolio | None, str | None]:
    """The keys are read straight from the environment and never stored on Settings, so they
    can't show up in a printed config."""
    key = (os.environ.get("KRAKEN_API_KEY") or "").strip()
    secret = (os.environ.get("KRAKEN_API_SECRET") or "").strip()
    if not key and not secret:
        return None, ("Kraken is not connected. Create a read-only API key on Kraken (only "
                      "'Query Funds' and 'Query Ledger Entries') and put it in .env as "
                      "KRAKEN_API_KEY and KRAKEN_API_SECRET.")
    try:
        return Portfolio(KrakenClient(key, secret), quote=s.kraken_quote), None
    except ValueError as e:
        return None, str(e)


@dataclass
class App:
    settings: Settings
    rpc: SolanaRpc
    graph: GraphStore
    box: Toolbox
    _agent: object = None

    @classmethod
    def create(cls, settings: Settings | None = None) -> "App":
        s = settings or Settings.load()
        cache = open_cache(s.redis_url) if s.cache else None
        rpc = SolanaRpc(s.rpc_url, cache=cache)
        graph = open_graph(s.pg_dsn, s.graph_name)
        box = Toolbox(rpc=rpc, graph=graph, market=KitMarket(), monitor=Monitor())
        box.docs, box.docs_error = open_docs(s)
        box.kraken, box.kraken_error = open_kraken(s)
        return cls(s, rpc, graph, box)

    @property
    def agent(self):
        if self._agent is None:
            from .agent import build_agent, default_model

            self._agent = build_agent(default_model(self.settings.model), self.box,
                                      self.settings.network)
        return self._agent
