"""Wiring shared by the CLI and the API: settings -> RPC, graph, market, agent."""

from __future__ import annotations

from dataclasses import dataclass

from .config import Settings
from .graph import GraphStore, open_graph
from .market import KitMarket
from .monitor import Monitor
from .rpc import SolanaRpc
from .tools import Toolbox


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
        rpc = SolanaRpc(s.rpc_url)
        graph = open_graph(s.pg_dsn, s.graph_name)
        box = Toolbox(rpc=rpc, graph=graph, market=KitMarket(), monitor=Monitor())
        return cls(s, rpc, graph, box)

    @property
    def agent(self):
        if self._agent is None:
            from .agent import build_agent, default_model

            self._agent = build_agent(default_model(self.settings.model), self.box,
                                      self.settings.network)
        return self._agent
