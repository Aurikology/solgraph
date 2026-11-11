"""The only things the AI can do. Every tool here is read-only.

The list is closed: `build_tools` returns exactly the names in `ALLOWED`, and a test fails if
anything else, in particular any of Solana Agent Kit's trade/transfer/stake/deploy tools, ever
appears. Inputs are Pydantic models, so a malformed address is rejected with a readable error
before any network call or graph query happens.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Protocol

import httpx
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from .graph import GraphStore, counterparties, trace_origin
from .ingest import ingest_wallet
from .kraken import KrakenError, Portfolio
from .models import Address, Mint
from .monitor import Monitor
from .rag import DocsIndex
from .rpc import RpcError, SolanaRpc

ALLOWED = frozenset({
    "wallet_balances", "recent_transactions", "wallet_counterparties",
    "trace_token_origin", "token_info", "check_wallet_changes", "graph_stats",
    "search_solana_docs", "my_kraken_portfolio",
})


class Market(Protocol):
    def token_info(self, query: str) -> dict: ...


# --- inputs --------------------------------------------------------------------------------------

class WalletIn(BaseModel):
    address: Address = Field(description="Solana wallet address (base58)")


class RecentIn(WalletIn):
    limit: int = Field(10, ge=1, le=25, description="How many recent transactions to load")


class TraceIn(WalletIn):
    mint: Mint = Field(description="Token mint address, or 'SOL' for native SOL")
    depth: int = Field(3, ge=1, le=5, description="How many hops back to follow")


class DocsIn(BaseModel):
    query: str = Field(min_length=3, max_length=300,
                       description="A question or topic, e.g. 'how are transaction fees calculated'")


class KrakenIn(BaseModel):
    asset: str | None = Field(None, pattern=r"^[A-Za-z0-9]{2,10}$",
                              description="Only this asset, e.g. 'SOL'. Leave empty for everything.")


class TokenIn(BaseModel):
    query: str = Field(min_length=1, max_length=64,
                       description="A token mint address or a ticker such as 'SOL' or 'USDC'")


# --- tools ---------------------------------------------------------------------------------------

@dataclass
class Toolbox:
    rpc: SolanaRpc
    graph: GraphStore
    market: Market | None = None
    monitor: Monitor | None = None
    docs: DocsIndex | None = None
    docs_error: str | None = None  # why there is no docs index, told to the model as is
    kraken: Portfolio | None = None
    kraken_error: str | None = None

    def wallet_balances(self, address: str) -> str:
        bal = self.rpc.balances(address)
        self.graph.set_holdings(bal)
        return bal.model_dump_json()

    def _with_kinds(self, edges) -> list[dict]:
        """Edges as dicts, with what each end is: a user wallet, a program, or a
        program-owned account such as a pool vault."""
        kinds = self.graph.kinds({a for e in edges for a in (e.source, e.destination)})
        return [vars(e) | {"source_kind": kinds[e.source],
                           "destination_kind": kinds[e.destination]} for e in edges]

    def recent_transactions(self, address: str, limit: int = 10) -> str:
        rep = ingest_wallet(self.rpc, self.graph, address, limit, with_balances=False)
        edges = self.graph.incoming(address) + self.graph.outgoing(address)
        edges.sort(key=lambda e: e.block_time or 0, reverse=True)
        return json.dumps({
            "loaded_transactions": rep.transactions,
            "could_not_load": len(rep.failed),
            "transfers": self._with_kinds(edges[:40]),
        })

    def wallet_counterparties(self, address: str) -> str:
        rows = counterparties(self.graph, address)
        if not rows:
            return json.dumps({"note": "No transfers for this wallet in the graph yet. "
                                       "Call recent_transactions first to load its history."})
        rows = rows[:25]
        kinds = self.graph.kinds({r["wallet"] for r in rows})
        return json.dumps([r | {"kind": kinds[r["wallet"]]} for r in rows])

    def trace_token_origin(self, address: str, mint: str, depth: int = 3) -> str:
        edges = trace_origin(self.graph, address, mint, depth)
        if not edges:
            return json.dumps({"note": "No incoming transfers of this asset in the graph. Load "
                                       "this wallet's history (and its senders') with "
                                       "recent_transactions, then trace again."})
        return json.dumps({
            "transfers": self._with_kinds(edges),
            "note": "The trace stops at program and program_account sources (pool vaults, "
                    "PDAs): tokens there came from a swap or a program, not from a person.",
        })

    def token_info(self, query: str) -> str:
        if self.market is None:
            return json.dumps({"error": "Token price lookup is not configured."})
        try:
            return json.dumps(self.market.token_info(query))
        except (RuntimeError, httpx.HTTPError) as e:
            return json.dumps({"error": f"Could not look up {query!r}: {e}. Try again in a minute."})

    def check_wallet_changes(self, address: str) -> str:
        mon = self.monitor or Monitor()
        return json.dumps(mon.check(self.rpc.balances(address)))

    def graph_stats(self) -> str:
        return json.dumps(self.graph.stats())

    def search_solana_docs(self, query: str) -> str:
        if self.docs is None:
            return json.dumps({"error": self.docs_error or "The docs index is not configured."})
        if self.docs.count() == 0:
            return json.dumps({"error": "The docs index is empty. Run "
                                        "`python -m solgraph docs ingest` once to build it."})
        hits = self.docs.search(query, k=4)
        return json.dumps({
            "results": [{"title": h.title, "section": h.section, "url": h.url,
                         "text": h.text[:1500]} for h in hits],
            "note": "From the official Solana docs. Answer from these passages and cite the url "
                    "of each page you use. If none of them answers the question, say so.",
        })

    def my_kraken_portfolio(self, asset: str | None = None) -> str:
        if self.kraken is None:
            return json.dumps({"error": self.kraken_error or "Kraken is not connected."})
        try:
            return json.dumps(self.kraken.summary(asset))
        except (KrakenError, httpx.HTTPError) as e:
            return json.dumps({"error": str(e)})


def _safe(fn):
    """Errors come back to the model as text it can explain, not as a crash."""
    def run(**kw):
        try:
            return fn(**kw)
        except RpcError as e:
            return json.dumps({"error": str(e)})
        except Exception as e:  # noqa: BLE001 - the agent must always get an answer back
            return json.dumps({"error": f"{type(e).__name__}: {e}"})
    run.__name__ = fn.__name__
    return run


def _explain_invalid(err) -> str:
    errs = err.errors() if hasattr(err, "errors") else [{"msg": str(err)}]
    return json.dumps({"error": "Invalid input: " + "; ".join(e["msg"] for e in errs)})


class _NoArgs(BaseModel):
    pass


def build_tools(box: Toolbox) -> list[StructuredTool]:
    spec = [
        ("wallet_balances", WalletIn, "SOL and token balances a wallet holds right now."),
        ("recent_transactions", RecentIn,
         "Load a wallet's recent transactions into the graph and list the transfers in and out."),
        ("wallet_counterparties", WalletIn,
         "Wallets this one has sent to or received from, with totals per asset and each "
         "counterparty's kind (user, program, program_account). Uses the graph."),
        ("trace_token_origin", TraceIn,
         "Follow an asset backwards through the graph to show where a wallet's tokens came from. "
         "Stops at programs and program-owned accounts such as DEX pool vaults."),
        ("token_info", TokenIn, "Name, symbol and USD price of a token, by mint address or ticker."),
        ("check_wallet_changes", WalletIn,
         "Compare a wallet's balances with the last time it was checked and report what changed."),
        ("graph_stats", _NoArgs, "How many wallets, tokens, transactions and transfers the graph holds."),
        ("search_solana_docs", DocsIn,
         "Search the official Solana documentation (core concepts, tokens, RPC methods) and "
         "return the most relevant passages with their page URLs, for questions about how "
         "Solana works rather than about a specific wallet."),
        ("my_kraken_portfolio", KrakenIn,
         "The person's OWN Kraken exchange account, read-only: what they hold (spot and "
         "staked), its value now, average buy price, staking rewards, deposits and withdrawals. "
         "Use for questions like 'how is my SOL doing'."),
    ]
    tools = [StructuredTool.from_function(func=_safe(getattr(box, name)), name=name,
                                          description=desc, args_schema=schema,
                                          handle_validation_error=_explain_invalid)
             for name, schema, desc in spec]
    assert {t.name for t in tools} == ALLOWED
    return tools
