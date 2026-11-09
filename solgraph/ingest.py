"""Pull a wallet's recent history from Solana into the graph."""

from __future__ import annotations

from dataclasses import dataclass, field

from .graph import GraphStore
from .rpc import RpcError, SolanaRpc


@dataclass
class IngestReport:
    address: str
    transactions: int = 0
    transfers: int = 0
    failed: list[str] = field(default_factory=list)


def ingest_wallet(rpc: SolanaRpc, graph: GraphStore, address: str, limit: int = 20,
                  with_balances: bool = True) -> IngestReport:
    rep = IngestReport(address)
    if with_balances:
        graph.set_holdings(rpc.balances(address))
    for sig in rpc.signatures(address, limit):
        try:
            tx = rpc.transaction(sig["signature"])
        except RpcError:
            rep.failed.append(sig["signature"])
            continue
        if tx is None:
            continue
        graph.add_transaction(tx)
        rep.transactions += 1
        rep.transfers += len(tx.transfers)
    return rep
