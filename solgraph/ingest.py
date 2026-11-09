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
    kinds_known: bool = True


def ingest_wallet(rpc: SolanaRpc, graph: GraphStore, address: str, limit: int = 20,
                  with_balances: bool = True) -> IngestReport:
    rep = IngestReport(address)
    seen = {address}
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
        seen.update(tx.signers)
        seen.update(a for t in tx.transfers for a in (t.source, t.destination))
    rep.kinds_known = classify_unknown(rpc, graph, seen)
    return rep


def classify_unknown(rpc: SolanaRpc, graph: GraphStore, addresses) -> bool:
    """Look up the kind of every address the graph does not know yet, in one batched read.
    Returns False if the lookup failed; the graph is unchanged and kinds fill in next time."""
    new = [a for a, k in graph.kinds(set(addresses)).items() if k == "unknown"]
    if not new:
        return True
    try:
        graph.set_kinds(rpc.account_kinds(new))
    except RpcError:
        return False
    return True
