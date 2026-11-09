"""Settings from the environment (or a .env file). Devnet by default, on purpose."""

from __future__ import annotations

import os
from dataclasses import dataclass

NETWORKS = {
    "devnet": "https://api.devnet.solana.com",
    "testnet": "https://api.testnet.solana.com",
    "mainnet": "https://api.mainnet-beta.solana.com",
}


def _load_dotenv(path: str = ".env") -> None:
    if not os.path.exists(path):
        return
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"'))


@dataclass(frozen=True)
class Settings:
    network: str
    rpc_url: str
    pg_dsn: str | None
    graph_name: str
    model: str

    @classmethod
    def load(cls) -> "Settings":
        _load_dotenv()
        network = os.environ.get("SOLGRAPH_NETWORK", "devnet")
        if network not in NETWORKS:
            raise ValueError(f"SOLGRAPH_NETWORK must be one of {sorted(NETWORKS)}, got {network!r}")
        return cls(
            network=network,
            rpc_url=os.environ.get("SOLGRAPH_RPC_URL", NETWORKS[network]),
            pg_dsn=os.environ.get("SOLGRAPH_PG_DSN"),  # unset = in-memory graph
            graph_name=os.environ.get("SOLGRAPH_GRAPH", "solgraph"),
            model=os.environ.get("SOLGRAPH_MODEL") or (
                "gemini-2.5-flash" if os.environ.get("GOOGLE_API_KEY")
                and not os.environ.get("ANTHROPIC_API_KEY") else "claude-sonnet-5-5"),
        )
