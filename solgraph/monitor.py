"""Wallet monitoring: snapshot balances, compare with the last snapshot, report what moved."""

from __future__ import annotations

import json
import time
from pathlib import Path

from .models import SOL_MINT, WalletBalances

DEFAULT_DIR = Path.home() / ".solgraph" / "snapshots"


def as_map(bal: WalletBalances) -> dict[str, float]:
    return {SOL_MINT: bal.sol} | {t.mint: t.amount for t in bal.tokens}


def diff(before: dict[str, float], after: dict[str, float], eps: float = 1e-9) -> list[dict]:
    out = []
    for mint in sorted(set(before) | set(after)):
        a, b = before.get(mint, 0.0), after.get(mint, 0.0)
        if abs(b - a) > eps:
            out.append({"mint": mint, "before": a, "after": b, "change": round(b - a, 9)})
    return out


class Monitor:
    def __init__(self, directory: Path = DEFAULT_DIR):
        self.dir = Path(directory)

    def _path(self, address: str) -> Path:
        return self.dir / f"{address}.json"

    def check(self, bal: WalletBalances) -> dict:
        """Compare against the stored snapshot, then store the new one."""
        path = self._path(bal.address)
        now = as_map(bal)
        prev = json.loads(path.read_text()) if path.exists() else None
        self.dir.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"taken_at": int(time.time()), "balances": now}))
        if prev is None:
            return {"first_snapshot": True, "changes": []}
        return {"first_snapshot": False, "since": prev["taken_at"],
                "changes": diff(prev["balances"], now)}
