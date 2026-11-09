"""python -m solgraph <command>

  balances <address>               what a wallet holds right now
  history  <address> [--limit N]   load recent transactions into the graph and list transfers
  origin   <address> <mint|SOL>    where this wallet's tokens came from (loads history first)
  watch    <address>               what changed since the last check
  token    <mint|ticker>           token name, symbol and price (mainnet tokens)
  ask      "<question>"            one question to the AI guide
  chat                             talk to the AI guide (needs ANTHROPIC_API_KEY)
"""

from __future__ import annotations

import argparse
import json
import sys

from pydantic import TypeAdapter, ValidationError

from .app import App
from .models import Address
from .rpc import RpcError


def _print(raw: str) -> None:
    print(json.dumps(json.loads(raw), indent=2))


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):  # Windows consoles default to cp1252; answers use emoji
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser(prog="solgraph", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("balances", "watch"):
        sub.add_parser(name).add_argument("address")
    h = sub.add_parser("history")
    h.add_argument("address")
    h.add_argument("--limit", type=int, default=10)
    o = sub.add_parser("origin")
    o.add_argument("address")
    o.add_argument("mint")
    o.add_argument("--depth", type=int, default=3)
    sub.add_parser("token").add_argument("query")
    sub.add_parser("ask").add_argument("question")
    sub.add_parser("chat")
    a = p.parse_args(argv)

    if hasattr(a, "address"):
        try:
            TypeAdapter(Address).validate_python(a.address)
        except ValidationError as e:
            print(e.errors()[0]["msg"], file=sys.stderr)
            return 2

    try:
        return _run(a)
    except RpcError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


def _run(a) -> int:
    app = App.create()
    box = app.box
    print(f"[solgraph] {app.settings.network}, graph: {type(app.graph).__name__}", file=sys.stderr)
    if a.cmd == "balances":
        _print(box.wallet_balances(a.address))
    elif a.cmd == "history":
        _print(box.recent_transactions(a.address, min(max(a.limit, 1), 25)))
    elif a.cmd == "origin":
        box.recent_transactions(a.address, 25)
        _print(box.trace_token_origin(a.address, a.mint, min(max(a.depth, 1), 5)))
    elif a.cmd == "watch":
        _print(box.check_wallet_changes(a.address))
    elif a.cmd == "token":
        _print(box.token_info(a.query))
    else:
        from .agent import ask

        if a.cmd == "ask":
            print(ask(app.agent, a.question))
        else:
            print("SolGraph chat (read-only). Empty line to quit.")
            while (q := input("> ").strip()):
                print(ask(app.agent, q), "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
