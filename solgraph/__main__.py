"""python -m solgraph <command>

  balances <address>               what a wallet holds right now
  history  <address> [--limit N]   load recent transactions into the graph and list transfers
  origin   <address> <mint|SOL>    where this wallet's tokens came from (loads history first)
  watch    <address>               what changed since the last check
  token    <mint|ticker>           token name, symbol and price (mainnet tokens)
  ask      "<question>"            one question to the AI guide
  chat                             talk to the AI guide (GOOGLE_API_KEY or ANTHROPIC_API_KEY)
  docs ingest [--sections a,b]     fetch the Solana docs, chunk, embed and index them
  docs search "<query>" [--mode M] search the docs index (hybrid, vector or keyword)
  docs eval [--k 5]                retrieval quality on evals/docs_retrieval.json, all modes
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pydantic import TypeAdapter, ValidationError

from .app import App
from .models import Address
from .rag import DEFAULT_SECTIONS, DOCS_REPO, MODES, evaluate, fetch_pages
from .rpc import RpcError

EVAL_CASES = Path(__file__).resolve().parent.parent / "evals" / "docs_retrieval.json"


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
    d = sub.add_parser("docs").add_subparsers(dest="docs_cmd", required=True)
    di = d.add_parser("ingest")
    di.add_argument("--sections", default=",".join(DEFAULT_SECTIONS))
    di.add_argument("--ref", default="main", help="branch, tag or commit of " + DOCS_REPO)
    ds = d.add_parser("search")
    ds.add_argument("query")
    ds.add_argument("--mode", choices=MODES, default="hybrid")
    ds.add_argument("--k", type=int, default=5)
    de = d.add_parser("eval")
    de.add_argument("--k", type=int, default=5)
    de.add_argument("--cases", default=str(EVAL_CASES))
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


def _docs(app: App, a) -> int:
    idx = app.box.docs
    if idx is None:
        print(f"error: {app.box.docs_error}", file=sys.stderr)
        return 1
    print(f"[solgraph] docs index: {type(idx.store).__name__}, {idx.embedder.model}", file=sys.stderr)
    if a.docs_cmd == "ingest":
        sections = [s for s in a.sections.split(",") if s.strip()]
        sha, pages = fetch_pages(sections, a.ref, progress=lambda i, n: print(
            f"\r  fetched {i}/{n} pages", end="", file=sys.stderr))
        print(file=sys.stderr)
        rep = idx.ingest(pages, sha)
        print(json.dumps(vars(rep) | {"source": f"{DOCS_REPO}@{sha[:10]}",
                                       "indexed_chunks": idx.count()}, indent=2))
    elif a.docs_cmd == "search":
        for i, h in enumerate(idx.search(a.query, min(max(a.k, 1), 10), a.mode), 1):
            text = h.text[:240].replace("\n", " ")
            print(f"{i}. [{h.score}] {h.title} > {h.section or '(intro)'}\n   {h.url}\n   {text}\n")
    else:
        cases = json.loads(Path(a.cases).read_text(encoding="utf-8"))
        res = evaluate(idx, cases, a.k)
        print(f"{len(cases)} questions, {idx.count()} chunks, k={a.k}")
        for mode, m in res.items():
            print(f"  {mode:8} hit@1 {m['hit@1']:.3f}  recall@{a.k} {m[f'recall@{a.k}']:.3f}  "
                  f"MRR {m['mrr']:.3f}  misses {len(m['misses'])}")
        for q in res["hybrid"]["misses"]:
            print(f"  hybrid miss: {q}")
    return 0


def _run(a) -> int:
    app = App.create()
    box = app.box
    if a.cmd == "docs":
        return _docs(app, a)
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
