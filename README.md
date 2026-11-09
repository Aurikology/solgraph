# SolGraph

[![tests](https://github.com/Aurikology/solgraph/actions/workflows/tests.yml/badge.svg)](https://github.com/Aurikology/solgraph/actions/workflows/tests.yml)

SolGraph is an AI-powered Solana assistant that uses LangChain and Apache AGE to understand, explore and explain relationships between wallets, tokens and transactions.

**It is read-only.** It holds no private keys, has no tool that signs or sends anything, and defaults to devnet. A test fails if a write tool ever reaches the AI.

```
question ─► LangChain agent (Gemini or Claude) ─► read-only tools ─► Solana RPC     (balances, history)
                                             │        └──► Jupiter / Agent Kit (token names, prices)
                                             ▼
                                   graph: Apache AGE on Postgres (or in memory)
                                   (:Wallet {kind})-[:TRANSFER {mint, amount}]->(:Wallet)
                                   (:Wallet)-[:HOLDS]->(:Token)   (:Wallet)-[:SIGNED]->(:Transaction)
```

## Status (v1.1, 2026-11-09)

| Part | Verified how |
|---|---|
| RPC reads, transfer parsing, in-memory graph, tracing, monitoring, CLI | 54 offline tests, and the CLI run against live Solana devnet |
| Token lookup and prices | Tests, and live lookups of USDC, JUP and BONK against Jupiter |
| FastAPI routes | In-process tests (`TestClient`) |
| RPC cache | Unit tests for every store, a live Redis round trip, and the cold/warm devnet timing below |
| Docker stack | `docker compose up -d --build`: all three containers healthy, page and `/ask` checked against live devnet, Redis serving 34 of 38 reads on the second load |
| Graph web page | Loaded a live devnet wallet through AGE in headless Chromium at 1366 px and 390 px: 16 addresses, 26 transfers, no console errors, no sideways scroll |
| LangChain agent | **Live on Gemini 3.5 Flash against devnet:** it answered "what does this wallet hold and where did its SOL come from" from real tool calls, and refused a request to send SOL (and warned that the pasted key was compromised) |
| Wallets vs programs | Live on devnet through AGE: 10 transactions gave 3 user wallets and 8 program-owned accounts, and the classifications were spot-checked against `getAccountInfo` |
| Apache AGE backend | **Verified on Postgres 16 + AGE 1.6.0 (Docker):** the AGE integration test passes (56/56 with AGE and Redis, none skipped), and a live devnet `origin` trace wrote its transfers into the `solgraph` graph, read back with Cypher |

## What it can answer

| Ask | Tool | Source |
|---|---|---|
| "What does this wallet hold?" | `wallet_balances` | RPC `getBalance`, `getTokenAccountsByOwner` (SPL Token and Token-2022) |
| "What has it done lately?" | `recent_transactions` | RPC history, parsed into owner-to-owner transfers and stored in the graph |
| "Who does it trade with?" | `wallet_counterparties` | Graph |
| "Where did these tokens come from?" | `trace_token_origin` | Graph, walked backwards hop by hop |
| "What's this token worth?" | `token_info` | Jupiter verified list and price API; Solana Agent Kit lookup as a fallback |
| "What changed since I last looked?" | `check_wallet_changes` | Balance snapshots in `~/.solgraph/` |

### How transfers are worked out

Solana transactions don't list transfers. They list instructions, and swaps or program calls can move tokens through accounts the instructions never name. SolGraph reads each transaction's **balance changes** instead: for every asset, owners whose balance fell are paired with owners whose balance rose, and the fee is added back first so it never shows up as a transfer. This works the same for plain sends, swaps and program-routed transfers. See `solgraph/rpc.py` (`parse_transaction`).

### Wallets vs programs

An address whose balance changed is not always a person. After loading a history, SolGraph asks the chain who owns each new address (one `getMultipleAccounts` call per 100 addresses, zero bytes of data) and stores a `kind` on the node:

| kind | Meaning | How it is decided |
|---|---|---|
| `user` | A wallet a keypair controls | Owned by the System Program |
| `program` | Program code | Executable |
| `program_account` | State a program controls: pool vaults, PDAs, token accounts | Owned by any other program |
| `unknown` | No account on chain (closed or never funded), or not looked up yet | |

Origin traces record a transfer out of a `program` or `program_account` but do not walk past it. Everything that ever went into a DEX pool vault is not where your tokens came from, so the trace answers "from a swap" instead of listing strangers.

## Run everything with Docker

```bash
cp .env.example .env              # optional: add GOOGLE_API_KEY (free) for the AI guide
docker compose up -d --build      # SolGraph + Postgres 16/Apache AGE + Redis
```

Open http://127.0.0.1:8000 for the graph page, or http://127.0.0.1:8000/docs for the API. The app runs as a non-root user, waits for both databases to be healthy, and reads only the AI key from `.env`: there is no wallet key to give it. Verified: the stack started from a clean build, the page loaded a live devnet wallet at desktop and phone widths, and `/ask` answered from real data (0.018010866 SOL, matching the page).

## Setup

Tested on Python 3.13.0. `solana-agent-kit-py` declares `requires-python ==3.13.0`, so installers may refuse it on other versions. [uv](https://docs.astral.sh/uv/) sets this up in four lines:

```bash
uv venv -p 3.13.0 .venv
uv pip install -p .venv --no-deps solana-agent-kit-py==1.4.0
uv pip install -p .venv -e ".[dev]"
cp .env.example .env            # add GOOGLE_API_KEY (free) or ANTHROPIC_API_KEY for chat
```

The kit goes in with `--no-deps` on purpose. Its declared pins (LangChain < 0.4, numpy < 2, docs tooling) would hold everything on LangChain 0.3, and that version's Gemini adapter fails on every Gemini 3 tool call ("missing a thought_signature"). The one kit module SolGraph uses needs only `requests`, `solders` and `construct`, which are listed in `pyproject.toml`.

A free Gemini key: https://aistudio.google.com/apikey. The default model is `gemini-3.5-flash`. On a new free key, `gemini-2.5-*` is retired and `gemini-3.8-flash` timed out, as of 2026-11-09.

### Try it without an API key

```bash
python -m solgraph balances <address>
python -m solgraph history  <address> --limit 10
python -m solgraph origin   <address> SOL
python -m solgraph token    USDC
python -m solgraph watch    <address>
```

### Ask the AI guide

```bash
python -m solgraph ask "Where did this wallet's SOL come from? <address>"
python -m solgraph chat
uvicorn solgraph.api:app        # then open http://127.0.0.1:8000/docs
```

### See the graph

```bash
uvicorn solgraph.api:app        # then open http://127.0.0.1:8000
```

Type a wallet and press Load: it pulls the last 15 transactions from Solana and draws every transfer within one to three hops, in rings by distance, coloured by what each address is. Repeated transfers between the same two addresses are merged into one edge (`12× 0.0323638 SOL` is the total). Click an address to open it in the Solana explorer or load its own history.

![SolGraph graph view of a devnet wallet: one user wallet funded it 12 times, and it paid account rent into 14 program-owned accounts](docs/graph-ui.png)

### Use Apache AGE instead of the in-memory graph

```bash
docker compose up -d age        # Postgres 16 + AGE 1.6.0 on 127.0.0.1:5455 (not "localhost": see below)
export SOLGRAPH_PG_DSN=postgresql://solgraph:solgraph@127.0.0.1:5455/solgraph
```

Use `127.0.0.1`, not `localhost`. Docker publishes the port on IPv4 only, and on Windows `localhost` tries IPv6 first: each connection then hangs for about two minutes before falling back (measured: 130 s against 0.02 s).

The graph then persists and can be queried directly in Cypher:

```sql
LOAD 'age'; SET search_path = ag_catalog, "$user", public;
SELECT * FROM cypher('solgraph', $$
  MATCH (a:Wallet)-[t:TRANSFER {mint: 'SOL'}]->(b:Wallet) RETURN a.address, b.address, t.amount
$$) AS (src agtype, dst agtype, amount agtype);
```

## Safety

- **No keys.** `SolanaAgentKit(...)` is never constructed: it requires a private key and logs it in plain text at startup (`solana_agent_kit/agent/__init__.py:100` in 1.4.0).
- **Closed tool list.** `tools.ALLOWED` is the whole list. `tests/test_agent.py` fails if any tool name looks like a write action.
- **Only read RPC methods.** A test checks that `rpc.py` calls only `getBalance`, `getTokenAccountsByOwner`, `getSignaturesForAddress` and `getTransaction`.
- **Validated input.** Every address goes through a base58 check (Pydantic) before it reaches the network. Values are checked again before going into a Cypher query, so a crafted string can't break out of it.
- **Copycat tokens.** Tickers aren't unique on Solana. Jupiter's verified list is checked first, and anything unverified comes with a warning.

## About Solana Agent Kit

The project depends on `solana-agent-kit-py` 1.4.0 as asked, but as of 2026-11-09 that package needs workarounds:

| Problem | What SolGraph does |
|---|---|
| `import solana_agent_kit` fails with a circular import (`agent` → `tools/use_3land` → `agent`). GitHub `main` is the same; last commit 2025-02-05. | Loads the one helper it needs (`TokenDataManager`) without running the broken package `__init__` (`market.py`) |
| `fetch_price` calls `api.jup.ag/price/v2`, which returns HTTP 404 | Uses Jupiter `lite-api.jup.ag/price/v3` |
| `get_token_data_by_address` calls `tokens.jup.ag`, which no longer resolves | Uses Jupiter `tokens/v2/search` |
| Its ticker lookup ranks by reported FDV and returned counterfeit USDC and USDT mints | Used only as a fallback after Jupiter's verified list, with a warning |
| Its pins force LangChain 0.3, numpy < 2 and Python 3.13.0, and LangChain 0.3 can't do Gemini 3 tool calls | Installed with `--no-deps`; SolGraph runs on LangChain 1.x and declares the three packages the helper imports |

## RPC cache

Every RPC read goes through a cache keyed by endpoint, method and parameters. How long an answer is kept depends on whether it can change:

| Read | Kept | Why |
|---|---|---|
| `getTransaction` | For good | A finalized transaction never changes |
| `getMultipleAccounts` (account kinds) | 1 hour | Owners almost never change |
| Balances, signature lists | 15 seconds | They change with every transaction; 15 s absorbs the burst one question causes |

The store is a SQLite file in `~/.solgraph/` by default. Set `SOLGRAPH_REDIS_URL` to use Redis instead (`docker compose up -d redis`, then `redis://127.0.0.1:6380/0`; `pip install "solgraph[redis]"`). `SOLGRAPH_CACHE=off` turns it off. `/health` reports hits and misses.

Measured on live devnet, loading the same 20-transaction history twice: **34.8 s cold, 2.7 s warm** with SQLite, and 33.0 s cold, 3.3 s warm with Redis. Most of the warm time is Python start-up.

## Tests

```bash
python -m pytest            # 54 tests against a fake RPC and a fake model, no network
SOLGRAPH_TEST_PG_DSN=postgresql://... SOLGRAPH_TEST_REDIS_URL=redis://... python -m pytest   # also the AGE and Redis tests
```

## Known limits

- Tracing goes back only as far as histories have been loaded. The agent loads a sender's history when a trace stops early.
- `kind` comes from the account's current owner. An account that was closed reads as `unknown` and is followed.
- Free public RPC and Jupiter endpoints rate-limit bursts. Set `SOLGRAPH_RPC_URL` to a private RPC for heavy use.

## Roadmap

1. Approval-gated writes on devnet only: transfer previews, a spending limit, and an explicit yes for every transaction.
2. pgvector document search over Solana docs, so the guide can cite them.

## License

MIT
