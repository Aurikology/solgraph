# SolGraph

[![tests](https://github.com/Aurikology/solgraph/actions/workflows/tests.yml/badge.svg)](https://github.com/Aurikology/solgraph/actions/workflows/tests.yml)

SolGraph is an AI-powered Solana assistant that uses LangChain and Apache AGE to understand, explore and explain relationships between wallets, tokens and transactions. For questions about how Solana itself works, it searches the official docs (RAG over pgvector) and cites the page.

**It is read-only.** It holds no private keys, has no tool that signs or sends anything, and defaults to devnet. A test fails if a write tool ever reaches the AI.

```
question ─► LangChain agent (Gemini or Claude) ─► read-only tools ─► Solana RPC     (balances, history)
                                             │        └──► Jupiter / Agent Kit (token names, prices)
                                             ▼
                                   graph: Apache AGE on Postgres (or in memory)
                                   (:Wallet {kind})-[:TRANSFER {mint, amount}]->(:Wallet)
                                   (:Wallet)-[:HOLDS]->(:Token)   (:Wallet)-[:SIGNED]->(:Transaction)

"how do fees work?" ─► search_solana_docs ─► embed query (Gemini) ─► pgvector HNSW (cosine) ─┐
                                                └─► Postgres full-text (tsvector, GIN) ──────┴─► RRF ─► top passages + URLs
                         one Postgres holds both: the AGE graph and the pgvector docs index
```

## Status (v1.3, 2026-11-11)

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
| Docs search (RAG) | 78 tests pass with AGE, pgvector and Redis running (store, chunker, embedder pacing, tool, `/kb/search`); full docs indexed in pgvector; retrieval measured on 35 labelled questions (below); agent answered a live question citing the docs; Docker stack healthy with the app reading the pgvector index (`/health`: `PgVectorStore`) |
| Kraken portfolio | 21 tests against a fake Kraken, including Kraken's own published signing example, the read-only guard, pagination, rate limits and the buy-price maths. **Live on a real account** with a read-only key: all 13 ledger entries read, and every asset's buys, sales, staking rewards and referral bonuses add up to the balance Kraken reports, to the last unit |
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
| "What is rent? How do PDAs work? What does `getSignaturesForAddress` return?" | `search_solana_docs` | The official Solana docs, indexed with pgvector; answers cite the page |
| "How is my SOL doing?" | `my_kraken_portfolio` | Your own Kraken account through a read-only API key (optional) |

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
docker compose up -d age        # Postgres 16 + AGE 1.6.0 + pgvector on 127.0.0.1:5455 (not "localhost": see below)
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
- **Kraken is read-only twice over.** The key you create only gets "Query Funds" and "Query Ledger Entries", and `kraken.py` refuses every private method except `Balance` and `Ledgers` before anything is sent (tested). The key and secret are read from the environment, never stored in settings or printed.
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

## Docs search (RAG)

`search_solana_docs` answers "how does Solana work" questions from the official docs instead of the model's memory, and the guide cites the page it used.

```bash
python -m solgraph docs ingest              # fetch, clean, chunk, embed, index (re-runs embed only what changed)
python -m solgraph docs search "why does an account need a minimum balance" --mode hybrid
python -m solgraph docs eval                # hit@1, recall@5 and MRR on 35 labelled questions, per mode
curl "http://127.0.0.1:8000/kb/search?q=priority+fees&mode=hybrid"
```

How it works (`solgraph/rag.py`):

- **Corpus.** The `intro`, `core`, `tokens` and `rpc/http` sections of [solana-foundation/solana-com](https://github.com/solana-foundation/solana-com), fetched at a pinned commit: 149 pages, 1,223 chunks. The docs are GPL-3.0, so they are fetched at ingest time and never committed here.
- **Cleaning and chunking.** MDX front matter, JSX components and client code samples are stripped. Pages are split at `##`/`###` headings, long sections again at paragraph breaks with one paragraph of overlap, and each chunk is embedded with its "Page > Section" title.
- **Embeddings.** `gemini-embedding-2` (with a free key, documents and queries embedded with their own task types, cut to 768 dimensions) or `bge-base-en-v1.5` run locally through ONNX (`pip install "solgraph[local]"`, no key). Pick with `SOLGRAPH_EMBED_MODEL`. Both fit one `vector(768)` column, and each row records its model, so the two indexes live side by side.
- **Store.** pgvector in the same Postgres as the Apache AGE graph (`db/Dockerfile` adds pgvector 0.8 to the AGE image). An HNSW index serves cosine search, a generated, weighted `tsvector` column with a GIN index serves full-text search. Without Postgres the index is a JSON file in `~/.solgraph/` with brute-force cosine and BM25.
- **Hybrid retrieval.** The top 40 by vector and the top 40 by full text are fused with Reciprocal Rank Fusion (k = 60) in one SQL statement. Embeddings catch paraphrases; full text catches exact names such as `getSignaturesForAddress`.
- **Incremental ingest.** A chunk's id hashes its model, page, heading and text, so a re-run embeds only new or changed chunks and deletes the stale ones.
- **Free-tier pacing.** Gemini's free tier counts every text as a request: 100 a minute and 1,000 a day per model. The embedder paces itself to 90 a minute (`SOLGRAPH_EMBED_PER_MINUTE`) and honours the API's retry delay.

### Measured retrieval quality

`evals/docs_retrieval.json` holds 35 questions written the way a beginner would ask them, each labelled with the pages that answer it (written before any results were seen). Against the full index (1,217 chunks, `bge-base-en-v1.5`, pgvector, top 5, one chunk per page):

| Mode | hit@1 | recall@5 | MRR |
|---|---|---|---|
| Hybrid (vector + full text, RRF) | **0.514** | 0.857 | 0.640 |
| Vector only (HNSW, cosine) | 0.486 | **0.914** | **0.651** |
| Keyword only (Postgres full text) | 0.143 | 0.629 | 0.320 |

Keeping one chunk per page raised recall@5 from 0.800 to 0.857 (hybrid) and from 0.829 to 0.914 (vector). Plain-language questions favour the embeddings: keyword search alone misses paraphrases, and on this set fusing it in helps the top hit but costs some recall. Hybrid stays the default because exact identifiers (`getSignaturesForAddress`) are where full text wins, and this set has few of them. The Gemini index has 900 of 1,223 chunks so far: the free tier's 1,000 embeddings a day ran out mid-ingest, and `docs ingest` will finish it after the reset, embedding only the rest.

Live: asked "why does a new account need a minimum SOL balance, and how is that amount worked out?", the agent (Gemini 3.5 Flash) called `search_solana_docs` and answered citing Account Structure, Modification Rules and Create a Token Account.

## Your Kraken account (optional)

Coins on an exchange are held in the exchange's books, not in an on-chain wallet under your name, so the blockchain tools can't see them. `my_kraken_portfolio` reads them from Kraken's API instead: what you hold (spot, staked, Earn), its value now, your average buy price, staking rewards, and deposits and withdrawals. The API is free; Kraken only charges for trades, and this key can't trade.

1. On Kraken: **Settings → API → Create API key**. Tick only **Query Funds** and **Query Ledger Entries**. Leave every trade, deposit, withdraw and Earn box unticked.
2. Add both values to `.env`:
   ```
   KRAKEN_API_KEY=...
   KRAKEN_API_SECRET=...
   SOLGRAPH_KRAKEN_QUOTE=CAD     # currency for values; USD is the fallback
   ```
3. `python -m solgraph kraken --asset SOL`, or ask the agent "how is my SOL doing?"

How the numbers are worked out (`solgraph/kraken.py`): ledger entries are paged in 50 at a time (up to 1,000, paced under Kraken's rate limit). Buys and sells are the trade and Buy Crypto entries that share a reference id, one crypto leg and one cash leg, with fees added to the cost. Crypto-to-crypto swaps and coins deposited from elsewhere have no cash price, so they don't move the average. It's an estimate for your own interest, not an adjusted cost base for taxes.

Privacy: what this tool returns goes to the chat model, so the model provider sees your balances. Use the `kraken` CLI command if you'd rather keep them local.

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
python -m pytest            # against a fake RPC, a fake model and a hashing embedder, no network
SOLGRAPH_TEST_PG_DSN=postgresql://... SOLGRAPH_TEST_REDIS_URL=redis://... python -m pytest   # also AGE, pgvector and Redis: 99 tests
```

## Known limits

- Tracing goes back only as far as histories have been loaded. The agent loads a sender's history when a trace stops early.
- `kind` comes from the account's current owner. An account that was closed reads as `unknown` and is followed.
- Free public RPC and Jupiter endpoints rate-limit bursts. Set `SOLGRAPH_RPC_URL` to a private RPC for heavy use.

## Roadmap

1. Approval-gated writes on devnet only: transfer previews, a spending limit, and an explicit yes for every transaction.
2. A reranking pass over the fused docs results, measured against the same eval set.

## License

MIT
