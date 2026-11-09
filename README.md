# SolGraph

SolGraph is an AI-powered Solana assistant that uses LangChain and Apache AGE to understand, explore and explain relationships between wallets, tokens and transactions.

**It is read-only.** It holds no private keys, has no tool that signs or sends anything, and defaults to devnet. A test fails if a write tool ever reaches the AI.

```
question ─► LangChain agent (Claude) ─► read-only tools ─► Solana RPC     (balances, history)
                                             │        └──► Jupiter / Agent Kit (token names, prices)
                                             ▼
                                   graph: Apache AGE on Postgres (or in memory)
                                   (:Wallet)-[:TRANSFER {mint, amount}]->(:Wallet)
                                   (:Wallet)-[:HOLDS]->(:Token)   (:Wallet)-[:SIGNED]->(:Transaction)
```

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

## Setup

You need Python **3.13.0 exactly**. `solana-agent-kit-py` declares `requires-python ==3.13.0`. [uv](https://docs.astral.sh/uv/) gets you that version in one step:

```bash
uv python install 3.13.0
uv venv -p 3.13.0 .venv
uv pip install -p .venv -e ".[dev]"
cp .env.example .env            # add ANTHROPIC_API_KEY for the chat features
```

On Windows, the first install builds numpy 1.26 from source (the kit pins numpy < 2) and needs the MSVC build tools. It takes a few minutes, once.

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

### Use Apache AGE instead of the in-memory graph

```bash
docker compose up -d            # Postgres 16 + AGE 1.6.0 on localhost:5455
export SOLGRAPH_PG_DSN=postgresql://solgraph:solgraph@localhost:5455/solgraph
```

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
| Its pins force Python 3.13.0, LangChain 0.3, numpy < 2, and an anchorpy pytest plugin that needs pytest-asyncio | Accepted; the plugin is disabled in `pyproject.toml` |

## Tests

```bash
python -m pytest            # 39 tests against a fake RPC and a fake model, no network
SOLGRAPH_TEST_PG_DSN=postgresql://... python -m pytest   # also runs the Apache AGE test
```

## Known limits

- `Wallet` nodes are any account whose balance changed, including program-owned accounts. Telling users from programs is not done yet.
- Tracing goes back only as far as histories have been loaded. The agent loads a sender's history when a trace stops early.
- Free public RPC and Jupiter endpoints rate-limit bursts. Set `SOLGRAPH_RPC_URL` to a private RPC for heavy use.

## Roadmap

1. Approval-gated writes on devnet only: transfer previews, a spending limit, and an explicit yes for every transaction.
2. Redis cache for RPC reads.
3. pgvector document search over Solana docs, so the guide can cite them.
4. A small web UI drawing the graph.
