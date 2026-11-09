"""The LangChain agent: Claude, the read-only toolbox, and instructions to explain simply."""

from __future__ import annotations

import inspect

from langchain_core.language_models import BaseChatModel
from langgraph.prebuilt import create_react_agent

from .tools import Toolbox, build_tools

SYSTEM = """You are SolGraph, a patient guide to the Solana blockchain for beginners.

You can only READ blockchain data. You cannot send, trade, stake, mint or sign anything, and you
never ask for a private key or seed phrase. If someone asks you to move money, say plainly that
SolGraph is read-only.

How to answer:
- Use the tools for every fact about a wallet or token. Never guess balances, prices or history.
- To explain where tokens came from: call recent_transactions for the wallet, then
  trace_token_origin. If the trace stops early, load the sender's history and trace again.
- Explain in plain language. Define a term the first time you use it (mint, lamport, token
  account, signature). Shorten addresses as the first 4 and last 4 characters.
- Say which network the data is from. Prices only exist for mainnet tokens.
- If a tool returns an error, explain what went wrong and what the person can do about it.
"""


def build_agent(model: BaseChatModel, box: Toolbox, network: str):
    prompt = SYSTEM + f"\nYou are connected to Solana {network}."
    params = inspect.signature(create_react_agent).parameters
    kw = {"prompt": prompt} if "prompt" in params else {"state_modifier": prompt}
    return create_react_agent(model, build_tools(box), **kw)


def default_model(name: str) -> BaseChatModel:
    from langchain_anthropic import ChatAnthropic

    return ChatAnthropic(model=name, max_tokens=2048, temperature=0)


def ask(agent, question: str) -> str:
    out = agent.invoke({"messages": [("user", question)]})
    return out["messages"][-1].content if isinstance(out["messages"][-1].content, str) else \
        "".join(b.get("text", "") for b in out["messages"][-1].content if isinstance(b, dict))
