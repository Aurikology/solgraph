import json

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from solgraph.agent import ask, build_agent
from solgraph.tools import ALLOWED, build_tools

from .conftest import ALICE, BOB, MINT

WRITE_WORDS = ("trade", "transfer", "send", "stake", "deploy", "mint_", "swap", "lend", "burn",
               "buy", "sell", "launch", "create", "faucet", "airdrop", "sign")


def test_toolset_is_exactly_the_read_only_allowlist(box):
    names = {t.name for t in build_tools(box)}
    assert names == ALLOWED
    assert not [n for n in names for w in WRITE_WORDS if w in n]


def test_bad_address_is_explained_without_a_network_call(box):
    calls_before = box.rpc._id
    tool = {t.name: t for t in build_tools(box)}["wallet_balances"]
    out = json.loads(tool.invoke({"address": "not-a-wallet"}))
    assert "not a Solana address" in out["error"]
    assert box.rpc._id == calls_before


def test_tools_end_to_end(box):
    tools = {t.name: t for t in build_tools(box)}
    bal = json.loads(tools["wallet_balances"].invoke({"address": ALICE}))
    assert bal["sol"] == 1.1 and bal["tokens"][0]["mint"] == MINT
    hist = json.loads(tools["recent_transactions"].invoke({"address": BOB, "limit": 5}))
    assert hist["loaded_transactions"] == 2
    tools["recent_transactions"].invoke({"address": ALICE})
    trace = json.loads(tools["trace_token_origin"].invoke({"address": BOB, "mint": MINT}))
    assert len(trace) == 2
    assert json.loads(tools["check_wallet_changes"].invoke({"address": ALICE}))["first_snapshot"]
    assert json.loads(tools["check_wallet_changes"].invoke({"address": ALICE}))["changes"] == []
    assert json.loads(tools["token_info"].invoke({"query": "USDC"}))["price_usd"] == 1.0
    assert json.loads(tools["graph_stats"].invoke({}))["transfers"] == 4


def test_limits_are_enforced(box):
    tool = {t.name: t for t in build_tools(box)}["recent_transactions"]
    assert "Invalid input" in tool.invoke({"address": ALICE, "limit": 500})


class ToolCallingFake(GenericFakeChatModel):
    def bind_tools(self, tools, **kw):
        return self


def test_agent_calls_a_tool_and_answers(box):
    model = ToolCallingFake(messages=iter([
        AIMessage(content="", tool_calls=[{"name": "wallet_balances", "args": {"address": ALICE},
                                           "id": "c1", "type": "tool_call"}]),
        AIMessage(content="That wallet holds 1.1 SOL and 75 USDC."),
    ]))
    agent = build_agent(model, box, "devnet")
    assert ask(agent, "what does this wallet hold?") == "That wallet holds 1.1 SOL and 75 USDC."
    assert box.graph.holdings(ALICE)["SOL"] == 1.1


def test_model_is_picked_from_the_key(monkeypatch):
    from solgraph.config import Settings

    for k in ("SOLGRAPH_MODEL", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "x")
    assert Settings.load().model.startswith("gemini")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "y")
    assert Settings.load().model.startswith("claude")
    monkeypatch.setenv("SOLGRAPH_MODEL", "gemini-2.5-pro")
    assert Settings.load().model == "gemini-2.5-pro"


def test_gemini_model_builds(monkeypatch):
    from solgraph.agent import default_model

    monkeypatch.setenv("GOOGLE_API_KEY", "x")
    assert type(default_model("gemini-2.5-flash")).__name__ == "ChatGoogleGenerativeAI"
