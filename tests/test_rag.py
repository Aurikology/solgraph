import json
import math
import os

import httpx
import pytest
from fastapi.testclient import TestClient

import solgraph.api as api
from solgraph.app import App
from solgraph.config import Settings
from solgraph.rag import (_retry_after, DocsIndex, GeminiEmbedder, HashEmbedder, LocalDocStore, Page,
                          PgVectorStore, chunk_page, clean_mdx, evaluate, page_url, rrf)
from solgraph.tools import build_tools

RAW = """---
title: Fee Structure
description: How fees work.
url: /docs/core/fees/fee-structure
---

{/* TOC: Base fee, Prioritization fee */}

<Callout type="info" title="Summary">
  Total fee = base fee + priority fee.
</Callout>

## Base fee

Every signature costs [5,000 lamports](https://github.com/anza-xyz/x/lib.rs#L1).

```ts title="Kit"
const fee = await rpc.getFeeForMessage(msg).send();
```

```text title="Base fee formula"
base_fee = signature_count * lamports_per_signature
```
"""

PAGES = [
    Page("https://solana.com/docs/core/fees", "Fees",
         "Intro to fees.\n\n## Base fee\n\nEvery transaction pays a base fee of 5000 lamports per "
         "signature. Half of the base fee is burned.\n\n## Priority fee\n\nA priority fee is "
         "compute unit price times compute unit limit, paid to the validator."),
    Page("https://solana.com/docs/core/accounts", "Accounts",
         "## Rent\n\nAn account must hold a minimum balance of lamports to be rent exempt, "
         "proportional to the bytes of data it stores."),
    Page("https://solana.com/docs/rpc/http/getsignaturesforaddress", "getSignaturesForAddress",
         "## Parameters\n\nReturns signatures for confirmed transactions that include the given "
         "address in their accountKeys list, newest first."),
]

CASES = [
    {"q": "base fee per signature", "urls": ["https://solana.com/docs/core/fees"]},
    {"q": "minimum balance rent exempt", "urls": ["https://solana.com/docs/core/accounts"]},
    {"q": "getSignaturesForAddress", "urls": ["https://solana.com/docs/rpc/http/getsignaturesforaddress"]},
]


def test_clean_mdx_keeps_prose_and_drops_components_and_client_code():
    title, text = clean_mdx(RAW)
    assert title == "Fee Structure"
    assert "description:" not in text and "TOC" not in text and "<Callout" not in text
    assert "Total fee = base fee + priority fee." in text
    assert "Every signature costs 5,000 lamports." in text  # link text kept, target dropped
    assert "getFeeForMessage" not in text  # client code sample dropped
    assert "base_fee = signature_count" in text  # formula kept


def test_page_urls():
    root = "apps/docs/content/docs/en/"
    assert page_url(root + "core/fees/fee-structure.mdx") == "https://solana.com/docs/core/fees/fee-structure"
    assert page_url(root + "core/fees/index.mdx") == "https://solana.com/docs/core/fees"
    assert page_url(root + "index.mdx") == "https://solana.com/docs/"


def test_chunks_follow_headings_and_ids_track_text_and_model():
    chunks = chunk_page(PAGES[0], "m1")
    assert [c.section for c in chunks] == ["Base fee", "Priority fee"]  # "Intro to fees." too short
    assert chunks[0].embed_text().startswith("Fees > Base fee\n\n")
    assert [c.id for c in chunk_page(PAGES[0], "m1")] == [c.id for c in chunks]
    assert chunk_page(PAGES[0], "m2")[0].id != chunks[0].id


def test_long_sections_split_with_overlap():
    body = "\n\n".join(f"Paragraph {i} " + "word " * 60 for i in range(12))
    chunks = chunk_page(Page("u", "T", "## Long\n\n" + body), "m", max_chars=1000)
    assert len(chunks) > 3 and all(len(c.text) <= 1400 for c in chunks)
    first_of_second = chunks[1].text.split("\n\n")[0]
    assert first_of_second in chunks[0].text  # the last paragraph carries over


def test_rrf():
    s = rrf(["a", "b"], ["b", "c"], k=60)
    assert s["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert max(s, key=s.get) == "b"


@pytest.fixture
def index(tmp_path):
    idx = DocsIndex(LocalDocStore(tmp_path / "docs.json"), HashEmbedder())
    idx.ingest(PAGES)
    return idx


@pytest.mark.parametrize("mode", ["hybrid", "vector", "keyword"])
def test_search_finds_the_right_page(index, mode):
    for case in CASES:
        assert index.search(case["q"], 3, mode)[0].url == case["urls"][0], (mode, case["q"])


def test_reingest_embeds_only_what_changed(index, tmp_path):
    rep = index.ingest(PAGES)
    assert rep.embedded == 0 and rep.removed == 0
    changed = PAGES[:1] + [Page(PAGES[1].url, "Accounts", "## Rent\n\nRent is now explained differently, "
                                                         "with a new paragraph of words.")]
    rep = index.ingest(changed)
    assert rep.embedded == 1 and rep.removed == 2  # new rent chunk in; old rent + RPC page out
    reopened = DocsIndex(LocalDocStore(tmp_path / "docs.json"), HashEmbedder())
    assert reopened.count() == rep.chunks == 3


def test_evaluate_reports_hit_recall_and_mrr(index):
    res = evaluate(index, CASES + [{"q": "staking rewards", "urls": ["https://nowhere"]}], k=3)
    assert set(res) == {"hybrid", "vector", "keyword"}
    h = res["hybrid"]
    assert h["hit@1"] == 0.75 and h["recall@3"] == 0.75 and h["mrr"] == 0.75
    assert h["misses"] == ["staking rewards"]


def test_docs_tool(box, index):
    tool = {t.name: t for t in build_tools(box)}["search_solana_docs"]
    assert "not configured" in json.loads(tool.invoke({"query": "rent"}))["error"]
    box.docs_error = "Docs search needs GOOGLE_API_KEY"
    assert json.loads(tool.invoke({"query": "rent"}))["error"] == box.docs_error
    box.docs = index
    out = json.loads(tool.invoke({"query": "rent exempt minimum balance"}))
    assert out["results"][0]["url"] == "https://solana.com/docs/core/accounts"
    assert "cite the url" in out["note"]
    assert "Invalid input" in tool.invoke({"query": "x"})


def test_docs_tool_on_an_empty_index(box, tmp_path):
    box.docs = DocsIndex(LocalDocStore(tmp_path / "empty.json"), HashEmbedder())
    tool = {t.name: t for t in build_tools(box)}["search_solana_docs"]
    assert "docs ingest" in json.loads(tool.invoke({"query": "rent"}))["error"]


def test_kb_route(box, index):
    box.docs = index
    api._state["app"] = App(Settings("devnet", "http://fake", None, "solgraph", "test"),
                            box.rpc, box.graph, box)
    try:
        c = TestClient(api.app)
        r = c.get("/kb/search", params={"q": "priority fee compute unit", "mode": "keyword"}).json()
        assert r["results"][0]["url"] == "https://solana.com/docs/core/fees"
        assert c.get("/kb/search", params={"q": "fee", "mode": "nope"}).status_code == 422
        assert c.get("/health").json()["docs_chunks"] == index.count()
        box.docs = None
        assert c.get("/kb/search", params={"q": "fees"}).status_code == 503
    finally:
        api._state.clear()


def test_gemini_embedder_batches_retries_and_normalizes():
    seen = []
    state = {"limited": 1}

    def handler(req):
        if state["limited"]:
            state["limited"] -= 1
            return httpx.Response(429, json={})
        body = json.loads(req.content)
        seen.append((len(body["requests"]), body["requests"][0]["taskType"],
                     body["requests"][0]["outputDimensionality"], req.headers["x-goog-api-key"]))
        return httpx.Response(200, json={"embeddings": [{"values": [3.0, 4.0]} for _ in body["requests"]]})

    e = GeminiEmbedder("k", batch=2, backoff=0, per_minute=None, client=httpx.Client(transport=httpx.MockTransport(handler)))
    vecs = e.embed_documents(["a", "b", "c"])
    assert len(vecs) == 3 and vecs[0] == [0.6, 0.8]
    assert seen == [(2, "RETRIEVAL_DOCUMENT", 768, "k"), (1, "RETRIEVAL_DOCUMENT", 768, "k")]
    e.embed_query("q")
    assert seen[-1][1] == "RETRIEVAL_QUERY"
    assert math.isclose(sum(x * x for x in vecs[2]), 1.0)


def test_gemini_without_a_key_is_a_clear_error(monkeypatch):
    from solgraph.app import open_docs

    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    s = Settings("devnet", "http://fake", None, "solgraph", "test", embed_model="gemini-embedding-2")
    idx, err = open_docs(s)
    assert idx is None and "GOOGLE_API_KEY" in err and "bge-base-en-v1.5" in err


def test_embed_model_defaults_to_local_without_a_google_key(monkeypatch):
    for k in ("SOLGRAPH_EMBED_MODEL", "GOOGLE_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.chdir(os.path.dirname(__file__))  # no .env here
    assert Settings.load().embed_model == "bge-base-en-v1.5"
    monkeypatch.setenv("GOOGLE_API_KEY", "x")
    assert Settings.load().embed_model == "gemini-embedding-2"


DSN = os.environ.get("SOLGRAPH_TEST_PG_DSN")


@pytest.mark.skipif(not DSN, reason="set SOLGRAPH_TEST_PG_DSN to a Postgres with pgvector")
@pytest.mark.parametrize("mode", ["hybrid", "vector", "keyword"])
def test_pgvector_store_matches_local(mode):
    store = PgVectorStore(DSN, table="solgraph_docs_test")
    with store.conn.cursor() as cur:
        cur.execute("TRUNCATE public.solgraph_docs_test")
    idx = DocsIndex(store, HashEmbedder())
    rep = idx.ingest(PAGES)
    assert rep.embedded == idx.count() == 4
    for case in CASES:
        assert idx.search(case["q"], 3, mode)[0].url == case["urls"][0], (mode, case["q"])
    assert idx.ingest(PAGES).embedded == 0
    assert idx.ingest(PAGES[:1]).removed == 2


def test_retry_delay_is_read_from_the_error():
    assert _retry_after(httpx.Response(429, text="Please retry in 50.7s.")) == pytest.approx(51.7)
    assert _retry_after(httpx.Response(429, headers={"retry-after": "3"})) == 4
    assert _retry_after(httpx.Response(429, json={})) is None


def test_pacing_waits_when_the_minute_is_full(monkeypatch):
    import solgraph.rag as rag

    clock = {"t": 1000.0}
    monkeypatch.setattr(rag.time, "monotonic", lambda: clock["t"])
    monkeypatch.setattr(rag.time, "sleep", lambda s: clock.__setitem__("t", clock["t"] + s))
    e = GeminiEmbedder("k", per_minute=50)
    e._pace(30)
    e._pace(20)
    assert clock["t"] == 1000.0  # 50 in the window: still allowed
    e._pace(10)
    assert clock["t"] >= 1060.0  # had to wait for the first batch to age out


def test_search_returns_one_chunk_per_page(index):
    hits = index.search("fee", 5, "keyword")
    assert len({h.url for h in hits}) == len(hits)
