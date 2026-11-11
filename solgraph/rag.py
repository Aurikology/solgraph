"""Retrieval over the official Solana docs, so the guide can cite a page instead of guessing.

    fetch (GitHub, pinned commit) -> clean MDX -> chunk by heading -> embed (Gemini)
        -> store (pgvector on the same Postgres as the graph, or a local file)
    question -> embed -> hybrid search: vector (cosine, HNSW) + full-text, fused by RRF -> top k

Why hybrid: embeddings find paraphrases ("how much does a transaction cost" -> Fee Structure),
full-text finds exact names an embedding blurs (`getSignaturesForAddress`, `Token-2022`).
Reciprocal Rank Fusion merges the two ranked lists without having to calibrate their scores
against each other. `evaluate` measures all three modes on a labelled question set.

The docs are fetched at ingest time and kept locally; they are not part of this repository
(solana-foundation/solana-com is GPL-3.0). Chunk ids hash the text and the embedding model, so
re-running ingest embeds only what changed.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import httpx

DOCS_REPO = "solana-foundation/solana-com"
DOCS_ROOT = "apps/docs/content/docs/en/"
DEFAULT_SECTIONS = ("intro", "core", "tokens", "rpc/http")
SITE = "https://solana.com/docs/"
DIM = 768
RRF_K = 60  # the constant from the original RRF paper (Cormack et al., 2009)
DEFAULT_PATH = Path.home() / ".solgraph" / "docs-index.json"


@dataclass
class Page:
    url: str
    title: str
    text: str


@dataclass
class Chunk:
    id: str
    url: str
    title: str
    section: str
    text: str

    def embed_text(self) -> str:
        return f"{self.title} > {self.section}\n\n{self.text}" if self.section else \
            f"{self.title}\n\n{self.text}"


@dataclass
class Hit:
    url: str
    title: str
    section: str
    text: str
    score: float


# --- fetch and clean -----------------------------------------------------------------------------

def list_doc_files(sections=DEFAULT_SECTIONS, ref: str = "main",
                   client: httpx.Client | None = None) -> tuple[str, list[str]]:
    """The commit `ref` resolves to, and the .mdx paths under the chosen sections."""
    c = client or httpx.Client(timeout=30)
    r = c.get(f"https://api.github.com/repos/{DOCS_REPO}/git/trees/{ref}", params={"recursive": 1})
    r.raise_for_status()
    tree = r.json()
    prefixes = tuple(DOCS_ROOT + s.strip("/") + "/" for s in sections)
    paths = sorted(t["path"] for t in tree["tree"] if t["path"].endswith(".mdx")
                   and t["path"].startswith(prefixes))
    return tree["sha"], paths


def page_url(path: str) -> str:
    rel = path[len(DOCS_ROOT):-len(".mdx")]
    rel = re.sub(r"(^|/)index$", "", rel)
    return SITE + rel


_FRONT = re.compile(r"\A---\n(.*?)\n---\n", re.S)
_COMMENT = re.compile(r"\{/\*.*?\*/\}", re.S)
_FENCE = re.compile(r"^```([^\n]*)\n(.*?)^```[ \t]*$", re.S | re.M)
_TAG = re.compile(r"</?[A-Z][A-Za-z0-9.]*(?:\s[^<>]*?)?/?>")
_LINK = re.compile(r"\[([^\]]+)\]\((?:[^()]|\([^)]*\))*\)")
_KEEP_CODE = {"text", "json", "jsonc", ""}


def clean_mdx(raw: str) -> tuple[str, str]:
    """(title, plain markdown). Drops front matter, JSX comments and components, client code
    samples in five languages (the prose explains them), and link targets."""
    title = ""
    m = _FRONT.match(raw)
    if m:
        t = re.search(r"^title:\s*(.+)$", m.group(1), re.M)
        title = t.group(1).strip().strip("\"'") if t else ""
        raw = raw[m.end():]
    raw = _COMMENT.sub("", raw)

    def fence(mm: re.Match) -> str:
        lang = mm.group(1).split()[0].lower() if mm.group(1).strip() else ""
        body = mm.group(2)
        if lang not in _KEEP_CODE or body.count("\n") > 25:
            return ""
        return "```\n" + re.sub(r"^\s*// !.*\n", "", body, flags=re.M) + "```"

    raw = _FENCE.sub(fence, raw)
    raw = _TAG.sub("", raw)
    raw = _LINK.sub(r"\1", raw)
    raw = re.sub(r"\n[ \t]+\n", "\n\n", raw)
    raw = re.sub(r"\n{3,}", "\n\n", raw).strip()
    return title, raw


def fetch_pages(sections=DEFAULT_SECTIONS, ref: str = "main", client: httpx.Client | None = None,
                progress=None) -> tuple[str, list[Page]]:
    c = client or httpx.Client(timeout=30)
    sha, paths = list_doc_files(sections, ref, c)
    pages = []
    for i, p in enumerate(paths, 1):
        r = c.get(f"https://raw.githubusercontent.com/{DOCS_REPO}/{sha}/{p}")
        r.raise_for_status()
        title, text = clean_mdx(r.text)
        if text:
            pages.append(Page(page_url(p), title or p.rsplit("/", 1)[-1], text))
        if progress:
            progress(i, len(paths))
    return sha, pages


# --- chunk ---------------------------------------------------------------------------------------

def chunk_page(page: Page, model: str, max_chars: int = 1600) -> list[Chunk]:
    """Split at ## and ### headings so each chunk is about one thing; a long section is split
    again at paragraph breaks, carrying the last paragraph over so no fact loses its context."""
    out: list[Chunk] = []
    parts = re.split(r"^(#{2,3} .+)$", page.text, flags=re.M)
    sections = [("", parts[0])] + [(parts[i].lstrip("# ").strip(), parts[i + 1])
                                   for i in range(1, len(parts) - 1, 2)]
    for heading, body in sections:
        body = body.strip()
        if len(body) < 40:
            continue
        paras = [p for p in re.split(r"\n\s*\n", body) if p.strip()]
        buf: list[str] = []
        for para in paras:
            if buf and sum(len(b) for b in buf) + len(para) > max_chars:
                out.append(_chunk(page, heading, "\n\n".join(buf), model))
                buf = buf[-1:] if len(buf[-1]) < max_chars // 3 else []
            buf.append(para[: max_chars * 2])
        if buf:
            out.append(_chunk(page, heading, "\n\n".join(buf), model))
    return out


def _chunk(page: Page, heading: str, text: str, model: str) -> Chunk:
    cid = hashlib.sha256(f"{model}\n{page.url}\n{heading}\n{text}".encode()).hexdigest()[:32]
    return Chunk(cid, page.url, page.title, heading, text)


# --- embed ---------------------------------------------------------------------------------------

class Embedder(Protocol):
    model: str
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


def _normalize(v: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def _retry_after(r: httpx.Response) -> float | None:
    if r.headers.get("retry-after", "").isdigit():
        return float(r.headers["retry-after"]) + 1
    m = re.search(r"retry in ([0-9.]+)s", r.text)
    return float(m.group(1)) + 1 if m else None


class GeminiEmbedder:
    """Google's embedding API (free tier with GOOGLE_API_KEY). Documents and queries are embedded
    with different task types, which Google trains for retrieval. Vectors are cut to 768
    dimensions (Matryoshka truncation) and re-normalised, which keeps them under pgvector's HNSW
    limit of 2,000 dimensions at a quarter of the storage."""

    URL = "https://generativelanguage.googleapis.com/v1beta/models/{m}:batchEmbedContents"

    def __init__(self, api_key: str, model: str = "gemini-embedding-2", batch: int = 25,
                 client: httpx.Client | None = None, backoff: float = 5.0,
                 per_minute: int | None = 90):
        self.model, self.key, self.batch, self.backoff = model, api_key, batch, backoff
        self.per_minute = per_minute  # the free tier allows 100 texts a minute; stay under it
        self.sent: list[tuple[float, int]] = []
        self.client = client or httpx.Client(timeout=60)

    def _pace(self, n: int) -> None:
        """Wait until sending n more texts keeps the last 60 seconds under the limit. Pacing up
        front matters: a rejected request seems to count against the quota too, so retrying a
        large batch can keep the window full forever."""
        if not self.per_minute:
            return
        while True:
            now = time.monotonic()
            self.sent = [(t, k) for t, k in self.sent if now - t < 60]
            if sum(k for _, k in self.sent) + n <= self.per_minute:
                self.sent.append((now, n))
                return
            time.sleep(60 - (now - self.sent[0][0]) + 0.5)

    def _embed(self, texts: list[str], task: str) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch):
            reqs = [{"model": f"models/{self.model}", "content": {"parts": [{"text": t}]},
                     "taskType": task, "outputDimensionality": DIM} for t in texts[i:i + self.batch]]
            for attempt in range(8):
                self._pace(len(reqs))
                r = self.client.post(self.URL.format(m=self.model), json={"requests": reqs},
                                     headers={"x-goog-api-key": self.key})
                if r.status_code not in (429, 500, 503):
                    break
                # the free tier counts every text as a request (100 a minute) and says how long
                # to wait; honour that, else back off exponentially
                time.sleep(_retry_after(r) or self.backoff * 2 ** min(attempt, 4))
            if r.status_code != 200:
                raise RuntimeError(f"Embedding request failed: HTTP {r.status_code} {r.text[:200]}")
            out += [_normalize(e["values"]) for e in r.json()["embeddings"]]
        return out

    def embed_documents(self, texts):
        return self._embed(texts, "RETRIEVAL_DOCUMENT")

    def embed_query(self, text):
        return self._embed([text], "RETRIEVAL_QUERY")[0]


class LocalEmbedder:
    """BAAI bge-base-en-v1.5 through fastembed (ONNX, CPU): 768 dimensions like the Gemini index,
    no API key and no quota. `pip install "solgraph[local]"`; the model (about 200 MB) downloads
    once into ~/.solgraph/models. Queries get BGE's retrieval instruction prefix, passages don't."""

    NAMES = {"bge-base-en-v1.5": "BAAI/bge-base-en-v1.5"}

    def __init__(self, model: str = "bge-base-en-v1.5", cache_dir: Path | str | None = None):
        from fastembed import TextEmbedding  # optional dependency

        self.model = model
        self._m = TextEmbedding(self.NAMES.get(model, model),
                                cache_dir=str(cache_dir or Path.home() / ".solgraph" / "models"))

    def embed_documents(self, texts):
        return [_normalize(v.tolist()) for v in self._m.passage_embed(texts, batch_size=32)]

    def embed_query(self, text):
        return _normalize(next(iter(self._m.query_embed([text]))).tolist())


def make_embedder(model: str, google_key: str | None = None, per_minute: int | None = 90) -> Embedder:
    """`gemini-*` uses Google's API; anything else is a local fastembed model."""
    if model.startswith("gemini"):
        if not google_key:
            raise ValueError(f"{model} needs GOOGLE_API_KEY (free: https://aistudio.google.com/apikey), "
                             "or set SOLGRAPH_EMBED_MODEL=bge-base-en-v1.5 to embed locally")
        return GeminiEmbedder(google_key, model, per_minute=per_minute)
    return LocalEmbedder(model)


_WORD = re.compile(r"[a-z0-9]+")


class HashEmbedder:
    """Feature hashing of words: deterministic and offline. For tests, not for real retrieval."""

    model = "hash-test"

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * DIM
        for w, n in Counter(_WORD.findall(text.lower())).items():
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % DIM] += n
        return _normalize(v)

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


# --- store and search ----------------------------------------------------------------------------

MODES = ("hybrid", "vector", "keyword")


def rrf(*rankings: list[str], k: int = RRF_K) -> dict[str, float]:
    """Reciprocal Rank Fusion: score(d) = sum over lists of 1 / (k + rank of d in that list)."""
    score: dict[str, float] = {}
    for ranking in rankings:
        for rank, cid in enumerate(ranking, 1):
            score[cid] = score.get(cid, 0.0) + 1.0 / (k + rank)
    return score


class DocStore(Protocol):
    def known_ids(self, model: str) -> set[str]: ...
    def upsert(self, chunks: list[Chunk], vectors: list[list[float]], model: str) -> None: ...
    def prune(self, keep: set[str], model: str) -> int: ...
    def search(self, qvec, qtext: str, model: str, k: int, mode: str) -> list[Hit]: ...
    def count(self, model: str | None = None) -> int: ...


def _terms(text: str) -> list[str]:
    return [w for w in _WORD.findall(text.lower()) if len(w) > 1]


class LocalDocStore:
    """A JSON file in ~/.solgraph. Brute-force cosine plus BM25 for the keyword side; fine for a
    few thousand chunks. Use pgvector for anything shared or larger."""

    def __init__(self, path: Path | str = DEFAULT_PATH):
        self.path = Path(path)
        self.rows: dict[str, dict] = {}
        if self.path.exists():
            self.rows = json.loads(self.path.read_text(encoding="utf-8"))

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.rows), encoding="utf-8")

    def known_ids(self, model):
        return {i for i, r in self.rows.items() if r["model"] == model}

    def upsert(self, chunks, vectors, model):
        for c, v in zip(chunks, vectors):
            self.rows[c.id] = asdict(c) | {"model": model, "vec": [round(x, 6) for x in v]}
        self._save()

    def prune(self, keep, model):
        gone = [i for i, r in self.rows.items() if r["model"] == model and i not in keep]
        for i in gone:
            del self.rows[i]
        self._save()
        return len(gone)

    def count(self, model=None):
        return sum(1 for r in self.rows.values() if model is None or r["model"] == model)

    def search(self, qvec, qtext, model, k=5, mode="hybrid", candidates=40):
        rows = [r for r in self.rows.values() if r["model"] == model]
        by_vec = sorted(rows, key=lambda r: -sum(a * b for a, b in zip(qvec, r["vec"])))
        vec_rank = [r["id"] for r in by_vec[:candidates]]
        kw_rank = self._bm25(rows, qtext)[:candidates]
        ranked = {"vector": rrf(vec_rank), "keyword": rrf(kw_rank),
                  "hybrid": rrf(vec_rank, kw_rank)}[mode]
        best = sorted(ranked, key=ranked.get, reverse=True)[:k]
        return [Hit(self.rows[i]["url"], self.rows[i]["title"], self.rows[i]["section"],
                    self.rows[i]["text"], round(ranked[i], 5)) for i in best]

    @staticmethod
    def _bm25(rows, qtext, k1=1.5, b=0.75) -> list[str]:
        q = set(_terms(qtext))
        if not q or not rows:
            return []
        docs = [(r["id"], Counter(_terms(f'{r["title"]} {r["section"]} {r["text"]}'))) for r in rows]
        avg = sum(sum(tf.values()) for _, tf in docs) / len(docs)
        df = Counter(t for _, tf in docs for t in q if t in tf)
        scored = []
        for cid, tf in docs:
            length = sum(tf.values())
            s = sum(math.log(1 + (len(docs) - df[t] + 0.5) / (df[t] + 0.5)) * tf[t] * (k1 + 1)
                    / (tf[t] + k1 * (1 - b + b * length / avg)) for t in q if t in tf)
            if s > 0:
                scored.append((s, cid))
        return [cid for _, cid in sorted(scored, reverse=True)]


def _vec_literal(v: list[float]) -> str:
    return "[" + ",".join(f"{x:.6f}" for x in v) + "]"


class PgVectorStore:
    """pgvector in the same PostgreSQL as the Apache AGE graph. An HNSW index serves the cosine
    search, a generated tsvector column with a GIN index serves the full-text search, and one SQL
    statement fuses the two rankings with RRF."""

    def __init__(self, dsn: str, table: str = "solgraph_docs", connect_timeout: int = 10):
        import psycopg

        if not re.fullmatch(r"[a-z_][a-z0-9_]*", table):
            raise ValueError("table name must be lowercase letters, digits and _")
        self.t = table
        try:
            self.conn = psycopg.connect(dsn, autocommit=True, connect_timeout=connect_timeout)
        except psycopg.OperationalError as e:
            raise ConnectionError(
                f"Could not connect to Postgres for the docs index ({e.__class__.__name__}). Is "
                "the database running (`docker compose up -d age`)? Use 127.0.0.1, not localhost.") from e
        with self.conn.cursor() as cur:
            # SCHEMA public: AGE makes a schema named after the graph, and with the default
            # search_path ("$user", public) the extension would otherwise land inside it
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector SCHEMA public")
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS public.{table} (
                    id text PRIMARY KEY, model text NOT NULL,
                    url text NOT NULL, title text NOT NULL, section text NOT NULL,
                    body text NOT NULL, embedding public.vector({DIM}) NOT NULL,
                    tsv tsvector GENERATED ALWAYS AS (
                        setweight(to_tsvector('english', title || ' ' || section), 'A') ||
                        setweight(to_tsvector('english', body), 'B')) STORED)""")
            cur.execute(f"CREATE INDEX IF NOT EXISTS {table}_hnsw ON public.{table} "
                        "USING hnsw (embedding public.vector_cosine_ops)")
            cur.execute(f"CREATE INDEX IF NOT EXISTS {table}_tsv ON public.{table} USING gin (tsv)")
            # a filtered HNSW scan can return fewer rows than asked for; pgvector 0.8 can keep
            # scanning until the LIMIT is met
            cur.execute("SET hnsw.ef_search = 100")
            cur.execute("SET hnsw.iterative_scan = relaxed_order")

    def known_ids(self, model):
        with self.conn.cursor() as cur:
            cur.execute(f"SELECT id FROM public.{self.t} WHERE model = %s", (model,))
            return {r[0] for r in cur.fetchall()}

    def upsert(self, chunks, vectors, model):
        with self.conn.cursor() as cur:
            cur.executemany(
                f"INSERT INTO public.{self.t} (id, model, url, title, section, body, embedding) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s::public.vector) ON CONFLICT (id) DO NOTHING",
                [(c.id, model, c.url, c.title, c.section, c.text, _vec_literal(v))
                 for c, v in zip(chunks, vectors)])

    def prune(self, keep, model):
        with self.conn.cursor() as cur:
            cur.execute(f"DELETE FROM public.{self.t} WHERE model = %s AND NOT (id = ANY(%s))",
                        (model, list(keep)))
            return cur.rowcount

    def count(self, model=None):
        with self.conn.cursor() as cur:
            if model is None:
                cur.execute(f"SELECT count(*) FROM public.{self.t}")
            else:
                cur.execute(f"SELECT count(*) FROM public.{self.t} WHERE model = %s", (model,))
            return cur.fetchone()[0]

    def search(self, qvec, qtext, model, k=5, mode="hybrid", candidates=40):
        use_vec, use_kw = mode in ("hybrid", "vector"), mode in ("hybrid", "keyword")
        sql = f"""
            WITH vec AS (
                SELECT id, row_number() OVER (ORDER BY embedding OPERATOR(public.<=>) %(q)s::public.vector) AS r
                FROM public.{self.t} WHERE %(use_vec)s AND model = %(m)s
                ORDER BY embedding OPERATOR(public.<=>) %(q)s::public.vector LIMIT %(n)s),
            kw AS (
                SELECT id, row_number() OVER (ORDER BY ts_rank_cd(tsv, query) DESC) AS r
                FROM public.{self.t}, websearch_to_tsquery('english', %(text)s) AS query
                WHERE %(use_kw)s AND model = %(m)s AND tsv @@ query
                ORDER BY ts_rank_cd(tsv, query) DESC LIMIT %(n)s),
            fused AS (
                SELECT id, COALESCE(1.0 / (%(rk)s + vec.r), 0) + COALESCE(1.0 / (%(rk)s + kw.r), 0) AS score
                FROM vec FULL OUTER JOIN kw USING (id))
            SELECT d.url, d.title, d.section, d.body, f.score
            FROM fused f JOIN public.{self.t} d USING (id)
            ORDER BY f.score DESC LIMIT %(k)s"""
        # websearch_to_tsquery ANDs every word; an OR of the words keeps a long question from
        # matching nothing at all
        text = " or ".join(t for t in _terms(qtext) if t not in ("or", "and", "not")) or "-"
        with self.conn.cursor() as cur:
            cur.execute(sql, {"q": _vec_literal(qvec), "m": model, "n": candidates, "k": k,
                              "text": text, "rk": RRF_K, "use_vec": use_vec, "use_kw": use_kw})
            return [Hit(u, t, s, b, round(float(sc), 5)) for u, t, s, b, sc in cur.fetchall()]


def open_docstore(dsn: str | None, path: Path | str = DEFAULT_PATH) -> DocStore:
    return PgVectorStore(dsn) if dsn else LocalDocStore(path)


# --- the index the app uses ----------------------------------------------------------------------

@dataclass
class IngestReport:
    commit: str
    pages: int
    chunks: int
    embedded: int
    removed: int


class DocsIndex:
    def __init__(self, store: DocStore, embedder: Embedder):
        self.store, self.embedder = store, embedder

    def ingest(self, pages: list[Page], commit: str = "", max_chars: int = 1600) -> IngestReport:
        m = self.embedder.model
        chunks = list({c.id: c for p in pages for c in chunk_page(p, m, max_chars)}.values())
        known = self.store.known_ids(m)
        new = [c for c in chunks if c.id not in known]
        for i in range(0, len(new), 100):  # saved batch by batch: a rate limit loses one batch
            part = new[i:i + 100]
            self.store.upsert(part, self.embedder.embed_documents([c.embed_text() for c in part]), m)
        removed = self.store.prune({c.id for c in chunks}, m)
        return IngestReport(commit, len(pages), len(chunks), len(new), removed)

    def search(self, query: str, k: int = 5, mode: str = "hybrid") -> list[Hit]:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        return self.search_vec(self.embedder.embed_query(query), query, k, mode)

    def search_vec(self, qvec, query: str, k: int = 5, mode: str = "hybrid") -> list[Hit]:
        """Best chunk per page: three sections of one page in the top five crowd out the other
        pages a question may need. Asks the store for 4k chunks, keeps the first per URL."""
        hits = self.store.search(qvec, query, self.embedder.model, k * 4, mode)
        seen: set[str] = set()
        return [h for h in hits if not (h.url in seen or seen.add(h.url))][:k]

    def count(self) -> int:
        return self.store.count(self.embedder.model)


def evaluate(index: DocsIndex, cases: list[dict], k: int = 5, modes=MODES) -> dict:
    """Retrieval quality on labelled questions: each case is {"q": ..., "urls": [pages that
    answer it]}. hit@1 and recall@k count questions with a right page at rank 1 / in the top k;
    MRR averages 1 / rank of the first right page (0 if none in the top k)."""
    qvecs = [index.embedder.embed_query(c["q"]) for c in cases]  # once, shared by every mode
    out = {}
    for mode in modes:
        h1 = hk = rr = 0.0
        misses = []
        for case, qvec in zip(cases, qvecs):
            urls = [h.url for h in index.search_vec(qvec, case["q"], k, mode)]
            rank = next((i for i, u in enumerate(urls, 1) if u in case["urls"]), None)
            h1 += rank == 1
            hk += rank is not None
            rr += 1 / rank if rank else 0
            if rank is None:
                misses.append(case["q"])
        n = len(cases) or 1
        out[mode] = {"hit@1": round(h1 / n, 3), f"recall@{k}": round(hk / n, 3),
                     "mrr": round(rr / n, 3), "misses": misses}
    return out
