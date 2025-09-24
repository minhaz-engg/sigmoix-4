# ============================================================
# Daraz Products RAG — Hybrid Retrieval + Re-ranker (Streamlit)
# ============================================================
# - Markdown corpus parsing with metadata (title/url/brand/price/rating)
# - Naive markdown chunking (training-aligned) with safe BM25 cleaning
# - BM25 baseline
# - Dense retrieval (fine-tuned BGE-small) + FAISS (IndexFlatIP, normalized)
# - RRF fusion (BM25 + Dense)
# - Cross-Encoder re-ranking (fine-tuned MiniLM)
# - Doc-aware ranking in the UI (max-pool best chunk per doc)
# - Grounded LLM answer (OpenAI Chat Completions) with [#] citations
# - Graceful fallbacks if models/index are missing
# ------------------------------------------------------------

import os
import re
import io
import json
import time
import math
import pickle
import hashlib
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple

import numpy as np
import streamlit as st
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi

# Optional deps (dense)
_HAS_ST = False
_HAS_FAISS = False
_HAS_CE = False
try:
    from sentence_transformers import SentenceTransformer, CrossEncoder
    _HAS_ST = True
except Exception:
    _HAS_ST = False
try:
    import faiss
    _HAS_FAISS = True
except Exception:
    _HAS_FAISS = False

# Optional LLM for answers
from openai import OpenAI

# Try to import Chonkie (we default to naive chunking anyway)
try:
    from chonkie import RecursiveChunker
    HAS_CHONKIE = True
except Exception:
    HAS_CHONKIE = False

load_dotenv()

# ----------------------------
# App Config & Defaults
# ----------------------------
st.set_page_config(page_title="Daraz RAG — Hybrid Retrieval + Re-ranker", layout="wide")
DEFAULT_MD_PATH = "out/daraz_products_corpus.md"
DEFAULT_MODELS_EMB = "models/embedder-ft"
DEFAULT_MODELS_RER = "models/reranker-ft-miniLM"
DEFAULT_INDEX_FAISS = "index/dense.faiss"
DEFAULT_INDEX_CHUNKS = "index/dense_chunks.json"
INDEX_DIR = "index"
os.makedirs(INDEX_DIR, exist_ok=True)

DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_TOP_DOCS = 8

STOPWORDS = set([
    "the","a","an","and","or","of","for","on","in","to","from","with","by","at","is","are","was","were",
    "this","that","these","those","it","its","as","be","can","will","has","have"
])

# ----------------------------
# Data structures
# ----------------------------
@dataclass
class ProductDoc:
    doc_id: str
    title: str
    url: Optional[str]
    category: Optional[str]
    brand: Optional[str]
    price_value: Optional[float]
    rating_avg: Optional[float]
    rating_cnt: Optional[int]
    raw_md: str

@dataclass
class ChunkRec:
    chunk_id: str
    doc_id: str
    title: str
    brand: Optional[str]
    category: Optional[str]
    text: str
    # optional extra fields preserved if available
    url: Optional[str] = None
    price_value: Optional[float] = None
    rating_avg: Optional[float] = None
    rating_cnt: Optional[int] = None

# ----------------------------
# Regex helpers
# ----------------------------
DOC_BLOCK_RE = re.compile(r"<!--DOC:START(?P<attrs>[^>]*)-->(?P<body>.*?)<!--DOC:END-->", re.DOTALL|re.IGNORECASE)
TITLE_RE     = re.compile(r"^##\s+(.+?)\s*(?:\s{2,}\n|\n|$)", re.MULTILINE)
URL_LINE_RE  = re.compile(r"\*\*URL:\*\*\s*(\S+)", re.IGNORECASE)
BRAND_RE     = re.compile(r"\*\*Brand:\*\*\s*(.+)", re.IGNORECASE)
PRICE_RE     = re.compile(r"\*\*Price:\*\*\s*(.+)", re.IGNORECASE)
RATING_RE    = re.compile(r"\*\*Rating:\*\*\s*([0-9.]+)\s*/\s*5(?:\s*\((\d+)\s*ratings\))?", re.IGNORECASE)
STOP_URL     = re.compile(r"\s+https?://\S+")

def _meta_from_header(attrs: str) -> Dict[str,str]:
    out = {}
    for kv in attrs.strip().split():
        if "=" in kv:
            k, v = kv.split("=", 1)
            out[k.strip()] = v.strip()
    return out

def _parse_price_value(s: str) -> Optional[float]:
    s = s.replace(",", "")
    nums = re.findall(r"(\d+(?:\.\d+)?)", s)
    if not nums: return None
    try: return min(float(x) for x in nums)
    except: return None

def clean_for_index(text: str) -> str:
    out = []
    for line in text.splitlines():
        ll = line.strip()
        if not ll: continue
        if ll.lower().startswith("**images"): continue
        if "http://" in ll or "https://" in ll:
            ll = STOP_URL.sub(" ", ll).strip()
            if not ll: continue
        out.append(ll)
    return "\n".join(out)

def naive_markdown_chunk(md: str, max_words=160, min_words=40) -> List[str]:
    parts = [p.strip() for p in re.split(r"\n{2,}", md) if p.strip()]
    chunks, buf, wc = [], [], 0
    for p in parts:
        w = p.split()
        if wc + len(w) <= max_words:
            buf.append(p); wc += len(w)
        else:
            if wc >= min_words: chunks.append("\n\n".join(buf))
            buf, wc = [p], len(w)
    if buf and wc >= min_words:
        chunks.append("\n\n".join(buf))
    if not chunks and parts:
        chunks = ["\n\n".join(parts[:3])]
    return [clean_for_index(c) for c in chunks if c.strip()]

def _tokenize(text: str) -> List[str]:
    toks = re.findall(r"[A-Za-z0-9_]+", text.lower())
    return [t for t in toks if t not in STOPWORDS]

def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

# ----------------------------
# Parsing
# ----------------------------
def parse_products_from_md(md_text: str) -> Tuple[List[ProductDoc], List[ChunkRec]]:
    products: List[ProductDoc] = []
    for m in DOC_BLOCK_RE.finditer(md_text):
        attrs = m.group("attrs") or ""
        body  = (m.group("body") or "").strip()
        meta  = _meta_from_header(attrs)
        doc_id   = meta.get("id") or f"doc_{len(products)+1}"
        category = meta.get("category")

        title_m = TITLE_RE.search(body)
        title   = (title_m.group(1).strip() if title_m else f"Product {doc_id}")
        url_m   = URL_LINE_RE.search(body)
        url     = url_m.group(1).strip() if url_m else None
        brand_m = BRAND_RE.search(body)
        brand   = brand_m.group(1).strip() if brand_m else None

        price_m = PRICE_RE.search(body)
        price_value = _parse_price_value(price_m.group(1)) if price_m else None

        rating_m = RATING_RE.search(body)
        rating_avg = float(rating_m.group(1)) if rating_m else None
        rating_cnt = int(rating_m.group(2)) if (rating_m and rating_m.group(2)) else None

        products.append(ProductDoc(
            doc_id=doc_id, title=title, url=url, category=category, brand=brand,
            price_value=price_value, rating_avg=rating_avg, rating_cnt=rating_cnt, raw_md=body
        ))

    # training-aligned chunker (naive); Chonkie available optionally in sidebar
    chunks: List[ChunkRec] = []
    for p in products:
        texts = naive_markdown_chunk(p.raw_md, max_words=160, min_words=40)
        for i, t in enumerate(texts):
            chunks.append(ChunkRec(
                chunk_id=f"{p.doc_id}::ch{i+1}",
                doc_id=p.doc_id,
                title=p.title,
                brand=p.brand,
                category=p.category,
                text=t,
                url=p.url,
                price_value=p.price_value,
                rating_avg=p.rating_avg,
                rating_cnt=p.rating_cnt,
            ))
    return products, chunks

# ----------------------------
# BM25
# ----------------------------
def build_bm25(chunks: List[ChunkRec]) -> Tuple[BM25Okapi, List[List[str]]]:
    tokenized = [_tokenize(c.text) for c in chunks]
    return BM25Okapi(tokenized), tokenized

def bm25_topk(bm25: BM25Okapi, chunks: List[ChunkRec], query: str, k=200,
              allowed_categories: Optional[set] = None, brand_filter: Optional[str] = None,
              price_min: Optional[float] = None, price_max: Optional[float] = None,
              rating_min: Optional[float] = None) -> List[Tuple[ChunkRec, float]]:
    q_tokens = _tokenize(query)
    scores = bm25.get_scores(q_tokens)
    pairs: List[Tuple[int, float]] = []
    for i, sc in enumerate(scores):
        c = chunks[i]
        if not _passes_filters(c, allowed_categories, brand_filter, price_min, price_max, rating_min):
            continue
        pairs.append((i, float(sc)))
    pairs.sort(key=lambda x: x[1], reverse=True)
    return [(chunks[i], s) for i, s in pairs[:k]]

def _passes_filters(chunk: ChunkRec,
                    allowed_categories: Optional[set],
                    brand_filter: Optional[str],
                    price_min: Optional[float],
                    price_max: Optional[float],
                    rating_min: Optional[float]) -> bool:
    if allowed_categories and (chunk.category not in allowed_categories):
        return False
    if brand_filter:
        b = (chunk.brand or "").lower()
        if brand_filter.lower() not in b:
            return False
    if price_min is not None and (chunk.price_value is not None) and (chunk.price_value < price_min):
        return False
    if price_max is not None and (chunk.price_value is not None) and (chunk.price_value > price_max):
        return False
    if rating_min is not None and (chunk.rating_avg is not None) and (chunk.rating_avg < rating_min):
        return False
    return True

# ----------------------------
# Dense (embedder + FAISS)
# ----------------------------
class DenseBackend:
    def __init__(self, model_dir: str, faiss_path: Optional[str] = None, chunks_path: Optional[str] = None):
        self.ok = False
        self.model_dir = model_dir
        self.faiss_path = faiss_path
        self.chunks_path = chunks_path
        self.embedder = None
        self.index = None
        self.chunks_for_index: Optional[List[ChunkRec]] = None
        if not _HAS_ST or not _HAS_FAISS:
            return
        try:
            self.embedder = SentenceTransformer(self.model_dir)
        except Exception as e:
            st.warning(f"Embedder load failed: {e}")
            return

        # Try prebuilt index + chunks
        if self.faiss_path and os.path.exists(self.faiss_path) and self.chunks_path and os.path.exists(self.chunks_path):
            try:
                self.index = faiss.read_index(self.faiss_path)
                # Load chunks metadata aligned to FAISS
                with open(self.chunks_path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                # tolerate both list[dict] or jsonl dump
                if isinstance(raw, list):
                    self.chunks_for_index = [ChunkRec(**c) if "chunk_id" in c else ChunkRec(
                        chunk_id=c.get("chunk_id") or f"{c['doc_id']}::?",
                        doc_id=c["doc_id"], title=c.get("title",""),
                        brand=c.get("brand"), category=c.get("category"),
                        text=c.get("text","")) for c in raw]
                else:
                    raise ValueError("dense_chunks.json does not contain a list.")
                self.ok = True
                return
            except Exception as e:
                st.info(f"Prebuilt FAISS present but could not load cleanly: {e}")

        # Otherwise we'll (re)build later when we know the runtime chunks
        self.ok = True

    def ensure_index(self, chunks_runtime: List[ChunkRec], encode_batch=256) -> Tuple[bool, Optional[str]]:
        if not self.ok or self.embedder is None:
            return False, "Dense backend not available."
        # If we already have a prebuilt index AND chunk list that we can reuse, only reuse if same length
        if self.index is not None and self.chunks_for_index is not None:
            if len(self.chunks_for_index) == len(chunks_runtime):
                return True, None  # we assume alignment (built from same corpus)
            else:
                st.info("Prebuilt FAISS length does not match runtime chunks; rebuilding…")
                self.index = None
                self.chunks_for_index = None

        # Build FAISS for current chunks
        texts = [c.text for c in chunks_runtime]
        try:
            vecs = self.embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True,
                                        batch_size=encode_batch, show_progress_bar=False)
            idx = faiss.IndexFlatIP(vecs.shape[1])
            idx.add(vecs.astype(np.float32))
            self.index = idx
            self.chunks_for_index = chunks_runtime
            return True, None
        except Exception as e:
            return False, f"Dense index build failed: {e}"

    def topk(self, query: str, k=200) -> List[Tuple[ChunkRec, float]]:
        if not (self.ok and self.embedder is not None and self.index is not None and self.chunks_for_index):
            return []
        qv = self.embedder.encode([query], normalize_embeddings=True)
        D, I = self.index.search(qv.astype(np.float32), k)
        return [(self.chunks_for_index[i], float(D[0][j])) for j, i in enumerate(I[0])]

# ----------------------------
# Cross-Encoder re-ranker
# ----------------------------
class Reranker:
    def __init__(self, model_dir: str, max_length: int = 320):
        self.ok = False
        self.model_dir = model_dir
        self.max_length = max_length
        if not _HAS_ST:
            return
        try:
            self.model = CrossEncoder(self.model_dir, num_labels=1, max_length=self.max_length)
            self.ok = True
        except Exception as e:
            st.warning(f"Reranker load failed: {e}")
            self.ok = False

    def rerank(self, query: str, candidates: List[Tuple[ChunkRec, float]], topk=200, batch_size=64,
               query_max_words=64, chunk_max_words=160) -> List[Tuple[ChunkRec, float]]:
        if not self.ok:
            return candidates[:topk]
        def clip_words(txt, n): 
            w=txt.split()
            return " ".join(w[:n])
        pairs = [(clip_words(query, query_max_words), clip_words(c.text, chunk_max_words)) for c,_ in candidates]
        try:
            scores = self.model.predict(pairs, convert_to_numpy=True, batch_size=batch_size)
            ranked = list(zip([c for c,_ in candidates], scores))
            ranked.sort(key=lambda x: x[1], reverse=True)
            return [(c, float(s)) for c, s in ranked[:topk]]
        except Exception as e:
            st.warning(f"Re-ranking failed: {e}")
            return candidates[:topk]

# ----------------------------
# Fusion & doc-aware ranking
# ----------------------------
def rrf_fuse(listA: List[Tuple[ChunkRec, float]],
             listB: List[Tuple[ChunkRec, float]],
             k: int = 60, topk: int = 200) -> List[Tuple[ChunkRec, float]]:
    rankA = {rec.chunk_id: r for r,(rec,_) in enumerate(listA, 1)}
    rankB = {rec.chunk_id: r for r,(rec,_) in enumerate(listB, 1)}
    ids = set(rankA) | set(rankB)
    fused = []
    for cid in ids:
        ra = rankA.get(cid, 10**9)
        rb = rankB.get(cid, 10**9)
        s = 1.0/(k+ra) + 1.0/(k+rb)
        # recover the object from whichever list we have (prefer A then B)
        obj = None
        if cid in rankA:
            obj = listA[ra-1][0]
        elif cid in rankB:
            obj = listB[rb-1][0]
        if obj is not None:
            fused.append((obj, s))
    fused.sort(key=lambda x: x[1], reverse=True)
    return fused[:topk]

def to_doc_ranking(chunk_ranked: List[Tuple[ChunkRec, float]], topk=100) -> List[Tuple[str, float, ChunkRec]]:
    # max-pool best chunk score per doc
    best: Dict[str, Tuple[float, ChunkRec]] = {}
    for c, s in chunk_ranked:
        prev = best.get(c.doc_id)
        if (prev is None) or (s > prev[0]):
            best[c.doc_id] = (s, c)
    items = sorted(((doc_id, sc, ck) for doc_id,(sc,ck) in best.items()), key=lambda x: x[1], reverse=True)
    return items[:topk]

# ----------------------------
# OpenAI helpers (answering)
# ----------------------------
def _ensure_client() -> Optional[OpenAI]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    return OpenAI()

def _build_messages(query: str, doc_results: List[Tuple[str, float, ChunkRec]]) -> List[Dict[str, str]]:
    blocks = []
    for i, (_, _, top_chunk) in enumerate(doc_results, 1):
        head = f"[{i}] {top_chunk.title} — DocID: {top_chunk.doc_id}" + (f" — {top_chunk.url}" if top_chunk.url else "")
        fields = []
        if top_chunk.brand: fields.append(f"Brand: {top_chunk.brand}")
        if top_chunk.category: fields.append(f"Category: {top_chunk.category}")
        if top_chunk.price_value is not None: fields.append(f"PriceValue: {int(top_chunk.price_value)}")
        if top_chunk.rating_avg is not None: fields.append(f"Rating: {top_chunk.rating_avg}/5")
        meta_line = " | ".join(fields)
        blocks.append(f"{head}\n{meta_line}\n---\n{top_chunk.text}\n")
    system = (
        "You are a precise product assistant. Answer ONLY from the provided context. "
        "If the answer isn't present, say you don't know. Include concise bullets. "
        "Cite as [#] with DocID and include URL when available."
    )
    user = f"Question:\n{query}\n\nContext:\n" + "\n\n".join(blocks)
    return [{"role":"system","content":system},{"role":"user","content":user}]

def stream_answer(model_name: str, messages: List[Dict[str,str]], temperature: float = 0.2):
    client = _ensure_client()
    if client is None:
        yield "⚠️ Set OPENAI_API_KEY to enable grounded answers."
        return
    resp = client.chat.completions.create(
        model=model_name,
        temperature=temperature,
        messages=messages,
        stream=True,
    )
    for ch in resp:
        delta = ch.choices[0].delta.content or ""
        if delta:
            yield delta

# ----------------------------
# Sidebar UI
# ----------------------------
st.title("Daraz Products RAG — Hybrid Retrieval + Re-ranker")

with st.sidebar:
    st.markdown("### ⚙️ Settings")
    openai_model = st.selectbox("Answering model (OpenAI)", [DEFAULT_OPENAI_MODEL, "gpt-4o"], index=0)
    top_docs = st.slider("Top docs (answer context)", 3, 15, DEFAULT_TOP_DOCS, 1)
    temperature = st.slider("Answer temperature", 0.0, 1.0, 0.2, 0.1)

    st.markdown("### 🔎 Retrieval Strategy")
    use_dense = st.checkbox("Enable Dense + RRF (recommended)", value=True)
    use_reranker = st.checkbox("Enable Cross-Encoder Re-ranking (recommended)", value=True)
    rrf_k = st.slider("RRF k", 10, 120, 60, 10)

    st.markdown("### 📦 Artifacts / Paths")
    md_path = st.text_input("Markdown file path", value=DEFAULT_MD_PATH)
    emb_dir = st.text_input("Embedder dir", value=DEFAULT_MODELS_EMB)
    rer_dir = st.text_input("Reranker dir", value=DEFAULT_MODELS_RER)
    faiss_path = st.text_input("FAISS index path (optional)", value=DEFAULT_INDEX_FAISS)
    dense_chunks_path = st.text_input("FAISS chunks path (optional)", value=DEFAULT_INDEX_CHUNKS)

    st.markdown("### 🧪 Filters")
    # Inputs for filters appear later after parsing
    st.caption("Set filters after the corpus is loaded.")

# ----------------------------
# Load corpus
# ----------------------------
md_text = None
if md_path and os.path.exists(md_path):
    md_text = open(md_path, "r", encoding="utf-8", errors="ignore").read()
uploaded = st.file_uploader("…or upload Markdown corpus (.md)", type=["md"])
if uploaded is not None:
    md_text = uploaded.read().decode("utf-8", errors="ignore")

if not md_text:
    st.info("Provide the `daraz_products_corpus.md` file path in the sidebar, or upload it here.")
    st.stop()

with st.spinner("Parsing products and building chunks…"):
    products, chunks = parse_products_from_md(md_text)

DOC_BY_ID = {p.doc_id: p for p in products}
st.success(f"Parsed **{len(products):,}** products → **{len(chunks):,}** chunks.")

# Facets
all_categories = sorted({p.category for p in products if p.category})
all_brands = sorted({(p.brand or "").strip() for p in products if p.brand})
with st.expander("Show some detected brands", expanded=False):
    st.write(", ".join(all_brands[:60]) + (" ..." if len(all_brands) > 60 else ""))

# ----------------------------
# Build BM25
# ----------------------------
with st.spinner("Building BM25 index…"):
    bm25, _tok = build_bm25(chunks)

# ----------------------------
# Dense backend
# ----------------------------
dense_backend = None
if use_dense and _HAS_ST and _HAS_FAISS:
    dense_backend = DenseBackend(model_dir=emb_dir, faiss_path=faiss_path, chunks_path=dense_chunks_path)
    ok, err = dense_backend.ensure_index(chunks, encode_batch=256)
    if not ok:
        st.warning(f"Dense disabled: {err}")
        dense_backend = None
else:
    if use_dense:
        st.warning("Dense toggled ON but sentence-transformers/faiss not available. Falling back to BM25.")

# ----------------------------
# Reranker
# ----------------------------
reranker = None
if use_reranker and _HAS_ST:
    reranker = Reranker(model_dir=rer_dir, max_length=320)
    if not reranker.ok:
        st.warning("Re-ranker toggled ON but could not load. Proceeding without.")
        reranker = None
else:
    if use_reranker:
        st.warning("Re-ranker toggled ON but sentence-transformers not available. Proceeding without.")

# ----------------------------
# Filters UI
# ----------------------------
ncols = st.columns([1.6, 1.2, 1.2, 1.2])
with ncols[0]:
    sel_categories = st.multiselect("Category", options=all_categories, default=[])
with ncols[1]:
    brand_filter = st.text_input("Brand contains", "")
with ncols[2]:
    price_max_ui = st.text_input("Max price (BDT)", "")
with ncols[3]:
    rating_min_ui = st.text_input("Min rating (0–5)", "")

def _to_float(x: str) -> Optional[float]:
    x = (x or "").strip().replace(",", "")
    if not x: return None
    m = re.match(r"^\d+(?:\.\d+)?$", x)
    return float(x) if m else None

price_max_filter = _to_float(price_max_ui)
rating_min_filter = _to_float(rating_min_ui)

allowed_categories = set(sel_categories) if sel_categories else None
brand_q = brand_filter if brand_filter.strip() else None
price_min = None
price_max = price_max_filter
rating_min = rating_min_filter

# ----------------------------
# Search box
# ----------------------------
st.markdown("---")
query = st.text_input("🔎 Ask about products (e.g., 'bedsheet under 2000 rating 4.5+')", "")
go = st.button("Search")

def search_pipeline(query: str, top_docs: int = 8) -> Tuple[List[Tuple[str, float, ChunkRec]], List[Tuple[ChunkRec, float]]]:
    # 1) BM25 candidates
    bm = bm25_topk(bm25, chunks, query, k=200,
                   allowed_categories=allowed_categories, brand_filter=brand_q,
                   price_min=price_min, price_max=price_max, rating_min=rating_min)
    # 2) Dense candidates (if available)
    hyb = bm
    if dense_backend is not None:
        dense = dense_backend.topk(query, k=200)
        hyb = rrf_fuse(bm, dense, k=rrf_k, topk=200)

    # 3) Re-ranking (if available)
    if reranker is not None:
        hyb = reranker.rerank(query, hyb, topk=200, batch_size=64)

    # 4) Doc-aware aggregation
    doc_rank = to_doc_ranking(hyb, topk=top_docs)
    return doc_rank, hyb

if go and query.strip():
    with st.spinner("Retrieving…"):
        doc_results, chunk_results = search_pipeline(query, top_docs=top_docs)

    if not doc_results:
        st.warning("No results matched your query/filters.")
        st.stop()

    # Two columns: left = top docs, right = grounded answer
    colL, colR = st.columns([0.55, 0.45], gap="large")

    with colL:
        st.subheader("Top documents (doc-aware)")
        for i, (doc_id, score, best_chunk) in enumerate(doc_results, 1):
            p = DOC_BY_ID.get(doc_id)
            meta_bits = []
            if p and p.brand: meta_bits.append(f"**Brand:** {p.brand}")
            if p and p.category: meta_bits.append(f"**Category:** {p.category}")
            if p and p.price_value is not None: meta_bits.append(f"**Price:** ~৳{int(p.price_value)}")
            if p and p.rating_avg is not None:
                rc = f" ({p.rating_cnt} ratings)" if p.rating_cnt is not None else ""
                meta_bits.append(f"**Rating:** {p.rating_avg}/5{rc}")

            st.markdown(
                f"**[{i}] {p.title if p else best_chunk.title}**  \n"
                f"DocID: `{doc_id}` • Score: `{score:.3f}`  \n"
                f"{'URL: ' + (p.url or '') if p and p.url else ''}  \n"
                + ("  \n".join(meta_bits) if meta_bits else "")
            )
            with st.expander("View best chunk"):
                st.write(best_chunk.text)

        # Export button (doc-level)
        export_rows = []
        for i,(doc_id,score,best_chunk) in enumerate(doc_results,1):
            p = DOC_BY_ID.get(doc_id)
            export_rows.append({
                "rank": i, "score": score,
                "doc_id": doc_id,
                "title": (p.title if p else best_chunk.title),
                "url": (p.url if p else best_chunk.url) or "",
                "category": (p.category if p else best_chunk.category) or "",
                "brand": (p.brand if p else best_chunk.brand) or "",
                "price_value": p.price_value if (p and p.price_value is not None) else "",
                "rating_avg": p.rating_avg if (p and p.rating_avg is not None) else "",
                "rating_cnt": p.rating_cnt if (p and p.rating_cnt is not None) else "",
                "best_chunk_text": best_chunk.text[:2000],
            })
        export_bytes = io.BytesIO()
        export_bytes.write(json.dumps(export_rows, ensure_ascii=False, indent=2).encode("utf-8"))
        export_bytes.seek(0)
        st.download_button("⬇️ Download results (JSON)", data=export_bytes, file_name="results.json", mime="application/json")

    with colR:
        st.subheader("Grounded answer")
        msgs = _build_messages(query, doc_results[:top_docs])
        try:
            st.write_stream(stream_answer(openai_model, msgs, temperature=temperature))
        except Exception as e:
            st.error(f"Answer generation error: {e}")

# Footer: show active backends
st.markdown("---")
st.caption(
    f"Backends — BM25 ✅ • Dense: {'✅' if dense_backend is not None else '—'} • Reranker: {'✅' if reranker is not None else '—'} • Doc-aware ranking ✅"
)
