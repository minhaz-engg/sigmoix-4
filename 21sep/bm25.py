#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RAG app for large product corpora (imports the Markdown file we generated earlier).

- Ingests: out/daraz_products_corpus.md
- Retrieval: BM25 (rank-bm25, local, no embeddings)
- Answers:   gpt-4o-mini (cheap)
- UI:        Streamlit

Run:
  1) pip install -r requirements.txt
  2) cp .env.example .env  # then put your OpenAI API key in .env
  3) streamlit run rag_products_app.py
"""

import os
import re
import json
import time
import pickle
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import streamlit as st
from dotenv import load_dotenv

# -------- OpenAI (>=1.x) --------
from openai import OpenAI

# -------- BM25 -----------
from rank_bm25 import BM25Okapi


# -----------------------------
# Config
# -----------------------------
MD_PATH = Path("./out/daraz_products_corpus.md")
INDEX_DIR = Path("index")
INDEX_DIR.mkdir(parents=True, exist_ok=True)

CHAT_MODEL = "gpt-4o-mini"

TOP_K = 8               # retrieved docs
SLEEP_BETWEEN = 0.0     # (not used for BM25, left for symmetry)
BM25_K1 = 1.5
BM25_B = 0.75


# -----------------------------
# Utilities
# -----------------------------
def sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:['\-][A-Za-z0-9]+)?", re.UNICODE)

def tokenize(text: str) -> List[str]:
    """
    Lightweight, fast tokenizer for product text:
    - lowercases
    - keeps alphanumerics and common word-internal ' or -
    """
    return [m.group(0).lower() for m in _TOKEN_RE.finditer(text)]


def parse_markdown_products(md_text: str) -> List[Dict[str, Any]]:
    """
    Parses products delimited by:
      <!--DOC:START id=... category=... -->
      ...
      <!--DOC:END-->
    Returns list of dicts:
      {
        'id': ..., 'category': ...,
        'text': core fields for retrieval,
        'url': optional, 'title': optional,
        'raw': full block for display
      }
    """
    pattern = re.compile(
        r"<!--DOC:START\s+([^>]+?)-->(.*?)<!--DOC:END-->",
        flags=re.DOTALL | re.IGNORECASE
    )
    items = []
    for m in pattern.finditer(md_text):
        header = m.group(1)
        body = m.group(2).strip()

        # Parse header key=val
        meta = {}
        for kv in header.split():
            if "=" in kv:
                k, v = kv.split("=", 1)
                meta[k.strip()] = v.strip()

        # Try to extract URL & Title from body (best-effort)
        title_match = re.search(r"^##\s+(.+?)\s{2,}\n\*\*DocID:\*\*\s*(.+)$", body, flags=re.MULTILINE)
        url_match = re.search(r"\*\*URL:\*\*\s*(\S+)", body)
        title = title_match.group(1).strip() if title_match else None
        url = url_match.group(1).strip() if url_match else None

        # Build concise retrieval text by picking core fields
        snippets = []
        if title:
            snippets.append(title)
        if meta.get("category"):
            snippets.append(f"Category: {meta['category']}")

        def grab(label: str, pattern_str: str, multiline=False, list_mode=False):
            pm = re.search(pattern_str, body, flags=re.DOTALL if multiline else 0)
            if not pm:
                return
            if multiline and list_mode:
                values = [ln.strip("- ").strip() for ln in pm.group(1).strip().splitlines() if ln.strip()]
                if values:
                    snippets.append(f"{label}: " + ", ".join(values))
            else:
                snippets.append(f"{label}: {pm.group(1).strip()}")

        grab("Brand", r"\*\*Brand:\*\*\s*(.+)")
        grab("Price", r"\*\*Price:\*\*\s*(.+)")
        grab("Original Price", r"\*\*Original:\*\*\s*(.+)")
        grab("Discount", r"\*\*Discount:\*\*\s*(.+)")
        rating_m = re.search(r"\*\*Rating:\*\*\s*([0-9.]+)/5(?:\s+\((\d+)\s+ratings\))?", body)
        if rating_m:
            r = rating_m.group(1); rc = rating_m.group(2)
            rtxt = f"Rating: {r}/5" + (f" ({rc} ratings)" if rc else "")
            snippets.append(rtxt)
        grab("Sales", r"\*\*Sales:\*\*\s*(.+)")
        grab("Colors", r"\*\*Colors:\*\*\s*(.+)")
        grab("Sizes", r"\*\*Sizes:\*\*\s*(.+)")
        grab("Variants", r"\*\*Variants:\*\*\n([\s\S]+?)(?:\n\n|\Z)", multiline=True, list_mode=True)
        grab("Description", r"\*\*Description:\*\*\n([\s\S]+?)(?:\n\n|\Z)", multiline=True)
        if url:
            snippets.append(f"URL: {url}")

        text_for_retrieval = "\n".join(snippets) if snippets else body

        items.append({
            "id": meta.get("id") or sha1(text_for_retrieval)[:12],
            "category": meta.get("category"),
            "title": title or "",
            "url": url,
            "text": text_for_retrieval.strip(),
            "raw": body,
        })
    return items


def ensure_client() -> OpenAI:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY. Put it in .env or your environment.")
    return OpenAI()


# -----------------------------
# BM25 index build/load
# -----------------------------
def build_or_load_bm25(products: List[Dict[str, Any]]) -> Tuple[BM25Okapi, List[Dict[str, Any]]]:
    """
    Builds BM25 index if not present, else loads it.
    Uses a stable hash over the input corpus to detect changes.
    """
    corpus_sig = sha1("\n".join([p["id"] + "\t" + p["text"] for p in products]))
    idx_stem = f"bm25_{corpus_sig}_k{BM25_K1}_b{BM25_B}"
    model_path = INDEX_DIR / f"{idx_stem}.pkl"
    meta_path = INDEX_DIR / f"{idx_stem}.jsonl"

    if model_path.exists() and meta_path.exists():
        with open(model_path, "rb") as f:
            bm25, tokenized_docs = pickle.load(f)
        with meta_path.open("r", encoding="utf-8") as f:
            meta = [json.loads(line) for line in f]
        # quick sanity
        assert len(tokenized_docs) == len(meta) == getattr(bm25, "nd", len(tokenized_docs))
        return bm25, meta

    # Fresh build
    tokenized_docs = [tokenize(p["text"]) for p in products]
    bm25 = BM25Okapi(tokenized_docs, k1=BM25_K1, b=BM25_B)

    # Persist
    with open(model_path, "wb") as f:
        pickle.dump((bm25, tokenized_docs), f)
    with meta_path.open("w", encoding="utf-8") as f:
        for p in products:
            f.write(json.dumps({
                "id": p["id"],
                "category": p.get("category"),
                "title": p.get("title"),
                "url": p.get("url"),
                "text": p["text"],
                "raw": p["raw"],
            }, ensure_ascii=False) + "\n")
    return bm25, products


def retrieve_bm25(bm25: BM25Okapi, meta: List[Dict[str, Any]], query: str, k: int = TOP_K):
    q_tokens = tokenize(query)
    scores = bm25.get_scores(q_tokens)  # numpy array
    if scores.size == 0:
        return []

    # Top-k indices by score (descending)
    top_idx = np.argsort(scores)[::-1][:k]
    hits = []
    for idx in top_idx:
        rec = meta[int(idx)]
        hits.append({
            "score": float(scores[int(idx)]),
            "id": rec["id"],
            "title": rec.get("title") or "",
            "url": rec.get("url"),
            "category": rec.get("category"),
            "snippet": rec.get("text"),
            "raw": rec.get("raw"),
        })
    return hits


def build_prompt(query: str, hits: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """
    Compact, citation-friendly prompt. Each context block carries [ID] and URL.
    """
    ctx_blocks = []
    for h in hits:
        title = h["title"] or "(no title)"
        url = f" | {h['url']}" if h.get("url") else ""
        cat = f" | Category: {h['category']}" if h.get("category") else ""
        block = f"[{h['id']}] {title}{cat}{url}\n{h['snippet']}\n"
        ctx_blocks.append(block[:1800])  # keep it tight per doc

    system = (
        "You are a precise product assistant. "
        "Answer only from the provided context. If unsure, say you don't know. "
        "Always include bullet citations like [ID] and the URL when available."
    )
    user = (
        f"Question: {query}\n\n"
        f"Context (top {len(hits)} results):\n"
        + "\n---\n".join(ctx_blocks)
        + "\n\nAnswer succinctly with factual details (price/brand/variants/seller) when relevant. "
          "Finish with a short bullet list of citations: [ID] Title – URL."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user}
    ]


def answer_with_llm(client: OpenAI, messages: List[Dict[str, str]]) -> str:
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0.2,
        messages=messages,
    )
    return resp.choices[0].message.content.strip()


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Products RAG (BM25)", page_icon="🛒", layout="wide")

st.title("🛒 Daraz Products RAG — BM25")

# Show available categories 
@st.cache_data(show_spinner=False)
def _extract_categories(products_list):
    categories = set()
    for product in products_list:
        if product.get("category"):
            categories.add(product["category"])
    return sorted(list(categories))

with st.sidebar:
    st.header("Index")
    st.write("This app ingests your Markdown corpus at:")
    st.code(str(MD_PATH), language="bash")
    st.write("Retrieval:")
    st.write(f"- **BM25** (rank-bm25)  k1={BM25_K1}, b={BM25_B}")
    st.write("LLM:")
    st.write(f"- Chat: `{CHAT_MODEL}`")
    st.divider()
    st.caption("Tip: First run will build a local BM25 index (pickled). Subsequent runs load instantly.")

# Load corpus
if not MD_PATH.exists():
    st.error(f"Markdown corpus not found at {MD_PATH}. Generate it first.")
    st.stop()

md_text = MD_PATH.read_text(encoding="utf-8")

@st.cache_data(show_spinner=True)
def _load_products(md_text: str):
    return parse_markdown_products(md_text)

products = _load_products(md_text)
st.write(f"**Loaded products**: {len(products):,}")

# Show available categories
categories = _extract_categories(products)
if categories:
    st.write(f"**Available categories** ({len(categories)}):")
    cols = st.columns(3)
    for i, category in enumerate(categories):
        with cols[i % 3]:
            st.caption(f"• {category}")
else:
    st.write("**Categories**: None found")

@st.cache_resource(show_spinner=True)
def _load_bm25(md_text: str):
    prods = parse_markdown_products(md_text)
    return build_or_load_bm25(prods)

bm25, meta = _load_bm25(md_text)

# Query box
query = st.text_input("Ask a question about your products (e.g., 'best budget bedding set under 2000 BDT with good ratings')", "")
btn = st.button("Search")

if btn and query.strip():
    try:
        client = ensure_client()
        with st.spinner("Retrieving…"):
            hits = retrieve_bm25(bm25, meta, query, k=TOP_K)

        if not hits:
            st.warning("No relevant products found.")
        else:
            # Show retrieved cards
            st.subheader("Top matches")
            for h in hits:
                with st.expander(f"[{h['id']}] {h['title'] or '(no title)'}  —  BM25 score {h['score']:.2f}"):
                    if h.get("url"):
                        st.markdown(f"[Open product]({h['url']})")
                    st.write(f"**Category:** {h.get('category') or '—'}")
                    st.text(h["snippet"])

            # LLM answer
            messages = build_prompt(query, hits)
            with st.spinner("Thinking…"):
                answer = answer_with_llm(client, messages)

            st.subheader("Answer")
            st.markdown(answer)

    except Exception as e:
        st.error(f"Error: {e}")
