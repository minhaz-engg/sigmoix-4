#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RAG app for large product corpora (imports the Markdown file we generated earlier).

- Ingests: out/daraz_products_corpus.md
- Index:    FAISS (local)
- Embeds:   text-embedding-3-small (cheap)
- Answers:  gpt-4o-mini (cheap)
- UI:       Streamlit

Run:
  1) pip install -r requirements.txt
  2) cp .env.example .env  # then put your OpenAI API key in .env
  3) streamlit run rag_products_app.py
"""

import os
import re
import json
import time
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import streamlit as st
from dotenv import load_dotenv

# -------- OpenAI (>=1.x) --------
from openai import OpenAI

# -------- Vector index (FAISS) ---
import faiss


# -----------------------------
# Config
# -----------------------------
MD_PATH = Path("./out/daraz_products_corpus.md")
INDEX_DIR = Path("index")
INDEX_DIR.mkdir(parents=True, exist_ok=True)

EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"  # or "gpt-4o-mini"

TOP_K = 8               # retrieved docs
EMBED_BATCH = 512       # batch size for embeddings
SLEEP_BETWEEN = 0.1     # gentle rate limiting for huge corpora
CHUNK_PER_PRODUCT = True  # products already structured; 1 vector per product is enough


# -----------------------------
# Utilities5  5  
# -----------------------------
def sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()

def parse_markdown_products(md_text: str) -> List[Dict[str, Any]]:
    """
    Parses products delimited by:
      <!--DOC:START id=... category=... -->
      ...
      <!--DOC:END-->
    Returns list of dicts:
      {
        'id': ..., 'category': ...,
        'text': full_product_text_for_embedding,
        'url': optional, 'title': optional
      }
    """
    # Grab blocks
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
        title = None
        if title_match:
            title = title_match.group(1).strip()
        url = url_match.group(1).strip() if url_match else None

        # Build concise embedding text (trim boilerplate)
        # Keep core facts: title, brand, category, price, rating, sales, colors, sizes, description, url
        snippets = []

        if title:
            snippets.append(title)
        if meta.get("category"):
            snippets.append(f"Category: {meta['category']}")

        brand_m = re.search(r"\*\*Brand:\*\*\s*(.+)", body)
        price_m = re.search(r"\*\*Price:\*\*\s*(.+)", body)
        original_price_m = re.search(r"\*\*Original:\*\*\s*(.+)", body)
        discount_percentage_m = re.search(r"\*\*Discount:\*\*\s*(.+)", body)
        rating_m = re.search(r"\*\*Rating:\*\*\s*([0-9.]+)/5(?:\s+\((\d+)\s+ratings\))?", body)
        sales_m = re.search(r"\*\*Sales:\*\*\s*(.+)", body)
        colors_m = re.search(r"\*\*Colors:\*\*\s*(.+)", body)
        sizes_m = re.search(r"\*\*Sizes:\*\*\s*(.+)", body)
        variants_m = re.search(r"\*\*Variants:\*\*\n([\s\S]+?)(?:\n\n|\Z)", body)
        desc_m = re.search(r"\*\*Description:\*\*\n([\s\S]+?)(?:\n\n|\Z)", body)

        if brand_m: snippets.append(f"Brand: {brand_m.group(1).strip()}")
        if price_m: snippets.append(f"Price: {price_m.group(1).strip()}")
        if original_price_m: snippets.append(f"Original Price: {original_price_m.group(1).strip()}")
        if discount_percentage_m: snippets.append(f"Discount: {discount_percentage_m.group(1).strip()}")
        if rating_m:
            r = rating_m.group(1)
            rc = rating_m.group(2)
            rtxt = f"Rating: {r}/5"
            if rc: rtxt += f" ({rc} ratings)"
            snippets.append(rtxt)
        if sales_m: snippets.append(f"Sales: {sales_m.group(1).strip()}")
        if colors_m: snippets.append(f"Colors: {colors_m.group(1).strip()}")
        if sizes_m: snippets.append(f"Sizes: {sizes_m.group(1).strip()}")
        if variants_m:
            variants = [line.strip("- ").strip() for line in variants_m.group(1).strip().splitlines() if line.strip()]
            if variants:
                snippets.append("Variants: " + ", ".join(variants))
        if desc_m:
            snippets.append("Description: " + desc_m.group(1).strip())
        if url:
            snippets.append(f"URL: {url}")

        text_for_embed = "\n".join(snippets) if snippets else body

        items.append({
            "id": meta.get("id") or sha1(text_for_embed)[:12],
            "category": meta.get("category"),
            "title": title or "",
            "url": url,
            "text": text_for_embed.strip(),
            "raw": body,  # for nice rendering later
        })
    return items

def ensure_client() -> OpenAI:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY. Put it in .env or your environment.")
    return OpenAI()

def embed_texts(client: OpenAI, texts: List[str]) -> np.ndarray:
    vectors = []
    for i in range(0, len(texts), EMBED_BATCH):
        batch = texts[i:i+EMBED_BATCH]
        resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
        vectors.extend([e.embedding for e in resp.data])
        time.sleep(SLEEP_BETWEEN)
    return np.array(vectors, dtype="float32")

def build_or_load_index(products: List[Dict[str, Any]]) -> Tuple[faiss.Index, List[Dict[str, Any]]]:
    """
    Builds FAISS index if not present, else loads it.
    Uses a stable hash over the input MD + model + TOP_K to detect changes.
    """
    # Hash corpus IDs + text
    corpus_sig = sha1("\n".join([p["id"] + "\t" + p["text"] for p in products]))
    idx_stem = f"products_{corpus_sig}_{EMBED_MODEL.replace(':','_')}"
    faiss_path = INDEX_DIR / f"{idx_stem}.faiss"
    meta_path = INDEX_DIR / f"{idx_stem}.jsonl"
    ids_path = INDEX_DIR / f"{idx_stem}.ids.npy"

    if faiss_path.exists() and meta_path.exists() and ids_path.exists():
        index = faiss.read_index(str(faiss_path))
        with meta_path.open("r", encoding="utf-8") as f:
            meta = [json.loads(line) for line in f]
        ids = np.load(ids_path)
        assert index.ntotal == len(meta) == len(ids)
        return index, meta

    # Build new
    client = ensure_client()
    texts = [p["text"] for p in products]
    embs = embed_texts(client, texts)

    # Normalize for cosine similarity (IndexFlatIP)
    faiss.normalize_L2(embs)
    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)

    faiss.write_index(index, str(faiss_path))
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
    np.save(ids_path, np.arange(len(products), dtype=np.int64))
    return index, products

def retrieve(index: faiss.Index, meta: List[Dict[str, Any]], client: OpenAI, query: str, k: int = TOP_K):
    q_emb = client.embeddings.create(model=EMBED_MODEL, input=[query]).data[0].embedding
    q = np.array([q_emb], dtype="float32")
    faiss.normalize_L2(q)
    D, I = index.search(q, k)
    hits = []
    for score, idx in zip(D[0], I[0]):
        if idx == -1: 
            continue
        rec = meta[idx]
        hits.append({
            "score": float(score),
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
st.set_page_config(page_title="Products RAG", page_icon="🛒", layout="wide")

st.title("🛒 Daraz Products RAG")

# Show available categories 
@st.cache_data(show_spinner=False)
def _extract_categories(products_list):
    """Extract unique categories from products."""
    categories = set()
    for product in products_list:
        if product.get("category"):
            categories.add(product["category"])
    return sorted(list(categories))

with st.sidebar:
    st.header("Index")
    st.write("This app ingests your Markdown corpus at:")
    st.code(str(MD_PATH), language="bash")
    st.write("Models:")
    st.write(f"- Embeddings: `{EMBED_MODEL}`")
    st.write(f"- Chat: `{CHAT_MODEL}`")
    st.divider()
    st.caption("Tip: First run will build a local FAISS index. Subsequent runs load instantly.")

# Load/Build index (cached across Streamlit reruns)
@st.cache_data(show_spinner=True)
def _load_products(md_text: str):
    return parse_markdown_products(md_text)

@st.cache_resource(show_spinner=True)
def _load_index(md_text: str):
    products = parse_markdown_products(md_text)
    index, meta = build_or_load_index(products)
    return index, meta

# Gate on presence of corpus
if not MD_PATH.exists():
    st.error(f"Markdown corpus not found at {MD_PATH}. Generate it first.")
    st.stop()

md_text = MD_PATH.read_text(encoding="utf-8")

# Quick stats
products = _load_products(md_text)
st.write(f"**Loaded products**: {len(products):,}")

# Show available categories
categories = _extract_categories(products)
if categories:
    st.write(f"**Available categories** ({len(categories)}):")
    cols = st.columns(3)  # Display in 3 columns for better layout
    for i, category in enumerate(categories):
        with cols[i % 3]:
            st.caption(f"• {category}")
else:
    st.write("**Categories**: None found")

index, meta = _load_index(md_text)

# Query box
query = st.text_input("Ask a question about your products (e.g., 'best budget bedding set under 2000 BDT with good ratings')", "")
btn = st.button("Search")

if btn and query.strip():
    try:
        client = ensure_client()
        with st.spinner("Retrieving…"):
            hits = retrieve(index, meta, client, query, k=TOP_K)

        if not hits:
            st.warning("No relevant products found.")
        else:
            # Show retrieved cards
            st.subheader("Top matches")
            for h in hits:
                with st.expander(f"[{h['id']}] {h['title'] or '(no title)'}  —  score {h['score']:.3f}"):
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
