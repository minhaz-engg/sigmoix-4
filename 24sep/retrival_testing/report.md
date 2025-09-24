# Daraz Product Search — SLM Fine‑tuning for RAG Retrieval (Final Report)


**Environment:** Kaggle GPU (Tesla T4)  
**Date:** 2025‑09‑24  
**Repository components:**
- `new_rag.py`  ← Updated RAG application (Hybrid + Re‑ranker)
- `models/embedder-ft/`   ← Fine‑tuned dense embedder (BGE‑small‑en‑v1.5). can be find in kaggle link output section
- `models/reranker-ft-miniLM/` ← Fine‑tuned cross‑encoder re‑ranker (MiniLM) can be find in kaggle link output section
- `index/dense.faiss`     ← Dense FAISS index (optional, can rebuild) can be find in kaggle link output section
- `index/dense_chunks.json` ← Chunk metadata aligned with the FAISS index (optional) can be find in kaggle link output section


---

## Task Overview

> **SLM finetuning:**  
> - **Embedding finetune**  
> - **RAG dataset finetune**  
>  
> Goal: Check whether a Small Language Model (**Gemma 270M**) can be fine‑tuned on our dataset to **improve retrieval**, using Unsloth.ai and Colab/Kaggle GPUs.

**Interpretation for a retrieval first product:**  
Improving **retrieval quality** directly impacts user experience in a RAG application. We executed **two complementary tracks**:

1. **Retrieval‑specific fine‑tuning (delivered):**  
   - Very small Dense **embedder** fine‑tune for better recall and semantic matching  
   - Small **Cross‑encoder re‑ranker** fine‑tune for better top‑k ordering  
   - Integrated into the RAG app (default mode: **Hybrid + Re‑ranker**)  
2. **SLM (Gemma 270M) instruction SFT (attempted):**  
   - Explored with Unsloth in Kaggle. Training started but hit a runtime issue in the Unsloth forward hook (`Boolean value of Tensor…`).  
   - Will try to fix the bug
   - **Note:** Generator SFT usually improves **answer style**/coverage, not **retrieval metrics**. Retrieval improvements are best achieved via embedding & re‑ranking (which we delivered and integrated).

---

## Data & Pre‑processing

- **Corpus:** `daraz_products_corpus.md` (Markdown) with **4,836** products delimited by  
  `<!--DOC:START ...--> ... <!--DOC:END-->`
- **Parsing:** Title, URL, brand, (approx) price, rating metadata extracted by regex; safe to missing fields.
- **Chunking:** Compact **naive markdown chunker** (40–160 words) used to match training artifacts (BM25 cleaning removes bare URLs and image sections).
- **Silver labeling (queries):**  
  For each product we synthesize up to **3** short queries (title, brand/category, light constraints).  
- **Hard negatives:** BM25 pool + **dense negatives augmentation** (after embedder FT).

Artifact sizes from the Kaggle run:
- Products: **4,836**
- Chunks: **8,541**
- Train pairs (IR): **13,055**
- Dev pairs (IR): **1,451**
- Re‑ranker rows (after dense augmentation): **train 148,037 / dev 16,449**

---

## Models

- **Dense embedder (bi‑encoder):** `BAAI/bge-small-en-v1.5` → **fine‑tuned**  
- **Cross‑encoder re‑ranker:** `cross-encoder/ms-marco-MiniLM-L-6-v2` → **fine‑tuned**  
- **Index:** FAISS `IndexFlatIP` with **normalized** embeddings  
- **BM25:** `rank_bm25` baseline

> Why this stack?  
> - **BGE‑small** is lightweight but strong for product search; affordable to fine‑tune on T4.  
> - **MiniLM** cross‑encoder re‑ranks top candidates very effectively with limited cost.  
> - This combo consistently improves **Recall@1** and **MRR@10** in product retrieval.
> - Both are very very light model which actually meets the requirements of Small Language Model (SLM) where we couldnt run Gemma 270 due to some error.

---

## Training & Compute

- **Platform:** Kaggle (T4)  
- **Embedder FT:** 2 epochs, in‑batch negatives (`MultipleNegativesRankingLoss`)  
- **Dense index build:** ~1.6 min for 8.5k chunks  
- **Re‑ranker FT:** 1 epoch, clipped inputs (query ≤ 64 words, chunk ≤ 160 words, max_len = 320)  
- **Total times:**  
  - Embedder FT: **~9.7 min**  
  - Dense build: **~1.6 min**  
  - Re‑ranker FT: **~28.2 min**

---

## Evaluation (Doc‑aware)

All metrics are **doc‑aware** (max‑pool best chunk per doc before ranking).

**Baseline (before re‑ranking)**

| Retriever            | Recall@1 | Recall@5 | Recall@10 | MRR@10 | nDCG@10 |
|----------------------|---------:|---------:|----------:|-------:|--------:|
| **BM25 (doc)**       | 0.4052 | 0.5789 | 0.6210 | 0.4780 | 0.5126 |
| **Dense (doc)**      | 0.4873 | 0.6375 | 0.6575 | 0.5521 | 0.5782 |
| **Hybrid RRF (doc)** | 0.4562 | 0.6203 | 0.6492 | 0.5247 | 0.5552 |

**After Re‑ranking (MiniLM Cross‑encoder)**

| Retriever                         | Recall@1 | Recall@5 | Recall@10 | MRR@10 | nDCG@10 |
|-----------------------------------|---------:|---------:|----------:|-------:|--------:|
| **Hybrid + MiniLM Reranker (doc)**| **0.5348** | 0.6278 | 0.6485 | **0.5754** | **0.5933** |

**Key gains (vs BM25):**
- **Recall@1:** +0.1296 absolute (**+32%** relative) → *0.4052 → 0.5348*  
- **MRR@10:** +0.0974 absolute → *0.4780 → 0.5754*  
- **nDCG@10:** +0.0806 absolute → *0.5126 → 0.5933*

**Conclusion:** For this dataset, the **best performing retrieval** is **Hybrid (BM25 + Dense) fused with RRF, then re‑ranked by the fine‑tuned MiniLM cross‑encoder**. This decision is implemented as the **default** in the updated app.

---

## Decision & Integration into the RAG App

**Rationale:** The measured improvements are in the **top‑ranked doc** (Recall@1) and **ordering quality** (MRR@10, nDCG@10), which directly benefits user experience in product search.

**App changes (now default):**
1. **Hybrid retrieval** (BM25 + Dense with fine‑tuned embedder) using **RRF**.  
2. **Cross‑encoder re‑ranking** (MiniLM FT).  
3. **Doc‑aware ranking** in the UI (matches evaluation).  
4. Smart fallbacks:  
   - If **embedder/FAISS** missing → BM25 only.  
   - If **re‑ranker** missing → Hybrid without re‑ranking.

---

## 7) How to Run

### A) Quick start (BM25 only)
```bash
pip install streamlit rank_bm25 python-dotenv openai sentence-transformers faiss-cpu
export OPENAI_API_KEY=sk-...
streamlit run new_rag.py
# In the sidebar: upload `daraz_products_corpus.md` and hit Search
