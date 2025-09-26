# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# Builds an advanced multi-task fine-tuning dataset from your Daraz RAG corpus.

# Input:
#     out/daraz_products_corpus.jsonl
#       Each line: {"id": str, "text": str, "metadata": {...fields...}}

# Outputs:
#     out/ft/ecom_sft_advanced.jsonl      # Supervised fine-tuning (chat format)
#     out/ft/ecom_sft_advanced.train.jsonl
#     out/ft/ecom_sft_advanced.val.jsonl
#     out/ft/ecom_dpo_pairs.jsonl         # Optional: preference pairs (chosen/rejected)
#     out/ft/stats.json                   # Basic dataset stats

# Design principles:
# - Deterministic labels from metadata only (no hallucination)
# - Multi-task:
#     1) Buyer summary (EN/BN)
#     2) Product JSON extraction (strict schema + normalized fields)
#     3) SEO title (<= 70 chars)
#     4) Feature bullets (Markdown)
#     5) FAQ (Q&A) from available fields
#     6) Compact Markdown table card
# - Style diversity and multilingual prompts
# - Clean, FT-ready chat format with system/user/assistant messages
# """

# from __future__ import annotations
# import json, re, random, math, argparse, os
# from pathlib import Path
# from typing import Dict, Any, List, Tuple, Optional

# # ----------------------------
# # Config
# # ----------------------------
# IN_JSONL = Path("out/daraz_products_corpus.jsonl")
# OUT_DIR  = Path("out/ft")
# TRAIN_F  = OUT_DIR / "ecom_sft_advanced.train.jsonl"
# VAL_F    = OUT_DIR / "ecom_sft_advanced.val.jsonl"
# ALL_F    = OUT_DIR / "ecom_sft_advanced.jsonl"
# DPO_F    = OUT_DIR / "ecom_dpo_pairs.jsonl"
# STATS_F  = OUT_DIR / "stats.json"

# SEED = 3407
# VAL_RATIO = 0.02                   # ~2% validation
# MAX_CONTEXT_CHARS = 3600           
# MAX_TITLE_CHARS = 70               # SEO constraint
# MAX_BULLETS = 6
# MIN_TASKS_PER_RECORD = 4           # we will sample from available tasks per record
# MAX_TASKS_PER_RECORD = 7

# # ----------------------------
# # Utilities & Normalizers
# # ----------------------------
# PRICE_RE = re.compile(
#     r"(?P<cur>৳|Tk|BDT|\$|USD|₹|Rs|INR|€|GBP|£)?\s*(?P<num>[0-9]{1,3}(?:[, 0-9]{0,})?(?:\.[0-9]+)?)",
#     re.IGNORECASE,
# )

# def _strip(s: Optional[str]) -> Optional[str]:
#     if s is None: return None
#     s = re.sub(r"\s+", " ", str(s)).strip()
#     return s or None

# def _as_float(x) -> Optional[float]:
#     try:
#         if x is None: return None
#         if isinstance(x, (int, float)): return float(x)
#         x = str(x).replace(",", "").strip()
#         return float(x) if x else None
#     except Exception:
#         return None

# def parse_price_display(s: Optional[str]) -> Dict[str, Any]:
#     """Parse '৳1,499' / 'BDT 1499' / '$12.99' → {currency, value} if possible."""
#     if not s: return {"currency": None, "value": None, "raw": None}
#     m = PRICE_RE.search(s)
#     if not m: return {"currency": None, "value": None, "raw": s}
#     cur = m.group("cur")
#     num = m.group("num")
#     val = _as_float(num)
#     # Normalize currency codes where obvious
#     if cur:
#         cur = cur.strip()
#         cur_map = {"৳": "BDT", "Tk": "BDT", "BDT": "BDT", "$": "USD", "₹": "INR", "Rs": "INR", "£": "GBP"}
#         cur = cur_map.get(cur, cur.upper())
#     return {"currency": cur or None, "value": val, "raw": s}

# def cap_len(text: str, n: int) -> str:
#     if text is None: return ""
#     text = text.strip()
#     return text if len(text) <= n else (text[:n-1] + "…")

# def join_nonempty(parts: List[str], sep=" ") -> str:
#     return sep.join([p for p in parts if p])

# def pick_lang_variants() -> List[str]:
#     """Randomize task language variants: 'en', 'bn', or 'mix' (bn prompt + en answer, etc.)."""
#     # Keep simple: equal probabilities
#     return random.choices(["en", "bn", "mix"], k=MAX_TASKS_PER_RECORD)

# # ----------------------------
# # Deterministic Target Builders
# # ----------------------------
# KEEP_KEYS_JSON = [
#     "title","brand","category","price_display","original_price_display",
#     "discount_display","discount_percent","rating_average","rating_count",
#     "sold_text","url"
# ]

# def make_short_summary_en(meta: Dict[str, Any]) -> str:
#     title  = _strip(meta.get("title"))
#     brand  = _strip(meta.get("brand"))
#     cat    = _strip(meta.get("category"))
#     price  = _strip(meta.get("price_display") or meta.get("original_price_display"))
#     r_avg  = meta.get("rating_average")
#     r_cnt  = meta.get("rating_count")
#     sales  = _strip(meta.get("sold_text"))
#     bits = []
#     if title: bits.append(title)
#     if brand: bits.append(f"by {brand}")
#     if cat:   bits.append(f"({cat})")
#     base = " ".join(bits) if bits else "Product"
#     if price: base += f". Price: {price}"
#     if r_avg is not None:
#         if r_cnt is not None: base += f". Avg rating {r_avg}/5 from {r_cnt} ratings"
#         else: base += f". Avg rating {r_avg}/5"
#     if sales: base += f". Sales: {sales}"
#     return base

# def make_short_summary_bn(meta: Dict[str, Any]) -> str:
#     title  = _strip(meta.get("title"))
#     brand  = _strip(meta.get("brand"))
#     cat    = _strip(meta.get("category"))
#     price  = _strip(meta.get("price_display") or meta.get("original_price_display"))
#     r_avg  = meta.get("rating_average")
#     r_cnt  = meta.get("rating_count")
#     sales  = _strip(meta.get("sold_text"))
#     bits = []
#     if title: bits.append(title)
#     if brand: bits.append(f"({brand})")
#     if cat:   bits.append(f"— {cat}")
#     base = " ".join(bits) if bits else "পণ্য"
#     if price: base += f"। দাম: {price}"
#     if r_avg is not None:
#         if r_cnt is not None: base += f"। গড় রেটিং {r_avg}/5 (মোট {r_cnt} রেটিং)"
#         else: base += f"। গড় রেটিং {r_avg}/5"
#     if sales: base += f"। বিক্রি: {sales}"
#     return base

# def make_json_norm(meta: Dict[str, Any]) -> Dict[str, Any]:
#     """Strict JSON extraction with normalization for price/discount/ratings."""
#     price_disp = _strip(meta.get("price_display") or meta.get("original_price_display"))
#     price_parsed = parse_price_display(price_disp)
#     out = {
#         "title": meta.get("title") or None,
#         "brand": meta.get("brand") or None,
#         "category": meta.get("category") or None,
#         "price_display": _strip(meta.get("price_display")),
#         "original_price_display": _strip(meta.get("original_price_display")),
#         "discount_display": _strip(meta.get("discount_display")),
#         "discount_percent": _as_float(meta.get("discount_percent")),
#         "rating_average": _as_float(meta.get("rating_average")),
#         "rating_count": _as_float(meta.get("rating_count")),
#         "sold_text": _strip(meta.get("sold_text")),
#         "url": _strip(meta.get("url")),
#         # Normalized extras:
#         "normalized": {
#             "price_value": price_parsed["value"],
#             "price_currency": price_parsed["currency"],
#         }
#     }
#     return out

# def make_seo_title(meta: Dict[str, Any], max_len=MAX_TITLE_CHARS) -> str:
#     title  = _strip(meta.get("title"))
#     brand  = _strip(meta.get("brand"))
#     cat    = _strip(meta.get("category"))
#     base = " | ".join([x for x in [title, brand, cat] if x])
#     if not base:
#         base = meta.get("title") or "Product"
#     return cap_len(base, max_len)

# def make_bullets(meta: Dict[str, Any], max_bullets=MAX_BULLETS) -> List[str]:
#     bullets: List[str] = []
#     title  = _strip(meta.get("title"))
#     brand  = _strip(meta.get("brand"))
#     cat    = _strip(meta.get("category"))
#     price  = _strip(meta.get("price_display") or meta.get("original_price_display"))
#     disc   = _strip(meta.get("discount_display"))
#     r_avg  = meta.get("rating_average")
#     r_cnt  = meta.get("rating_count")
#     sales  = _strip(meta.get("sold_text"))
#     url    = _strip(meta.get("url"))

#     if title: bullets.append(f"**Product:** {title}")
#     if brand: bullets.append(f"**Brand:** {brand}")
#     if cat:   bullets.append(f"**Category:** {cat}")
#     if price: bullets.append(f"**Price:** {price}")
#     if disc:  bullets.append(f"**Discount:** {disc}")
#     if r_avg is not None:
#         if r_cnt is not None: bullets.append(f"**Rating:** {r_avg}/5 ({int(r_cnt)} ratings)")
#         else: bullets.append(f"**Rating:** {r_avg}/5")
#     if sales: bullets.append(f"**Sales:** {sales}")
#     if url:   bullets.append(f"**Link:** {url}")
#     return bullets[:max_bullets] if bullets else ["**Product:** Information unavailable"]

# def make_bullets_bn(meta: Dict[str, Any], max_bullets=MAX_BULLETS) -> List[str]:
#     en = make_bullets(meta, max_bullets=max_bullets)
#     bn = []
#     for b in en:
#         b = b.replace("**Product:**", "**পণ্য:**") \
#              .replace("**Brand:**", "**ব্র্যান্ড:**") \
#              .replace("**Category:**", "**ক্যাটাগরি:**") \
#              .replace("**Price:**", "**দাম:**") \
#              .replace("**Discount:**", "**ডিসকাউন্ট:**") \
#              .replace("**Rating:**", "**রেটিং:**") \
#              .replace("**Sales:**", "**বিক্রি:**") \
#              .replace("**Link:**", "**লিংক:**")
#         bn.append(b)
#     return bn

# def make_faq(meta: Dict[str, Any]) -> List[Tuple[str, str]]:
#     """Deterministic Q&A from available fields."""
#     QAs: List[Tuple[str,str]] = []
#     if meta.get("price_display") or meta.get("original_price_display"):
#         QAs.append(("What is the price?",
#                     _strip(meta.get("price_display") or meta.get("original_price_display")) or "Not available"))
#     if meta.get("brand"):
#         QAs.append(("Which brand is it?", meta["brand"]))
#     if meta.get("category"):
#         QAs.append(("Which category does it belong to?", meta["category"]))
#     if meta.get("discount_display"):
#         QAs.append(("Is there any discount?",
#                     f"Yes — {meta['discount_display']}"))
#     if meta.get("rating_average") is not None:
#         if meta.get("rating_count") is not None:
#             QAs.append(("How is it rated?",
#                         f"{meta['rating_average']}/5 from {int(meta['rating_count'])} ratings"))
#         else:
#             QAs.append(("How is it rated?", f"{meta['rating_average']}/5"))
#     if meta.get("sold_text"):
#         QAs.append(("How many sold?", meta["sold_text"]))
#     if meta.get("url"):
#         QAs.append(("Where can I view it?", meta["url"]))
#     # Fallback
#     if not QAs:
#         QAs.append(("What is this product?", _strip(meta.get("title")) or "Not available"))
#     return QAs

# def make_faq_bn(meta: Dict[str, Any]) -> List[Tuple[str, str]]:
#     en = make_faq(meta)
#     bn: List[Tuple[str,str]] = []
#     trans_q = {
#         "What is the price?": "দাম কত?",
#         "Which brand is it?": "এটি কোন ব্র্যান্ডের?",
#         "Which category does it belong to?": "এটি কোন ক্যাটাগরির পণ্য?",
#         "Is there any discount?": "কোন ডিসকাউন্ট আছে কি?",
#         "How is it rated?": "রেটিং কেমন?",
#         "How many sold?": "কতটি বিক্রি হয়েছে?",
#         "Where can I view it?": "কোথায় দেখতে পারি?",
#         "What is this product?": "এটি কী পণ্য?"
#     }
#     for q, a in en:
#         bn.append((trans_q.get(q, "এই পণ্য সম্পর্কে বলুন"), a))
#     return bn

# def make_markdown_table(meta: Dict[str, Any]) -> str:
#     rows = [
#         ("Title", _strip(meta.get("title")) or "-"),
#         ("Brand", _strip(meta.get("brand")) or "-"),
#         ("Category", _strip(meta.get("category")) or "-"),
#         ("Price", _strip(meta.get("price_display") or meta.get("original_price_display")) or "-"),
#         ("Discount", _strip(meta.get("discount_display")) or "-"),
#         ("Rating", f"{meta.get('rating_average')}/5" if meta.get("rating_average") is not None else "-"),
#         ("Ratings Count", str(int(meta["rating_count"])) if meta.get("rating_count") is not None else "-"),
#         ("Sales", _strip(meta.get("sold_text")) or "-"),
#         ("URL", _strip(meta.get("url")) or "-"),
#     ]
#     header = "| Field | Value |\n|---|---|"
#     body = "\n".join([f"| {k} | {v} |" for k,v in rows])
#     return header + "\n" + body

# # ----------------------------
# # Builders for Chat Samples
# # ----------------------------
# def build_samples_for_record(rec: Dict[str, Any]) -> List[Dict[str, Any]]:
#     """Return a list of chat-format samples (system/user/assistant) for one product."""
#     meta = rec.get("metadata") or {}
#     text = (rec.get("text") or "").strip()[:MAX_CONTEXT_CHARS]
#     if not meta and not text:
#         return []

#     samples: List[Dict[str, Any]] = []
#     langs = pick_lang_variants()

#     # Task 1: Buyer summary (EN/BN)
#     sum_en = make_short_summary_en(meta)
#     sum_bn = make_short_summary_bn(meta)

#     # English prompt
#     samples.append({
#         "messages": [
#             {"role":"system","content":"You are a helpful e-commerce assistant."},
#             {"role":"user","content":"Write a concise 2–3 sentence buyer summary from the context.\n\n=== PRODUCT CONTEXT ===\n"+text},
#             {"role":"assistant","content":sum_en},
#         ]
#     })
#     # Bangla prompt
#     samples.append({
#         "messages": [
#             {"role":"system","content":"আপনি একজন সহায়ক ই-কমার্স সহকারী।"},
#             {"role":"user","content":"নিচের কনটেক্সট থেকে ২–৩ বাক্যের সংক্ষিপ্ত ক্রেতা-সংক্ষেপ লিখুন।\n\n=== পণ্য কনটেক্সট ===\n"+text},
#             {"role":"assistant","content":sum_bn},
#         ]
#     })

#     # Task 2: Strict JSON extraction with normalization (EN)
#     gold_json = make_json_norm(meta)
#     samples.append({
#         "messages":[
#             {"role":"system","content":"You convert product context into strict JSON following the schema and do light normalization."},
#             {"role":"user","content":
#                 "Extract JSON with keys: [title, brand, category, price_display, original_price_display, "
#                 "discount_display, discount_percent, rating_average, rating_count, sold_text, url, "
#                 "normalized: {price_value, price_currency}].\n"
#                 "Rules: null if unknown; numbers as numbers; keep strings verbatim.\n\n=== PRODUCT CONTEXT ===\n"+text},
#             {"role":"assistant","content":json.dumps(gold_json, ensure_ascii=False)}
#         ]
#     })

#     # Task 3: SEO title (<= 70 chars)
#     seo = make_seo_title(meta, MAX_TITLE_CHARS)
#     samples.append({
#         "messages":[
#             {"role":"system","content":"You write short, SEO-friendly, non-clickbait titles."},
#             {"role":"user","content":f"Create an SEO title under {MAX_TITLE_CHARS} characters for this product.\n\n=== PRODUCT CONTEXT ===\n{text}"},
#             {"role":"assistant","content":seo}
#         ]
#     })

#     # Task 4: Feature bullets (Markdown). EN and BN
#     bullets_en = make_bullets(meta)
#     bullets_bn = make_bullets_bn(meta)
#     samples.append({
#         "messages":[
#             {"role":"system","content":"You write crisp product bullets in Markdown."},
#             {"role":"user","content":"List up to 6 concise product bullets in Markdown based on the context.\n\n=== PRODUCT CONTEXT ===\n"+text},
#             {"role":"assistant","content":"\n".join(f"- {b}" for b in bullets_en)}
#         ]
#     })
#     samples.append({
#         "messages":[
#             {"role":"system","content":"আপনি Markdown এ সংক্ষিপ্ত বুলেট তৈরি করেন।"},
#             {"role":"user","content":"কনটেক্সট দেখে সর্বোচ্চ ৬টি সংক্ষিপ্ত বুলেট লিস্ট তৈরি করুন।\n\n=== পণ্য কনটেক্সট ===\n"+text},
#             {"role":"assistant","content":"\n".join(f"- {b}" for b in bullets_bn)}
#         ]
#     })

#     # Task 5: FAQ (Q&A) EN & BN
#     for q, a in make_faq(meta):
#         samples.append({
#             "messages":[
#                 {"role":"system","content":"You answer clearly and concisely."},
#                 {"role":"user","content":"Q: "+q+"\n\nUse the product context only.\n\n=== PRODUCT CONTEXT ===\n"+text},
#                 {"role":"assistant","content":a}
#             ]
#         })
#     for q, a in make_faq_bn(meta):
#         samples.append({
#             "messages":[
#                 {"role":"system","content":"আপনি পরিষ্কারভাবে সংক্ষিপ্ত উত্তর দেন।"},
#                 {"role":"user","content":"প্রশ্ন: "+q+"\n\nশুধু কনটেক্সট ব্যবহার করুন।\n\n=== পণ্য কনটেক্সট ===\n"+text},
#                 {"role":"assistant","content":a}
#             ]
#         })

#     # Task 6: Compact Markdown table card
#     table_md = make_markdown_table(meta)
#     samples.append({
#         "messages":[
#             {"role":"system","content":"You produce compact Markdown tables without extra text."},
#             {"role":"user","content":"Make a concise two-column Markdown table (Field | Value) for the product.\n\n=== PRODUCT CONTEXT ===\n"+text},
#             {"role":"assistant","content":table_md}
#         ]
#     })

#     # Random capping of tasks per record for size control
#     random.shuffle(samples)
#     n = random.randint(MIN_TASKS_PER_RECORD, min(MAX_TASKS_PER_RECORD, len(samples)))
#     return samples[:n]

# # ----------------------------
# # DPO Pairs (optional)
# # ----------------------------
# def make_dpo_pair(meta: Dict[str, Any]) -> Optional[Dict[str, str]]:
#     """Create a deterministic pair: both obey content, but one violates clear constraints (rejected)."""
#     title_good = make_seo_title(meta, MAX_TITLE_CHARS)
#     title_bad  = (make_seo_title(meta, 120) + " ⭐🔥 SUPER SALE!!!!")  # too long + emojis + clickbait
#     prompt = (
#         "Create an SEO-friendly title under 70 characters for this product. "
#         "No emojis, no clickbait, do not exceed 70 characters."
#     )
#     return {
#         "prompt": prompt,
#         "chosen": title_good,
#         "rejected": title_bad,
#     }

# # ----------------------------
# # IO
# # ----------------------------
# def iter_jsonl(path: Path):
#     with path.open("r", encoding="utf-8") as f:
#         for line in f:
#             line = line.strip()
#             if not line: continue
#             try:
#                 yield json.loads(line)
#             except Exception:
#                 continue

# def write_jsonl(path: Path, rows: List[Dict[str, Any]]):
#     path.parent.mkdir(parents=True, exist_ok=True)
#     with path.open("w", encoding="utf-8") as w:
#         for r in rows:
#             w.write(json.dumps(r, ensure_ascii=False) + "\n")

# # ----------------------------
# # Main
# # ----------------------------
# def main():
#     random.seed(SEED)
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--in_jsonl", type=str, default=str(IN_JSONL))
#     parser.add_argument("--out_dir", type=str, default=str(OUT_DIR))
#     parser.add_argument("--make_dpo", action="store_true", help="Also emit ecom_dpo_pairs.jsonl")
#     parser.add_argument("--max_products", type=int, default=None, help="Cap products processed")
#     args = parser.parse_args()

#     in_path = Path(args.in_jsonl)
#     out_dir = Path(args.out_dir)
#     out_dir.mkdir(parents=True, exist_ok=True)

#     if not in_path.exists():
#         raise SystemExit(f"[ERR] Input not found: {in_path}")

#     # Load products
#     records = []
#     for rec in iter_jsonl(in_path):
#         if isinstance(rec, dict) and "metadata" in rec and "text" in rec:
#             records.append(rec)
#     if args.max_products:
#         records = random.sample(records, min(args.max_products, len(records)))

#     # Build SFT samples
#     sft_rows = []
#     dpo_rows = []
#     total_products = 0
#     for r in records:
#         total_products += 1
#         samples = build_samples_for_record(r)
#         for s in samples:
#             # Store as chat format (messages field). Keep a small meta anchor.
#             sft_rows.append({
#                 "id": r.get("id"),
#                 "messages": s["messages"],
#             })
#         if args.make_dpo:
#             pair = make_dpo_pair(r.get("metadata") or {})
#             if pair:
#                 dpo_rows.append(pair)

#     # Shuffle and split
#     random.shuffle(sft_rows)
#     n_val = max(1, int(len(sft_rows) * VAL_RATIO))
#     val_rows = sft_rows[:n_val]
#     train_rows = sft_rows[n_val:]

#     # Write files
#     write_jsonl(ALL_F, sft_rows)
#     write_jsonl(TRAIN_F, train_rows)
#     write_jsonl(VAL_F, val_rows)
#     if args.make_dpo and dpo_rows:
#         write_jsonl(DPO_F, dpo_rows)

#     # Stats
#     stats = {
#         "total_products_seen": total_products,
#         "total_sft_samples": len(sft_rows),
#         "train_samples": len(train_rows),
#         "val_samples": len(val_rows),
#         "dpo_pairs": len(dpo_rows),
#         "val_ratio": VAL_RATIO,
#         "max_context_chars": MAX_CONTEXT_CHARS,
#         "tasks": [
#             "Buyer summary (EN/BN)",
#             "Strict JSON extraction with normalization",
#             f"SEO title (<= {MAX_TITLE_CHARS} chars)",
#             "Feature bullets (Markdown, EN/BN)",
#             "FAQ Q&A (EN/BN)",
#             "Compact Markdown table",
#         ],
#     }
#     with STATS_F.open("w", encoding="utf-8") as w:
#         json.dump(stats, w, ensure_ascii=False, indent=2)

#     print("[OK] Wrote:")
#     print("  SFT all :", ALL_F)
#     print("  SFT train:", TRAIN_F)
#     print("  SFT val  :", VAL_F)
#     if args.make_dpo:
#         print("  DPO pairs:", DPO_F)
#     print("  Stats    :", STATS_F)

# if __name__ == "__main__":
#     main()


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Builds an advanced EN-only multi-task fine-tuning dataset from your Daraz RAG corpus.

Input:
    out/daraz_products_corpus.jsonl
      Each line: {"id": str, "text": str, "metadata": {...fields...}}

Outputs:
    out/ft/ecom_sft_advanced.jsonl      # Supervised fine-tuning (chat format)
    out/ft/ecom_sft_advanced.train.jsonl
    out/ft/ecom_sft_advanced.val.jsonl
    out/ft/ecom_dpo_pairs.jsonl         # Optional: preference pairs (chosen/rejected)
    out/ft/stats.json                   # Basic dataset stats

Design principles:
- Deterministic labels from metadata only (no hallucination)
- Multi-task (EN only):
    1) Buyer summary
    2) Product JSON extraction (strict schema + normalized fields)
    3) SEO title (<= 70 chars)
    4) Feature bullets (Markdown)
    5) FAQ (Q&A) from available fields
    6) Compact Markdown table card
- Clean, FT-ready chat format with system/user/assistant messages
"""

from __future__ import annotations
import json, re, random, argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

# ----------------------------
# Config
# ----------------------------
IN_JSONL = Path("out/daraz_products_corpus.jsonl")
OUT_DIR  = Path("out/ft")
TRAIN_F  = OUT_DIR / "ecom_sft_advanced.train.jsonl"
VAL_F    = OUT_DIR / "ecom_sft_advanced.val.jsonl"
ALL_F    = OUT_DIR / "ecom_sft_advanced.jsonl"
DPO_F    = OUT_DIR / "ecom_dpo_pairs.jsonl"
STATS_F  = OUT_DIR / "stats.json"

SEED = 3407
VAL_RATIO = 0.02                   # ~2% validation
MAX_CONTEXT_CHARS = 1800
MAX_TITLE_CHARS = 70               # SEO constraint
MAX_BULLETS = 6
MIN_TASKS_PER_RECORD = 3           # sample from available tasks per record
MAX_TASKS_PER_RECORD = 5

# ----------------------------
# Utilities & Normalizers
# ----------------------------
PRICE_RE = re.compile(
    r"(?P<cur>৳|Tk|BDT|\$|USD|₹|Rs|INR|€|GBP|£)?\s*(?P<num>[0-9]{1,3}(?:[, 0-9]{0,})?(?:\.[0-9]+)?)",
    re.IGNORECASE,
)

def _strip(s: Optional[str]) -> Optional[str]:
    if s is None: return None
    s = re.sub(r"\s+", " ", str(s)).strip()
    return s or None

def _as_float(x) -> Optional[float]:
    try:
        if x is None: return None
        if isinstance(x, (int, float)): return float(x)
        x = str(x).replace(",", "").strip()
        return float(x) if x else None
    except Exception:
        return None

def parse_price_display(s: Optional[str]) -> Dict[str, Any]:
    """Parse '৳1,499' / 'BDT 1499' / '$12.99' → {currency, value} if possible."""
    if not s: return {"currency": None, "value": None, "raw": None}
    m = PRICE_RE.search(s)
    if not m: return {"currency": None, "value": None, "raw": s}
    cur = m.group("cur")
    num = m.group("num")
    val = _as_float(num)
    if cur:
        cur = cur.strip()
        cur_map = {"৳": "BDT", "Tk": "BDT", "BDT": "BDT", "$": "USD", "₹": "INR", "Rs": "INR", "£": "GBP"}
        cur = cur_map.get(cur, cur.upper())
    return {"currency": cur or None, "value": val, "raw": s}

def cap_len(text: str, n: int) -> str:
    if text is None: return ""
    text = text.strip()
    return text if len(text) <= n else (text[:n-1] + "…")

# ----------------------------
# Deterministic Target Builders
# ----------------------------
KEEP_KEYS_JSON = [
    "title","brand","category","price_display","original_price_display",
    "discount_display","discount_percent","rating_average","rating_count",
    "sold_text","url"
]

def make_short_summary(meta: Dict[str, Any]) -> str:
    title  = _strip(meta.get("title"))
    brand  = _strip(meta.get("brand"))
    cat    = _strip(meta.get("category"))
    price  = _strip(meta.get("price_display") or meta.get("original_price_display"))
    r_avg  = meta.get("rating_average")
    r_cnt  = meta.get("rating_count")
    sales  = _strip(meta.get("sold_text"))
    bits = []
    if title: bits.append(title)
    if brand: bits.append(f"by {brand}")
    if cat:   bits.append(f"({cat})")
    base = " ".join(bits) if bits else "Product"
    if price: base += f". Price: {price}"
    if r_avg is not None:
        if r_cnt is not None: base += f". Avg rating {r_avg}/5 from {r_cnt} ratings"
        else: base += f". Avg rating {r_avg}/5"
    if sales: base += f". Sales: {sales}"
    return base

def make_json_norm(meta: Dict[str, Any]) -> Dict[str, Any]:
    """Strict JSON extraction with normalization for price/discount/ratings."""
    price_disp = _strip(meta.get("price_display") or meta.get("original_price_display"))
    price_parsed = parse_price_display(price_disp)
    out = {
        "title": meta.get("title") or None,
        "brand": meta.get("brand") or None,
        "category": meta.get("category") or None,
        "price_display": _strip(meta.get("price_display")),
        "original_price_display": _strip(meta.get("original_price_display")),
        "discount_display": _strip(meta.get("discount_display")),
        "discount_percent": _as_float(meta.get("discount_percent")),
        "rating_average": _as_float(meta.get("rating_average")),
        "rating_count": _as_float(meta.get("rating_count")),
        "sold_text": _strip(meta.get("sold_text")),
        "url": _strip(meta.get("url")),
        "normalized": {
            "price_value": price_parsed["value"],
            "price_currency": price_parsed["currency"],
        }
    }
    return out

def make_seo_title(meta: Dict[str, Any], max_len=MAX_TITLE_CHARS) -> str:
    title  = _strip(meta.get("title"))
    brand  = _strip(meta.get("brand"))
    cat    = _strip(meta.get("category"))
    base = " | ".join([x for x in [title, brand, cat] if x]) or (meta.get("title") or "Product")
    return cap_len(base, max_len)

def make_bullets(meta: Dict[str, Any], max_bullets=MAX_BULLETS) -> List[str]:
    bullets: List[str] = []
    title  = _strip(meta.get("title"))
    brand  = _strip(meta.get("brand"))
    cat    = _strip(meta.get("category"))
    price  = _strip(meta.get("price_display") or meta.get("original_price_display"))
    disc   = _strip(meta.get("discount_display"))
    r_avg  = meta.get("rating_average")
    r_cnt  = meta.get("rating_count")
    sales  = _strip(meta.get("sold_text"))
    url    = _strip(meta.get("url"))

    if title: bullets.append(f"**Product:** {title}")
    if brand: bullets.append(f"**Brand:** {brand}")
    if cat:   bullets.append(f"**Category:** {cat}")
    if price: bullets.append(f"**Price:** {price}")
    if disc:  bullets.append(f"**Discount:** {disc}")
    if r_avg is not None:
        if r_cnt is not None: bullets.append(f"**Rating:** {r_avg}/5 ({int(r_cnt)} ratings)")
        else: bullets.append(f"**Rating:** {r_avg}/5")
    if sales: bullets.append(f"**Sales:** {sales}")
    if url:   bullets.append(f"**Link:** {url}")
    return bullets[:max_bullets] if bullets else ["**Product:** Information unavailable"]

def make_faq(meta: Dict[str, Any]) -> List[Tuple[str, str]]:
    """Deterministic Q&A from available fields."""
    QAs: List[Tuple[str,str]] = []
    if meta.get("price_display") or meta.get("original_price_display"):
        QAs.append(("What is the price?",
                    _strip(meta.get("price_display") or meta.get("original_price_display")) or "Not available"))
    if meta.get("brand"):
        QAs.append(("Which brand is it?", meta["brand"]))
    if meta.get("category"):
        QAs.append(("Which category does it belong to?", meta["category"]))
    if meta.get("discount_display"):
        QAs.append(("Is there any discount?", f"Yes — {meta['discount_display']}"))
    if meta.get("rating_average") is not None:
        if meta.get("rating_count") is not None:
            QAs.append(("How is it rated?",
                        f"{meta['rating_average']}/5 from {int(meta['rating_count'])} ratings"))
        else:
            QAs.append(("How is it rated?", f"{meta['rating_average']}/5"))
    if meta.get("sold_text"):
        QAs.append(("How many sold?", meta["sold_text"]))
    if meta.get("url"):
        QAs.append(("Where can I view it?", meta["url"]))
    if not QAs:
        QAs.append(("What is this product?", _strip(meta.get("title")) or "Not available"))
    return QAs

def make_markdown_table(meta: Dict[str, Any]) -> str:
    rows = [
        ("Title", _strip(meta.get("title")) or "-"),
        ("Brand", _strip(meta.get("brand")) or "-"),
        ("Category", _strip(meta.get("category")) or "-"),
        ("Price", _strip(meta.get("price_display") or meta.get("original_price_display")) or "-"),
        ("Discount", _strip(meta.get("discount_display")) or "-"),
        ("Rating", f"{meta.get('rating_average')}/5" if meta.get("rating_average") is not None else "-"),
        ("Ratings Count", str(int(meta["rating_count"])) if meta.get("rating_count") is not None else "-"),
        ("Sales", _strip(meta.get("sold_text")) or "-"),
        ("URL", _strip(meta.get("url")) or "-"),
    ]
    header = "| Field | Value |\n|---|---|"
    body = "\n".join([f"| {k} | {v} |" for k,v in rows])
    return header + "\n" + body

# ----------------------------
# Builders for Chat Samples
# ----------------------------
def build_samples_for_record(rec: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return a list of chat-format samples (system/user/assistant) for one product."""
    meta = rec.get("metadata") or {}
    text = (rec.get("text") or "").strip()[:MAX_CONTEXT_CHARS]
    if not meta and not text:
        return []

    samples: List[Dict[str, Any]] = []

    # Task 1: Buyer summary (EN)
    sum_en = make_short_summary(meta)
    samples.append({
        "messages": [
            {"role":"system","content":"You are a helpful e-commerce assistant."},
            {"role":"user","content":"Write a concise 2–3 sentence buyer summary from the context.\n\n=== PRODUCT CONTEXT ===\n"+text},
            {"role":"assistant","content":sum_en},
        ]
    })

    # Task 2: Strict JSON extraction with normalization (EN)
    gold_json = make_json_norm(meta)
    samples.append({
        "messages":[
            {"role":"system","content":"You convert product context into strict JSON following the schema and do light normalization."},
            {"role":"user","content":
                "Extract JSON with keys: [title, brand, category, price_display, original_price_display, "
                "discount_display, discount_percent, rating_average, rating_count, sold_text, url, "
                "normalized: {price_value, price_currency}].\n"
                "Rules: null if unknown; numbers as numbers; keep strings verbatim.\n\n=== PRODUCT CONTEXT ===\n"+text},
            {"role":"assistant","content":json.dumps(gold_json, ensure_ascii=False)}
        ]
    })

    # Task 3: SEO title (<= 70 chars)
    seo = make_seo_title(meta, MAX_TITLE_CHARS)
    samples.append({
        "messages":[
            {"role":"system","content":"You write short, SEO-friendly, non-clickbait titles."},
            {"role":"user","content":f"Create an SEO title under {MAX_TITLE_CHARS} characters for this product.\n\n=== PRODUCT CONTEXT ===\n{text}"},
            {"role":"assistant","content":seo}
        ]
    })

    # Task 4: Feature bullets (Markdown) EN
    bullets_en = make_bullets(meta)
    samples.append({
        "messages":[
            {"role":"system","content":"You write crisp product bullets in Markdown."},
            {"role":"user","content":"List up to 6 concise product bullets in Markdown based on the context.\n\n=== PRODUCT CONTEXT ===\n"+text},
            {"role":"assistant","content":"\n".join(f"- {b}" for b in bullets_en)}
        ]
    })

    # Task 5: FAQ (Q&A) EN
    for q, a in make_faq(meta):
        samples.append({
            "messages":[
                {"role":"system","content":"You answer clearly and concisely."},
                {"role":"user","content":"Q: "+q+"\n\nUse the product context only.\n\n=== PRODUCT CONTEXT ===\n"+text},
                {"role":"assistant","content":a}
            ]
        })

    # Task 6: Compact Markdown table card
    table_md = make_markdown_table(meta)
    samples.append({
        "messages":[
            {"role":"system","content":"You produce compact Markdown tables without extra text."},
            {"role":"user","content":"Make a concise two-column Markdown table (Field | Value) for the product.\n\n=== PRODUCT CONTEXT ===\n"+text},
            {"role":"assistant","content":table_md}
        ]
    })

    # Random cap for size control
    random.shuffle(samples)
    n = random.randint(MIN_TASKS_PER_RECORD, min(MAX_TASKS_PER_RECORD, len(samples)))
    return samples[:n]

# ----------------------------
# DPO Pairs (optional)
# ----------------------------
def make_dpo_pair(meta: Dict[str, Any]) -> Optional[Dict[str, str]]:
    """Create a deterministic pair: both obey content, but one violates clear constraints (rejected)."""
    title_good = make_seo_title(meta, MAX_TITLE_CHARS)
    title_bad  = (make_seo_title(meta, 120) + " ⭐🔥 SUPER SALE!!!!")  # too long + emojis + clickbait
    prompt = (
        "Create an SEO-friendly title under 70 characters for this product. "
        "No emojis, no clickbait, do not exceed 70 characters."
    )
    return {"prompt": prompt, "chosen": title_good, "rejected": title_bad}

# ----------------------------
# IO
# ----------------------------
def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                yield json.loads(line)
            except Exception:
                continue

def write_jsonl(path: Path, rows: List[Dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as w:
        for r in rows:
            w.write(json.dumps(r, ensure_ascii=False) + "\n")

# ----------------------------
# Main
# ----------------------------
def main():
    random.seed(SEED)
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_jsonl", type=str, default=str(IN_JSONL))
    parser.add_argument("--out_dir", type=str, default=str(OUT_DIR))
    parser.add_argument("--make_dpo", action="store_true", help="Also emit ecom_dpo_pairs.jsonl")
    parser.add_argument("--max_products", type=int, default=None, help="Cap products processed")
    args = parser.parse_args()

    in_path = Path(args.in_jsonl)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not in_path.exists():
        raise SystemExit(f"[ERR] Input not found: {in_path}")

    # Load products
    records = []
    for rec in iter_jsonl(in_path):
        if isinstance(rec, dict) and "metadata" in rec and "text" in rec:
            records.append(rec)
    if args.max_products:
        records = random.sample(records, min(args.max_products, len(records)))

    # Build SFT samples
    sft_rows = []
    dpo_rows = []
    total_products = 0
    for r in records:
        total_products += 1
        samples = build_samples_for_record(r)
        for s in samples:
            sft_rows.append({"id": r.get("id"), "messages": s["messages"]})
        if args.make_dpo:
            pair = make_dpo_pair(r.get("metadata") or {})
            if pair:
                dpo_rows.append(pair)

    # Shuffle and split
    random.shuffle(sft_rows)
    n_val = max(1, int(len(sft_rows) * VAL_RATIO))
    val_rows = sft_rows[:n_val]
    train_rows = sft_rows[n_val:]

    # Write files
    write_jsonl(ALL_F, sft_rows)
    write_jsonl(TRAIN_F, train_rows)
    write_jsonl(VAL_F, val_rows)
    if args.make_dpo and dpo_rows:
        write_jsonl(DPO_F, dpo_rows)

    # Stats
    stats = {
        "total_products_seen": total_products,
        "total_sft_samples": len(sft_rows),
        "train_samples": len(train_rows),
        "val_samples": len(val_rows),
        "dpo_pairs": len(dpo_rows),
        "val_ratio": VAL_RATIO,
        "max_context_chars": MAX_CONTEXT_CHARS,
        "tasks": [
            "Buyer summary (EN)",
            "Strict JSON extraction with normalization (EN)",
            f"SEO title (<= {MAX_TITLE_CHARS} chars) (EN)",
            "Feature bullets (Markdown, EN)",
            "FAQ Q&A (EN)",
            "Compact Markdown table (EN)",
        ],
    }
    with STATS_F.open("w", encoding="utf-8") as w:
        json.dump(stats, w, ensure_ascii=False, indent=2)

    print("[OK] Wrote:")
    print("  SFT all :", ALL_F)
    print("  SFT train:", TRAIN_F)
    print("  SFT val  :", VAL_F)
    if args.make_dpo:
        print("  DPO pairs:", DPO_F)
    print("  Stats    :", STATS_F)

if __name__ == "__main__":
    main()
