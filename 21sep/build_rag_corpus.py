#!/usr/bin/env python3
"""
Merge Daraz category `products.json` files into a single RAG-friendly corpus.

- Input  : result/<category>/products.json
- Output : one of {Markdown (.md), JSONL (.jsonl), Plain text (.txt)}

Examples:
  python build_rag_corpus.py --input result --format md    --output daraz_products_corpus.md
  python build_rag_corpus.py --input result --format jsonl --output daraz_products_corpus.jsonl
  python build_rag_corpus.py --input result --format txt   --output daraz_products_corpus.txt
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

def normalize_url(u: Optional[str]) -> Optional[str]:
    if not u:
        return None
    u = u.strip()
    if u.startswith("//"):
        return "https:" + u
    if u.startswith("http://") or u.startswith("https://"):
        return u
    # sometimes product_detail_url and detail_url are full https already; otherwise leave as-is
    return u

def category_readable_name(folder: str) -> str:
    # e.g. "www_daraz_com_bd_shop_bedding_sets" -> "shop bedding sets"
    # fall back: replace underscores with spaces
    slug = folder
    prefix = "www_daraz_com_bd_"
    if folder.startswith(prefix):
        slug = folder[len(prefix):]
    return slug.replace("_", " ").strip()

def safe_get(d: Dict[str, Any], *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

def clean_text(t: Optional[str]) -> Optional[str]:
    if not t:
        return t
    # Collapse excessive whitespace
    t = re.sub(r"\s+\n", "\n", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    t = re.sub(r"[ \t]{2,}", " ", t)
    return t.strip()

def list_preview(items: List[str], max_n: int = 10) -> List[str]:
    out = []
    for s in items[:max_n]:
        if isinstance(s, str):
            s = s.strip()
            if s:
                out.append(s)
    return out

def unique_stable_key(prod: Dict[str, Any]) -> str:
    # priority: data_item_id -> detail.url -> detail_url -> data_sku_simple -> product_title + image_url
    candidates = [
        prod.get("data_item_id"),
        safe_get(prod, "detail", "url"),
        prod.get("detail_url"),
        prod.get("product_detail_url"),
        prod.get("data_sku_simple"),
    ]
    for c in candidates:
        if c:
            return str(c)
    # last resort (unlikely collisions, but better than drop)
    return (prod.get("product_title") or "unknown") + "||" + (prod.get("image_url") or "unknown")

def unify_product(prod: Dict[str, Any], category_folder: str, products_path: Path) -> Dict[str, Any]:
    detail = prod.get("detail") or {}

    title = detail.get("name") or prod.get("product_title") or "Unknown Product"
    brand = detail.get("brand") or None

    # Prices
    price = detail.get("price") or {}
    price_disp = price.get("display") or prod.get("product_price") or None
    price_val = price.get("value")  # numeric or None
    orig_disp = price.get("original_display") or None
    orig_val = price.get("original_value")  # numeric or None
    discount_disp = price.get("discount_display") or None
    discount_pct = price.get("discount_percent")  # numeric or None

    # Ratings
    rating = detail.get("rating") or {}
    rating_avg = rating.get("average")
    rating_count = rating.get("count")
    rating_raw = rating.get("raw")

    # URLs / images
    url = normalize_url(detail.get("url") or prod.get("detail_url") or prod.get("product_detail_url"))
    image_url = prod.get("image_url")
    images = detail.get("images") or ([image_url] if image_url else [])
    images = [normalize_url(i) for i in images if isinstance(i, str) and i and i.startswith(("http://", "https://", "//"))]
    images = list_preview([i for i in images if i], max_n=10)

    # Variations / options
    colors = detail.get("colors") or []
    colors = [c for c in colors if isinstance(c, str)]
    sizes = detail.get("sizes") or []
    sizes = [s for s in sizes if isinstance(s, str)]

    variants = detail.get("variants") or []
    variant_summaries = []
    for v in variants[:20]:  # cap for readability
        v_color = v.get("color")
        v_price = safe_get(v, "price", "display") or safe_get(v, "price", "value")
        vsizes = v.get("sizes")
        bits = []
        if v_color:
            bits.append(str(v_color))
        if v_price:
            bits.append(f"price {v_price}")
        if vsizes and isinstance(vsizes, list):
            bits.append("sizes: " + ", ".join([str(x) for x in vsizes[:5]]))
        if bits:
            variant_summaries.append(" — ".join(bits))

    # Delivery / returns / seller
    delivery_options = detail.get("delivery_options") or []
    ret_warranty = detail.get("return_and_warranty") or []

    seller_name = safe_get(detail, "seller", "name")
    seller_link = normalize_url(safe_get(detail, "seller", "link"))
    seller_metrics = safe_get(detail, "seller", "metrics") or {}

    description_text = clean_text(safe_get(detail, "details", "description_text") or safe_get(detail, "details", "raw_text"))
    highlights = safe_get(detail, "details", "highlights") or []
    specs = safe_get(detail, "details", "specifications") or []

    # other helpful top-level bits
    sku = prod.get("data_sku_simple")
    item_id = prod.get("data_item_id")
    sold_text = prod.get("location")  # e.g., "5.3K sold"
    category_slug = category_readable_name(Path(category_folder).name)

    return {
        "id": str(item_id) if item_id else unique_stable_key(prod),
        "title": str(title),
        "brand": brand,
        "category": category_slug,
        "category_dir": str(Path(category_folder).name),
        "source_file": str(products_path),
        "url": url,
        "sku": sku,
        "image_url": image_url if (isinstance(image_url, str) and image_url.startswith(("http://", "https://", "//"))) else None,
        "images": images,

        "listing_price_display": prod.get("product_price"),
        "price_display": price_disp,
        "price_value": price_val,
        "original_price_display": orig_disp,
        "original_price_value": orig_val,
        "discount_display": discount_disp,
        "discount_percent": discount_pct,

        "rating_average": rating_avg,
        "rating_count": rating_count,
        "rating_raw": rating_raw,

        "sold_text": sold_text,

        "colors": colors,
        "sizes": sizes,
        "variants": variant_summaries,

        "delivery_options": delivery_options,
        "return_and_warranty": ret_warranty,

        "seller_name": seller_name,
        "seller_link": seller_link,
        "seller_metrics": seller_metrics,

        "highlights": highlights,
        "specifications": specs,
        "description": description_text,
    }

def product_to_markdown(p: Dict[str, Any]) -> str:
    lines = []
    lines.append(f"## {p['title']} — Item ID: {p['id']}")
    lines.append("")
    # Core facts
    core = []
    if p.get("category"): core.append(f"**Category:** {p['category']}")
    if p.get("brand"): core.append(f"**Brand:** {p['brand']}")
    if p.get("sku"): core.append(f"**SKU:** {p['sku']}")
    if p.get("url"): core.append(f"**URL:** {p['url']}")
    if p.get("sold_text"): core.append(f"**Sales:** {p['sold_text']}")
    if core:
        lines.append("  \n".join(core))
        lines.append("")
    # Price & rating
    price_bits = []
    pd = p.get("price_display") or p.get("listing_price_display")
    if pd: price_bits.append(f"**Price:** {pd}")
    if p.get("original_price_display"): price_bits.append(f"**Original:** {p['original_price_display']}")
    if p.get("discount_display"): price_bits.append(f"**Discount:** {p['discount_display']}")
    rating_bits = []
    if p.get("rating_average") is not None:
        rc = p.get("rating_count")
        if rc is not None:
            rating_bits.append(f"**Rating:** {p['rating_average']}/5 ({rc} ratings)")
        else:
            rating_bits.append(f"**Rating:** {p['rating_average']}/5")
    if price_bits or rating_bits:
        lines.append("  \n".join(price_bits + rating_bits))
        lines.append("")

    # Options
    if p.get("colors"):
        lines.append("**Colors:** " + ", ".join(p["colors"]))
    if p.get("sizes"):
        lines.append("**Sizes:** " + ", ".join(p["sizes"]))
    if p.get("variants"):
        lines.append("**Variants (sample):**")
        for v in p["variants"]:
            lines.append(f"- {v}")
    if p.get("colors") or p.get("sizes") or p.get("variants"):
        lines.append("")

    # Delivery / Warranty
    if p.get("delivery_options"):
        lines.append("**Delivery Options:**")
        for d in p["delivery_options"]:
            if isinstance(d, dict):
                title = d.get("title")
                time = d.get("time")
                fee = d.get("fee")
                bits = [title] if title else []
                if time: bits.append(time)
                if fee: bits.append(fee)
                if bits:
                    lines.append("- " + " — ".join(bits))
        lines.append("")
    if p.get("return_and_warranty"):
        lines.append("**Return & Warranty:**")
        for rw in p["return_and_warranty"]:
            if rw: lines.append(f"- {rw}")
        lines.append("")

    # Seller
    if p.get("seller_name") or p.get("seller_metrics"):
        seller_line = []
        if p.get("seller_name"): seller_line.append(f"**Seller:** {p['seller_name']}")
        if p.get("seller_link"): seller_line.append(f"[Store Link]({p['seller_link']})")
        if seller_line:
            lines.append(" ".join(seller_line))
        metrics = p.get("seller_metrics") or {}
        if metrics:
            mparts = []
            for k, v in metrics.items():
                if v: mparts.append(f"{k}: {v}")
            if mparts:
                lines.append("- " + " | ".join(mparts))
        lines.append("")

    # Description
    if p.get("description"):
        lines.append("**Description:**")
        lines.append(p["description"])
        lines.append("")

    # Images
    if p.get("images"):
        lines.append("**Images (sample):**")
        for im in p["images"]:
            lines.append(f"- {im}")
        lines.append("")

    # Traceability
    lines.append(f"_Source: {p.get('category_dir')} / products.json_")
    lines.append("")
    lines.append("---")
    lines.append("")
    return "\n".join(lines)

def product_to_text(p: Dict[str, Any]) -> str:
    # Similar to markdown but without **bold** / headings
    lines = []
    lines.append(f"PRODUCT: {p['title']}  |  ID: {p['id']}")
    if p.get("url"): lines.append(f"URL: {p['url']}")
    if p.get("category"): lines.append(f"CATEGORY: {p['category']}")
    if p.get("brand"): lines.append(f"BRAND: {p['brand']}")
    if p.get("sku"): lines.append(f"SKU: {p['sku']}")
    if p.get("sold_text"): lines.append(f"SALES: {p['sold_text']}")
    pd = p.get("price_display") or p.get("listing_price_display")
    if pd: lines.append(f"PRICE: {pd}")
    if p.get("original_price_display"): lines.append(f"ORIGINAL: {p['original_price_display']}")
    if p.get("discount_display"): lines.append(f"DISCOUNT: {p['discount_display']}")
    if p.get("rating_average") is not None:
        rc = p.get("rating_count")
        lines.append(f"RATING: {p['rating_average']}/5" + (f" ({rc} ratings)" if rc is not None else ""))
    if p.get("colors"): lines.append("COLORS: " + ", ".join(p["colors"]))
    if p.get("sizes"): lines.append("SIZES: " + ", ".join(p["sizes"]))
    if p.get("variants"):
        lines.append("VARIANTS:")
        for v in p["variants"]:
            lines.append(f"- {v}")
    if p.get("delivery_options"):
        lines.append("DELIVERY:")
        for d in p["delivery_options"]:
            if isinstance(d, dict):
                bits = []
                if d.get("title"): bits.append(d["title"])
                if d.get("time"): bits.append(d["time"])
                if d.get("fee"): bits.append(d["fee"])
                if bits: lines.append("- " + " — ".join(bits))
    if p.get("return_and_warranty"):
        lines.append("RETURNS/WARRANTY:")
        for rw in p["return_and_warranty"]:
            if rw: lines.append(f"- {rw}")
    if p.get("seller_name"):
        line = f"SELLER: {p['seller_name']}"
        if p.get("seller_link"): line += f" ({p['seller_link']})"
        lines.append(line)
    metrics = p.get("seller_metrics") or {}
    if metrics:
        mparts = []
        for k, v in metrics.items():
            if v: mparts.append(f"{k}: {v}")
        if mparts: lines.append("SELLER METRICS: " + " | ".join(mparts))
    if p.get("description"):
        lines.append("DESCRIPTION:")
        lines.append(p["description"])
    if p.get("images"):
        lines.append("IMAGES:")
        for im in p["images"]:
            lines.append(f"- {im}")
    lines.append(f"SOURCE: {p.get('category_dir')} / products.json")
    lines.append("-----")
    return "\n".join(lines) + "\n\n"

def product_to_jsonl_record(p: Dict[str, Any]) -> Dict[str, Any]:
    # One JSON per line: { "id":..., "text":..., "metadata":{...} }
    # The "text" is a compact version of the markdown content.
    text_lines = []
    text_lines.append(f"{p['title']} (ID: {p['id']})")
    if p.get("brand"): text_lines.append(f"Brand: {p['brand']}")
    if p.get("category"): text_lines.append(f"Category: {p['category']}")
    pd = p.get("price_display") or p.get("listing_price_display")
    if pd: text_lines.append(f"Price: {pd}")
    if p.get("discount_display"): text_lines.append(f"Discount: {p['discount_display']}")
    if p.get("rating_average") is not None:
        rc = p.get("rating_count")
        text_lines.append(f"Rating: {p['rating_average']}/5" + (f" ({rc} ratings)" if rc is not None else ""))
    if p.get("sold_text"): text_lines.append(f"Sales: {p['sold_text']}")
    if p.get("colors"): text_lines.append("Colors: " + ", ".join(p["colors"]))
    if p.get("sizes"): text_lines.append("Sizes: " + ", ".join(p["sizes"]))
    if p.get("description"): text_lines.append("Description: " + p["description"])
    if p.get("url"): text_lines.append(f"URL: {p['url']}")

    metadata = {
        "id": p.get("id"),
        "title": p.get("title"),
        "brand": p.get("brand"),
        "category": p.get("category"),
        "sku": p.get("sku"),
        "url": p.get("url"),
        "price_display": p.get("price_display") or p.get("listing_price_display"),
        "original_price_display": p.get("original_price_display"),
        "discount_display": p.get("discount_display"),
        "rating_average": p.get("rating_average"),
        "rating_count": p.get("rating_count"),
        "sold_text": p.get("sold_text"),
        "seller_name": p.get("seller_name"),
        "category_dir": p.get("category_dir"),
        "source_file": p.get("source_file"),
    }
    return {"id": p["id"], "text": "\n".join(text_lines), "metadata": metadata}

def find_products_json_files(root: Path) -> List[Path]:
    return sorted(root.glob("*/products.json"))

def load_products(products_path: Path) -> List[Dict[str, Any]]:
    try:
        with products_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        else:
            return []
    except Exception as e:
        print(f"[WARN] Failed to parse {products_path}: {e}")
        return []

def main():
    ap = argparse.ArgumentParser(description="Build a RAG-friendly corpus from Daraz products.json files.")
    ap.add_argument("--input", type=str, default="./layer_23/result", help="Root folder containing category subfolders (default: result)")
    ap.add_argument("--format", type=str, default="md", choices=["md", "jsonl", "txt"], help="Output format (md/jsonl/txt). Default: md")
    ap.add_argument("--output", type=str, required=True, help="Output file path")
    ap.add_argument("--max-images", type=int, default=10, help="Max images to include per product (default: 10)")
    args = ap.parse_args()

    root = Path(args.input)
    if not root.exists():
        raise SystemExit(f"Input folder not found: {root}")

    products_files = find_products_json_files(root)
    if not products_files:
        raise SystemExit(f"No products.json files found under: {root}")

    print(f"Found {len(products_files)} products.json files")

    # load + normalize + dedupe
    seen: Dict[str, Dict[str, Any]] = {}
    categories_covered = set()

    for pfile in products_files:
        cat_dir = pfile.parent  # result/<category>
        categories_covered.add(cat_dir.name)
        products = load_products(pfile)
        for raw in products:
            uid = unique_stable_key(raw)
            unified = unify_product(raw, str(cat_dir), pfile)
            # If duplicate seen, prefer to preserve the one with brand/title/price if the new one is richer.
            if uid in seen:
                old = seen[uid]
                # naive merge: keep fields that are missing in old
                for k, v in unified.items():
                    if (old.get(k) in (None, "", [], {})) and v not in (None, "", [], {}):
                        old[k] = v
                # merge categories if different
                if old.get("category") and unified.get("category") and old["category"] != unified["category"]:
                    old["category"] = old["category"] + " | " + unified["category"]
            else:
                seen[uid] = unified

    all_products = list(seen.values())
    print(f"Merged products (deduped): {len(all_products)} from {len(categories_covered)} categories")

    # Trim images to user preference
    max_images = max(0, int(args.max_images))
    for p in all_products:
        if isinstance(p.get("images"), list):
            p["images"] = p["images"][:max_images]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.format == "md":
        with out_path.open("w", encoding="utf-8") as f:
            f.write("# Daraz Product Corpus\n\n")
            for p in all_products:
                f.write(product_to_markdown(p))
        print(f"[OK] Wrote Markdown corpus: {out_path}")

    elif args.format == "txt":
        with out_path.open("w", encoding="utf-8") as f:
            f.write("DARAZ PRODUCT CORPUS\n\n")
            for p in all_products:
                f.write(product_to_text(p))
        print(f"[OK] Wrote plain text corpus: {out_path}")

    elif args.format == "jsonl":
        with out_path.open("w", encoding="utf-8") as f:
            for p in all_products:
                rec = product_to_jsonl_record(p)
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[OK] Wrote JSONL corpus: {out_path}")

if __name__ == "__main__":
    main()
