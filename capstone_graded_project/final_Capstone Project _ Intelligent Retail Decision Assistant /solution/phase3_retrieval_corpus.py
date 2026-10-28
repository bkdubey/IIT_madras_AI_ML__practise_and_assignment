"""Phase 3: enrich structured products and policy snippets into retrieval documents."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from phase1_data_preparation import POLICY_DIR, ROOT, load_interactions, write_policies


def build_product_documents(frame: pd.DataFrame | None = None) -> list[dict[str, str | float]]:
    """Create one retrieval document per unique product from interaction records."""
    data = frame if frame is not None else load_interactions()
    required = {"product_id", "product_title", "product_description", "brand", "category", "style_tags", "base_price"}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Dataset is missing retrieval columns: {', '.join(missing)}")
    columns = sorted(required)
    products = data[columns].drop_duplicates("product_id").sort_values("product_id")
    docs: list[dict[str, str | float]] = []
    for row in products.to_dict("records"):
        text = (
            f"{row['product_title']}. {row['product_description']} "
            f"Brand {row['brand']}. Category {row['category']}. "
            f"Style and occasions: {row['style_tags']}. Price ₹{float(row['base_price']):.0f}."
        )
        docs.append({
            "product_id": str(row["product_id"]),
            "product_title": str(row["product_title"]),
            "brand": str(row["brand"]),
            "category": str(row["category"]),
            "style_tags": str(row["style_tags"]),
            "base_price": float(row["base_price"]),
            "text": text,
        })
    return docs


def build_policy_documents(policy_path: Path | None = None) -> list[dict[str, str]]:
    """Load policy documents and validate the citation fields."""
    path = policy_path or write_policies(POLICY_DIR)
    policies = json.loads(path.read_text(encoding="utf-8"))
    for item in policies:
        if not all(item.get(field) for field in ("policy_doc_id", "title", "policy_text")):
            raise ValueError("Every policy must include policy_doc_id, title, and policy_text.")
    return policies


def prepare_retrieval_corpus() -> dict[str, int]:
    """Persist product and policy corpora for inspection and downstream retrieval."""
    products = build_product_documents()
    policies = build_policy_documents()
    output = ROOT / "data" / "processed"
    output.mkdir(parents=True, exist_ok=True)
    (output / "product_documents.json").write_text(json.dumps(products, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "policy_documents.json").write_text(json.dumps(policies, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"products": len(products), "policies": len(policies)}


if __name__ == "__main__":
    print(json.dumps(prepare_retrieval_corpus(), indent=2))
