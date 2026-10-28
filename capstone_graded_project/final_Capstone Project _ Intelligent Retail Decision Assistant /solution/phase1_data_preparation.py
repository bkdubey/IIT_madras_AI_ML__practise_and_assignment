"""Phase 1: generate and inspect a reproducible retail interaction dataset."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
POLICY_DIR = DATA_DIR / "policies"
RANDOM_SEED = 42

PRODUCTS = [
    ("P001", "Everyday Mesh Runner", "Shoes", "Sneakers", "Stride", "Black", "Mesh", 2799, "casual walking daily sports"),
    ("P002", "Cloudfoam Walking Sneaker", "Shoes", "Sneakers", "Northstar", "White", "Knit", 3499, "casual walking daily travel"),
    ("P003", "Urban Canvas Low-Top", "Shoes", "Sneakers", "Streetwise", "Black", "Canvas", 1899, "casual everyday travel"),
    ("P004", "Trail Grip Running Shoe", "Shoes", "Sports Shoes", "Peak", "Blue", "Mesh", 3899, "running sports outdoor"),
    ("P005", "Classic Leather Formal Oxford", "Shoes", "Formal Shoes", "Regent", "Black", "Leather", 4999, "formal office business wedding"),
    ("P006", "Office Comfort Derby", "Shoes", "Formal Shoes", "Regent", "Black", "Leather", 3799, "formal office business"),
    ("P007", "Brown Suede Loafer", "Shoes", "Loafers", "Marlow", "Brown", "Suede", 4299, "formal office smart casual"),
    ("P008", "Lightweight Gym Trainer", "Shoes", "Sports Shoes", "Peak", "Grey", "Mesh", 2999, "gym training sports running"),
    ("P009", "Minimal White Court Sneaker", "Shoes", "Sneakers", "Streetwise", "White", "Synthetic", 3199, "casual everyday office travel"),
    ("P010", "Comfort Knit Slip-On", "Shoes", "Slip-ons", "Northstar", "Navy", "Knit", 2499, "casual walking travel daily"),
    ("P011", "Waterproof Hiker Boot", "Shoes", "Boots", "Peak", "Brown", "Leather", 5999, "outdoor hiking travel winter"),
    ("P012", "Everyday Sandal", "Shoes", "Sandals", "Coast", "Tan", "Synthetic", 1299, "casual summer daily travel"),
    ("P013", "Tailored Office Chino", "Apparel", "Trousers", "Regent", "Navy", "Cotton", 2599, "formal office business"),
    ("P014", "Classic Cotton Oxford Shirt", "Apparel", "Shirts", "Regent", "White", "Cotton", 2299, "formal office business"),
    ("P015", "Weekend Relaxed Tee", "Apparel", "T-Shirts", "Streetwise", "Black", "Cotton", 999, "casual everyday travel"),
    ("P016", "Lightweight Running Jacket", "Apparel", "Jackets", "Peak", "Black", "Polyester", 3499, "running sports outdoor"),
    ("P017", "Everyday Leather Belt", "Accessories", "Belts", "Marlow", "Black", "Leather", 1499, "formal office everyday"),
    ("P018", "Compact Travel Backpack", "Accessories", "Bags", "Northstar", "Black", "Polyester", 2899, "travel daily casual"),
    ("P019", "Classic Analog Watch", "Accessories", "Watches", "Regent", "Black", "Steel", 4599, "formal office business"),
    ("P020", "Performance Sports Socks", "Accessories", "Socks", "Peak", "White", "Cotton", 599, "running gym sports daily"),
    ("P021", "Soft Knit Winter Scarf", "Accessories", "Scarves", "Marlow", "Grey", "Wool", 1799, "winter casual travel"),
    ("P022", "Everyday Slip-Resistant Clog", "Shoes", "Clogs", "Coast", "Black", "Synthetic", 1999, "casual walking daily"),
    ("P023", "Premium Suede Chelsea Boot", "Shoes", "Boots", "Marlow", "Black", "Suede", 6999, "formal office winter"),
    ("P024", "Cushioned Daily Trainer", "Shoes", "Sneakers", "Stride", "Grey", "Mesh", 3299, "casual walking running daily"),
]

POLICIES = [
    {"policy_doc_id": "RET_01", "topic": "returns", "title": "Returns", "policy_text": "Unused products may be returned within 7 calendar days of delivery. Items must retain original tags and packaging. Products marked final sale are not returnable."},
    {"policy_doc_id": "RET_02", "topic": "discounted returns", "title": "Discounted item returns", "policy_text": "Discounted products can be returned within 7 calendar days of delivery if unused and in original packaging with tags attached. After 7 days they are not eligible for return."},
    {"policy_doc_id": "RET_03", "topic": "refunds", "title": "Refunds", "policy_text": "Approved refunds are issued to the original payment method within 5 to 7 business days after the returned item passes inspection."},
    {"policy_doc_id": "RET_04", "topic": "shipping", "title": "Shipping", "policy_text": "Standard delivery takes 3 to 5 business days. Orders over ₹2,500 qualify for free standard shipping; other orders have a ₹99 standard shipping fee."},
    {"policy_doc_id": "RET_05", "topic": "exchange", "title": "Exchanges", "policy_text": "Size exchanges may be requested within 7 calendar days of delivery for unused items with tags attached. Exchanges are subject to stock availability."},
    {"policy_doc_id": "RET_06", "topic": "warranty", "title": "Warranty", "policy_text": "Manufacturing defects are covered for 90 days from delivery. Normal wear, accidental damage, and improper care are not covered."},
]


def generate_interactions(n_rows: int = 1200, random_seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Create realistic, deterministic interaction rows when source data is unavailable."""
    rng = np.random.default_rng(random_seed)
    n_customers = max(120, n_rows // 6)
    ages = rng.integers(18, 66, n_customers)
    incomes = rng.choice(["value", "mid", "premium"], n_customers, p=[0.35, 0.45, 0.20])
    members = rng.choice(["standard", "silver", "gold"], n_customers, p=[0.55, 0.30, 0.15])
    city_tiers = rng.choice(["tier_1", "tier_2", "tier_3"], n_customers, p=[0.40, 0.35, 0.25])
    profiles = pd.DataFrame({
        "customer_id": [f"C{i:04d}" for i in range(n_customers)],
        "age": ages,
        "income_band": incomes,
        "membership_level": members,
        "city_tier": city_tiers,
        "total_orders": rng.poisson(5, n_customers) + 1,
        "avg_order_value": np.round(rng.lognormal(7.7, 0.45, n_customers), 2),
        "days_since_last_purchase": rng.integers(1, 181, n_customers),
        "preferred_channel": rng.choice(["web", "app", "store"], n_customers, p=[0.42, 0.48, 0.10]),
    })
    p = pd.DataFrame(PRODUCTS, columns=[
        "product_id", "product_title", "category", "sub_category", "brand",
        "color", "material", "base_price", "style_tags",
    ])
    p["season"] = rng.choice(["all-season", "summer", "winter"], len(p))
    p["size"] = rng.choice(["S", "M", "L", "7", "8", "9", "10"], len(p))
    p["product_description"] = p.apply(
        lambda r: f"{r['product_title']} by {r['brand']}. {r['material']} {r['sub_category'].lower()} in {r['color'].lower()}. Ideal for {r['style_tags']}.",
        axis=1,
    )
    customer_rows = rng.integers(0, n_customers, n_rows)
    product_rows = rng.integers(0, len(p), n_rows)
    frame = profiles.iloc[customer_rows].reset_index(drop=True).copy()
    products = p.iloc[product_rows].reset_index(drop=True)
    frame = pd.concat([frame, products], axis=1)
    frame["interaction_id"] = [f"I{i:05d}" for i in range(n_rows)]
    frame["discount_percent"] = rng.choice([0, 5, 10, 15, 20, 25, 30], n_rows, p=[.16, .12, .18, .18, .16, .12, .08])
    frame["final_price"] = (frame["base_price"] * (1 - frame["discount_percent"] / 100)).round(2)
    frame["stock_level"] = rng.integers(0, 81, n_rows)
    frame["avg_product_rating"] = np.round(rng.uniform(2.8, 5.0, n_rows), 1)
    frame["payment_mode"] = rng.choice(["UPI", "card", "cash_on_delivery", "wallet"], n_rows)
    frame["delivery_type"] = rng.choice(["standard", "express"], n_rows, p=[.78, .22])
    frame["time_on_page"] = np.round(rng.gamma(2.1, 34, n_rows).clip(5, 360), 1)
    frame["added_to_cart"] = rng.binomial(1, .28, n_rows)
    frame["wishlist_flag"] = rng.binomial(1, .22, n_rows)
    member_effect = frame["membership_level"].map({"standard": 0, "silver": .10, "gold": .20})
    logit = -1.0 + 1.4 * frame["added_to_cart"] + .006 * frame["time_on_page"] + .035 * frame["discount_percent"] + .25 * member_effect
    purchase_probability = 1 / (1 + np.exp(-logit))
    frame["purchase_flag"] = rng.binomial(1, purchase_probability)
    frame["quantity"] = np.where(frame["purchase_flag"].eq(1), rng.integers(1, 4, n_rows), 0)
    frame["returned_flag"] = np.where(
        frame["purchase_flag"].eq(1),
        rng.binomial(1, .12 + .06 * (frame["discount_percent"] >= 25)),
        0,
    )
    frame["rating_given"] = np.where(frame["purchase_flag"].eq(1), rng.integers(1, 6, n_rows), np.nan)
    frame["customer_review_text"] = np.where(
        frame["purchase_flag"].eq(1),
        rng.choice(["Comfortable and good quality.", "Good value for everyday use.", "Fit was not quite right.", "Looks as described and feels durable."], n_rows),
        "",
    )
    frame["customer_query"] = frame.apply(
        lambda r: f"Looking for {r['color'].lower()} {r['sub_category'].lower()} for {r['style_tags'].split()[0]}",
        axis=1,
    )
    frame["policy_doc_id"] = "RET_01"
    frame["policy_text_snippet"] = POLICIES[0]["policy_text"]
    frame["occasion_tags"] = frame["style_tags"]
    frame["image_path"] = ""
    return frame


def ensure_source_data(data_path: Path | None = None) -> Path:
    """Use the supplied CSV if present; otherwise persist the documented synthetic dataset."""
    path = data_path or RAW_DIR / "retail_interactions.csv"
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    generate_interactions().to_csv(path, index=False)
    return path


def load_interactions(data_path: Path | None = None) -> pd.DataFrame:
    """Load interactions and validate required columns."""
    path = ensure_source_data(data_path)
    frame = pd.read_csv(path)
    required = {"customer_id", "product_id", "purchase_flag", "base_price", "final_price"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing)}")
    if frame.empty:
        raise ValueError("Retail interaction dataset must contain at least one row.")
    return frame


def write_policies(policy_dir: Path = POLICY_DIR) -> Path:
    """Create the demo policy corpus only when no policy file exists."""
    policy_dir.mkdir(parents=True, exist_ok=True)
    path = policy_dir / "retail_policies.json"
    if not path.exists():
        path.write_text(json.dumps(POLICIES, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def prepare_dataset(data_path: Path | None = None) -> dict[str, object]:
    """Persist Phase 1 inputs and a compact dataset inspection report."""
    path = ensure_source_data(data_path)
    write_policies()
    frame = pd.read_csv(path)
    report = {
        "rows": int(len(frame)),
        "columns": int(frame.shape[1]),
        "missing_values": {k: int(v) for k, v in frame.isna().sum().items() if v},
        "purchase_flag_balance": {str(k): int(v) for k, v in frame["purchase_flag"].value_counts().sort_index().items()},
        "source_note": "Synthetic deterministic data generated for this submission; the referenced source CSV was not included.",
    }
    (ROOT / "outputs").mkdir(exist_ok=True)
    (ROOT / "outputs" / "phase1_data_profile.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(prepare_dataset(), indent=2))
