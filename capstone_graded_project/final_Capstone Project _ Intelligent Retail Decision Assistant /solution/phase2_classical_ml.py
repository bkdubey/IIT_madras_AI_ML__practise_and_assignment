"""Phase 2: supervised purchase prediction and interpretable behavioral segments."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from phase1_data_preparation import ROOT, load_interactions

RANDOM_SEED = 42
NUMERIC_FEATURES = [
    "age", "days_since_last_purchase", "avg_order_value", "total_orders",
    "base_price", "discount_percent", "final_price", "stock_level",
    "avg_product_rating", "time_on_page", "added_to_cart", "wishlist_flag",
]
CATEGORICAL_FEATURES = [
    "income_band", "membership_level", "city_tier", "preferred_channel",
    "category", "sub_category", "brand", "color", "material", "season",
    "payment_mode", "delivery_type",
]
CLUSTER_FEATURES = [
    "days_since_last_purchase", "avg_order_value", "total_orders",
    "final_price", "discount_percent", "time_on_page", "wishlist_flag",
]


def train_purchase_model(frame: pd.DataFrame | None = None) -> tuple[dict[str, float], pd.DataFrame]:
    """Train/evaluate a leakage-aware logistic baseline with a stratified holdout."""
    data = frame if frame is not None else load_interactions()
    features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    absent = sorted(set(features + ["purchase_flag"]) - set(data.columns))
    if absent:
        raise ValueError(f"Dataset is missing model columns: {', '.join(absent)}")
    X = data[features].copy()
    y = data["purchase_flag"].astype(int)
    if y.nunique() != 2:
        raise ValueError("purchase_flag must contain both binary classes.")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y,
    )
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocess = ColumnTransformer([
        ("numeric", numeric, NUMERIC_FEATURES),
        ("categorical", categorical, CATEGORICAL_FEATURES),
    ])
    model = Pipeline([
        ("preprocess", preprocess),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_SEED)),
    ])
    model.fit(X_train, y_train)
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= .5).astype(int)
    metrics = {
        "test_rows": int(len(y_test)),
        "positive_rate": round(float(y.mean()), 4),
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "precision": round(float(precision_score(y_test, predictions, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, predictions, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, predictions, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, probabilities)), 4),
        "baseline": "class-weighted logistic regression; 80/20 stratified holdout, random_state=42",
    }
    feature_names = model.named_steps["preprocess"].get_feature_names_out()
    coefficients = model.named_steps["classifier"].coef_[0]
    importance = pd.DataFrame({"feature": feature_names, "coefficient": coefficients})
    importance["absolute_coefficient"] = importance["coefficient"].abs()
    importance = importance.sort_values("absolute_coefficient", ascending=False).head(15).reset_index(drop=True)
    return metrics, importance


def segment_customers(frame: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict[str, object]]:
    """Aggregate interactions to customer profiles, then cluster customer behavior."""
    data = frame if frame is not None else load_interactions()
    absent = sorted(set(CLUSTER_FEATURES + ["customer_id"]) - set(data.columns))
    if absent:
        raise ValueError(f"Dataset is missing clustering columns: {', '.join(absent)}")
    customer_profiles = data.groupby("customer_id", as_index=False).agg(
        days_since_last_purchase=("days_since_last_purchase", "first"),
        avg_order_value=("avg_order_value", "first"),
        total_orders=("total_orders", "first"),
        final_price=("final_price", "mean"),
        discount_percent=("discount_percent", "mean"),
        time_on_page=("time_on_page", "mean"),
        wishlist_flag=("wishlist_flag", "mean"),
        interaction_count=("customer_id", "size"),
    )
    if len(customer_profiles) < 3:
        raise ValueError("At least three distinct customers are required for segmentation.")
    X = customer_profiles[CLUSTER_FEATURES].copy()
    X = SimpleImputer(strategy="median").fit_transform(X)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(X)
    best: tuple[float, int, KMeans, np.ndarray] | None = None
    for k in range(2, min(6, len(customer_profiles))):
        candidate = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_SEED)
        labels = candidate.fit_predict(scaled)
        score = float(silhouette_score(scaled, labels))
        if best is None or score > best[0]:
            best = (score, k, candidate, labels)
    if best is None:
        raise ValueError("At least three interaction rows are required for segmentation.")
    score, k, _, labels = best
    segments = customer_profiles.copy()
    segments["segment"] = labels
    summary = segments.groupby("segment").agg(
        customers=("customer_id", "size"),
        mean_interactions=("interaction_count", "mean"),
        mean_final_price=("final_price", "mean"),
        mean_discount_percent=("discount_percent", "mean"),
        mean_avg_order_value=("avg_order_value", "mean"),
        mean_days_since_last_purchase=("days_since_last_purchase", "mean"),
        mean_total_orders=("total_orders", "mean"),
        mean_time_on_page=("time_on_page", "mean"),
        wishlist_rate=("wishlist_flag", "mean"),
    ).round(2)
    centers = segments.groupby("segment")[CLUSTER_FEATURES].mean()
    feature_descriptions = {
        "days_since_last_purchase": ("less recent purchases", "more recent purchases"),
        "avg_order_value": ("lower historical order value", "higher historical order value"),
        "total_orders": ("fewer prior orders", "more prior orders"),
        "final_price": ("lower-priced interactions", "higher-priced interactions"),
        "discount_percent": ("lower discount exposure", "higher discount exposure"),
        "time_on_page": ("shorter product-page visits", "longer product-page visits"),
        "wishlist_flag": ("less wishlist activity", "more wishlist activity"),
    }
    normalized_spread = (centers.max() - centers.min()) / customer_profiles[CLUSTER_FEATURES].std().replace(0, np.nan)
    distinguishing_feature = str(normalized_spread.fillna(0).idxmax())
    low_cluster = centers[distinguishing_feature].idxmin()
    high_cluster = centers[distinguishing_feature].idxmax()
    low_label, high_label = feature_descriptions[distinguishing_feature]
    descriptions = {
        str(segment_id): (
            f"{high_label} (cluster {segment_id})"
            if segment_id == high_cluster
            else f"{low_label} (cluster {segment_id})"
        )
        for segment_id in summary.index
    }
    result = {
        "clusters": k,
        "silhouette_score": round(score, 4),
        "unit_of_analysis": "customer profiles aggregated from their interaction rows",
        "features": CLUSTER_FEATURES,
        "distinguishing_feature": distinguishing_feature,
        "segments": descriptions,
        "segment_details": {
            str(segment_id): {key: float(value) for key, value in row.items()}
            for segment_id, row in summary.iterrows()
        },
        "caveat": "Clusters describe synthetic customer profiles, not verified customer identities or causal behavior.",
    }
    (ROOT / "outputs").mkdir(exist_ok=True)
    segments.to_csv(ROOT / "outputs" / "customer_segments.csv", index=False)
    summary.to_csv(ROOT / "outputs" / "segment_summary.csv")
    return segments, result


def run_classical_ml() -> dict[str, object]:
    """Run Phase 2 and persist the model evaluation artifacts."""
    (ROOT / "outputs").mkdir(exist_ok=True)
    metrics, importance = train_purchase_model()
    _, segments = segment_customers()
    (ROOT / "outputs" / "supervised_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    importance.to_csv(ROOT / "outputs" / "purchase_feature_importance.csv", index=False)
    (ROOT / "outputs" / "segmentation_summary.json").write_text(json.dumps(segments, indent=2) + "\n", encoding="utf-8")
    return {"supervised": metrics, "segmentation": segments, "important_features": importance.to_dict("records")}


if __name__ == "__main__":
    print(json.dumps(run_classical_ml(), indent=2))
