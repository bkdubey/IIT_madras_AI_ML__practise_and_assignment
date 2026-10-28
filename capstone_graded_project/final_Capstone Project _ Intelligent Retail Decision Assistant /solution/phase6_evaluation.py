"""Final phase: reproducible end-to-end checks and persisted demo outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from phase1_data_preparation import ROOT, prepare_dataset
from phase2_classical_ml import run_classical_ml
from phase3_retrieval_corpus import prepare_retrieval_corpus
from phase4_rag import RetailKnowledgeBase
from phase5_controller import RetailAssistant

CURATED_QUERIES = [
    "Recommend casual sneakers under ₹4000 for daily wear",
    "Can I return a discounted product after 7 days?",
    "Suggest products for a premium customer who prefers black formal shoes",
    "Which customer segment seems most price-sensitive?",
    "Summarise the top three casual daily-wear products and explain why they fit",
]


def run_evaluation() -> dict[str, Any]:
    """Run all phases in order, assert core behavioral guarantees, and save results."""
    outputs = ROOT / "outputs"
    outputs.mkdir(exist_ok=True)
    phase1 = prepare_dataset()
    phase2 = run_classical_ml()
    phase3 = prepare_retrieval_corpus()
    kb = RetailKnowledgeBase()
    assistant = RetailAssistant(kb)
    results = [assistant.answer(query) for query in CURATED_QUERIES]
    assert all(result["route"] for result in results)
    assert results[0]["route"] == "product_search"
    assert results[1]["route"] == "policy_search"
    assert results[3]["route"] == "retail_analytics"
    assert "RET_02" in results[1]["sources"]
    assert all(float(item["base_price"]) <= 4000 for item in results[0]["products"])
    assert len(results[4]["products"]) == 3
    report = {
        "phase1": phase1,
        "phase2": {
            "supervised_metrics": phase2["supervised"],
            "segmentation": phase2["segmentation"],
        },
        "phase3": phase3,
        "phase4_index": "TF-IDF text embeddings with FAISS IndexFlatIP when available; sklearn cosine fallback otherwise.",
        "phase5_queries": results,
        "regression_checks": {
            "five_queries": len(results) >= 5,
            "product_budget_respected": all(float(item["base_price"]) <= 4000 for item in results[0]["products"]),
            "policy_source_cited": "RET_02" in results[1]["sources"],
            "controller_routes_policy": results[1]["route"] == "policy_search",
            "controller_routes_analytics": results[3]["route"] == "retail_analytics",
        },
    }
    (outputs / "evaluation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (outputs / "example_queries.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    report = run_evaluation()
    print(json.dumps(report["regression_checks"], indent=2))
