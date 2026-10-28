"""Regression checks for the complete retail decision assistant."""

import json
import sys
from pathlib import Path

import pytest

SOLUTION_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOLUTION_DIR))

from phase1_data_preparation import generate_interactions, write_policies  # noqa: E402
from phase2_classical_ml import segment_customers, train_purchase_model  # noqa: E402
from phase4_rag import RetailKnowledgeBase, parse_budget  # noqa: E402
from phase5_controller import RetailAssistant  # noqa: E402


@pytest.fixture(scope="module")
def interactions():
    return generate_interactions(n_rows=360)


def test_generated_dataset_is_reproducible_and_complete(interactions):
    assert interactions.equals(generate_interactions(n_rows=360))
    assert len(interactions) == 360
    assert interactions["purchase_flag"].isin([0, 1]).all()
    assert {"product_description", "customer_query", "policy_text_snippet"} <= set(interactions.columns)


def test_supervised_pipeline_and_segmentation(interactions):
    metrics, importance = train_purchase_model(interactions)
    assert 0 <= metrics["roc_auc"] <= 1
    assert metrics["test_rows"] == 72
    assert not importance.empty
    segments, summary = segment_customers(interactions)
    assert segments["segment"].nunique() == summary["clusters"]
    assert len(segments) == interactions["customer_id"].nunique()
    assert summary["unit_of_analysis"].startswith("customer profiles")
    assert summary["silhouette_score"] >= -1
    assert len(set(summary["segments"].values())) == summary["clusters"]


def test_segmentation_requires_three_distinct_customers(interactions):
    few_customers = interactions[interactions["customer_id"].isin(["C0000", "C0001"])].copy()
    with pytest.raises(ValueError, match="three distinct customers"):
        segment_customers(few_customers)


def test_product_budget_and_policy_evidence():
    kb = RetailKnowledgeBase()
    products = kb.search_products("casual sneakers under ₹3000", top_k=5)
    assert products
    assert all(product["base_price"] <= 3000 for product in products)
    answer = kb.answer_policy("Can I return discounted shoes after 7 days?", use_llm=False)
    assert "RET_02" in answer["sources"]
    assert "7 calendar days" in answer["answer"]


def test_controller_routes_policy_products_and_analytics():
    assistant = RetailAssistant()
    assert assistant.answer("What is the shipping policy?")["route"] == "policy_search"
    assert assistant.answer("Recommend black formal shoes")["route"] == "product_search"
    assert assistant.answer("Which segment is price sensitive?")["route"] == "retail_analytics"


def test_empty_or_unanswerable_requests_are_safe():
    assistant = RetailAssistant()
    with pytest.raises(ValueError, match="must not be empty"):
        assistant.answer("   ")
    answer = assistant.knowledge_base.answer_policy("What is the moon policy?", use_llm=False)
    assert answer["answer"] == "I cannot answer from the available policy documents."
    assert answer["sources"] == []


def test_budget_parser():
    assert parse_budget("under ₹4,000 please") == 4000
    assert parse_budget("Rs. 2500 maximum") == 2500
    assert parse_budget("no budget") is None


def test_existing_policy_file_is_not_overwritten(tmp_path):
    policy_dir = tmp_path / "policies"
    policy_dir.mkdir()
    policy_path = policy_dir / "retail_policies.json"
    supplied = [{"policy_doc_id": "COURSE_01", "title": "Course policy", "topic": "test", "policy_text": "Provided source."}]
    policy_path.write_text(json.dumps(supplied), encoding="utf-8")

    assert write_policies(policy_dir) == policy_path
    assert json.loads(policy_path.read_text(encoding="utf-8")) == supplied
