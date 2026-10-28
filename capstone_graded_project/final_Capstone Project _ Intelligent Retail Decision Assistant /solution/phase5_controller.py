"""Phase 5: bounded, inspectable routing between retail tools."""

from __future__ import annotations

from typing import Any

from phase2_classical_ml import run_classical_ml
from phase4_rag import RetailKnowledgeBase

POLICY_TERMS = ("return", "refund", "shipping", "delivery policy", "exchange", "warranty", "policy")
ANALYTICS_TERMS = ("segment", "price-sensitive", "price sensitive", "cluster", "purchase prediction", "conversion")


class RetailAssistant:
    """Route each request to one tool and return a human-readable decision trace."""

    def __init__(self, knowledge_base: RetailKnowledgeBase | None = None, use_llm: bool = False):
        self.knowledge_base = knowledge_base or RetailKnowledgeBase()
        self.use_llm = use_llm

    def answer(self, query: str) -> dict[str, Any]:
        """Choose a single tool via explicit keyword rules; do not chain tools."""
        normalized = query.casefold().strip()
        if not normalized:
            raise ValueError("Query must not be empty.")
        if any(term in normalized for term in POLICY_TERMS):
            route = "policy_search"
            result = self.knowledge_base.answer_policy(query, use_llm=self.use_llm)
            reason = "Policy-related keyword matched; retrieve policy evidence only."
        elif any(term in normalized for term in ANALYTICS_TERMS):
            route = "retail_analytics"
            metrics = run_classical_ml()
            segment_details = metrics["segmentation"]["segment_details"]
            most_discount_exposed = max(
                segment_details,
                key=lambda segment_id: segment_details[segment_id]["mean_discount_percent"],
            )
            other_segments = [
                segment_id for segment_id in segment_details
                if segment_id != most_discount_exposed
            ]
            comparison = ""
            if other_segments:
                comparison_id = other_segments[0]
                comparison = (
                    f" The other segment's mean was "
                    f"{segment_details[comparison_id]['mean_discount_percent']:.1f}%."
                )
            result = {
                "answer": (
                    f"Segment {most_discount_exposed} had the highest observed average discount "
                    f"({segment_details[most_discount_exposed]['mean_discount_percent']:.1f}%).{comparison} "
                    "This is discount exposure in synthetic interactions, not evidence of causal price sensitivity. "
                    f"Segments are described by {metrics['segmentation']['distinguishing_feature']}; "
                    f"silhouette score: {metrics['segmentation']['silhouette_score']}."
                ),
                "sources": ["outputs/segmentation_summary.json", "outputs/segment_summary.csv"],
            }
            reason = "Analytics keyword matched; report the reproducible segmentation summary."
        else:
            route = "product_search"
            result = self.knowledge_base.answer_products(query, use_llm=self.use_llm)
            reason = "No policy or analytics keyword matched; search the product catalog."
        return {
            "route": route,
            "trace": [{"step": 1, "decision": route, "reason": reason}],
            **result,
        }


if __name__ == "__main__":
    assistant = RetailAssistant()
    for example in (
        "Recommend casual sneakers under ₹4000 for daily wear",
        "Can I return a discounted product after 7 days?",
        "Which customer segment seems most price-sensitive?",
    ):
        print(assistant.answer(example))
