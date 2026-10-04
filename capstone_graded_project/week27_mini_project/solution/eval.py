from __future__ import annotations

from typing import List

from graph import PolicyGraph
from tools import extract_doc_id

CANNED_QUESTIONS = [
    "What is our remote work limit? Cite and quote.",
    "How many annual leave days and can they be carried over? Cite and quote.",
    "What’s the travel meals cap (INR) and any receipt rule? Cite and quote.",
    "Is MFA required for VPN? Cite and quote.",
    "Read policy remote_work_v9.9 and answer the limit.",
    "Compute 12 * 4 and then tell me meals cap with citation.",
]


def run_batch() -> List[dict]:
    graph = PolicyGraph(max_steps=5)
    results = graph.evaluate(CANNED_QUESTIONS)
    print(f"{'#':>2} {'Question':<55} {'doc_id':<8} {'quote':<6} {'valid':<6} {'steps':<5} {'time(s)':>8}")
    for item in results:
        print(f"{item['id']:>2} {item['question']:<55} {str(item['doc_id_present']):<8} {str(item['quote_present']):<6} {str(item['valid']):<6} {str(item['steps']):<5} {item['elapsed']:>8.3f}")
    return results


if __name__ == "__main__":
    run_batch()
