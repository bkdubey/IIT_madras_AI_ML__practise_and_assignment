from __future__ import annotations

import time
from typing import Any, Dict, List

from memory import VerifiedMemory
from prompts import SYSTEM_PROMPT
from tools import PolicyNotFoundError, _extract_quote, extract_doc_id, policy_search, read_policy


class PolicyAssistantAgent:
    def __init__(self, memory: VerifiedMemory | None = None):
        self.memory = memory or VerifiedMemory()

    def _pick_best_policy(self, question: str) -> List[dict]:
        doc_id = extract_doc_id(question)
        if doc_id:
            try:
                text = read_policy(doc_id)
                return [{"doc_id": doc_id, "text": text, "score": 1.0}]
            except PolicyNotFoundError:
                pass
        return [
            {"doc_id": item.doc_id, "text": item.snippet, "score": float(item.score)}
            for item in policy_search(question, k=3)
        ]

    def answer_question(self, question: str) -> Dict[str, Any]:
        start = time.perf_counter()
        memory_hints = self.memory.read_hints(question, k=2)
        policy_hits = self._pick_best_policy(question)

        context_parts = [SYSTEM_PROMPT]
        if memory_hints:
            context_parts.append("Verified memory hints:\n- " + "\n- ".join(memory_hints))
        context_parts.append("Relevant policy snippets:\n")
        for hit in policy_hits:
            doc_id = hit["doc_id"]
            text = hit["text"]
            context_parts.append(f"doc_id={doc_id}\n{text}\n")

        context = "\n".join(context_parts)
        answer = self._compose_answer(question, context)
        elapsed = time.perf_counter() - start
        return {
            "answer": answer,
            "doc_id": extract_doc_id(answer) or (policy_hits[0]["doc_id"] if policy_hits else None),
            "steps": len(policy_hits) + 1,
            "memory_hints": memory_hints,
            "elapsed": elapsed,
            "context": context,
        }

    def _compose_answer(self, question: str, context: str) -> str:
        doc_id = extract_doc_id(question)
        if doc_id:
            try:
                text = read_policy(doc_id)
                quote = _extract_quote(text)
                return f"{doc_id}: \"{quote}\""
            except PolicyNotFoundError:
                pass

        policy_hits = []
        for line in context.split("doc_id="):
            if "\n" in line:
                doc, text = line.split("\n", 1)
                if doc.strip():
                    policy_hits.append((doc.strip(), text.strip()))

        if not policy_hits:
            return "No grounded policy found."

        doc_id, text = policy_hits[0]
        quote = _extract_quote(text)
        return f"{doc_id}: \"{quote}\""


if __name__ == "__main__":
    agent = PolicyAssistantAgent()
    print(agent.answer_question("What is our remote work limit? Cite and quote."))
