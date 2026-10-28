"""Phase 4: local vector retrieval and evidence-grounded answer generation."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from phase1_data_preparation import ROOT
from phase3_retrieval_corpus import build_policy_documents, build_product_documents

try:
    import faiss  # type: ignore[import-not-found]
except ImportError:
    faiss = None


class TextVectorIndex:
    """TF-IDF text embeddings indexed with FAISS when installed or sklearn otherwise."""

    def __init__(self, documents: list[dict[str, Any]], text_key: str = "text") -> None:
        if not documents:
            raise ValueError("Cannot build a retrieval index with no documents.")
        self.documents = documents
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), strip_accents="unicode")
        self.matrix = self.vectorizer.fit_transform([str(doc[text_key]) for doc in documents])
        self.dense = self.matrix.toarray().astype("float32")
        norms = np.linalg.norm(self.dense, axis=1, keepdims=True)
        self.dense = self.dense / np.maximum(norms, 1e-12)
        self.index = None
        if faiss is not None:
            self.index = faiss.IndexFlatIP(self.dense.shape[1])
            self.index.add(self.dense)

    def search(self, query: str, top_k: int = 3, allowed: list[int] | None = None) -> list[dict[str, Any]]:
        """Return cosine-ranked evidence; allowed filters indices after vector search."""
        if not query.strip():
            return []
        if top_k < 1:
            raise ValueError("top_k must be at least one.")
        vector = self.vectorizer.transform([query]).toarray().astype("float32")
        norm = np.linalg.norm(vector)
        if norm:
            vector /= norm
        if self.index is not None:
            scores, ids = self.index.search(vector, len(self.documents))
            ranked = [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i >= 0]
        else:
            scores = cosine_similarity(vector, self.dense)[0]
            ranked = sorted(enumerate(scores), key=lambda pair: pair[1], reverse=True)
        if allowed is not None:
            allowed_set = set(allowed)
            ranked = [(i, s) for i, s in ranked if i in allowed_set]
        return [
            {**self.documents[i], "relevance_score": round(score, 4)}
            for i, score in ranked[:top_k] if score > 0
        ]


def parse_budget(query: str) -> float | None:
    """Extract a simple explicit rupee/Rs budget from a user request."""
    match = re.search(r"(?:₹|rs\.?\s*)(\d[\d,]*(?:\.\d+)?)", query, flags=re.IGNORECASE)
    if not match:
        return None
    return float(match.group(1).replace(",", ""))


class RetailKnowledgeBase:
    """Product and policy retrievers with explicit local grounding."""

    def __init__(self, products: list[dict[str, Any]] | None = None, policies: list[dict[str, Any]] | None = None):
        self.products = products if products is not None else build_product_documents()
        self.policies = policies if policies is not None else build_policy_documents()
        self.product_index = TextVectorIndex(self.products)
        policy_docs = [
            {**doc, "text": f"{doc['title']}. {doc['topic']}. {doc['policy_text']}"}
            for doc in self.policies
        ]
        self.policy_index = TextVectorIndex(policy_docs)

    def search_products(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        budget = parse_budget(query)
        allowed = None
        if budget is not None:
            allowed = [i for i, p in enumerate(self.products) if float(p["base_price"]) <= budget]
            if not allowed:
                return []
        results = self.product_index.search(query, top_k=top_k, allowed=allowed)
        if not results:
            return []
        minimum_score = max(.03, results[0]["relevance_score"] * .15)
        return [result for result in results if result["relevance_score"] >= minimum_score]

    def search_policies(self, query: str, top_k: int = 2) -> list[dict[str, Any]]:
        return self.policy_index.search(query, top_k=top_k)

    def answer_products(self, query: str, top_k: int = 3, use_llm: bool = True) -> dict[str, Any]:
        results = self.search_products(query, top_k)
        if not results:
            answer = "I could not find matching products within the stated constraints."
        else:
            answer = _generate_answer(
                "Answer the product request using only this catalog evidence. Do not invent features or prices. "
                "Mention why each item fits, show the product ID and price in INR, and state if evidence is limited.",
                query, results, use_llm,
            )
        return {"answer": answer, "products": results, "sources": [p["product_id"] for p in results]}

    def answer_policy(self, query: str, top_k: int = 2, use_llm: bool = True) -> dict[str, Any]:
        results = self.search_policies(query, top_k)
        stop_words = {
            "what", "is", "the", "a", "an", "can", "i", "do", "does", "my", "your",
            "how", "after", "before", "to", "from", "of", "on", "in", "for", "and",
            "or", "policy", "please", "product", "item", "items",
        }
        query_terms = set(re.findall(r"[a-z]+", query.casefold())) - stop_words
        best_score = max((item["relevance_score"] for item in results), default=0.0)
        grounded = [
            item for item in results
            if item["relevance_score"] >= max(.08, best_score * .25)
            and query_terms.intersection(set(re.findall(r"[a-z]+", item["policy_text"].casefold())))
        ]
        if not grounded:
            return {"answer": "I cannot answer from the available policy documents.", "evidence": [], "sources": []}
        answer = _generate_answer(
            "Answer the policy question using only the retrieved policy evidence. If the evidence does not resolve "
            "the question, say so. Never add a policy rule that is not stated. Cite each relevant policy_doc_id.",
            query, grounded, use_llm,
        )
        return {
            "answer": answer,
            "evidence": [{"policy_doc_id": p["policy_doc_id"], "text": p["policy_text"], "relevance_score": p["relevance_score"]} for p in grounded],
            "sources": [p["policy_doc_id"] for p in grounded],
        }


def _generate_answer(instructions: str, query: str, evidence: list[dict[str, Any]], use_llm: bool) -> str:
    """Use an optional chat model; default to an extractive, evidence-only response."""
    api_key = os.getenv("OPENAI_API_KEY")
    if use_llm and api_key:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                temperature=0,
                messages=[
                    {"role": "system", "content": instructions},
                    {"role": "user", "content": json.dumps({"query": query, "evidence": evidence}, ensure_ascii=False)},
                ],
            )
            content = response.choices[0].message.content
            if content:
                return content
        except Exception as exc:
            raise RuntimeError(f"LLM answer generation failed: {exc}") from exc
    # This offline response intentionally quotes retrieved evidence rather than filling gaps.
    if evidence and "policy_doc_id" in evidence[0]:
        return " ".join(
            f"{doc['policy_text']} [Source: {doc['policy_doc_id']}]" for doc in evidence
        )
    return "Retrieved catalog evidence: " + " ".join(
        f"{doc['product_title']} ({doc['product_id']}, ₹{float(doc['base_price']):.0f}) — {doc['text']}"
        for doc in evidence
    )


def prepare_indexes() -> dict[str, int]:
    """Build retrieval documents to validate the Phase 3/4 handoff."""
    products, policies = build_product_documents(), build_policy_documents()
    return {"product_embeddings": len(products), "policy_embeddings": len(policies)}


if __name__ == "__main__":
    kb = RetailKnowledgeBase()
    print(json.dumps(kb.answer_products("Recommend casual sneakers under ₹4000 for daily wear", use_llm=False), ensure_ascii=False, indent=2))
    print(json.dumps(kb.answer_policy("Can I return a discounted product after 7 days?", use_llm=False), ensure_ascii=False, indent=2))
