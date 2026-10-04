from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any, Dict, List

from models import PolicyDoc, SearchResult

DATA_FILE = Path(__file__).resolve().parent / "data" / "policies.jsonl"


class PolicyNotFoundError(ValueError):
    """Raised when a requested policy doc_id does not exist."""


class PolicySearchError(RuntimeError):
    """Raised when a local search query cannot be processed."""


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9_.-]+", text.lower())


def load_policies() -> List[PolicyDoc]:
    docs: List[PolicyDoc] = []
    if not DATA_FILE.exists():
        return docs
    with DATA_FILE.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            docs.append(PolicyDoc(doc_id=row["doc_id"], text=row["text"], date=row["date"]))
    return docs


def read_policy(doc_id: str) -> str:
    for doc in load_policies():
        if doc.doc_id.lower() == doc_id.lower():
            return doc.text
    raise PolicyNotFoundError(f"Unknown policy doc_id: {doc_id}")


def _extract_quote(text: str, max_len: int = 140) -> str:
    matches = re.findall(r'"([^"]+)"', text)
    if matches:
        quote = matches[0]
        return quote[:max_len]
    return text[:max_len].strip()


def policy_search(query: str, k: int = 3) -> List[SearchResult]:
    if not query or not query.strip():
        raise PolicySearchError("Search query cannot be empty.")

    q_tokens = set(_tokenize(query))
    scored: List[tuple[float, SearchResult]] = []
    for doc in load_policies():
        tokens = set(_tokenize(doc.text))
        overlap = len(q_tokens & tokens)
        if not overlap:
            phrase_bonus = 0.1 if any(token in doc.text.lower() for token in _tokenize(query)) else 0.0
            score = phrase_bonus
        else:
            score = overlap + 0.05 * len(q_tokens)
        snippet = _extract_quote(doc.text)
        scored.append((score, SearchResult(doc_id=doc.doc_id, score=score, snippet=snippet, date=doc.date)))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [result for _, result in scored[:k]]


def extract_doc_id(question: str) -> str | None:
    match = re.search(r"([A-Za-z0-9_]+\.[A-Za-z0-9_]+|[A-Za-z0-9_]+)", question)
    if not match:
        return None
    candidate = match.group(1)
    docs = {doc.doc_id.lower() for doc in load_policies()}
    if candidate.lower() in docs:
        return candidate
    return None


def safe_calculator(expression: str) -> float:
    valid_ops = {ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b,
                 ast.Mult: lambda a, b: a * b, ast.Div: lambda a, b: a / b,
                 ast.Pow: lambda a, b: a ** b}

    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Invalid calculation expression: {expression}") from exc

    def _eval(node):
        if isinstance(node, ast.BinOp) and type(node.op) in valid_ops:
            left = _eval(node.left)
            right = _eval(node.right)
            if isinstance(node.op, ast.Div) and right == 0:
                raise ZeroDivisionError("Division by zero is not allowed.")
            return valid_ops[type(node.op)](left, right)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -_eval(node.operand)
        raise ValueError(f"Unsupported expression: {expression}")

    return float(_eval(tree.body))


def safe_divide(a: float, b: float) -> float:
    if b == 0:
        raise ZeroDivisionError("Division by zero is not allowed.")
    return float(a / b)
