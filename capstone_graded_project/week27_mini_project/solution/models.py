from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class PolicyDoc:
    doc_id: str
    text: str
    date: str


@dataclass
class SearchResult:
    doc_id: str
    score: float
    snippet: str
    date: str


@dataclass
class CriticResult:
    valid: bool
    issues: List[str] = field(default_factory=list)


@dataclass
class GraphState:
    question: str
    context: str = ""
    memory_hints: List[str] = field(default_factory=list)
    answer: str = ""
    critic: Optional[CriticResult] = None
    doc_id: Optional[str] = None
    steps: int = 0
    retry_count: int = 0
    max_steps: int = 5
    stop_reason: str = ""
    search_results: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class EvalRecord:
    question: str
    has_doc_id: bool
    quote_present: bool
    valid: bool
    steps: int
    elapsed: float
