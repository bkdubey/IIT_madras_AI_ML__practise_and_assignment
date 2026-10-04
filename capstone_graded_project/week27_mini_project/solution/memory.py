from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List, Sequence

import numpy as np

try:
    import faiss  # type: ignore
except ImportError:  # pragma: no cover
    faiss = None


class VerifiedMemory:
    """Lightweight FAISS-like memory store for verified policy facts."""

    def __init__(self, path: str | None = None):
        self.path = Path(path) if path else Path(__file__).resolve().parent / "data" / "memory.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.entries: List[dict] = []
        self._load()
        self._index = None
        if faiss is not None:
            self._index = faiss.IndexFlatL2(32)
            self._rebuild_index()

    def _load(self) -> None:
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                self.entries.append(json.loads(line))

    def _save(self) -> None:
        with self.path.open("w", encoding="utf-8") as handle:
            for entry in self.entries:
                handle.write(json.dumps(entry) + "\n")

    @staticmethod
    def _embed(text: str) -> np.ndarray:
        tokens = re.findall(r"[a-zA-Z0-9_]+", text.lower())
        vec = np.zeros(32, dtype="float32")
        for i, token in enumerate(tokens[:32]):
            vec[i % 32] += (ord(token[0]) if token else 1) / 10.0
        return vec

    def _rebuild_index(self) -> None:
        if faiss is None or not self.entries:
            return
        arr = np.stack([self._embed(entry["fact"]) for entry in self.entries]).astype("float32")
        self._index.reset()
        self._index.add(arr)

    def write_fact(self, doc_id: str, fact: str) -> None:
        if not doc_id or not doc_id.strip():
            raise ValueError("Memory writes require a valid doc_id.")
        entry = {"doc_id": doc_id, "fact": fact.strip()}
        self.entries.append(entry)
        self._save()
        if self._index is not None:
            self._rebuild_index()

    def read_hints(self, question: str, k: int = 2) -> List[str]:
        if not self.entries:
            return []
        question_vec = self._embed(question)
        if self._index is not None:
            D, I = self._index.search(question_vec.reshape(1, -1), min(k, len(self.entries)))
            hits = [self.entries[i] for i in I[0] if i >= 0]
        else:
            hits = []
            for entry in self.entries:
                score = float(np.dot(self._embed(question), self._embed(entry["fact"])))
                hits.append((score, entry))
            hits.sort(key=lambda item: item[0], reverse=True)
            hits = [entry for _, entry in hits[:k]]
        return [entry["fact"] for entry in hits]

    def cheap_sufficient_hint(self, question: str) -> str | None:
        hints = self.read_hints(question, k=1)
        return hints[0] if hints else None
