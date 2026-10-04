from __future__ import annotations

import json
import time
from typing import Any, Dict, List

from agent import PolicyAssistantAgent
from memory import VerifiedMemory
from models import CriticResult, GraphState
from prompts import CRITIC_PROMPT
from tools import extract_doc_id


class PolicyGraph:
    def __init__(self, memory: VerifiedMemory | None = None, max_steps: int = 5):
        self.agent = PolicyAssistantAgent(memory=memory)
        self.memory = memory or self.agent.memory
        self.max_steps = max_steps

    @staticmethod
    def _critic(answer: str) -> CriticResult:
        issues: List[str] = []
        if not extract_doc_id(answer):
            issues.append("Missing doc_id in answer.")
        if '"' not in answer:
            issues.append("Missing quoted evidence.")
        if "No grounded policy found" in answer:
            issues.append("No grounded policy found.")
        return CriticResult(valid=not issues, issues=issues)

    def execute(self, question: str) -> GraphState:
        state = GraphState(question=question, max_steps=self.max_steps)
        start = time.perf_counter()
        for step in range(self.max_steps):
            state.steps = step + 1
            if step == 0:
                state.memory_hints = self.memory.read_hints(question, k=2)
                state.context = "memory_hints=" + json.dumps(state.memory_hints)
            elif step == 1:
                result = self.agent.answer_question(question)
                state.answer = result["answer"]
                state.doc_id = result["doc_id"]
                state.search_results = [{"doc_id": result["doc_id"], "answer": result["answer"]}]
            elif step == 2:
                state.critic = self._critic(state.answer)
                if not state.critic.valid:
                    state.retry_count += 1
                    if state.retry_count <= 1:
                        state.stop_reason = "retrying_after_critic"
                        continue
                    state.stop_reason = "critic_failed_after_retry"
                    break
            elif step == 3:
                if state.doc_id and state.answer:
                    self.memory.write_fact(state.doc_id, state.answer)
                    state.stop_reason = "memory_written"
                    break
            else:
                state.stop_reason = "budget_exhausted"
                break

        if not state.stop_reason:
            state.stop_reason = "completed"
        state.steps = min(state.steps, self.max_steps)
        state.elapsed = time.perf_counter() - start
        return state

    def evaluate(self, questions: List[str]) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for idx, question in enumerate(questions, start=1):
            state = self.execute(question)
            rows.append({
                "id": idx,
                "question": question,
                "valid": self._critic(state.answer).valid,
                "doc_id_present": bool(extract_doc_id(state.answer)),
                "quote_present": '"' in state.answer,
                "steps": state.steps,
                "elapsed": round(state.elapsed, 3),
                "stop_reason": state.stop_reason,
            })
        return rows
