from __future__ import annotations

import argparse

from agent import PolicyAssistantAgent
from eval import run_batch
from graph import PolicyGraph


def main() -> int:
    parser = argparse.ArgumentParser(description="Policy Assistant Agent")
    parser.add_argument("--mode", choices=["ask", "eval"], default="ask")
    parser.add_argument("--question", default="What is our remote work limit? Cite and quote.")
    args = parser.parse_args()

    if args.mode == "eval":
        run_batch()
        return 0

    agent = PolicyAssistantAgent()
    answer = agent.answer_question(args.question)
    print(answer["answer"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
