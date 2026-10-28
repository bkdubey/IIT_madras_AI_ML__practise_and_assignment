"""Run every project phase and write the final evaluation artifacts."""

from __future__ import annotations

import json

from phase6_evaluation import run_evaluation


if __name__ == "__main__":
    report = run_evaluation()
    print(json.dumps(report, ensure_ascii=False, indent=2))
