# Week 27 Mini Project: Policy Assistant Agent

This solution implements a lightweight, local-first policy assistant that follows the assignment brief for the Week 27 graded project. The design mirrors the problem-solving pattern used in earlier projects like Week 24 (retrieval-augmented generation) and Week 19 (model training workflow): define a small corpus, build deterministic retrieval, enforce grounded prompting, and add a simple evaluation harness.

## Architecture

The project is organized as a modular agent system:

- `data/policies.jsonl`: internal policy corpus
- `models.py`: state/data models
- `tools.py`: policy search, read-by-doc-id, and safe calculator
- `prompts.py`: system and critic instructions
- `memory.py`: FAISS-backed verified fact memory with a fallback search path
- `agent.py`: ReAct-style policy question answering
- `graph.py`: step-based orchestration with a retry budget
- `eval.py`: canned Q&A acceptance checks and summary printing
- `app.py`: CLI entry point
- `tests/`: unit and acceptance tests

## Quick start

1. Create a virtual environment:

```bash
cd capstone_graded_project/week27_mini_project/solution
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Run the evaluation set:

```bash
python app.py --mode eval
```

3. Ask a single question:

```bash
python app.py --mode ask --question "What is our remote work limit? Cite and quote."
```

4. Run tests:

```bash
pytest -q
```

## Design notes

The project is intentionally deterministic and local. Retrieval is done against a small in-project JSONL corpus, and the agent is forced to ground answers in a policy `doc_id` and a short direct quote. The memory layer only writes verified facts when a valid `doc_id` is present, which matches the assignment requirement.

## Outputs

The evaluation script prints a compact table with:

- question
- doc_id present
- quote present
- valid result
- steps used
- runtime in seconds

This makes the workflow easy to assess without any external service or paid API.
