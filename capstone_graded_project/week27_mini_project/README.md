# Week 27 Mini Project: Policy Assistant Agent

This project implements a local, deterministic policy assistant that answers grounded questions using an internal policy corpus without relying on external services.

## Project Status

Status: complete and validated.

## Structure

- `prob_definition/` — assignment brief and template files
- `solution/` — working implementation
  - `data/policies.jsonl`
  - `models.py`
  - `tools.py`
  - `prompts.py`
  - `memory.py`
  - `agent.py`
  - `graph.py`
  - `eval.py`
  - `app.py`
  - `tests/`
  - `README.md`

## Run locally

```bash
cd capstone_graded_project/week27_mini_project/solution
python -m pip install -r requirements.txt
python app.py --mode eval
python app.py --mode ask --question "What is our remote work limit? Cite and quote."
```

## Validation

The implementation was checked with:

```bash
python -m pytest -q
```

Result: 7 passed.

## Notes

The app is intentionally deterministic and grounded to the local policy corpus. Every answer is expected to include a valid `doc_id` and a short quote from the relevant policy text.
