# Week 24 Mini Project: Grounded Answers from a Small Knowledge Pack

This solution implements a lightweight Retrieval-Augmented Generation (RAG) pipeline for a curated knowledge base on responsible AI. The project follows the graded assignment brief in the Week 24 problem definition.

- a small local corpus of 6 documents on one topic
- chunking experiments across multiple chunk sizes and overlaps
- a deterministic local embedding baseline and FAISS retrieval
- two prompt templates (P1 and P2)
- conversation memory for follow-up questions
- JSONL/CSV logging and a manual evaluation rubric

## Project structure

```text
solution/
├── README.md
├── requirements.txt
├── week24_rag_analysis.ipynb
├── knowledge_base/
│   ├── 01_fairness_and_bias.md
│   ├── 02_privacy_and_data_minimization.md
│   ├── 03_transparency_and_explainability.md
│   ├── 04_human_oversight.md
│   ├── 05_monitoring_and_governance.md
│   └── 06_risk_management.md
└── results/
	├── experiment_runs.jsonl
	└── evaluation_summary.csv
```

## Objective

Build a minimal but realistic RAG pipeline that tests how retrieval and prompting choices affect:

- groundedness
- citation correctness
- answer relevance
- answer conciseness
- conversational continuity in multi-turn follow-up questions

## Quick start

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r capstone_graded_project/week24_mini_project/solution/requirements.txt
```

3. Open and run all cells in sequence in the notebook:

```bash
jupyter notebook capstone_graded_project/week24_mini_project/solution/week24_rag_analysis.ipynb
```

4. Review generated artifacts after execution:

```bash
ls capstone_graded_project/week24_mini_project/solution/results
```

## RAG design

The pipeline follows the assignment flow:

1. Load a small curated corpus from the `knowledge_base` folder.
2. Split documents with a configurable chunk size and overlap.
3. Embed chunks using a deterministic local hashing embedder.
4. Store vectors in FAISS.
5. Retrieve the most relevant chunks for each question.
6. Assemble a prompt that enforces citations and strict grounding.
7. Maintain a short conversation buffer so follow-up questions use prior context.
8. Log every answer and manually score it using the rubric described in the assignment.

## Prompt templates

The project includes two variants:

- P1: concise response under 120 tokens, cite context like `[Title#page]`, refuse when unsure.
- P2: concise but structured with `'Evidence:'` and `'Answer:'` sections and a brief enforcement to ground the answer to the snippet set.

## Experiment grid

The project runs a compact A/B-style experiment set that varies one factor at a time:

- chunk sizes: 300, 600, 1000
- overlap: 10%, 20%
- embedding baseline: deterministic local hashing vectors
- prompts: P1 vs P2

This yields a reproducible comparison table without the full Cartesian explosion.

## Notes

- The assignment requires a local knowledge pack, a retrieval layer, and prompt-engineering constraints. This solution implements all of them in a fully executable project package.
- The notebook is self-contained; the Markdown files are source data and `requirements.txt` is the environment manifest.
- The final experiment results are generated as JSONL logs and a summary CSV file when the final cells run.

## Output artifacts

- `results/experiment_runs.jsonl`: per-answer logs including `run_id`, `question`, `answer`, `citations`, `used_chunks`, tokens, latency, and score metadata.
- `results/evaluation_summary.csv`: aggregated evaluation metrics per run, including relevance, groundedness, citation correctness, conciseness, memory continuity, and refusal correctness.
