# Week 29 Mini Project: Reliable Retail Assistant

This solution builds a compact retail assistant that supports text retrieval and multimodal image-to-query retrieval for a catalog of consumer products. The system combines a product catalog, FAISS-backed semantic retrieval, schema-validated LLM output, budgeted logging, and a lightweight regression harness.

## Project structure

- `retail_assistant.py`: complete end-to-end implementation
- `requirements.txt`: Python dependencies
- `results/example_runs.json`: demo outputs and evaluation summary

## Features

- Product catalog with 30+ products across shoes, phones, laptops, headphones, and earbuds
- Retrieval documents built from structured product specs and descriptions
- FAISS vector index with cosine-similarity style normalization
- Optional OpenAI integration with graceful local fallback when no API key is available
- Strict JSON schema validation using Pydantic
- Observability logs for each agent step
- Unified controller for text-only and image-based product queries
- Regression checks spanning 5+ categories and retrieval scenarios

## Run locally

```bash
cd capstone_graded_project/week29_mini_project/solution
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python retail_assistant.py
```

## Notes

The implementation works fully offline by default. If `OPENAI_API_KEY` is available, it automatically prefers the OpenAI embedding and chat APIs for better natural-language behavior, but the code still runs correctly with the built-in local retrieval and recommendation fallback.
