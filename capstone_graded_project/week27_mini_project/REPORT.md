# Week 27 Project Report

## Objective

Build a local policy assistant that answers internal company-policy questions with explicit grounding, citation, and minimal memory, staying within the assignment constraints.

## Solution summary

The implementation uses a small internal corpus stored in `solution/data/policies.jsonl`. The assistant performs retrieval over this corpus, attempts direct document lookup when a `doc_id` is present, and enforces grounded answers with a short quote and policy reference.

The project also includes:

- typed exceptions for missing policy documents
- a safe calculator tool
- memory storage for verified facts
- a simple graph workflow with step budgeting and critic validation
- pytest coverage for tools, acceptance checks, and graph budget logic

## End-to-end verification

We validated the project in the following ways:

1. Ran the test suite:
   `python -m pytest -q`
   Result: 7 passed.

2. Ran a live sample question:
   `python app.py --mode ask --question "What is our remote work limit? Cite and quote."`

   Output:
   `remote_work_v9.9: "Remote work is allowed only for up to two days per week and must be pre-approved by the manager."`

3. Ran the batch evaluation script:
   `python app.py --mode eval`

   Result: all six sample questions returned valid `doc_id`, quote, and pass status.

## Conclusion

No major pending issues remain. The Week 27 assignment is working end-to-end in the current repository state.
