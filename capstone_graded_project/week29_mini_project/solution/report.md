# Week 29 Mini Project Report

## Problem summary
The Week 29 graded mini project requires building a reliable retail assistant that supports both text-based and image-based product discovery. The assistant must retrieve products from a catalog, provide grounded recommendations, enforce strict schemas, maintain logs, and pass a lightweight evaluation harness.

The main challenge is building a trustworthy AI workflow instead of a simple prompt. The system must use retrieval to ground recommendations in product context, not freeform hallucination, and must keep a controlled behavior under budgeted steps and validation checks.

## Why RAG is needed
Retail queries are naturally variable and underspecified. Typical requests include:
- “Recommend running shoes under $100”
- “Which phone supports wireless charging under $600?”
- “Find similar products from a shoe image”

A single LLM prompt alone is not reliable enough because it may hallucinate product names, prices, or specs. By retrieving only relevant product documents from a catalog and then grounding the recommendation on that retrieved context, the system becomes more accurate, explainable, and easier to validate.

## System design
The solution uses a lightweight product catalog with 31 entries across shoes, phones, laptops, headphones, and earbuds. Each product is converted into a compact text document containing:
- product ID
- name
- category
- brand
- price
- specs
- description

These documents are embedded and indexed in FAISS for fast similarity search. For a user query, the agent retrieves the top-k entries and passes them as context to a recommendation step. The output is forced into strict JSON using Pydantic validation.

## Image-to-query design
For image inputs, the system does not embed the image directly. Instead, it follows the required pattern:
- image → extract structured query → text retrieval → recommendation

This design is chosen because it is more robust and more controllable in a classroom/vocareum environment. The image is converted into a query such as category, brand hints, and key attributes; this query is then used to retrieve matching products from the same catalog. If the vision call fails, the system gracefully uses a fallback query so the workflow remains operational.

## Reliability controls
The solution includes the required safeguards:
- schema validation using Pydantic
- strict JSON parsing with safe failure on malformed outputs
- step budget enforcement in the agent controller
- observability logging for each request
- evaluation cases with include/exclude product IDs
- regression threshold assertion

These controls reduce the chance of silent failure and make the assistant suitable for a production-minded workflow.

## Evaluation results
The evaluation harness includes 6 test queries covering sports shoes, phones, gaming laptops, travel headphones, earbuds, and lightweight laptops.

Observed pass rate: 66.67%
Regression guardrail: pass rate >= 0.60
Result: PASS

This demonstrates that the retrieval and recommendation logic remains stable across multiple product categories.

## Limitations and future improvements
The current system is intentionally lightweight and does not depend on heavy model training or a full production UI. The key limitations are:
- reliance on a curated catalog rather than a live catalog
- simple fallback logic for image extraction when vision is unavailable
- local retrieval quality depends on the quality of the product descriptions and query phrasing

Future improvements could include:
- richer product metadata and pricing filters
- hybrid retrieval using BM25 + vector search
- better multimodal prompt tuning for image queries
- automatic quality dashboards and golden test tracking

## Conclusion
This project demonstrates a practical RAG-based retail assistant that supports both text and image discovery flows. It follows the assignment’s core requirements: retrieving relevant products, generating grounded recommendations, validating the output schema, logging the workflow, and testing the system with a lightweight regression harness.
