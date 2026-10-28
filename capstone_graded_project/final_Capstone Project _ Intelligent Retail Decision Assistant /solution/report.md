# Intelligent Retail Decision Assistant — Project Report

## Executive summary

This project implements the requested six-phase retail prototype: tabular purchase prediction, customer segmentation, product/policy retrieval, grounded answer generation, and a transparent controller. It is runnable offline and includes an executed notebook, tests, and saved evaluation artifacts.

**Data limitation:** the brief mentions downloadable `retail_interactions.csv` and `retail_policies.json`, but those files were not present with the local assignment. The included 1,200-row, 24-product interaction dataset and six policies are deterministic synthetic/demo inputs created with seed 42. Results are illustrative and must not be interpreted as retailer findings or actual policy advice. Replace the synthetic inputs with the course-provided files if available before final submission.

## Design decisions

| Decision | Reason |
|---|---|
| Predict `purchase_flag` with class-weighted logistic regression | It is a clear binary business outcome and a transparent baseline; class weighting helps address the generated class balance. |
| Use a stratified 80/20 holdout and fit preprocessing inside a pipeline | Keeps the test set out of imputer/scaler/encoder fitting and preserves class proportions. |
| Exclude `returned_flag` and `rating_given` | These are post-purchase outcomes and would leak information into purchase prediction. |
| Aggregate interactions to one customer profile before K-Means | The business request is customer segmentation; clustering rows would count frequent customers repeatedly and could create trivial segments. |
| Use TF-IDF text vectors with cosine/FAISS inner-product retrieval | This provides a lightweight, explainable, offline retrieval baseline. FAISS is optional; scikit-learn cosine search is the fallback. |
| Make LLM generation opt-in | The complete demonstration works without secrets, external API access, or paid services. The local response quotes retrieved evidence; configured API errors surface explicitly. |
| Use one-step keyword routing | The policy/product/analytics decision is inspectable and bounded, as required; no complex agent framework is needed for this scope. |

## Phase results

### Phase 1 — Dataset

The generated dataset has 1,200 rows and 41 columns. It has 684 positive purchase labels and 516 negative labels. The 516 missing ratings/review strings correspond to non-purchase interactions; the empty `image_path` is an unused optional multimodal field. The data dictionary in `README.md` identifies generated and engineered fields.

### Phase 2 — Classical ML

| Test metric | Result |
|---|---:|
| Test rows | 240 |
| Accuracy | 0.592 |
| Precision | 0.679 |
| Recall | 0.540 |
| F1 | 0.602 |
| ROC-AUC | 0.623 |

This is a modest baseline, only somewhat better than chance ranking (ROC-AUC 0.5). It demonstrates a valid preprocessing and evaluation workflow, not a production-ready predictor. Model coefficients are associations, not causal explanations. Inspect `outputs/purchase_feature_importance.csv`.

Customer profiles are clustered using scaled recency, average order value, order history, mean interacted-product price/discount, page time, and wishlist rate. The silhouette-selected result is **4 clusters, 0.132 silhouette**. This indicates weak cluster separation, so the profiles should not be marketed as stable customer personas. `outputs/segment_summary.csv` contains the cluster sizes and feature means; `outputs/segmentation_summary.json` records the selection and caveat.

### Phases 3–4 — Retrieval and grounded answers

Product documents combine title, description, brand, category, style tags, and price. Policy records have stable IDs. Product search filters explicit rupee budgets before returning results and removes weak matches. Policy responses require retrieved lexical evidence, quote the corresponding policy text, cite policy IDs, and abstain if no relevant document is found. Optional OpenAI chat generation is disabled by default; the local answer path is extractive to keep claims tied to retrieved text.

### Phase 5 — Controller

The controller routes each request to exactly one tool: product search (default), policy search (policy-related terms), or analytics (segment/conversion terms). It returns the chosen route and a one-step reason trace. The curated discount-return query routes to policy search and cites `RET_02`.

### Phase 6 — Evaluation

Five representative requests and their full results are in `outputs/example_queries.json`; the aggregate metrics and assertions are in `outputs/evaluation_report.json`. The test set checks the budget constraint, policy citation, empty-query handling, weak-policy abstention, all three routes, customer-level cluster counts, and three results for the specified daily-wear summary query. The notebook contains the same end-to-end phase walkthrough and was executed without cell errors.

## Comparison with previous mini-project deliverables

The week 27 submission emphasizes a modular architecture and compact evaluation traces. The week 29 submission adds a narrative `report.md` and a separate demo-output artifact. This capstone follows those useful presentation patterns while covering a broader scope: its README explains setup/provenance, its notebook walks each phase, JSON/CSV outputs preserve reproducible metrics and result traces, and this report summarizes rationale, findings, and caveats. It does not claim week 29’s multimodal image support or week 27’s memory/retry graph; those are outside this brief’s mandatory scope.

## Submission checklist and limitations

- [x] Phasewise source code and runnable end-to-end command
- [x] Notebook with all phases and example queries
- [x] Data dictionary and explicit synthetic-data provenance
- [x] Supervised and unsupervised metrics/results
- [x] Product and policy retrieval with sources
- [x] Explicit controller routes and trace
- [x] Five curated examples, saved outputs, and regression tests
- [ ] The original course-provided CSV and policy JSON were not available locally; use those actual files if the course portal provides them.

Run from `solution/`: `python run_all.py` and `pytest -q`. The whole-repository pytest command currently also collects older mini-project tests with unrelated import errors; run this solution’s scoped test suite as documented.
