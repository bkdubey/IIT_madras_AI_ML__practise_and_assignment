# Intelligent Retail Decision Assistant

An end-to-end, phasewise retail decision-support prototype. It connects a tabular purchase model and behavioral segmentation to product retrieval, evidence-grounded policy answers, and an inspectable query controller.

## Reproducibility and data provenance

The assignment references `retail_interactions.csv` and `retail_policies.json`, but neither supporting file was included alongside the project brief. To keep this submission runnable without inventing an external dataset source, Phase 1 generates a deterministic synthetic dataset (1,200 interaction rows, 24 products, fixed seed 42) and a small illustrative policy corpus. The data is persisted under `data/raw/` and `data/policies/` when the pipeline runs. **These data are synthetic; metrics and policy answers are a demonstration, not real business findings or legal guidance.**

The generator is `phase1_data_preparation.py`. The product catalog and policy statements are authored for this prototype, not collected from a retailer. A future real dataset can be supplied as `data/raw/retail_interactions.csv`; required field checks will fail explicitly if it does not support the pipeline.

### Data dictionary and provenance

All records are synthetic. “Generated” fields simulate a plausible base customer-product interaction; “engineered” fields are derived or added to support the assignment's retrieval and outcome tasks.

| Fields | Type | Provenance / purpose |
|---|---|---|
| `interaction_id`, `customer_id`, `age`, `income_band`, `membership_level`, `city_tier`, `total_orders`, `avg_order_value`, `days_since_last_purchase`, `preferred_channel` | Mixed | Generated customer and interaction context |
| `product_id`, `product_title`, `category`, `sub_category`, `brand`, `color`, `material`, `size`, `season`, `base_price`, `product_description`, `style_tags` | Mixed | Generated product catalog attributes; description and tags are authored for retrieval |
| `discount_percent`, `final_price`, `stock_level`, `avg_product_rating`, `payment_mode`, `delivery_type`, `time_on_page`, `added_to_cart`, `wishlist_flag` | Mixed | Generated interaction/commercial features |
| `purchase_flag`, `quantity`, `returned_flag`, `rating_given`, `customer_review_text` | Mixed | Generated outcomes and illustrative review text |
| `customer_query`, `policy_doc_id`, `policy_text_snippet`, `occasion_tags`, `image_path` | Text | Engineered/example extensions for retrieval; `image_path` is intentionally empty |
| `data/policies/retail_policies.json` | JSON | Six authored demonstration policies with stable IDs `RET_01`–`RET_06` |

## Phasewise implementation

1. **Problem framing and dataset preparation** — `phase1_data_preparation.py` generates/loads inputs, validates schema, reports missingness and class balance.
2. **Classical ML** — `phase2_classical_ml.py` trains class-weighted logistic regression using a stratified holdout and explicit numeric/categorical preprocessing; it also aggregates interactions to one profile per customer and selects K-Means customer segments using silhouette score.
3. **Retrieval corpus** — `phase3_retrieval_corpus.py` creates one text document per unique product and validates policy references.
4. **RAG and grounded generation** — `phase4_rag.py` uses TF-IDF vectors as local text embeddings, with FAISS inner-product search when available and a scikit-learn cosine-search fallback. Policy answers cite retrieved policy IDs and refuse when no relevant policy evidence is found. Product queries with a parsed rupee budget strictly filter out over-budget items.
5. **Agent/controller** — `phase5_controller.py` applies explicit keyword rules and makes exactly one route to product search, policy search, or descriptive retail analytics; responses include a one-step trace.
6. **Evaluation and packaging** — `phase6_evaluation.py` runs five curated queries, regression assertions, and saves reports and example outputs under `outputs/`.

The submission-facing narrative, rationale, results, limitations, and checklist are in [`report.md`](report.md). The executed notebook and saved machine-readable outputs provide the reproducible walkthrough and evidence behind that report.

The optional OpenAI chat completion is opt-in (`RetailAssistant(use_llm=True)`) and requires `OPENAI_API_KEY`. By default the assistant produces extractive responses from retrieved evidence and runs offline, even if a key is present. A configured API failure is reported rather than silently falling back. No image understanding is implemented; the image field is optional in the brief.

## Run

From this directory:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# Optional FAISS index and OpenAI generation:
pip install faiss-cpu openai
python run_all.py
pytest -q
```

Or open and run `notebooks/Intelligent_Retail_Decision_Assistant.ipynb` from this directory. Generated input data, processed retrieval documents, ML metrics, customer-level segment files, and five example assistant runs are written to `data/` and `outputs/`.

## Evaluation notes and limitations

The purchase target is predicted from customer, product, and interaction features available at interaction time; post-purchase columns such as `returned_flag` and `rating_given` are excluded to prevent target leakage. A stratified 80/20 test split and ROC-AUC, precision, recall, F1, and accuracy are reported. Class weighting addresses the generated class distribution. The coefficient table is a linear-model association, not causal feature importance.

On the fixed-seed 1,200-row synthetic run, the holdout baseline reports accuracy 0.592, F1 0.602, and ROC-AUC 0.623 (240 test rows). The selected four-cluster customer-profile solution has silhouette 0.132 and is most separated by average order value. The weak separation indicates limited segment structure; neither these clusters nor the model results should be presented as strong evidence about real customers. The five curated requests route to product search, policy search, or analytics, the discounted-return example cites `RET_02`, and the summary query returns three relevant daily-wear products.

Clustering is performed on customer profiles aggregated from interaction rows, using scaled recency, spend, order history, mean interacted-product price, discounts, page time, and wishlist activity. The highest silhouette score among `k=2..5` is selected. Cluster descriptions use the feature with the greatest standardized separation; they should not be treated as validated personas or causal customer behavior.

TF-IDF is an interpretable, offline embedding baseline rather than a neural semantic embedding model. FAISS is an optional approximate ecosystem component here (`IndexFlatIP` performs exact inner-product search). LLM generation, if enabled, cannot guarantee factuality; answers include retrieved sources and should be checked against the evidence. The local fallback is intentionally extractive. This prototype is not a production recommendation, pricing, privacy, or policy system.
