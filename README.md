# churn-analytics: tabular ML, end to end

Predicting customer churn on the public Telco dataset (7,043 customers, 26.5% churn).
Pandas cleaning, feature engineering, scikit-learn pipelines, stratified 5-fold CV, a cost-based
decision threshold and a static dashboard.

**Live dashboard:** https://shaurya55555.github.io/churn-analytics/

## Results

Held-out test split (20%, stratified, 1,409 rows). Models compared with 5-fold CV on the training split only.

| model | CV ROC-AUC | CV PR-AUC | CV F1@0.5 |
|---|---|---|---|
| dummy (prior) | 0.500 | 0.265 | 0.000 |
| logistic regression | **0.847** | **0.662** | 0.594 |
| random forest | 0.843 | 0.657 | 0.573 |
| hist gradient boosting | 0.844 | 0.658 | 0.581 |

Selected model: logistic regression. **Test ROC-AUC 0.845, PR-AUC 0.653** (dummy baseline 0.5 / 0.265).

Business framing: a missed churner costs 5x a wasted retention offer. The threshold (0.19) is chosen on
out-of-fold predictions of the training split, then applied once to the test split:
**68% lower cost than offering nobody, 43% lower than offering everyone.**

## Findings worth knowing
- The three model families are within 0.5 AUC points of each other. Feature engineering and threshold choice
  moved the business outcome far more than model choice, and the simplest model won.
- Month-to-month contracts, short tenure and fiber internet drive churn (see permutation importance in the dashboard).
- `TotalCharges` is blank for tenure-0 customers; imputed as 0 (documented in `load_clean`).

## Features added (`src/pipeline.py: add_features`)
`n_services` (count of add-on services), `avg_monthly_spend` (TotalCharges / tenure),
`charge_gap` (current price minus historical average, a price-hike signal), `tenure_bucket`, `autopay`.

## Run
    pip install pandas numpy scikit-learn matplotlib pytest
    python src/pipeline.py        # writes reports/metrics.json and docs/index.html
    pytest

Data: OpenML id 42178 (cached to `data/telco_churn.csv`).

## Limits
Single dataset, one random split plus CV, no calibration analysis, no drift or fairness checks.
The 5:1 cost ratio is an assumption, not measured.
