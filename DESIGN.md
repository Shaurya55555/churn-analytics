# Design decisions and likely questions: churn-analytics

**Why ROC-AUC and PR-AUC and not accuracy?**
Only 26.5% of customers churn, so predicting "no churn" for everyone scores 73.5% accuracy. ROC-AUC measures ranking
quality across thresholds; PR-AUC is more informative on the minority class. The dummy model gets AUC 0.5.

**Why did logistic regression win?**
The three families are within 0.5 AUC points of each other. With about 7,000 rows and mostly categorical
features, a regularized linear model is hard to beat, and it is explainable. Feature engineering and the
threshold mattered more than the model.

**How did you avoid leakage?**
The test split is held out before anything else. Model comparison uses 5-fold CV on the training split only. The
decision threshold is chosen on out-of-fold predictions of the training split, then applied once to the test split.
(My first version tuned the threshold on the test set; I caught that and fixed it.)

**Where does the 68% come from?**
Assume a missed churner costs 5 and a wasted retention offer costs 1. At the chosen threshold the test-set cost is
68% lower than offering nobody and 43% lower than offering everyone. The 5:1 ratio is an assumption, not data.

**What features did you add and why?**
`n_services` (engagement), `avg_monthly_spend` and `charge_gap` (a price-hike signal: current monthly charge minus
historical average), `tenure_bucket` (churn is non-linear in tenure), `autopay`.

**Data issue:** `TotalCharges` is blank for tenure-0 customers; imputed as 0 because they have not been billed yet.

**What would you do next?**
Calibration check, SHAP for per-customer explanations, a time-based split if dates were available, and measuring
the real cost ratio with the business.
