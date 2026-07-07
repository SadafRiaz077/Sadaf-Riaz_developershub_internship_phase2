# Customer Churn Prediction — End-to-End ML Pipeline

**Task 2 — DevelopersHub Corporation AI/ML Engineering Internship (Advanced Task Set)**

## Objective
Build a reusable, production-ready machine learning pipeline that predicts telecom customer
churn using the scikit-learn `Pipeline` API — combining preprocessing (scaling + encoding),
model training, hyperparameter tuning, and export into a single deployable artifact.

## Dataset
IBM Telco Customer Churn Dataset — 7,043 customers, 21 columns (demographics, account info,
services subscribed, and churn label). Loaded directly at runtime from IBM's public dataset
repository, so no manual download step is required.

## Methodology / Approach
1. **Data cleaning:**
   - Converted `TotalCharges` from string to numeric — this exposed 11 missing values
     (blank strings for customers with 0 tenure), which were left as `NaN` and handled by
     the pipeline's imputer rather than dropped manually.
   - Dropped `customerID` (non-predictive identifier).
   - Encoded the target `Churn` column as 1 (Yes) / 0 (No). Class distribution: **73.5% No
     churn / 26.5% Churn** — a moderately imbalanced target.
2. **Feature split:**
   - Numeric features: `tenure`, `MonthlyCharges`, `TotalCharges`
   - Categorical features: remaining 16 columns (`gender`, `Contract`, `PaymentMethod`, etc.)
3. **Preprocessing pipeline (`ColumnTransformer` inside `Pipeline`):**
   - Numeric → median imputation + `StandardScaler`
   - Categorical → most-frequent imputation + `OneHotEncoder`
4. **Train/test split:** 80/20 stratified split → 5,634 training rows, 1,409 test rows.
5. **Model training:** Two full pipelines (preprocessing + classifier) built for Logistic
   Regression and Random Forest.
6. **Hyperparameter tuning:** 5-fold `GridSearchCV` (scored on F1):
   - Logistic Regression: 8 candidate combinations (40 fits)
   - Random Forest: 12 candidate combinations (60 fits)
7. **Evaluation:** Both tuned models compared on the held-out test set using Accuracy,
   Precision, Recall, F1-score, and ROC-AUC.
8. **Export:** The better-performing pipeline was exported with `joblib` as
   `churn_prediction_pipeline.joblib` — reloadable and directly usable on raw customer data.

## Key Results / Observations

**Best hyperparameters found:**
| Model | Best Parameters | Best CV F1-Score |
|---|---|---|
| Logistic Regression | `C=10`, `penalty='l2'`, `solver='lbfgs'` | 0.5994 |
| Random Forest | `max_depth=10`, `min_samples_split=5`, `n_estimators=100` | 0.5833 |

**Test set performance:**
| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.8055 | 0.6572 | 0.5588 | 0.6040 | **0.8411** |
| Random Forest | 0.8062 | 0.6772 | 0.5160 | 0.5857 | 0.8401 |

**Observations:**

- **Data quality:** `TotalCharges` was stored as a string in the raw dataset and had 11
  missing values, all belonging to customers with `tenure = 0` (brand-new customers who
  hadn't been billed yet). Rather than dropping these rows, the pipeline's median imputer
  handled them automatically — this keeps the fix reproducible and avoids losing data.

- **Class imbalance:** The dataset is imbalanced (73.5% did not churn vs 26.5% churned).
  Because of this, **accuracy is not a reliable metric on its own** — a model predicting
  "No Churn" for every customer would already score ~73.5% accuracy while being useless for
  the business. This is why F1-score and ROC-AUC (which account for the minority/churn
  class) were used as the primary comparison metrics instead of accuracy alone.

- **Both models performed similarly on accuracy (~80.5–80.6%), but differed meaningfully on
  recall and F1:**
  - **Logistic Regression:** Accuracy 0.8055, Precision 0.6572, Recall 0.5588,
    F1-Score 0.6040, ROC-AUC 0.8411
  - **Random Forest:** Accuracy 0.8062, Precision 0.6772, Recall 0.5160,
    F1-Score 0.5857, ROC-AUC 0.8401

- **Logistic Regression was selected as the final model** because it achieved the higher
  F1-score (0.604 vs 0.586) and the higher ROC-AUC (0.841 vs 0.840). The main driver of this
  was **recall**: Logistic Regression correctly identified 55.9% of actual churners vs only
  51.6% for Random Forest. In a churn-prediction use case, missing an at-risk customer
  (false negative) is usually more costly than a false alarm, so higher recall on the churn
  class was prioritized over Random Forest's slightly better precision.

- **Random Forest's trade-off:** it was more "cautious" — when it predicted churn, it was
  right slightly more often (67.7% precision vs 65.7%), but it also missed more churners
  overall (lower recall). This makes it a weaker fit for a retention campaign where the goal
  is to catch as many at-risk customers as possible, even at the cost of a few extra false
  alarms.

- **Per-class breakdown for the final model (Logistic Regression):**
  - Non-churn class (0): Precision 0.85, Recall 0.89, F1 0.87 — the model is very reliable
    at identifying customers who will stay.
  - Churn class (1): Precision 0.66, Recall 0.56, F1 0.60 — noticeably weaker, which is the
    expected effect of class imbalance (fewer churn examples to learn from during training).

- **Hyperparameter tuning impact:** GridSearchCV tested 8 combinations for Logistic
  Regression (40 fits across 5 folds) and 12 combinations for Random Forest (60 fits). The
  best Logistic Regression configuration (`C=10`, `solver='lbfgs'`) suggests the model
  benefited from weaker regularization (larger `C`), i.e. it needed more flexibility to fit
  the churn patterns rather than being constrained too heavily.

- **Practical takeaway:** on its own, this pipeline is a reasonable first-pass churn model
  (ROC-AUC of 0.84 indicates decent separability between churners and non-churners), but the
  weaker recall/F1 on the churn class shows there is room for improvement — primarily
  through better handling of class imbalance (see Future Improvements below) rather than
  further tuning of the current models.

## Files in this repository
- `Task2_Customer_Churn_Pipeline.ipynb` — full notebook (preprocessing, training, tuning,
  evaluation, export) with all outputs already executed
- `churn_prediction_pipeline.joblib` — exported, ready-to-use trained pipeline
  (Logistic Regression, tuned)
- `README.md` — this file

## How to reuse the exported pipeline
```python
import joblib
pipeline = joblib.load("churn_prediction_pipeline.joblib")
predictions = pipeline.predict(new_customer_dataframe)  # raw data, no preprocessing needed
```

## Possible Future Improvements
- Address class imbalance with `class_weight='balanced'` or SMOTE to improve recall further
- Add feature importance analysis (particularly for Random Forest) to identify top churn drivers
- Deploy the exported pipeline behind a Streamlit form for live churn prediction
