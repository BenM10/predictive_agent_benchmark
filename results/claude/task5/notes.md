# Task 5 – Experiment Notes: Data Leakage Detection and Pipeline Correction

## Issues Identified in Original Pipeline (`task/task5_pipeline.py`)

| # | Issue | Severity |
|---|-------|----------|
| 1 | **Wrong data file path** – loads `airline_passenger_satisfaction_full.csv` instead of `Claude_clean.csv` | High (runtime error) |
| 2 | **Data leakage** – `StandardScaler.fit_transform()` called on the full dataset before train/test split | Critical |
| 3 | **Categorical variables not encoded** – `Gender`, `Customer Type`, `Type of Travel`, `Class` are object dtype; `StandardScaler` would raise `ValueError` | High |
| 4 | **ID column not dropped** – `id` is a row identifier with no predictive signal | Medium |
| 5 | **No cross-validation** – single random split is sensitive to seed choice | Medium |
| 6 | **Only accuracy reported** – insufficient for real-world evaluation | Low–Medium |

## Why Data Leakage is Critical

When `StandardScaler` is fit on the entire dataset before splitting, the mean and standard deviation computed by the scaler include information from the test set. This means:
- The model indirectly "sees" the test set during training via the scaler parameters.
- Reported test accuracy is optimistically biased — it does not reflect true out-of-sample performance.
- In this dataset (numeric features only), the leakage effect was measurable though small (typically <0.001), because the dataset is large and the leakage is subtle. However, in smaller datasets or with features that have high variance, the effect can be much larger.

## Corrections Applied

1. **Data path** corrected to `data/processed/Claude_clean.csv`.
2. **Split-first principle enforced**: `train_test_split` called on raw features; `StandardScaler` fit only on `X_train`, then `transform` applied to `X_test`.
3. **Categorical encoding**: `pd.get_dummies` applied to encode `Gender`, `Customer Type`, `Type of Travel`, `Class` before modelling.
4. **`id` column dropped** before modelling.
5. **Stratified 5-fold cross-validation** using `sklearn.pipeline.Pipeline` (which scopes the scaler fit to each fold's training data automatically).
6. **Extended evaluation**: Accuracy, ROC-AUC, precision, recall, F1-score, confusion matrix, and ROC curve all reported.

## Results

| Metric | Leaky (numeric only) | Correct (numeric only) | Corrected Full Pipeline |
|--------|---------------------|----------------------|------------------------|
| Accuracy (test) | 0.8262 | 0.8262 | 0.8754 |
| ROC-AUC (test) | 0.8849 | 0.8849 | 0.9282 |
| CV ROC-AUC (5-fold) | — | — | 0.9265 ± 0.0021 |

The leakage effect is numerically negligible on this large dataset (129k rows) for the numeric-only comparison, confirming that the dataset is too large for the leakage to shift summary statistics meaningfully. However, **the principle remains critical**: leakage can be significant with smaller datasets or features with high variance. The performance improvement when moving from numeric-only to the full corrected pipeline (0.8262 → 0.8754 accuracy, 0.8849 → 0.9282 ROC-AUC) demonstrates the value of properly encoding categorical features.

## Figures Produced

- `fig_17_leakage_comparison.png` – bar chart comparing leaky vs correct pipeline on accuracy and ROC-AUC
- `fig_18_cv_scores_corrected.png` – 5-fold cross-validation ROC-AUC scores
- `fig_19_confusion_matrix_and_roc.png` – confusion matrix and ROC curve for the corrected pipeline
