# Task 3 – Baseline Model Training
**Agent:** Claude (claude-sonnet-4-6)
**Date:** 2026-03-15

---

## Experiment Setup

| Setting | Value |
|---------|-------|
| Dataset | `data/processed/Claude_clean.csv` |
| Train/test split | 80% / 20% (random_state=42, stratified) |
| Train size | 103,589 rows |
| Test size | 25,898 rows |
| Positive rate (test) | 43.5% satisfied |

### Feature Preparation
- Dropped `id` column
- One-hot encoded 4 categorical columns: `Gender`, `Customer Type`, `Type of Travel`, `Class`
- Final feature matrix: **23 features**
- Target encoded: `satisfied=1`, `neutral or dissatisfied=0`
- Logistic Regression uses `StandardScaler` inside a Pipeline

---

## Models Trained

| Model | Config |
|-------|--------|
| Logistic Regression | `max_iter=1000`, `StandardScaler` |
| Random Forest | `n_estimators=200`, `max_depth=15` |
| HistGradientBoosting | `max_iter=200`, `max_depth=6`, `lr=0.1` |

---

## Results

| Model | Accuracy | F1 Score | ROC AUC |
|-------|----------|----------|---------|
| Logistic Regression | 0.8744 | 0.8529 | 0.9270 |
| Random Forest | 0.9596 | 0.9530 | 0.9933 |
| **HistGradientBoosting** | **0.9617** | **0.9552** | **0.9948** |

---

## Key Observations

1. **All three models are strong baselines.** Even Logistic Regression achieves 87.4% accuracy and ROC AUC 0.927, confirming the features have good linear separability.

2. **Tree-based models dominate.** Both Random Forest (96.0%) and HistGradientBoosting (96.2%) achieve near-identical performance, well above logistic regression. This suggests meaningful non-linear interactions between features.

3. **HistGradientBoosting is the best overall** — marginal edge in F1 (0.955 vs 0.953) and AUC (0.9948 vs 0.9933), with faster training than Random Forest at this scale.

4. **Top features (Random Forest importances):**
   - `Online boarding`, `Inflight entertainment`, `Type of Travel_Personal Travel`, and `Flight Distance` are the most discriminating features — consistent with EDA findings.

5. **Confusion matrix:** Logistic Regression has more false negatives (missing satisfied passengers) than the tree models. Tree models are balanced in both error directions.

6. **No overfitting concern at first glance** — high test-set AUC without any regularisation tuning suggests the signal is strong.

---

## Modelling Implications

- HistGradientBoosting is recommended as the primary model for further development (Task 4 hyperparameter tuning).
- Random Forest provides a useful secondary model and gives clean feature importance estimates.
- Logistic Regression remains a useful interpretable baseline for comparison.
- Consider cross-validation (e.g. 5-fold) in Task 4 to confirm generalisation.

---

## Figures Produced

| Figure | File | Description |
|--------|------|-------------|
| Fig 8 | `fig_08_model_metric_comparison.png` | Grouped bar chart: Accuracy, F1, AUC by model |
| Fig 9 | `fig_09_roc_curves.png` | ROC curves for all 3 models |
| Fig 10 | `fig_10_confusion_matrices.png` | Confusion matrices (3 subplots) |
| Fig 11 | `fig_11_feature_importance_rf.png` | Top 20 features by Random Forest Gini importance |

All figures saved to `figures/Claude/`.
