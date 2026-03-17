# Task 3 – Baseline model training (codex)

This experiment trains several simple baseline models to predict passenger satisfaction on `data/processed/codex_clean.csv`.

Key details:
- Target: `satisfaction` (positive class treated as `"satisfied"`)
- ID column `id` is dropped before modelling
- Categorical features are one‑hot encoded; numeric features are passed through unchanged
- Train/test split: 80% / 20%, `random_state=42`, stratified by target

Models trained:
- Logistic Regression (`class_weight="balanced"`)
- RandomForestClassifier (200 trees)
- HistGradientBoostingClassifier

Evaluation:
- Metrics computed on the held‑out test set: accuracy, F1 (positive class), ROC AUC
- Per‑model ROC curves and confusion matrix heatmaps are saved to `figures/codex/` via `save_and_show`
- A tabular summary of metrics is saved to `results/codex/task3/metrics.csv`

These models provide a reasonable starting point for further feature engineering, hyperparameter tuning and model comparison.

