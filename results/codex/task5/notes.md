# Task 5 Notes

- Reviewed `task/task5_pipeline.py` for leakage and evaluation problems.
- Identified leakage from fitting `StandardScaler` before the train/test split, along with missing-value handling, categorical preprocessing, retained `id`, lack of stratification, and limited evaluation metrics.
- Implemented a corrected logistic-regression pipeline with train-only preprocessing inside an sklearn `Pipeline`.
- Evaluated the corrected pipeline on a held-out stratified test set with accuracy, F1, and ROC AUC.
- Saved Task 5 figures to `figures/codex/`.
