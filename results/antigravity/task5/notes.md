# Task 5 Notes

## Issues Identified in the Original Pipeline
1. **Data Leakage:** `StandardScaler` was fit and transformed on the entire dataset `X` prior to splitting into train/test sets, leaking test set statistics into the training pipeline.
2. **Preprocessing Flaws:**
   - Attempted to apply `StandardScaler` to categorical string variables which breaks the execution. No categorical encoder like `OneHotEncoder` was provided.
   - Included the `id` column as a feature, which has zero predictive meaning and adds noise to the model learning.
   - Loaded an incorrect raw CSV file (`airline_passenger_satisfaction_full.csv`) rather than the target cleaned dataset.
3. **Evaluation Methodology:**
   - Evaluated the model using only the `accuracy_score` metric, which is insufficient to measure the real classification performance on potentially imbalanced data. Essential metrics like Precision, Recall, F1-Score, and ROC-AUC were omitted.

## Corrected Pipeline
- Corrected the dataset loading step to read `antigravity_clean.csv`.
- Dropped the `id` column.
- Handled categorical representations by segregating variables into `categorical_cols` and `numerical_cols`. Passed these through `ColumnTransformer` with `OneHotEncoder` and `StandardScaler` concurrently.
- Used `Pipeline` to ensure all fitting operations occur rigorously.
- Executed `train_test_split` securely *before* handing the split datasets to the pipeline logic.
- Generated comprehensive metrics through a comprehensive classification report and a ROC AUC score/curve plot. Produced code module efficiently logs to `figures/antigravity/` and execution context.
