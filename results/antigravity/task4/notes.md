# Experiment Outcome - Task 4: Improving Model Performance

## Summary
The objective of this task was to improve upon the baseline model performance for the Airline Passenger Satisfaction dataset. We achieved this through a combination of feature engineering and hyperparameter tuning of a `HistGradientBoostingClassifier`, using the initial baseline of `LogisticRegression` for direct comparison.

## Changes Made
1. **Feature Engineering**: 
   - Created a `Total Delay` feature by summing `Departure Delay in Minutes` and `Arrival Delay in Minutes`.
   - Created an `Average Service Score` feature by calculating the mean of all 14 satisfaction-related service ratings (e.g., Inflight wifi, Food and drink, Seat comfort).
2. **Missing Value Imputation**: Handled natively for categories via mode imputation and median for numerical columns prior to advanced algorithms.
3. **Hyperparameter Tuning**: Used `RandomizedSearchCV` (10 iterations, 3-fold CV) to tune the `HistGradientBoostingClassifier` focusing on `learning_rate`, `max_leaf_nodes`, `min_samples_leaf`, `max_iter`, and `l2_regularization`.

## Model Performance Comparison (on 20% test set)

| Model                               | Accuracy | F1 Score | ROC AUC |
|-------------------------------------|----------|----------|---------|
| Baseline (Logistic Regression)      | 0.8762   | 0.8555   | 0.9257  |
| **Improved (Tuned HGB)**            | **0.9644** | **0.9584** | **0.9951** |

## Conclusion
The improved modeling pipeline successfully outperformed the baseline across all classification metrics. The engineered features and tuned hyperparameter settings (best: `min_samples_leaf=100`, `max_leaf_nodes=63`, `max_iter=100`, `learning_rate=0.1`, `l2_regularization=1.0`) allowed the algorithm to better capture the complex non-linear relationships in the data.

## Artifacts Produced
- **Generated Code**: `results/antigravity/task4/generated_code.py`
- **Notebook**: `notebooks/task4/task4_antigravity.ipynb`
- **Output Log**: `results/antigravity/task4/output_log.txt`
- **Prompt**: `results/antigravity/task4/prompt.txt`
- **Figures**:
    - `01_model_comparison_bar.png`
    - `02_roc_comparison.png`
