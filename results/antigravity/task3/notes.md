# Experiment Outcome - Task 3: Baseline Model Training

## Summary
The goal was to establish a baseline performance for predicting passenger satisfaction. We trained three models: Logistic Regression (as the primary baseline), Random Forest, and HistGradientBoosting.

## Model Performance (on test set)

| Model                 | Accuracy | F1 Score | ROC AUC |
|-----------------------|----------|----------|---------|
| Logistic Regression   | 0.8761   | 0.8554   | 0.9257  |
| Random Forest         | 0.9625   | 0.9562   | 0.9937  |
| HistGradientBoosting  | 0.9633   | 0.9572   | 0.9948  |

## Key Findings
- **Logistic Regression** provides a solid baseline with ~87.6% accuracy.
- **HistGradientBoosting** achieved the highest performance (accuracy ~96.3%) and is highly efficient for this dataset.
- The high ROC AUC (>0.99) for tree-based models suggests that the features provided are highly predictive of satisfaction.
- Missing values in `Arrival Delay in Minutes` were imputed with the median.

## Artifacts Produced
- **Generated Code**: `results/antigravity/task3/generated_code.py`
- **Notebook**: `notebooks/task3/task3_antigravity.ipynb`
- **Figures**:
    - `01_model_performance_comparison.png`
    - `02_roc_curves.png`
- **Prompt**: `results/antigravity/task3/prompt.txt`
