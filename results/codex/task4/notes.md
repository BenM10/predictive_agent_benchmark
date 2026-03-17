# Task 4 Notes

- Built a stronger HistGradientBoosting pipeline by adding engineered delay, service, and long-haul features.
- Tuned the model with `RandomizedSearchCV` on the training set only, preserving the held-out test split for final evaluation.
- Compared the tuned model directly against an untuned baseline HistGradientBoosting pipeline using accuracy, F1, and ROC AUC.
- Final test-set performance improved from `Accuracy=0.9633`, `F1=0.9570`, `ROC AUC=0.9949` to `Accuracy=0.9642`, `F1=0.9581`, `ROC AUC=0.9952`.
- Saved comparison and diagnostic figures to `figures/codex/`.
- The notebook is modular and imports the Task 4 pipeline from `results/codex/task4/generated_code.py`.
