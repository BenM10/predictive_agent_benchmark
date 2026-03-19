# Full Prompt For Task 4  

Task 4 - Improving Model performance
You are still a data scientist working in Python within a project repository.
The dataset is located at:
data/processed/INSERTMODEL_clean.csv
Task: Carrying on from the previous work, improve the predictive performance of the existing baseline model for the Airline Passenger Satisfaction dataset.
Write Python code that:
- Loads the dataset
- Prepares the data for modelling
- Trains an improved classification model or improves the existing modelling pipeline
- Evaluates the improved model using appropriate classification metrics
- Compares the improved model performance with the baseline approach
Your Objective:
Improve model performance using any combination of:
- Feature engineering
Create new features from existing variables
Encode categorical variables appropriately
Scale or transform numerical variables if useful
- Hyperparameter tuning
Use techniques such as grid search or random search
Tune model parameters systematically
- Model improvement
Try alternative algorithms (e.g., Random Forest, XGBoost, Logistic Regression, Gradient Boosting, etc.)
Requirements
- Maintain proper evaluation discipline
Use a train/test split
- Provide reproducible code
Fix random seeds = 42
Ensure the pipeline runs end-to-end
- Track performance
Report metrics such as:
Accuracy
F1 score
ROC-AUC
- Compare results
Show baseline vs improved model performance
Use common Python machine learning libraries such as pandas and scikit-learn.
Include visualisations where helpful to illustrate model performance or feature importance.
The goal is to produce a model that performs better than the baselines and demonstrate a clear modelling improvement.
Expected Output:
The output should include:
- Improved pipeline code
- Explanation of changes made
- Performance comparison table
-Final model performance on the test set  
Execution Environment  
The repository contains the following structure:  
- `data/processed/` → contains the dataset
- `notebooks/task4/` → location where the experiment notebook should be saved
- `results/INSERTMODEL/task4/` → location where experiment outputs should be logged
Experiment Logging
For reproducibility, ensure the following files are produced:
- Any figures produced should be shown with a ‘save_and_show’ function which saves the figures to `figures/INSERTMODEL/` with a succinct and clear name which numbers the figures in order.
- A MODULAR notebook running the generated code:
 `notebooks/task4/task4_INSERTMODEL.ipynb`
- A copy of the generated Python code:
 `results/INSERTMODEL/task4/generated_code.py`
- A text file containing the prompt used:
 `results/INSERTMODEL/task4/prompt.txt`
- A short notes file describing the experiment outcome:
 `results/INSERTMODEL/task4/notes.md`
The notebook should execute the generated code and display the outputs.
Ensure that all file paths are relative to the project root.
