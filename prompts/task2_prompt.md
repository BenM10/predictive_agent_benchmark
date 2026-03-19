# Full Prompt For Task 2  

Task 2 — Exploratory Data Analysis

You are still a data scientist working in Python within a project repository.

The dataset is located at:

data/processed/INSERTMODEL_clean.csv

Your task is to perform exploratory data analysis (EDA) to better understand the dataset and identify patterns related to passenger satisfaction.

Write Python code that:

1. Loads the dataset
2. Explores the structure of the dataset
3. Identifies and reports important variables and distributions
4. Generates a maximum of 7 visualisations that help understand relationships in the data
5. Highlights any interesting patterns or potential predictive features related to passenger satisfaction

Use common Python data science libraries where appropriate (e.g. pandas, matplotlib, seaborn).

The goal is to produce an initial exploratory analysis that would help guide later modelling decisions.

Execution Environment

The repository contains the following structure:

- `data/processed/` → contains the dataset
- `notebooks/task2/` → location where the experiment notebook should be saved
- `results/INSERTMODEL/task2/` → location where experiment outputs should be logged

Experiment Logging

For reproducibility, ensure the following files are produced:

Any figures produced should be shown with a ‘save_and_show’ function which saves the figures to `figures/INSERTMODEL/` with a succinct and clear name which numbers the figures in order.

- A notebook running the generated code:
  `notebooks/task2/task2_INSERTMODEL.ipynb`

- A copy of the generated Python code:
  `results/INSERTMODEL/task2/generated_code.py`

- A text file containing the prompt used:
  `results/INSERTMODEL/task2/prompt.txt`

- A short notes file describing the experiment outcome:
  `results/INSERTMODEL/task2/notes.md`

The notebook should execute the generated code and display the outputs.

Ensure that all file paths are relative to the project root.
