# Full Prompt For Task 1  

Task 1 - Dataset Ingestion and Validation
You are a data scientist working in Python within a project repository.

The dataset is located at:
data/processed/airline_passenger_satisfaction_full.csv
Your task is to perform a basic dataset inspection to understand the structure and quality of the data.

Write Python code that performs the following steps:
1. Data ingestion
Load the dataset using pandas
Print the dataset shape

2. Schema inspection
Display column names
Display data types

3. Missing value check
Identify missing values for each column

4. Summary statistics
Produce basic summary statistics for numeric columns

5. Data preview
Display the first few rows of the dataset
The goal is to allow a data scientist to quickly understand the dataset structure and quality.

Execution Environment
The repository contains the following structure:
data/processed/          → contains the dataset
notebooks/task1/         → location where the experiment notebook should be saved
results/INSERTMODEL/task1/ → location where experiment outputs should be logged
Ensure all file paths are relative to the project root.

Experiment Logging
For reproducibility, ensure the following files are produced:
Notebook running the generated code
notebooks/task1/task1_INSERTMODEL.ipynb
Save the cleaned dataset as INSERTMODEL_clean.csv to data/processed/

Generated Python code
results/INSERTMODEL/task1/generated_code.py
Prompt used for the experiment
results/INSERTMODEL/task1/prompt.txt

Experiment notes
results/INSERTMODEL/task1/notes.md
The notebook should execute the generated code and display the outputs.
