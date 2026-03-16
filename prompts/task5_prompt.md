# Full Prompt for Task 5  

Task 5 — Detecting Data Leakage and Evaluation Issues

You are again a data scientist reviewing a machine learning pipeline.

The dataset is located at:
data/processed/antigravity_clean.csv

A modelling pipeline has been written in the file:

task/task5_pipeline.py

Your task is to review this pipeline and determine whether it contains any methodological problems or risks of data leakage.

Write Python code or explanations that:

Identify any issues in the pipeline
Explain why these issues are problematic
Provide a corrected version of the pipeline that follows good machine learning practice

Focus particularly on:

- data leakage
- preprocessing steps
- evaluation methodology

The goal is to ensure that the modelling workflow follows sound statistical practice.
Execution Environment
The repository contains the following structure:
- `data/processed/` → contains the dataset
- `notebooks/task5/` → location where the experiment notebook should be saved
- `results/antigravity/task5/` → location where experiment outputs should be logged
Experiment Logging
For reproducibility, ensure the following files are produced:
- Any figures produced should be shown with a ‘save_and_show’ function which saves the figures to `figures/antigravity/` with a succinct and clear name which numbers the figures in order.
- A MODULAR notebook containing the generated code in logical steps, with short markdown blocks explaining each step:
 `notebooks/task5/task5_antigravity.ipynb`
- A copy of the generated Python code:
 `results/antigravity/task5/generated_code.py`
- A text file containing the prompt used:
 `results/antigravity/task5/prompt.txt`
- A short notes file describing the experiment outcome:
 `results/antigravity/task5/notes.md`
The notebook should execute the generated code and display the outputs.
Ensure that all file paths are relative to the project root.
