## Task 2 - Exploratory Data Analysis (codex)

- **Objective**: Perform exploratory data analysis on `data/processed/codex_clean.csv` to understand structure, key variables, and patterns related to passenger satisfaction.
- **Approach**:
  - Loaded the dataset and reported shape, column names, data types, missing values, and summary statistics for numeric variables.
  - Heuristically detected a satisfaction/target column (if present) and examined its distribution.
  - Distinguished numeric vs categorical features to guide visualisation choices.
  - Generated up to seven figures saved under `figures/codex/` using a `save_and_show` helper, including satisfaction distribution, distributions of key numeric features, relationships between satisfaction and important categorical variables, and a correlation heatmap.
  - Reported example aggregated statistics such as average age and flight distance by satisfaction, and class distribution by satisfaction where possible.
- **Reproducibility**: Re-run `notebooks/task2/task2_codex.ipynb` from the project root to reproduce all console outputs and regenerate the figures in `figures/codex/`.

