# Task 1 Notes

- Implemented a pandas-based inspection script for ingestion, schema review, missing values, summary statistics, and data preview.
- Saved `data/processed/codex_clean.csv` as a reproducibility artifact. No transformations were applied because this task is limited to inspection and validation.
- During validation, the dataset loaded successfully with shape `(129880, 24)`.
- Missing values were only observed in `Arrival Delay in Minutes` with `393` null entries.
