from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "airline_passenger_satisfaction_full.csv"
CLEAN_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "codex_clean.csv"


def main() -> None:
    df = pd.read_csv(DATASET_PATH)

    print("Dataset shape:")
    print(df.shape)
    print()

    print("Column names:")
    print(df.columns.tolist())
    print()

    print("Data types:")
    print(df.dtypes)
    print()

    print("Missing values by column:")
    print(df.isna().sum())
    print()

    print("Summary statistics for numeric columns:")
    print(df.describe())
    print()

    print("First five rows:")
    print(df.head())
    print()

    # Task 1 is inspection-focused, so the cleaned export preserves the
    # ingested dataset while standardizing the benchmark output artifact.
    df.to_csv(CLEAN_OUTPUT_PATH, index=False)
    print(f"Saved cleaned dataset to: {CLEAN_OUTPUT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
