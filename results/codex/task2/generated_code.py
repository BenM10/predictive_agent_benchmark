from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "codex_clean.csv"
FIGURES_DIR = PROJECT_ROOT / "figures" / "codex"


def detect_satisfaction_column(df: pd.DataFrame) -> Optional[str]:
    """Heuristically detect the satisfaction/target column."""
    candidates = {"satisfaction", "satisfied", "target", "label"}
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in lower_map:
            return lower_map[cand]
    return None


def save_and_show(fig: plt.Figure, idx: int, name: str) -> None:
    """
    Save a matplotlib figure to figures/codex with a numbered, succinct name
    and also display it inline.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = name.replace(" ", "_").lower()
    filename = f"fig_{idx:02d}_{safe_name}.png"
    output_path = FIGURES_DIR / filename
    fig.savefig(output_path, bbox_inches="tight")
    print(f"Saved figure {idx} to: {output_path.relative_to(PROJECT_ROOT)}")
    plt.show(fig)


def main() -> None:
    print(f"Loading dataset from: {DATASET_PATH.relative_to(PROJECT_ROOT)}")
    df = pd.read_csv(DATASET_PATH)

    print("\n=== Basic Structure ===")
    print("Shape (rows, columns):", df.shape)
    print("\nColumns:")
    print(df.columns.tolist())
    print("\nData types:")
    print(df.dtypes)

    print("\n=== Missing Values by Column ===")
    print(df.isna().sum())

    print("\n=== Summary Statistics (Numeric Columns) ===")
    print(df.describe())

    print("\n=== Preview (first 5 rows) ===")
    print(df.head())

    satisfaction_col = detect_satisfaction_column(df)
    if satisfaction_col is None:
        print(
            "\nWARNING: Could not automatically detect a satisfaction column. "
            "Downstream analyses that depend on satisfaction will be skipped."
        )

    # Identify basic column types
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [
        c for c in df.columns
        if c not in numeric_cols
    ]

    print("\n=== Column Type Summary ===")
    print("Numeric columns:", numeric_cols)
    print("Categorical columns:", categorical_cols)

    if satisfaction_col is not None:
        print(f"\nDetected satisfaction column: {satisfaction_col}")
        print("\nValue counts for satisfaction:")
        print(df[satisfaction_col].value_counts(dropna=False))

    # ---- Visualisations (max 7) ----
    fig_idx = 1
    sns.set_theme(style="whitegrid")

    # 1. Distribution of satisfaction
    if satisfaction_col is not None:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(data=df, x=satisfaction_col, ax=ax)
        ax.set_title("Distribution of Satisfaction")
        save_and_show(fig, fig_idx, "satisfaction_distribution")
        fig_idx += 1

    # 2. Age distribution
    if "Age" in df.columns and fig_idx <= 7:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(data=df, x="Age", kde=True, bins=30, ax=ax)
        ax.set_title("Age Distribution")
        save_and_show(fig, fig_idx, "age_distribution")
        fig_idx += 1

    # 3. Flight Distance distribution
    if "Flight Distance" in df.columns and fig_idx <= 7:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(data=df, x="Flight Distance", bins=40, ax=ax)
        ax.set_title("Flight Distance Distribution")
        save_and_show(fig, fig_idx, "flight_distance_distribution")
        fig_idx += 1

    # 4. Satisfaction by Customer Type
    if satisfaction_col is not None and "Customer Type" in df.columns and fig_idx <= 7:
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.countplot(
            data=df,
            x="Customer Type",
            hue=satisfaction_col,
            ax=ax,
        )
        ax.set_title("Satisfaction by Customer Type")
        ax.tick_params(axis="x", rotation=30)
        save_and_show(fig, fig_idx, "satisfaction_by_customer_type")
        fig_idx += 1

    # 5. Satisfaction by Class
    if satisfaction_col is not None and "Class" in df.columns and fig_idx <= 7:
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.countplot(
            data=df,
            x="Class",
            hue=satisfaction_col,
            ax=ax,
        )
        ax.set_title("Satisfaction by Travel Class")
        ax.tick_params(axis="x", rotation=30)
        save_and_show(fig, fig_idx, "satisfaction_by_class")
        fig_idx += 1

    # 6. Correlation heatmap for numeric features
    if numeric_cols and fig_idx <= 7:
        corr = df[numeric_cols].corr()
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax)
        ax.set_title("Correlation Heatmap (Numeric Features)")
        save_and_show(fig, fig_idx, "correlation_heatmap")
        fig_idx += 1

    # 7. Mean of a service rating by satisfaction (if available)
    service_candidates = [
        "Inflight wifi service",
        "Online boarding",
        "Seat comfort",
        "Inflight entertainment",
        "Cleanliness",
    ]
    if satisfaction_col is not None and fig_idx <= 7:
        service_col = next((c for c in service_candidates if c in df.columns), None)
        if service_col is not None:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.barplot(
                data=df,
                x=satisfaction_col,
                y=service_col,
                estimator="mean",
                errorbar=None,
                ax=ax,
            )
            ax.set_title(f"Average {service_col} Rating by Satisfaction")
            save_and_show(fig, fig_idx, "service_rating_by_satisfaction")
            fig_idx += 1

    # ---- Highlight interesting patterns ----
    if satisfaction_col is not None:
        print("\n=== Potentially Interesting Patterns ===")
        # Example: average selected service rating by satisfaction
        if "Flight Distance" in df.columns:
            grouped = df.groupby(satisfaction_col)["Flight Distance"].mean()
            print("\nAverage Flight Distance by Satisfaction:")
            print(grouped)

        if "Age" in df.columns:
            grouped_age = df.groupby(satisfaction_col)["Age"].mean()
            print("\nAverage Age by Satisfaction:")
            print(grouped_age)

        if "Class" in df.columns:
            class_dist = (
                df.groupby(["Class", satisfaction_col])
                .size()
                .unstack(fill_value=0)
            )
            print("\nClass distribution by Satisfaction:")
            print(class_dist)

    print("\nEDA complete.")


if __name__ == "__main__":
    main()

