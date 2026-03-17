import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "codex_clean.csv"
PIPELINE_PATH = PROJECT_ROOT / "task" / "task5_pipeline.py"
FIGURES_DIR = PROJECT_ROOT / "figures" / "codex"
TARGET_COLUMN = "satisfaction"
RANDOM_STATE = 42


def save_and_show(fig: plt.Figure, fig_id: int, name: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    output_path = FIGURES_DIR / f"fig_{fig_id:02d}_{name}.png"
    fig.savefig(output_path, bbox_inches="tight")
    print(f"Saved figure {fig_id} to: {output_path.relative_to(PROJECT_ROOT)}")
    display(fig)
    plt.close(fig)


def load_source() -> str:
    return PIPELINE_PATH.read_text()


def review_pipeline() -> pd.DataFrame:
    issues = [
        {
            "issue": "Preprocessing before the train/test split",
            "why_problematic": (
                "The scaler is fit on the full dataset before splitting, so information from the "
                "test set leaks into the training workflow."
            ),
            "correction": (
                "Fit preprocessing only on the training data by placing it inside an sklearn Pipeline "
                "or ColumnTransformer after the split."
            ),
        },
        {
            "issue": "Mixed data types are passed directly to StandardScaler",
            "why_problematic": (
                "The dataset contains categorical string columns such as Gender and Class, so the "
                "current code will fail instead of producing a valid model."
            ),
            "correction": (
                "Handle categorical and numeric features separately, for example with one-hot encoding "
                "for categoricals and scaling only numeric features."
            ),
        },
        {
            "issue": "Missing values are not handled",
            "why_problematic": (
                "Arrival Delay in Minutes contains missing values, and LogisticRegression cannot train "
                "on NaNs without imputation."
            ),
            "correction": (
                "Impute missing values inside the preprocessing pipeline using train-only statistics."
            ),
        },
        {
            "issue": "Identifier column is left in the feature set",
            "why_problematic": (
                "The id column is not a meaningful predictive feature and can inject row-specific noise "
                "or spurious signal."
            ),
            "correction": "Drop the id column before modelling.",
        },
        {
            "issue": "Train/test split is not stratified",
            "why_problematic": (
                "Without stratification, the class balance can drift between train and test splits, "
                "especially in classification tasks."
            ),
            "correction": "Use train_test_split(..., stratify=y, random_state=42).",
        },
        {
            "issue": "Evaluation uses accuracy only",
            "why_problematic": (
                "Accuracy alone can hide important trade-offs. For classification, F1 and ROC AUC "
                "provide a more reliable picture of model quality."
            ),
            "correction": "Report accuracy, F1, and ROC AUC on the held-out test set.",
        },
    ]
    issues_df = pd.DataFrame(issues)
    print("=== Reviewed Pipeline ===")
    print(PIPELINE_PATH.relative_to(PROJECT_ROOT))
    print()
    print("=== Identified Issues ===")
    print(issues_df.to_string(index=False))
    return issues_df


def load_data() -> pd.DataFrame:
    print(f"\nLoading dataset from: {DATASET_PATH.relative_to(PROJECT_ROOT)}")
    df = pd.read_csv(DATASET_PATH)
    print(f"Dataset shape: {df.shape}")
    return df


def build_corrected_pipeline(X: pd.DataFrame) -> Pipeline:
    categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", LogisticRegression(max_iter=5000, random_state=RANDOM_STATE)),
        ]
    )


def run_corrected_pipeline() -> tuple[pd.DataFrame, Pipeline, pd.DataFrame, pd.Series]:
    df = load_data()
    model_df = df.drop(columns=["id"], errors="ignore").copy()
    y = (model_df[TARGET_COLUMN] == "satisfied").astype(int)
    X = model_df.drop(columns=[TARGET_COLUMN])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    pipeline = build_corrected_pipeline(X)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics_df = pd.DataFrame(
        [
            {
                "model": "Corrected Logistic Regression",
                "accuracy": accuracy_score(y_test, y_pred),
                "f1": f1_score(y_test, y_pred),
                "roc_auc": roc_auc_score(y_test, y_proba),
            }
        ]
    )

    print("\n=== Corrected Pipeline Metrics ===")
    print(metrics_df.to_string(index=False))
    return metrics_df, pipeline, X_test, y_test


def plot_metrics(metrics_df: pd.DataFrame, fig_id: int) -> None:
    metric_names = ["accuracy", "f1", "roc_auc"]
    scores = metrics_df.iloc[0][metric_names].to_list()

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(metric_names, scores, color=["#4c78a8", "#f58518", "#54a24b"])
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Figure 1: Corrected pipeline evaluation metrics")
    save_and_show(fig, fig_id, "task5_corrected_pipeline_metrics")


def plot_confusion_matrix(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, fig_id: int) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_estimator(pipeline, X_test, y_test, ax=ax, cmap="Blues")
    ax.set_title("Figure 2: Confusion matrix for corrected pipeline")
    save_and_show(fig, fig_id, "task5_corrected_confusion_matrix")


def corrected_pipeline_code() -> str:
    return """df = pd.read_csv("data/processed/codex_clean.csv")
df = df.drop(columns=["id"])

X = df.drop(columns=["satisfaction"])
y = (df["satisfaction"] == "satisfied").astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()
numeric_features = X.select_dtypes(include=["number"]).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(handle_unknown="ignore")),
                ]
            ),
            categorical_features,
        ),
        (
            "numeric",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            ),
            numeric_features,
        ),
    ]
)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(max_iter=5000, random_state=42)),
    ]
)

pipeline.fit(X_train, y_train)"""


def main():
    issues_df = review_pipeline()
    metrics_df, pipeline, X_test, y_test = run_corrected_pipeline()
    plot_metrics(metrics_df, fig_id=1)
    plot_confusion_matrix(pipeline, X_test, y_test, fig_id=2)

    print("\n=== Corrected Pipeline Template ===")
    print(corrected_pipeline_code())
    return issues_df, metrics_df


if __name__ == "__main__":
    main()
