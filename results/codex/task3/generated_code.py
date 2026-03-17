from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    RocCurveDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "codex_clean.csv"
FIGURES_DIR = PROJECT_ROOT / "figures" / "codex"
RESULTS_DIR = PROJECT_ROOT / "results" / "codex" / "task3"


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


def load_data() -> pd.DataFrame:
    print(f"Loading dataset from: {DATASET_PATH.relative_to(PROJECT_ROOT)}")
    df = pd.read_csv(DATASET_PATH)
    print("Initial shape:", df.shape)
    return df


def prepare_data(
    df: pd.DataFrame, target_col: str = "satisfaction"
) -> Tuple[np.ndarray, np.ndarray, ColumnTransformer]:
    # Drop ID column if present
    if "id" in df.columns:
        df = df.drop(columns=["id"])

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe.")

    y = df[target_col].copy()
    X = df.drop(columns=[target_col])

    # Identify column types
    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]

    print("Numeric features:", numeric_features)
    print("Categorical features:", categorical_features)

    # Define preprocessing
    numeric_transformer = "passthrough"
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor: ColumnTransformer = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    X_processed = preprocessor.fit_transform(X)
    return X_processed, y.to_numpy(), preprocessor


def train_models(
    X_train: np.ndarray, y_train: np.ndarray
) -> Dict[str, Pipeline]:
    models: Dict[str, Pipeline] = {}

    # Baseline logistic regression
    log_reg = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        n_jobs=-1,
    )
    models["logistic_regression"] = log_reg

    # Optional Random Forest
    rf = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    )
    models["random_forest"] = rf

    # Optional HistGradientBoosting
    hgb = HistGradientBoostingClassifier(
        learning_rate=0.1,
        max_depth=None,
        random_state=42,
    )
    models["hist_gradient_boosting"] = hgb

    fitted_models: Dict[str, Pipeline] = {}
    for name, clf in models.items():
        print(f"\nTraining model: {name}")
        clf.fit(X_train, y_train)
        fitted_models[name] = clf

    return fitted_models


def evaluate_models(
    models: Dict[str, Pipeline],
    X_test: np.ndarray,
    y_test: np.ndarray,
    positive_label: str = "satisfied",
) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    metrics_rows = []
    fig_idx = 1
    sns.set_theme(style="whitegrid")

    for name, model in models.items():
        print(f"\n=== Evaluation for {name} ===")
        y_pred = model.predict(X_test)

        # Probabilities for ROC AUC if supported
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            # Some classifiers expose decision_function instead
            scores = model.decision_function(X_test)
            # Scale scores to [0, 1] for ROC curve consistency
            y_proba = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
        else:
            y_proba = None

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, pos_label=positive_label)
        print(f"Accuracy: {acc:.3f}")
        print(f"F1 ({positive_label}): {f1:.3f}")

        if y_proba is not None:
            # Ensure we know the index of the positive label
            roc_auc = roc_auc_score(
                (y_test == positive_label).astype(int),
                y_proba,
            )
            print(f"ROC AUC: {roc_auc:.3f}")
        else:
            roc_auc = np.nan
            print("ROC AUC: N/A (model does not support probability scores)")

        metrics_rows.append(
            {
                "model": name,
                "accuracy": acc,
                "f1_positive": f1,
                "roc_auc": roc_auc,
            }
        )

        # ROC Curve
        if y_proba is not None:
            fig, ax = plt.subplots(figsize=(6, 4))
            RocCurveDisplay.from_predictions(
                (y_test == positive_label).astype(int),
                y_proba,
                name=name,
                ax=ax,
            )
            ax.set_title(f"ROC Curve - {name}")
            save_and_show(fig, fig_idx, f"roc_curve_{name}")
            fig_idx += 1

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred, labels=[positive_label, "neutral or dissatisfied"])
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=[positive_label, "neutral or dissatisfied"],
        )
        fig, ax = plt.subplots(figsize=(5, 4))
        disp.plot(ax=ax, cmap="Blues", colorbar=False)
        ax.set_title(f"Confusion Matrix - {name}")
        save_and_show(fig, fig_idx, f"confusion_matrix_{name}")
        fig_idx += 1

    metrics_df = pd.DataFrame(metrics_rows)
    print("\n=== Summary metrics on test set ===")
    print(metrics_df)

    metrics_path = RESULTS_DIR / "metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\nSaved metrics to: {metrics_path.relative_to(PROJECT_ROOT)}")


def main() -> None:
    df = load_data()
    X, y, preprocessor = prepare_data(df, target_col="satisfaction")

    print("\nSplitting data into train and test sets (80/20, random_state=42)")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    print("Train shape:", X_train.shape)
    print("Test shape:", X_test.shape)

    models = train_models(X_train, y_train)
    evaluate_models(models, X_test, y_test, positive_label="satisfied")


if __name__ == "__main__":
    main()

