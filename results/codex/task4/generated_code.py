import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "codex_clean.csv"
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


class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        if {
            "Departure Delay in Minutes",
            "Arrival Delay in Minutes",
        }.issubset(X.columns):
            dep = X["Departure Delay in Minutes"].fillna(0)
            arr = X["Arrival Delay in Minutes"].fillna(0)
            X["Total Delay"] = dep + arr
            X["Delay Difference"] = arr - dep
            X["Any Significant Delay"] = ((dep > 15) | (arr > 15)).astype(int)

        if "Flight Distance" in X.columns:
            X["Long Haul Flight"] = (X["Flight Distance"] >= 1500).astype(int)

        service_cols = [
            "Inflight wifi service",
            "Ease of Online booking",
            "Online boarding",
            "Seat comfort",
            "Inflight entertainment",
            "On-board service",
            "Leg room service",
            "Baggage handling",
            "Checkin service",
            "Inflight service",
            "Cleanliness",
            "Food and drink",
        ]
        available_service_cols = [col for col in service_cols if col in X.columns]
        if available_service_cols:
            X["Overall Service Mean"] = X[available_service_cols].mean(axis=1)

        digital_cols = [
            "Inflight wifi service",
            "Ease of Online booking",
            "Online boarding",
        ]
        available_digital_cols = [col for col in digital_cols if col in X.columns]
        if available_digital_cols:
            X["Digital Experience Mean"] = X[available_digital_cols].mean(axis=1)

        comfort_cols = [
            "Seat comfort",
            "Leg room service",
            "Inflight entertainment",
            "Cleanliness",
        ]
        available_comfort_cols = [col for col in comfort_cols if col in X.columns]
        if available_comfort_cols:
            X["Comfort Mean"] = X[available_comfort_cols].mean(axis=1)

        return X


def load_data() -> pd.DataFrame:
    print(f"Loading dataset from: {DATASET_PATH.relative_to(PROJECT_ROOT)}")
    df = pd.read_csv(DATASET_PATH)
    print(f"Dataset shape: {df.shape}")
    return df


def prepare_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    model_df = df.drop(columns=["id"], errors="ignore").copy()
    y = (model_df[TARGET_COLUMN] == "satisfied").astype(int)
    X = model_df.drop(columns=[TARGET_COLUMN])
    return X, y


def split_data(X: pd.DataFrame, y: pd.Series):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    print(f"Training rows: {X_train.shape[0]}, Test rows: {X_test.shape[0]}")
    return X_train, X_test, y_train, y_test


def build_preprocessor(feature_names: list[str]) -> ColumnTransformer:
    categorical_features = [
        col for col in feature_names if col in {"Gender", "Customer Type", "Type of Travel", "Class"}
    ]
    numeric_features = [col for col in feature_names if col not in categorical_features]

    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OrdinalEncoder(
                                handle_unknown="use_encoded_value",
                                unknown_value=-1,
                            ),
                        ),
                    ]
                ),
                categorical_features,
            ),
            (
                "numeric",
                Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]),
                numeric_features,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_baseline_pipeline(X_train: pd.DataFrame) -> Pipeline:
    preprocessor = build_preprocessor(X_train.columns.tolist())
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", HistGradientBoostingClassifier(random_state=RANDOM_STATE)),
        ]
    )


def build_improved_pipeline(X_train: pd.DataFrame) -> Pipeline:
    engineered_columns = FeatureEngineer().fit_transform(X_train).columns.tolist()
    preprocessor = build_preprocessor(engineered_columns)
    return Pipeline(
        steps=[
            ("feature_engineering", FeatureEngineer()),
            ("preprocessor", preprocessor),
            ("model", HistGradientBoostingClassifier(random_state=RANDOM_STATE)),
        ]
    )


def evaluate_model(name: str, pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    return {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }


def fit_models(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series):
    baseline_pipeline = build_baseline_pipeline(X_train)
    baseline_pipeline.fit(X_train, y_train)
    baseline_metrics = evaluate_model("Baseline HistGradientBoosting", baseline_pipeline, X_test, y_test)

    improved_pipeline = build_improved_pipeline(X_train)
    param_distributions = {
        "model__learning_rate": [0.03, 0.05, 0.08, 0.1],
        "model__max_leaf_nodes": [15, 31, 63, 127],
        "model__max_depth": [None, 6, 10],
        "model__min_samples_leaf": [20, 40, 80],
        "model__l2_regularization": [0.0, 0.01, 0.1, 1.0],
        "model__max_bins": [127, 191, 255],
    }

    search = RandomizedSearchCV(
        estimator=improved_pipeline,
        param_distributions=param_distributions,
        n_iter=12,
        scoring="roc_auc",
        cv=3,
        random_state=RANDOM_STATE,
        n_jobs=1,
        refit=True,
        verbose=0,
    )
    search.fit(X_train, y_train)
    improved_metrics = evaluate_model("Improved Tuned HistGradientBoosting", search.best_estimator_, X_test, y_test)

    comparison_df = pd.DataFrame([baseline_metrics, improved_metrics])
    comparison_df["accuracy_delta_vs_baseline"] = comparison_df["accuracy"] - baseline_metrics["accuracy"]
    comparison_df["f1_delta_vs_baseline"] = comparison_df["f1"] - baseline_metrics["f1"]
    comparison_df["roc_auc_delta_vs_baseline"] = comparison_df["roc_auc"] - baseline_metrics["roc_auc"]

    print("\n=== Best Hyperparameters ===")
    print(search.best_params_)
    print("\n=== Baseline vs Improved Performance ===")
    print(comparison_df.to_string(index=False))

    return baseline_pipeline, search.best_estimator_, comparison_df, search.best_params_


def plot_metric_comparison(comparison_df: pd.DataFrame, fig_id: int) -> None:
    metrics = ["accuracy", "f1", "roc_auc"]
    x = np.arange(len(metrics))
    width = 0.35

    baseline_scores = comparison_df.iloc[0][metrics].to_numpy(dtype=float)
    improved_scores = comparison_df.iloc[1][metrics].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, baseline_scores, width, label=comparison_df.iloc[0]["model"])
    ax.bar(x + width / 2, improved_scores, width, label=comparison_df.iloc[1]["model"])
    ax.set_xticks(x)
    ax.set_xticklabels([metric.upper() for metric in metrics])
    ax.set_ylim(0.85, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Figure 1: Baseline vs improved model performance")
    ax.legend()
    save_and_show(fig, fig_id, "baseline_vs_improved_metrics")


def plot_roc_comparison(
    baseline_pipeline: Pipeline,
    improved_pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    fig_id: int,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, pipeline in [
        ("Baseline HistGradientBoosting", baseline_pipeline),
        ("Improved Tuned HistGradientBoosting", improved_pipeline),
    ]:
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc_score = roc_auc_score(y_test, y_proba)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc_score:.4f})")

    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Figure 2: ROC comparison on the held-out test set")
    ax.legend(loc="lower right")
    save_and_show(fig, fig_id, "roc_comparison")


def plot_permutation_importance(
    improved_pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    fig_id: int,
) -> None:
    result = permutation_importance(
        improved_pipeline,
        X_test,
        y_test,
        n_repeats=5,
        random_state=RANDOM_STATE,
        scoring="roc_auc",
        n_jobs=1,
    )
    importances = pd.Series(result.importances_mean, index=X_test.columns).sort_values(ascending=False).head(10)

    fig, ax = plt.subplots(figsize=(8, 5))
    importances.sort_values().plot(kind="barh", ax=ax, color="#2c7fb8")
    ax.set_title("Figure 3: Top permutation importances for the improved model")
    ax.set_xlabel("Mean ROC AUC decrease")
    save_and_show(fig, fig_id, "improved_model_permutation_importance")


def summarize_results(comparison_df: pd.DataFrame, best_params: dict) -> None:
    baseline_row = comparison_df.iloc[0]
    improved_row = comparison_df.iloc[1]
    print("\n=== Changes Made ===")
    print("- Added engineered delay, service-quality, digital-experience, and long-haul features.")
    print("- Tuned HistGradientBoosting hyperparameters with RandomizedSearchCV on the training split only.")
    print("- Preserved a held-out test set for final comparison against the untuned baseline.")

    print("\n=== Final Model Performance On Test Set ===")
    print(
        f"Accuracy={improved_row['accuracy']:.4f}, "
        f"F1={improved_row['f1']:.4f}, "
        f"ROC AUC={improved_row['roc_auc']:.4f}"
    )

    print("\n=== Improvement Over Baseline ===")
    print(
        f"Accuracy delta={improved_row['accuracy_delta_vs_baseline']:.4f}, "
        f"F1 delta={improved_row['f1_delta_vs_baseline']:.4f}, "
        f"ROC AUC delta={improved_row['roc_auc_delta_vs_baseline']:.4f}"
    )

    print("\n=== Selected Hyperparameters ===")
    print(best_params)


def main():
    df = load_data()
    X, y = prepare_target(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    baseline_pipeline, improved_pipeline, comparison_df, best_params = fit_models(
        X_train, y_train, X_test, y_test
    )
    plot_metric_comparison(comparison_df, fig_id=1)
    plot_roc_comparison(baseline_pipeline, improved_pipeline, X_test, y_test, fig_id=2)
    plot_permutation_importance(improved_pipeline, X_test, y_test, fig_id=3)
    summarize_results(comparison_df, best_params)
    return comparison_df, best_params


if __name__ == "__main__":
    main()
