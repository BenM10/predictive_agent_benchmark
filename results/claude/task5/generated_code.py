# Task 5: Detecting Data Leakage and Evaluation Issues
# Corrected ML pipeline with proper preprocessing and evaluation

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve, ConfusionMatrixDisplay
)
from sklearn.pipeline import Pipeline

# ── Figure counter ──────────────────────────────────────────────────────────
FIG_DIR = "figures/Claude"
os.makedirs(FIG_DIR, exist_ok=True)

_fig_counter = [17]  # previous tasks used 1-16

def save_and_show(fig, name):
    """Save figure to figures/Claude/ with sequential numbering."""
    n = _fig_counter[0]
    path = os.path.join(FIG_DIR, f"fig_{n:02d}_{name}.png")
    fig.savefig(path, bbox_inches="tight", dpi=150)
    _fig_counter[0] += 1
    plt.show()
    print(f"Saved: {path}")

# ════════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ════════════════════════════════════════════════════════════════════════════
df = pd.read_csv("data/processed/Claude_clean.csv")
print("Dataset shape:", df.shape)
print("\nColumn dtypes:\n", df.dtypes)
print("\nMissing values:\n", df.isnull().sum())
print("\nTarget distribution:\n", df["satisfaction"].value_counts())

# ════════════════════════════════════════════════════════════════════════════
# 2. IDENTIFY ISSUES IN ORIGINAL PIPELINE
# ════════════════════════════════════════════════════════════════════════════
issues = {
    "Issue 1 – Wrong data file": (
        "The pipeline loads 'airline_passenger_satisfaction_full.csv' instead of "
        "'Claude_clean.csv'. This would cause a FileNotFoundError at runtime."
    ),
    "Issue 2 – Data Leakage (Critical)": (
        "StandardScaler is fit_transform()ed on the FULL dataset before the "
        "train/test split. This means the scaler learns mean/std from the test set, "
        "which is information the model should never see during training. This "
        "artificially inflates test performance and gives an overly optimistic "
        "estimate of real-world accuracy."
    ),
    "Issue 3 – Categorical Variables Not Encoded": (
        "Columns Gender, Customer Type, Type of Travel, and Class are object dtype. "
        "StandardScaler cannot process strings; this would raise a ValueError. "
        "These must be encoded (e.g., via one-hot or ordinal encoding) before scaling."
    ),
    "Issue 4 – ID Column Included as Feature": (
        "The 'id' column is a row identifier with no predictive value. Including it "
        "as a feature adds noise and may cause the model to spuriously memorise "
        "row indices."
    ),
    "Issue 5 – No Cross-Validation": (
        "A single train/test split produces a point estimate of performance that "
        "is sensitive to the random seed. Stratified k-fold cross-validation gives "
        "a more reliable estimate of generalisation performance."
    ),
    "Issue 6 – Only Accuracy Reported": (
        "Accuracy alone is insufficient for imbalanced or real-world classification. "
        "Precision, recall, F1, and ROC-AUC should also be reported."
    ),
}

print("\n" + "="*70)
print("IDENTIFIED ISSUES IN ORIGINAL PIPELINE")
print("="*70)
for title, desc in issues.items():
    print(f"\n{title}:\n  {desc}")

# ════════════════════════════════════════════════════════════════════════════
# 3. DEMONSTRATE DATA LEAKAGE EFFECT
# ════════════════════════════════════════════════════════════════════════════
# We prepare a simple numeric-only subset to show the leakage effect numerically.

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols = [c for c in numeric_cols if c not in ["id"]]

X_num = df[numeric_cols]
y = (df["satisfaction"] == "satisfied").astype(int)

# ── LEAKY pipeline (reproduces the original bug) ─────────────────────────
scaler_leaky = StandardScaler()
X_leaky = scaler_leaky.fit_transform(X_num)          # fit on ALL data
X_train_leaky, X_test_leaky, y_train, y_test = train_test_split(
    X_leaky, y, test_size=0.25, random_state=42
)
model_leaky = LogisticRegression(max_iter=1000, random_state=42)
model_leaky.fit(X_train_leaky, y_train)
acc_leaky = accuracy_score(y_test, model_leaky.predict(X_test_leaky))
auc_leaky = roc_auc_score(y_test, model_leaky.predict_proba(X_test_leaky)[:, 1])

# ── CORRECT pipeline (split first, then fit scaler only on train) ─────────
X_train_raw, X_test_raw, y_train_c, y_test_c = train_test_split(
    X_num, y, test_size=0.25, random_state=42
)
scaler_correct = StandardScaler()
X_train_c = scaler_correct.fit_transform(X_train_raw)   # fit ONLY on train
X_test_c  = scaler_correct.transform(X_test_raw)        # transform test using train stats
model_correct = LogisticRegression(max_iter=1000, random_state=42)
model_correct.fit(X_train_c, y_train_c)
acc_correct = accuracy_score(y_test_c, model_correct.predict(X_test_c))
auc_correct = roc_auc_score(y_test_c, model_correct.predict_proba(X_test_c)[:, 1])

print("\n" + "="*70)
print("LEAKAGE DEMONSTRATION (numeric features only)")
print("="*70)
print(f"  Leaky pipeline   — Accuracy: {acc_leaky:.4f} | ROC-AUC: {auc_leaky:.4f}")
print(f"  Correct pipeline — Accuracy: {acc_correct:.4f} | ROC-AUC: {auc_correct:.4f}")
print(f"  Accuracy difference: {acc_leaky - acc_correct:+.4f}")

# Visualise the comparison
fig, ax = plt.subplots(figsize=(7, 4))
metrics = ["Accuracy", "ROC-AUC"]
leaky_vals   = [acc_leaky,   auc_leaky]
correct_vals = [acc_correct, auc_correct]
x = np.arange(len(metrics))
width = 0.35
ax.bar(x - width/2, leaky_vals,   width, label="Leaky (original)", color="#e74c3c", alpha=0.85)
ax.bar(x + width/2, correct_vals, width, label="Correct",           color="#2ecc71", alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.set_ylim(0.5, 1.0)
ax.set_ylabel("Score")
ax.set_title("Effect of Data Leakage on Reported Performance\n(numeric features only)")
ax.legend()
ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.3f"))
plt.tight_layout()
save_and_show(fig, "leakage_comparison")

# ════════════════════════════════════════════════════════════════════════════
# 4. CORRECTED FULL PIPELINE
# ════════════════════════════════════════════════════════════════════════════
# Drop id; encode categoricals; split FIRST; fit scaler only on train.

# 4a. Feature engineering
df_model = df.drop(columns=["id", "satisfaction"])
y_full = (df["satisfaction"] == "satisfied").astype(int)

# One-hot encode categorical columns
cat_cols = df_model.select_dtypes(include="object").columns.tolist()
df_encoded = pd.get_dummies(df_model, columns=cat_cols, drop_first=True)
print(f"\nFeatures after encoding: {df_encoded.shape[1]}")

# 4b. Train/test split BEFORE any fitting
X_tr, X_te, y_tr, y_te = train_test_split(
    df_encoded, y_full, test_size=0.25, random_state=42, stratify=y_full
)

# 4c. Scale ONLY on training data; apply same transform to test
scaler_full = StandardScaler()
X_tr_sc = scaler_full.fit_transform(X_tr)
X_te_sc = scaler_full.transform(X_te)

# 4d. Train model
model_full = LogisticRegression(max_iter=1000, random_state=42)
model_full.fit(X_tr_sc, y_tr)

# 4e. Evaluate
y_pred = model_full.predict(X_te_sc)
y_prob = model_full.predict_proba(X_te_sc)[:, 1]

print("\n" + "="*70)
print("CORRECTED PIPELINE – EVALUATION RESULTS")
print("="*70)
print(f"  Accuracy : {accuracy_score(y_te, y_pred):.4f}")
print(f"  ROC-AUC  : {roc_auc_score(y_te, y_prob):.4f}")
print("\nClassification Report:")
print(classification_report(y_te, y_pred, target_names=["neutral/dissatisfied", "satisfied"]))

# ════════════════════════════════════════════════════════════════════════════
# 5. STRATIFIED CROSS-VALIDATION
# ════════════════════════════════════════════════════════════════════════════
# Use a sklearn Pipeline to prevent leakage across folds automatically.
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf",    LogisticRegression(max_iter=1000, random_state=42))
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(pipe, df_encoded, y_full, cv=cv, scoring="roc_auc", n_jobs=-1)
print("\n" + "="*70)
print("5-FOLD STRATIFIED CROSS-VALIDATION (sklearn Pipeline)")
print("="*70)
print(f"  ROC-AUC per fold: {cv_scores.round(4)}")
print(f"  Mean ± Std:       {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# Visualise CV scores
fig2, ax2 = plt.subplots(figsize=(7, 4))
ax2.bar(range(1, 6), cv_scores, color="#3498db", alpha=0.85)
ax2.axhline(cv_scores.mean(), color="red", linestyle="--", label=f"Mean = {cv_scores.mean():.4f}")
ax2.set_xticks(range(1, 6))
ax2.set_xticklabels([f"Fold {i}" for i in range(1, 6)])
ax2.set_ylabel("ROC-AUC")
ax2.set_ylim(0.7, 1.0)
ax2.set_title("5-Fold Stratified Cross-Validation ROC-AUC\n(Corrected Pipeline with sklearn Pipeline)")
ax2.legend()
plt.tight_layout()
save_and_show(fig2, "cv_scores_corrected")

# ════════════════════════════════════════════════════════════════════════════
# 6. CONFUSION MATRIX & ROC CURVE
# ════════════════════════════════════════════════════════════════════════════
fig3, axes = plt.subplots(1, 2, figsize=(13, 5))

# Confusion matrix
cm = confusion_matrix(y_te, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["neutral/dissatisfied", "satisfied"])
disp.plot(ax=axes[0], colorbar=False, cmap="Blues")
axes[0].set_title("Confusion Matrix – Corrected Pipeline")

# ROC curve
fpr, tpr, _ = roc_curve(y_te, y_prob)
auc_val = roc_auc_score(y_te, y_prob)
axes[1].plot(fpr, tpr, color="#2c3e50", lw=2, label=f"ROC-AUC = {auc_val:.4f}")
axes[1].plot([0, 1], [0, 1], "k--", lw=1)
axes[1].set_xlabel("False Positive Rate")
axes[1].set_ylabel("True Positive Rate")
axes[1].set_title("ROC Curve – Corrected Pipeline")
axes[1].legend(loc="lower right")

plt.tight_layout()
save_and_show(fig3, "confusion_matrix_and_roc")

print("\nAll outputs saved successfully.")
