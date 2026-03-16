import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, RocCurveDisplay
import matplotlib.pyplot as plt
import os

# Create figures directory if it doesn't exist
os.makedirs("figures/antigravity", exist_ok=True)

figure_counter = 1
def save_and_show(fig, filename_base):
    global figure_counter
    filepath = f"figures/antigravity/{figure_counter:02d}_{filename_base}.png"
    fig.savefig(filepath, bbox_inches='tight', dpi=300)
    print(f"Figure saved to {filepath}")
    plt.show()
    figure_counter += 1

# 1. Identify issues in the pipeline
issues_explanation = """
Issues Identified in the Original Pipeline:

1. Data Leakage: 
The original pipeline applies `StandardScaler` to the entire feature set `X` before performing the `train_test_split`. This means that the mean and variance of the test set are used to scale the training set, leaking information from the test set into the model training phase. The correct approach is to split the data first, and then fit the scaler only on the training data.

2. Preprocessing Steps:
a) The pipeline fails to handle categorical features. It attempts to scale string columns directly, which will cause an error. Categorical variables must be encoded (e.g., using OneHotEncoder) before scaling and training.
b) The `id` column is included as a feature. An arbitrary identifier should not be used for prediction.
c) The script was loading the wrong dataset file (`airline_passenger_satisfaction_full.csv` instead of `antigravity_clean.csv`).

3. Evaluation Methodology:
The pipeline solely relies on `accuracy_score`. For classification tasks, accuracy can be misleading, especially if classes are imbalanced. A comprehensive evaluation should include Precision, Recall, F1-Score, ROC-AUC, and visual metrics like the ROC Curve.
"""
print(issues_explanation)

# Corrected Pipeline
print("Running Corrected Pipeline...")

# Load dataset
df = pd.read_csv("data/processed/antigravity_clean.csv")

# Drop the 'id' column as it has no predictive power
if "id" in df.columns:
    df = df.drop("id", axis=1)

# Handle potential missing values (if any)
df = df.dropna()

# Separate features and target
X = df.drop("satisfaction", axis=1)
y = df["satisfaction"]

# Encode the target variable for ROC AUC evaluation
le = LabelEncoder()
y = le.fit_transform(y)

# Identify numerical and categorical columns
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Create preprocessing steps
# We use ColumnTransformer to apply OneHotEncoder to categorical subsets and StandardScaler to numerical subsets
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_cols)
    ]
)

# Create a full pipeline that includes preprocessing and the model
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000, random_state=42))
])

# Perform train/test split BEFORE preprocessing to prevent data leakage
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Train the model (pipeline handles fitting preprocessor on train data only)
pipeline.fit(X_train, y_train)

# Evaluate the model
y_pred = pipeline.predict(X_test)
y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"ROC AUC Score: {roc_auc:.4f}")

# Plot and save ROC Curve
fig, ax = plt.subplots(figsize=(8, 6))
RocCurveDisplay.from_estimator(pipeline, X_test, y_test, ax=ax, name='Logistic Regression')
ax.set_title("Receiver Operating Characteristic (ROC) Curve")
save_and_show(fig, "roc_curve")
