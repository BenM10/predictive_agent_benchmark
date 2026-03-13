import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve, auc, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Configuration
DATA_PATH = 'data/processed/antigravity_clean.csv'
FIGURES_DIR = 'figures/antigravity/'
RESULTS_DIR = 'results/antigravity/task3/'
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

figure_counter = 1

def save_and_show(fig, name):
    global figure_counter
    filename = f"{figure_counter:02d}_{name}.png"
    filepath = os.path.join(FIGURES_DIR, filename)
    fig.savefig(filepath, bbox_inches='tight', dpi=300)
    plt.show()
    print(f"Figure saved to: {filepath}")
    figure_counter += 1

def load_and_preprocess(path):
    df = pd.read_csv(path)
    # Remove 'id' column
    if 'id' in df.columns:
        df = df.drop(columns=['id'])
    
    # Handle missing values - HistGradientBoosting handles them, but for others we might need to fill
    # Actually, let's see which columns have NaNs
    nan_cols = df.columns[df.isna().any()].tolist()
    for col in nan_cols:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(df[col].median())

    # Encode target
    le_target = LabelEncoder()
    df['satisfaction'] = le_target.fit_transform(df['satisfaction']) # 0: neutral or dissatisfied, 1: satisfied
    
    # Encode categorical features
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        
    return df

def train_baseline_models(df):
    X = df.drop(columns=['satisfaction'])
    y = df['satisfaction']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale data for Logistic Regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "HistGradientBoosting": HistGradientBoostingClassifier(random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"Training {name}...")
        if name == "Logistic Regression":
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_proba = model.predict_proba(X_test_scaled)[:, 1]
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
            
        results[name] = {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba),
            "y_test": y_test,
            "y_proba": y_proba
        }
    
    return results

def visualize_results(results):
    # 1. Performance Comparison Bar Chart
    metrics_df = pd.DataFrame(results).T[['accuracy', 'f1', 'roc_auc']]
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    metrics_df.plot(kind='bar', ax=ax1)
    ax1.set_title("Baseline Model Performance Comparison")
    ax1.set_ylabel("Score")
    ax1.set_ylim(0, 1.1)
    ax1.legend(loc='lower right')
    plt.xticks(rotation=0)
    save_and_show(fig1, "model_performance_comparison")
    
    # 2. ROC Curves
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    for name, res in results.items():
        fpr, tpr, _ = roc_curve(res['y_test'], res['y_proba'])
        roc_auc = auc(fpr, tpr)
        ax2.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})')
        
    ax2.plot([0, 1], [0, 1], 'k--')
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax2.legend(loc="lower right")
    save_and_show(fig2, "roc_curves")

if __name__ == "__main__":
    data = load_and_preprocess(DATA_PATH)
    results = train_baseline_models(data)
    visualize_results(results)
    
    # Print results for the log
    for name, res in results.items():
        print(f"\n{name} Results:")
        print(f"Accuracy: {res['accuracy']:.4f}")
        print(f"F1 Score: {res['f1']:.4f}")
        print(f"ROC AUC: {res['roc_auc']:.4f}")
