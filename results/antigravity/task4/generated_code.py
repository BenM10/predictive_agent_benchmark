import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve, auc, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Configuration
DATA_PATH = 'data/processed/antigravity_clean.csv'
FIGURES_DIR = 'figures/antigravity/'
RESULTS_DIR = 'results/antigravity/task4/'
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
    
    # Missing values filling
    nan_cols = df.columns[df.isna().any()].tolist()
    for col in nan_cols:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(df[col].median())

    # --- Feature Engineering ---
    # 1. Total Delay Feature
    if 'Departure Delay in Minutes' in df.columns and 'Arrival Delay in Minutes' in df.columns:
        df['Total Delay'] = df['Departure Delay in Minutes'] + df['Arrival Delay in Minutes']
        
    # 2. Overall Service Score (Average of all satisfaction ratings)
    service_cols = ['Inflight wifi service', 'Departure/Arrival time convenient', 'Ease of Online booking', 
                    'Gate location', 'Food and drink', 'Online boarding', 'Seat comfort', 
                    'Inflight entertainment', 'On-board service', 'Leg room service', 
                    'Baggage handling', 'Checkin service', 'Inflight service', 'Cleanliness']
    
    available_service_cols = [c for c in service_cols if c in df.columns]
    if available_service_cols:
        df['Average Service Score'] = df[available_service_cols].mean(axis=1)

    # Encode target
    le_target = LabelEncoder()
    df['satisfaction'] = le_target.fit_transform(df['satisfaction'])
    
    # Encode categorical features
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        
    return df

def train_baseline_and_improved(df):
    X = df.drop(columns=['satisfaction'])
    y = df['satisfaction']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scaler for Logistic Regression (Baseline)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    results = {}
    models_to_return = {}
    
    # 1. Baseline Model (Logistic Regression)
    print("Training Baseline (Logistic Regression)...")
    baseline = LogisticRegression(max_iter=1000, random_state=42)
    baseline.fit(X_train_scaled, y_train)
    y_pred_base = baseline.predict(X_test_scaled)
    y_proba_base = baseline.predict_proba(X_test_scaled)[:, 1]
    
    results['Baseline (Logistic Regression)'] = {
        "accuracy": accuracy_score(y_test, y_pred_base),
        "f1": f1_score(y_test, y_pred_base),
        "roc_auc": roc_auc_score(y_test, y_proba_base),
        "y_test": y_test,
        "y_proba": y_proba_base,
        "y_pred": y_pred_base
    }
    models_to_return['Baseline'] = baseline
    
    # 2. Improved Model (Tuned HistGradientBoosting)
    print("Training Improved Model (Tuned HistGradientBoosting)...")
    # Identify categorical columns indices for native support in HistGradientBoosting
    # But since they are already label-encoded and scaled differently, we just let it run on X_train.
    # HistGradientBoosting handles unscaled features perfectly.
    
    param_distributions = {
        'learning_rate': [0.05, 0.1, 0.2],
        'max_iter': [100, 200],
        'max_leaf_nodes': [31, 63, 127],
        'min_samples_leaf': [20, 50, 100],
        'l2_regularization': [0.0, 0.1, 1.0]
    }
    
    base_hgb = HistGradientBoostingClassifier(random_state=42)
    search = RandomizedSearchCV(base_hgb, param_distributions, n_iter=10, 
                                scoring='roc_auc', cv=3, random_state=42, n_jobs=-1, verbose=1)
    
    search.fit(X_train, y_train)
    print(f"Best Improved Model Params: {search.best_params_}")
    best_improved = search.best_estimator_
    
    y_pred_imp = best_improved.predict(X_test)
    y_proba_imp = best_improved.predict_proba(X_test)[:, 1]
    
    results['Improved (Tuned HGB)'] = {
        "accuracy": accuracy_score(y_test, y_pred_imp),
        "f1": f1_score(y_test, y_pred_imp),
        "roc_auc": roc_auc_score(y_test, y_proba_imp),
        "y_test": y_test,
        "y_proba": y_proba_imp,
        "y_pred": y_pred_imp
    }
    models_to_return['Improved'] = best_improved
    
    return results, models_to_return, X_test.columns

def visualize_results(results):
    # Performance Comparison Bar Chart
    metrics_df = pd.DataFrame(results).T[['accuracy', 'f1', 'roc_auc']]
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    metrics_df.plot(kind='bar', ax=ax1, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax1.set_title("Performance Comparison: Baseline vs Improved Model")
    ax1.set_ylabel("Score")
    ax1.set_ylim(0, 1.1)
    ax1.legend(loc='lower right')
    plt.xticks(rotation=0)
    for p in ax1.patches:
        ax1.annotate(f"{p.get_height():.4f}", (p.get_x() * 1.005, p.get_height() * 1.005), fontsize=8, rotation=90)
    save_and_show(fig1, "model_comparison_bar")
    
    # ROC Curves
    fig2, ax2 = plt.subplots(figsize=(8, 8))
    for name, res in results.items():
        fpr, tpr, _ = roc_curve(res['y_test'], res['y_proba'])
        roc_auc = auc(fpr, tpr)
        ax2.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.4f})')
        
    ax2.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title('ROC Curve: Baseline vs Improved Model')
    ax2.legend(loc="lower right")
    save_and_show(fig2, "roc_comparison")

if __name__ == "__main__":
    print("Loading and preprocessing data...")
    data = load_and_preprocess(DATA_PATH)
    
    print("Training models...")
    res, models, feature_names = train_baseline_and_improved(data)
    
    print("Visualizing results...")
    visualize_results(res)
    
    print("\n--- Final Performance Summary ---")
    for name, metrics in res.items():
        print(f"\n{name} Results:")
        print(f"Accuracy : {metrics['accuracy']:.4f}")
        print(f"F1 Score : {metrics['f1']:.4f}")
        print(f"ROC AUC  : {metrics['roc_auc']:.4f}")
        print("Classification Report:")
        print(classification_report(metrics['y_test'], metrics['y_pred']))
