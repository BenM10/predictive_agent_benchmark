import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create figures directory if it doesn't exist
figures_dir = 'figures/antigravity'
os.makedirs(figures_dir, exist_ok=True)

figure_counter = 1

def save_and_show(filename, title=None):
    global figure_counter
    save_path = os.path.join(figures_dir, f"{figure_counter:02d}_{filename}.png")
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Figure {figure_counter} saved to {save_path}")
    plt.show()
    figure_counter += 1

# 1. Load the dataset
data_path = 'data/processed/antigravity_clean.csv'
df = pd.read_csv(data_path)

# 2. Explore the structure of the dataset
print(f"Dataset shape: {df.shape}")
print("\nFirst 5 rows:")
print(df.head())
print("\nInfo:")
print(df.info())

# 3. Identifies and reports important variables and distributions
print("\nValue counts for 'satisfaction':")
print(df['satisfaction'].value_counts(normalize=True))

# 4. Generates a maximum of 7 visualisations

# Visualization 1: Target distribution
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='satisfaction', palette='viridis')
save_and_show("satisfaction_distribution", "Distribution of Passenger Satisfaction")

# Visualization 2: Age distribution by satisfaction
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='Age', hue='satisfaction', kde=True, palette='magma')
save_and_show("age_distribution_by_satisfaction", "Age Distribution by Satisfaction")

# Visualization 3: Type of Travel vs Satisfaction
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='Type of Travel', hue='satisfaction', palette='Set2')
save_and_show("travel_type_vs_satisfaction", "Type of Travel vs Satisfaction")

# Visualization 4: Class vs Satisfaction
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='Class', hue='satisfaction', order=['Eco', 'Eco Plus', 'Business'], palette='Set1')
save_and_show("class_vs_satisfaction", "Class vs Satisfaction")

# Visualization 5: Inflight wifi service vs Satisfaction
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='Inflight wifi service', hue='satisfaction', palette='coolwarm')
save_and_show("wifi_service_vs_satisfaction", "Inflight WiFi Service vs Satisfaction")

# Visualization 6: Flight Distance vs Satisfaction
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='satisfaction', y='Flight Distance', palette='husl')
save_and_show("flight_distance_vs_satisfaction", "Flight Distance vs Satisfaction")

# Visualization 7: Correlation heatmap of numeric features
plt.figure(figsize=(14, 10))
numeric_df = df.select_dtypes(include=['int64', 'float64']).drop(columns=['id'], errors='ignore')
sns.heatmap(numeric_df.corr(), annot=False, cmap='RdBu', center=0)
save_and_show("numeric_correlation_heatmap", "Correlation Heatmap of Numeric Features")

# 5. Highlights any interesting patterns or potential predictive features
print("\nTop 5 correlations with 'Arrival Delay in Minutes':")
print(numeric_df.corr()['Arrival Delay in Minutes'].sort_values(ascending=False).head(6))
