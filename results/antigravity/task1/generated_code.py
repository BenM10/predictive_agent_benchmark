import pandas as pd
import os

# 1. Data ingestion
data_path = 'data/processed/airline_passenger_satisfaction_full.csv'
df = pd.read_csv(data_path)
print(f"Dataset shape: {df.shape}")

# 2. Schema inspection
print("\nColumn names:")
print(df.columns.tolist())
print("\nData types:")
print(df.dtypes)

# 3. Missing value check
print("\nMissing values per column:")
print(df.isnull().sum())

# 4. Summary statistics
print("\nSummary statistics for numeric columns:")
print(df.describe())

# 5. Data preview
print("\nFirst few rows:")
print(df.head())

# Save the cleaned dataset (basic cleaning for now: none requested, so just save)
output_path = 'data/processed/antigravity_clean.csv'
df.to_csv(output_path, index=False)
print(f"\nSaved dataset to {output_path}")
