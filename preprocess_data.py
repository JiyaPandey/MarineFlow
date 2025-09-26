# MarineFlow Data Preprocessing - Phase 2
import pandas as pd
import numpy as np

print("MarineFlow Data Preprocessing Started...")
print("="*50)

# Load dataset
print("Loading dataset...")
df = pd.read_csv('marineflow_demurrage_synth.csv')

print(f"Dataset loaded successfully!")
print(f"Shape: {df.shape}")
print(f"Columns: {len(df.columns)} features")

print("\nTarget Variable Analysis:")
print("Demurrage Flag Distribution:")
print(df['demurrage_flag'].value_counts())

print("\nBASIC ANALYSIS COMPLETE!")

# Step 2: Missing Values Analysis
print("\n" + "="*50)
print("Step 2: Missing Values Analysis")
print("="*50)

missing_values = df.isnull().sum()
missing_percent = (missing_values / len(df)) * 100

print(f"Total missing values in dataset: {missing_values.sum()}")

if missing_values.sum() > 0:
    print("\nColumns with missing values:")
    for col in df.columns:
        if missing_values[col] > 0:
            print(f"   {col}: {missing_values[col]} missing ({missing_percent[col]:.1f}%)")
else:
    print("No missing values found in any column!")

# Show summary of missing data
missing_summary = pd.DataFrame({
    'Column': missing_values.index,
    'Missing_Count': missing_values.values,
    'Missing_Percent': missing_percent.values
})
top_missing = missing_summary[missing_summary['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False).head()

if len(top_missing) > 0:
    print(f"\nTop columns with most missing data:")
    print(top_missing.to_string(index=False))

print("\nMISSING VALUES ANALYSIS COMPLETE!")

# Step 3: Data Type Analysis
print("\n" + "="*50)
print("Step 3: Data Type Analysis")
print("="*50)

print("Current data types:")
dtype_counts = df.dtypes.value_counts()
for dtype, count in dtype_counts.items():
    print(f"   {dtype}: {count} columns")

print(f"\nDetailed data types:")
data_types = df.dtypes
for col in df.columns:
    print(f"   {col}: {data_types[col]}")

# Check timestamp columns
timestamp_cols = ['depart_ts', 'arrival_ts', 'nor_tender_ts', 'nor_accepted_ts']
print(f"\nTimestamp columns analysis:")
for col in timestamp_cols:
    if col in df.columns:
        print(f"   {col}: {df[col].dtype}")
        print(f"      Sample value: {df[col].iloc[0]}")

# Check numeric columns that should be numeric
numeric_cols = ['port_congestion_index', 'port_efficiency_index', 'allowed_laytime_h', 
                'used_laytime_h', 'demurrage_amount_usd']
print(f"\nNumerical columns check:")
for col in numeric_cols:
    if col in df.columns:
        print(f"   {col}: {df[col].dtype}")
        if df[col].dtype == 'object':
            print(f"      WARNING: Should be numeric but is object type")

print("\nDATA TYPE ANALYSIS COMPLETE!")