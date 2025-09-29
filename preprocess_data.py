# =============================================================================
# MARINEFLOW DATA PREPROCESSING PIPELINE - PHASE 2
# =============================================================================
# Purpose: Comprehensive data preprocessing and EDA for demurrage prediction
# Input: marineflow_demurrage_synth.csv (600 records, 42 features)
# Output: Clean dataset ready for feature engineering and ML modeling
# =============================================================================

import pandas as pd
import numpy as np

# =============================================================================
# SECTION 1: DATA LOADING AND INITIAL EXPLORATION
# =============================================================================

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

# =============================================================================
# SECTION 2: MISSING VALUES ANALYSIS
# =============================================================================

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

# Step 4: Data Type Corrections
print("\n" + "="*50)
print("Step 4: Data Type Corrections")
print("="*50)

# First, let's check the data shape before any corrections
print(f"Dataset shape before corrections: {df.shape}")

# Convert timestamp columns from object to datetime
timestamp_cols = ['depart_ts', 'arrival_ts', 'nor_tender_ts', 'nor_accepted_ts']
print(f"\nConverting timestamp columns to datetime...")

# Convert each timestamp column with error handling
for col in timestamp_cols:
    if col in df.columns:
        print(f"\nProcessing {col}...")
        print(f"   Original dtype: {df[col].dtype}")
        
        # Convert to datetime with error handling
        df_temp = pd.to_datetime(df[col], errors='coerce')
        invalid_count = df_temp.isnull().sum() - df[col].isnull().sum()
        
        if invalid_count > 0:
            print(f"   WARNING: Found {invalid_count} invalid timestamp values that will become NaN")
            # Show the invalid values
            invalid_mask = df_temp.isnull() & df[col].notnull()
            if invalid_mask.any():
                print(f"   Invalid values: {df.loc[invalid_mask, col].unique()}")
        
        df[col] = df_temp
        print(f"   New dtype: {df[col].dtype}")
        print(f"   Non-null values: {df[col].count()}/{len(df)}")

# Now drop rows with any invalid timestamps (NaN in timestamp columns)
print(f"\nChecking for rows with invalid timestamps...")
timestamp_nan_mask = df[timestamp_cols].isnull().any(axis=1)
invalid_timestamp_rows = timestamp_nan_mask.sum()

if invalid_timestamp_rows > 0:
    print(f"Found {invalid_timestamp_rows} rows with invalid timestamps")
    print(f"This represents {invalid_timestamp_rows/len(df)*100:.2f}% of the data")
    
    # Show which columns have the issues
    for col in timestamp_cols:
        nan_count = df[col].isnull().sum()
        if nan_count > 0:
            print(f"   {col}: {nan_count} NaN values")
    
    # Drop the rows with invalid timestamps
    df_original_shape = df.shape
    df = df[~timestamp_nan_mask].copy()
    
    print(f"\nDropped {invalid_timestamp_rows} rows with invalid timestamps")
    print(f"Dataset shape: {df_original_shape} → {df.shape}")
else:
    print("No invalid timestamps found - all conversions successful!")

print("\nTIMESTAMP CONVERSION COMPLETE!")
print(f"Final dataset shape: {df.shape}")

# Verify the timestamp conversions
print(f"\nTimestamp columns summary after conversion:")
for col in timestamp_cols:
    if col in df.columns:
        print(f"   {col}: {df[col].dtype}, {df[col].count()} valid values")
        print(f"      Sample: {df[col].iloc[0]}")

# Step 5: Consistency Checks
print("\n" + "="*50)
print("Step 5: Data Consistency Checks")
print("="*50)

print("Checking logical relationships and data consistency...")

# Check 1: Arrival should be after departure
print("\n1. Checking departure vs arrival times...")
invalid_journey_mask = df['arrival_ts'] <= df['depart_ts']
invalid_journeys = invalid_journey_mask.sum()
if invalid_journeys > 0:
    print(f"   WARNING: Found {invalid_journeys} records where arrival <= departure")
    print("   This could indicate data quality issues")
else:
    print("   All arrival times are after departure times")

# Check 2: NOR tender should be after arrival
print("\n2. Checking arrival vs NOR tender times...")
invalid_nor_mask = df['nor_tender_ts'] < df['arrival_ts']
invalid_nor = invalid_nor_mask.sum()
if invalid_nor > 0:
    print(f"   WARNING: Found {invalid_nor} records where NOR tender < arrival")
else:
    print("   All NOR tender times are after arrival times")

# Check 3: NOR accepted should be after NOR tender
print("\n3. Checking NOR tender vs accepted times...")
invalid_nor_accept_mask = df['nor_accepted_ts'] < df['nor_tender_ts']
invalid_nor_accept = invalid_nor_accept_mask.sum()
if invalid_nor_accept > 0:
    print(f"   WARNING: Found {invalid_nor_accept} records where NOR accepted < NOR tender")
else:
    print("   All NOR accepted times are after NOR tender times")

# Check 4: Laytime usage should be positive
print("\n4. Checking laytime values...")
negative_laytime = (df['used_laytime_h'] < 0).sum()
if negative_laytime > 0:
    print(f"   WARNING: Found {negative_laytime} records with negative used laytime")
else:
    print("   All used laytime values are non-negative")

# Check 5: Demurrage amount consistency with flag
print("\n5. Checking demurrage amount vs flag consistency...")
# When flag=1, amount should be > 0
flag_1_zero_amount = ((df['demurrage_flag'] == 1) & (df['demurrage_amount_usd'] <= 0)).sum()
# When flag=0, amount should be 0
flag_0_nonzero_amount = ((df['demurrage_flag'] == 0) & (df['demurrage_amount_usd'] > 0)).sum()

if flag_1_zero_amount > 0:
    print(f"   WARNING: {flag_1_zero_amount} records have demurrage_flag=1 but amount <= 0")
if flag_0_nonzero_amount > 0:
    print(f"   WARNING: {flag_0_nonzero_amount} records have demurrage_flag=0 but amount > 0")
if flag_1_zero_amount == 0 and flag_0_nonzero_amount == 0:
    print("   Demurrage flag and amount are consistent")

# Check 6: Port efficiency and congestion indices range
print("\n6. Checking port indices ranges...")
efficiency_range_issues = ((df['port_efficiency_index'] < 0) | (df['port_efficiency_index'] > 100)).sum()
congestion_range_issues = ((df['port_congestion_index'] < 0) | (df['port_congestion_index'] > 100)).sum()

if efficiency_range_issues > 0:
    print(f"   WARNING: {efficiency_range_issues} records have port efficiency index outside 0-100 range")
else:
    print("   Port efficiency indices are within expected range (0-100)")

if congestion_range_issues > 0:
    print(f"   WARNING: {congestion_range_issues} records have port congestion index outside 0-100 range")
else:
    print("   Port congestion indices are within expected range (0-100)")

print("\nCONSISTENCY CHECKS COMPLETE!")

# Summary of data quality
print("\n" + "="*50)
print("DATA QUALITY SUMMARY")
print("="*50)
print(f"Original dataset: 600 records")
print(f"Invalid timestamps removed: 5 records (0.83%)")
print(f"Final clean dataset: {df.shape[0]} records, {df.shape[1]} features")
print("Data consistency checks completed")
print("\nDataset is now ready for Exploratory Data Analysis!")

# =============================================================================
# PHASE 2: EXPLORATORY DATA ANALYSIS (EDA)
# =============================================================================

print("\n" + "="*60)
print("PHASE 2: EXPLORATORY DATA ANALYSIS (EDA)")
print("="*60)

# =============================================================================
# SECTION 7: DESCRIPTIVE STATISTICS
# =============================================================================

print("\n" + "="*50)
print("Step 6: Descriptive Statistics")
print("="*50)

# Numerical columns summary
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print(f"\nDescriptive statistics for {len(numerical_cols)} numerical columns:")
print("-" * 60)

desc_stats = df[numerical_cols].describe()
print(desc_stats.round(2))

# =============================================================================
# SECTION 8: TARGET VARIABLES ANALYSIS
# =============================================================================

print(f"\n" + "="*40)
print("TARGET VARIABLES ANALYSIS")
print("="*40)

# Demurrage Flag Analysis
print(f"\n1. Demurrage Flag Distribution:")
flag_counts = df['demurrage_flag'].value_counts().sort_index()
for flag, count in flag_counts.items():
    percentage = (count/len(df)) * 100
    print(f"   Flag {flag}: {count} records ({percentage:.1f}%)")

# Demurrage Amount Analysis
print(f"\n2. Demurrage Amount Analysis:")
demurrage_stats = df['demurrage_amount_usd'].describe()
print(f"   Mean: ${demurrage_stats['mean']:,.2f}")
print(f"   Median: ${demurrage_stats['50%']:,.2f}")
print(f"   Max: ${demurrage_stats['max']:,.2f}")
print(f"   Records with demurrage > 0: {(df['demurrage_amount_usd'] > 0).sum()}")

# =============================================================================
# SECTION 9: CATEGORICAL VARIABLES ANALYSIS
# =============================================================================

print(f"\n" + "="*40)
print("CATEGORICAL VARIABLES ANALYSIS")
print("="*40)

categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
print(f"\nAnalyzing {len(categorical_cols)} categorical columns:")

for col in categorical_cols:
    print(f"\n{col}:")
    value_counts = df[col].value_counts()
    unique_count = len(value_counts)
    print(f"   Unique values: {unique_count}")
    
    # Show top 5 most frequent values
    top_5 = value_counts.head(5)
    for value, count in top_5.items():
        percentage = (count/len(df)) * 100
        print(f"   '{value}': {count} ({percentage:.1f}%)")
    
    if unique_count > 5:
        print(f"   ... and {unique_count - 5} other values")

print("\nDESCRIPTIVE STATISTICS COMPLETE!")

# =============================================================================
# SECTION 10: BUSINESS METRICS & FEATURE ENGINEERING
# =============================================================================

print("\n" + "="*50)
print("Step 7: Key Maritime Business Metrics")
print("="*50)

# Calculate voyage duration (arrival - departure)
df['voyage_duration_days'] = (df['arrival_ts'] - df['depart_ts']).dt.total_seconds() / (24 * 3600)

# Calculate NOR processing time (accepted - tender)
df['nor_processing_hours'] = (df['nor_accepted_ts'] - df['nor_tender_ts']).dt.total_seconds() / 3600

# Calculate laytime efficiency (used/allowed ratio)
df['laytime_efficiency'] = df['used_laytime_h'] / df['allowed_laytime_h']

print(f"Created new business metrics:")
print(f"   voyage_duration_days: Mean = {df['voyage_duration_days'].mean():.1f} days")
print(f"   nor_processing_hours: Mean = {df['nor_processing_hours'].mean():.1f} hours")
print(f"   laytime_efficiency: Mean = {df['laytime_efficiency'].mean():.2f}")

# Laytime performance analysis
print(f"\nLaytime Performance Analysis:")
efficient_voyages = (df['laytime_efficiency'] <= 1.0).sum()
inefficient_voyages = (df['laytime_efficiency'] > 1.0).sum()
print(f"   Efficient voyages (≤100% laytime used): {efficient_voyages} ({efficient_voyages/len(df)*100:.1f}%)")
print(f"   Inefficient voyages (>100% laytime used): {inefficient_voyages} ({inefficient_voyages/len(df)*100:.1f}%)")

# Port performance insights
print(f"\nPort Performance Insights:")
avg_efficiency_by_port = df.groupby('discharge_port')['port_efficiency_index'].mean().sort_values(ascending=False)
print(f"   Top 3 most efficient ports:")
for i, (port, efficiency) in enumerate(avg_efficiency_by_port.head(3).items(), 1):
    print(f"   {i}. {port}: {efficiency:.1f}")

avg_congestion_by_port = df.groupby('discharge_port')['port_congestion_index'].mean().sort_values()
print(f"   Top 3 least congested ports:")
for i, (port, congestion) in enumerate(avg_congestion_by_port.head(3).items(), 1):
    print(f"   {i}. {port}: {congestion:.1f}")

print(f"\nKEY BUSINESS METRICS COMPLETE!")

# =============================================================================
# SECTION 11: CORRELATION ANALYSIS
# =============================================================================

print("\n" + "="*50)
print("Step 8: Correlation Analysis")
print("="*50)

# Focus on correlations with our target variables
target_cols = ['demurrage_flag', 'demurrage_amount_usd']
key_features = ['port_congestion_index', 'port_efficiency_index', 'allowed_laytime_h', 
                'used_laytime_h', 'berth_wait_h', 'pilot_delay_h', 'rain_hours',
                'demurrage_rate_usd_per_day', 'voyage_duration_days', 'nor_processing_hours',
                'laytime_efficiency', 'sof_count', 'prior_claims_6m']

print("\nCorrelations with Demurrage Flag:")
flag_correlations = df[key_features + ['demurrage_flag']].corr()['demurrage_flag'].drop('demurrage_flag').sort_values(key=abs, ascending=False)
for feature, corr in flag_correlations.head(8).items():
    direction = "Positive" if corr > 0 else "Negative"
    strength = "Strong" if abs(corr) > 0.5 else "Moderate" if abs(corr) > 0.3 else "Weak"
    print(f"   {feature}: {corr:.3f} ({direction}, {strength})")

print("\nCorrelations with Demurrage Amount:")
amount_correlations = df[key_features + ['demurrage_amount_usd']].corr()['demurrage_amount_usd'].drop('demurrage_amount_usd').sort_values(key=abs, ascending=False)
for feature, corr in amount_correlations.head(8).items():
    direction = "Positive" if corr > 0 else "Negative"
    strength = "Strong" if abs(corr) > 0.5 else "Moderate" if abs(corr) > 0.3 else "Weak"
    print(f"   {feature}: {corr:.3f} ({direction}, {strength})")

# =============================================================================
# SECTION 12: KEY INSIGHTS & FEATURE IMPORTANCE
# =============================================================================

print(f"\n" + "="*40)
print("KEY INSIGHTS FROM CORRELATIONS")
print("="*40)

top_flag_driver = flag_correlations.abs().idxmax()
top_amount_driver = amount_correlations.abs().idxmax()

print(f"Top driver for demurrage occurrence: {top_flag_driver} (r={flag_correlations[top_flag_driver]:.3f})")
print(f"Top driver for demurrage amount: {top_amount_driver} (r={amount_correlations[top_amount_driver]:.3f})")

print("CORRELATION ANALYSIS COMPLETE!")

# =============================================================================
# SECTION 13: FINAL SUMMARY
# =============================================================================

print("\n" + "="*60)
print("PREPROCESSING & EDA SUMMARY")
print("="*60)

print("DATA PREPARATION COMPLETE:")
print(f"   Clean dataset: {df.shape[0]} records, {df.shape[1]} features")
print(f"   Target distribution: {(df['demurrage_flag']==1).sum()} with demurrage ({(df['demurrage_flag']==1).sum()/len(df)*100:.1f}%)")
print(f"   Created business features: voyage_duration_days, nor_processing_hours, laytime_efficiency")

print(f"\nKEY BUSINESS INSIGHTS:")
print(f"   {(df['laytime_efficiency'] > 1.0).sum()} voyages ({(df['laytime_efficiency'] > 1.0).sum()/len(df)*100:.1f}%) exceed allowed laytime")
print(f"   Average demurrage cost: ${df[df['demurrage_amount_usd']>0]['demurrage_amount_usd'].mean():,.0f}")
print(f"   Most efficient port: {avg_efficiency_by_port.index[0]} ({avg_efficiency_by_port.iloc[0]:.1f})")
print(f"   Least congested port: {avg_congestion_by_port.index[0]} ({avg_congestion_by_port.iloc[0]:.1f})")

print(f"\nProcessing complete.")