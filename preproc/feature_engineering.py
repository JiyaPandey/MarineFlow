# =============================================================================
# MARINEFLOW FEATURE ENGINEERING PIPELINE - PHASE 2.2
# =============================================================================
# Purpose: Advanced feature engineering for demurrage prediction
# Input: Preprocessed data from preprocess_data.py
# Output: ML-ready datasets with engineered features
# =============================================================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
import pickle
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# SECTION 1: LOAD PREPROCESSED DATA
# =============================================================================

# Load and recreate the preprocessed dataset
df = pd.read_csv('marineflow_demurrage_synth.csv')

# Apply same preprocessing steps from preprocess_data.py
timestamp_cols = ['depart_ts', 'arrival_ts', 'nor_tender_ts', 'nor_accepted_ts']
for col in timestamp_cols:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')

# Remove invalid timestamp rows
timestamp_nan_mask = df[timestamp_cols].isnull().any(axis=1)
df = df[~timestamp_nan_mask].copy()

# Recreate basic features
df['voyage_duration_days'] = (df['arrival_ts'] - df['depart_ts']).dt.total_seconds() / (24 * 3600)
df['nor_processing_hours'] = (df['nor_accepted_ts'] - df['nor_tender_ts']).dt.total_seconds() / 3600
df['laytime_efficiency'] = df['used_laytime_h'] / df['allowed_laytime_h']

# =============================================================================
# SECTION 2: ADVANCED TEMPORAL FEATURES
# =============================================================================

# Month and seasonal patterns
df['departure_month'] = df['depart_ts'].dt.month
df['departure_quarter'] = df['depart_ts'].dt.quarter
df['departure_day_of_week'] = df['depart_ts'].dt.dayofweek
df['departure_day_of_year'] = df['depart_ts'].dt.dayofyear

# Arrival patterns
df['arrival_month'] = df['arrival_ts'].dt.month
df['arrival_quarter'] = df['arrival_ts'].dt.quarter
df['arrival_day_of_week'] = df['arrival_ts'].dt.dayofweek

# Seasonal indicators
df['is_peak_season'] = df['departure_quarter'].isin([2, 3]).astype(int)  # Q2, Q3 typically busy
df['is_year_end'] = df['departure_month'].isin([11, 12]).astype(int)  # Year-end rush
df['is_weekend_departure'] = df['departure_day_of_week'].isin([5, 6]).astype(int)

# Time-based efficiency metrics
df['laytime_per_day'] = df['used_laytime_h'] / df['voyage_duration_days']
df['berth_wait_ratio'] = df['berth_wait_h'] / df['voyage_duration_days']
df['pilot_delay_ratio'] = df['pilot_delay_h'] / df['voyage_duration_days']

# Delay indicators
df['has_berth_wait'] = (df['berth_wait_h'] > 0).astype(int)
df['has_pilot_delay'] = (df['pilot_delay_h'] > 0).astype(int)
df['has_rain_delay'] = (df['rain_hours'] > 0).astype(int)

# =============================================================================
# SECTION 3: OPERATIONAL EFFICIENCY FEATURES
# =============================================================================

# Port efficiency rankings
port_efficiency_rank = df.groupby('discharge_port')['port_efficiency_index'].mean().rank(ascending=False)
port_congestion_rank = df.groupby('discharge_port')['port_congestion_index'].mean().rank(ascending=True)

df['discharge_port_efficiency_rank'] = df['discharge_port'].map(port_efficiency_rank)
df['discharge_port_congestion_rank'] = df['discharge_port'].map(port_congestion_rank)

# Load port rankings
load_port_efficiency_rank = df.groupby('load_port')['port_efficiency_index'].mean().rank(ascending=False)
df['load_port_efficiency_rank'] = df['load_port'].map(load_port_efficiency_rank)

# Combined port performance score
df['port_performance_score'] = (df['port_efficiency_index'] - df['port_congestion_index']) / 100

# Vessel and cargo efficiency
vessel_avg_efficiency = df.groupby('vessel_class')['laytime_efficiency'].mean()
cargo_avg_efficiency = df.groupby('cargo_type')['laytime_efficiency'].mean()

df['vessel_class_avg_efficiency'] = df['vessel_class'].map(vessel_avg_efficiency)
df['cargo_type_avg_efficiency'] = df['cargo_type'].map(cargo_avg_efficiency)

# Weather impact features
df['weather_impact_score'] = df['weather_severity'].map({
    'Low': 0.1, 'Moderate': 0.3, 'High': 0.7, 'Extreme': 1.0
})

# SOF (Statement of Facts) aggregations
sof_columns = [col for col in df.columns if col.startswith('sof_')]
df['total_sof_events'] = df[sof_columns].sum(axis=1)
df['sof_event_rate'] = df['total_sof_events'] / df['voyage_duration_days']

# =============================================================================
# SECTION 4: FINANCIAL AND PERFORMANCE FEATURES
# =============================================================================

# Rate efficiency
df['demurrage_rate_efficiency'] = df['demurrage_rate_usd_per_day'] / df['demurrage_rate_usd_per_day'].median()

# Historical performance indicators
df['has_prior_claims'] = (df['prior_claims_6m'] > 0).astype(int)
df['prior_claim_frequency'] = df['prior_claims_6m'] / 6  # claims per month

# Charterer performance
charterer_avg_demurrage = df.groupby('charterer')['demurrage_amount_usd'].mean()
df['charterer_avg_demurrage'] = df['charterer'].map(charterer_avg_demurrage)

# Performance ratios
df['efficiency_vs_congestion'] = df['port_efficiency_index'] / (df['port_congestion_index'] + 1)
df['laytime_vs_allowed'] = df['used_laytime_h'] / df['allowed_laytime_h']

# =============================================================================
# SECTION 5: CATEGORICAL ENCODING
# =============================================================================

# Identify categorical columns for encoding
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

# Remove voyage_id as it's an identifier, not a feature
if 'voyage_id' in categorical_cols:
    categorical_cols.remove('voyage_id')

# Strategy: One-hot encode low cardinality, label encode high cardinality
one_hot_cols = []
label_encode_cols = []

for col in categorical_cols:
    unique_count = df[col].nunique()
    if unique_count <= 10:  # One-hot encode if 10 or fewer categories
        one_hot_cols.append(col)
    else:
        label_encode_cols.append(col)

# Apply one-hot encoding
for col in one_hot_cols:
    # Create dummy variables
    dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
    df = pd.concat([df, dummies], axis=1)
    # Drop original column
    df.drop(col, axis=1, inplace=True)

# Apply label encoding
label_encoders = {}
for col in label_encode_cols:
    le = LabelEncoder()
    df[col + '_encoded'] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le
    # Keep original for reference but will drop later
    
# Create binary flags for operational indicators
df['rain_excluded_flag'] = (df['rain_excluded_from_laytime'] > 0).astype(int)
df['strike_excluded_flag'] = (df['strike_excluded_from_laytime'] > 0).astype(int)

# =============================================================================
# SECTION 6: FEATURE SELECTION AND CORRELATION ANALYSIS
# =============================================================================

# Get numeric columns for correlation analysis
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# Remove target variables from feature correlation analysis
feature_cols = [col for col in numeric_cols if col not in ['demurrage_flag', 'demurrage_amount_usd']]

# Calculate correlation matrix
correlation_matrix = df[feature_cols].corr()

# Find highly correlated features
high_corr_pairs = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        if abs(correlation_matrix.iloc[i, j]) > 0.95:
            colname_i = correlation_matrix.columns[i]
            colname_j = correlation_matrix.columns[j]
            high_corr_pairs.append((colname_i, colname_j, correlation_matrix.iloc[i, j]))

# Remove one feature from each highly correlated pair
features_to_drop = []
for col1, col2, corr in high_corr_pairs:
    # Keep the feature with higher correlation to target
    if col1 not in features_to_drop and col2 not in features_to_drop:
        corr_1_target = abs(df[col1].corr(df['demurrage_flag']))
        corr_2_target = abs(df[col2].corr(df['demurrage_flag']))
        
        if corr_1_target > corr_2_target:
            features_to_drop.append(col2)
        else:
            features_to_drop.append(col1)

if features_to_drop:
    df.drop(features_to_drop, axis=1, inplace=True)

# Calculate feature importance using mutual information
final_feature_cols = [col for col in df.columns if col not in ['demurrage_flag', 'demurrage_amount_usd', 'voyage_id'] and df[col].dtype in ['int64', 'float64']]

# Mutual information for classification (demurrage_flag)
mi_class = mutual_info_classif(df[final_feature_cols].fillna(0), df['demurrage_flag'])
feature_importance_class = pd.DataFrame({
    'feature': final_feature_cols,
    'importance_class': mi_class
}).sort_values('importance_class', ascending=False)

# =============================================================================
# SECTION 7: DATA PREPARATION AND TRAIN-TEST SPLIT
# =============================================================================

# Clean up unnecessary columns
columns_to_keep = final_feature_cols + ['demurrage_flag', 'demurrage_amount_usd']
if 'voyage_id' in df.columns:
    columns_to_keep.append('voyage_id')

# Drop original categorical columns that were encoded
original_categoricals = ['vessel_class', 'cargo_type', 'charterer', 'load_port', 'discharge_port', 'weather_severity', 'cp_laytime_basis', 'cp_weather_clause', 'nor_basis']
for col in original_categoricals:
    if col in df.columns:
        df.drop(col, axis=1, inplace=True)

# Final feature matrix
X = df[final_feature_cols].fillna(0)
y_classification = df['demurrage_flag']
y_regression = df['demurrage_amount_usd']

# Train-validation-test split (70-15-15)

# First split: 70% train, 30% temp
X_train, X_temp, y_class_train, y_class_temp, y_reg_train, y_reg_temp = train_test_split(
    X, y_classification, y_regression, 
    test_size=0.3, 
    random_state=42, 
    stratify=y_classification
)

# Second split: 15% validation, 15% test from the 30% temp
X_val, X_test, y_class_val, y_class_test, y_reg_val, y_reg_test = train_test_split(
    X_temp, y_class_temp, y_reg_temp,
    test_size=0.5, 
    random_state=42, 
    stratify=y_class_temp
)

# =============================================================================
# SECTION 8: SCALING AND PREPROCESSING PIPELINE
# =============================================================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Convert back to DataFrames with feature names
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
X_val_scaled = pd.DataFrame(X_val_scaled, columns=X_val.columns, index=X_val.index)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

# =============================================================================
# SECTION 9: SAVE DATASETS AND PREPROCESSING PIPELINE
# =============================================================================

# Training data
train_data = pd.concat([X_train_scaled, y_class_train, y_reg_train], axis=1)
train_data.to_csv('marineflow_train.csv', index=False)

# Validation data
val_data = pd.concat([X_val_scaled, y_class_val, y_reg_val], axis=1)
val_data.to_csv('marineflow_validation.csv', index=False)

# Test data
test_data = pd.concat([X_test_scaled, y_class_test, y_reg_test], axis=1)
test_data.to_csv('marineflow_test.csv', index=False)

# Save preprocessing components

preprocessing_pipeline = {
    'scaler': scaler,
    'label_encoders': label_encoders,
    'feature_names': list(X_train.columns),
    'target_names': ['demurrage_flag', 'demurrage_amount_usd'],
    'feature_importance': feature_importance_class
}

with open('preprocessing_pipeline.pkl', 'wb') as f:
    pickle.dump(preprocessing_pipeline, f)

# Create feature documentation
feature_docs = {
    'total_features': len(final_feature_cols),
    'original_features': 42,
    'engineered_features': len(final_feature_cols) - 42,
    'temporal_features': 12,
    'efficiency_features': 10,
    'financial_features': 6,
    'categorical_encoded': len(one_hot_cols) + len(label_encode_cols),
    'train_samples': len(X_train),
    'val_samples': len(X_val),
    'test_samples': len(X_test)
}

with open('feature_documentation.txt', 'w') as f:
    f.write("MARINEFLOW FEATURE ENGINEERING SUMMARY\n")
    f.write("="*50 + "\n\n")
    for key, value in feature_docs.items():
        f.write(f"{key}: {value}\n")
    
    f.write(f"\nTop 10 Most Important Features:\n")
    for idx, row in feature_importance_class.head(10).iterrows():
        f.write(f"{row['feature']}: {row['importance_class']:.4f}\n")

logger.info("Feature engineering pipeline complete!")