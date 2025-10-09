# =============================================================================
# MARINEFLOW FEATURE BUILDER - FEATURE ENGINEERING AND TRANSFORMATION FUNCTIONS
# =============================================================================
# Purpose: Advanced feature engineering, categorical encoding, and feature selection
# Functions: create_temporal_features, create_operational_features, etc.
# =============================================================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif
import logging

# Configure logging
logger = logging.getLogger(__name__)

def create_temporal_features(df):
    """
    Create advanced temporal features from timestamp columns
    
    Args:
        df (pd.DataFrame): Input dataset with converted timestamp columns
        
    Returns:
        pd.DataFrame: Dataset with new temporal features added
    """
    logger.info("\n" + "="*50)
    logger.info("SECTION 2: ADVANCED TEMPORAL FEATURES")
    logger.info("="*50)
    
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
    
    logger.info("Temporal features created successfully!")
    return df

def create_operational_features(df):
    """
    Create operational efficiency and port performance features
    
    Args:
        df (pd.DataFrame): Input dataset
        
    Returns:
        pd.DataFrame: Dataset with operational features added
    """
    logger.info("\n" + "="*50)
    logger.info("SECTION 3: OPERATIONAL EFFICIENCY FEATURES")
    logger.info("="*50)
    
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
    weather_impact_map = {
        'Low': 0.1, 'Moderate': 0.3, 'High': 0.7, 'Extreme': 1.0
    }
    df['weather_impact_score'] = df['weather_severity'].map(weather_impact_map).fillna(0.5)
    
    # SOF (Statement of Facts) aggregations
    df['total_sof_events'] = df['sof_count']
    df['sof_event_rate'] = df['total_sof_events'] / df['voyage_duration_days']
    
    logger.info("Operational features created successfully!")
    return df

def create_financial_features(df):
    """
    Create financial and performance-related features
    
    Args:
        df (pd.DataFrame): Input dataset
        
    Returns:
        pd.DataFrame: Dataset with financial features added
    """
    logger.info("\n" + "="*50)
    logger.info("SECTION 4: FINANCIAL AND PERFORMANCE FEATURES")
    logger.info("="*50)
    
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
    
    logger.info("Financial features created successfully!")
    return df

def encode_categorical_features(df):
    """
    Encode categorical variables using one-hot and label encoding strategies
    
    Args:
        df (pd.DataFrame): Input dataset
        
    Returns:
        tuple: (df, label_encoders) - Modified dataset and label encoder dict
    """
    logger.info("\n" + "="*50)
    logger.info("SECTION 5: CATEGORICAL ENCODING")
    logger.info("="*50)
    
    # Identify categorical columns for encoding
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # Remove voyage_id as it's an identifier, not a feature
    if 'voyage_id' in categorical_cols:
        categorical_cols.remove('voyage_id')
    
    # Strategy: One-hot encode low cardinality, label encode high cardinality
    one_hot_cols = []
    label_encode_cols = []
    
    for col in categorical_cols:
        if col in df.columns:
            unique_count = df[col].nunique()
            if unique_count <= 10:  # One-hot encode if 10 or fewer categories
                one_hot_cols.append(col)
            else:
                label_encode_cols.append(col)
    
    # Apply one-hot encoding
    for col in one_hot_cols:
        if col in df.columns:
            # Create dummy variables
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
            df = pd.concat([df, dummies], axis=1)
            # Drop original column
            df.drop(col, axis=1, inplace=True)
    
    # Apply label encoding
    label_encoders = {}
    for col in label_encode_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col + '_encoded'] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
            # Drop original column
            df.drop(col, axis=1, inplace=True)
    
    # Create binary flags for operational indicators (if columns exist)
    if 'rain_excluded_from_laytime' in df.columns:
        df['rain_excluded_flag'] = (df['rain_excluded_from_laytime'] > 0).astype(int)
    if 'strike_excluded_from_laytime' in df.columns:
        df['strike_excluded_flag'] = (df['strike_excluded_from_laytime'] > 0).astype(int)
    
    logger.info(f"Encoded {len(one_hot_cols)} columns with one-hot encoding")
    logger.info(f"Encoded {len(label_encode_cols)} columns with label encoding")
    
    return df, label_encoders

def select_features_and_analyze_correlation(df):
    """
    Perform feature selection and correlation analysis
    
    Args:
        df (pd.DataFrame): Input dataset with encoded features
        
    Returns:
        tuple: (df, final_feature_cols, feature_importance_class) - Cleaned dataset, feature list, importance scores
    """
    logger.info("\n" + "="*50)
    logger.info("SECTION 6: FEATURE SELECTION AND CORRELATION ANALYSIS")
    logger.info("="*50)
    
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
        logger.info(f"Dropping {len(features_to_drop)} highly correlated features")
        df.drop(features_to_drop, axis=1, inplace=True)
    
    # Calculate feature importance using mutual information
    final_feature_cols = [col for col in df.columns 
                         if col not in ['demurrage_flag', 'demurrage_amount_usd', 'voyage_id'] 
                         and df[col].dtype in ['int64', 'float64']]
    
    # Mutual information for classification (demurrage_flag)
    mi_class = mutual_info_classif(df[final_feature_cols].fillna(0), df['demurrage_flag'])
    feature_importance_class = pd.DataFrame({
        'feature': final_feature_cols,
        'importance_class': mi_class
    }).sort_values('importance_class', ascending=False)
    
    logger.info(f"Feature selection complete. Final feature count: {len(final_feature_cols)}")
    
    return df, final_feature_cols, feature_importance_class

def prepare_train_test_split(df, final_feature_cols):
    """
    Prepare data for train-validation-test split
    
    Args:
        df (pd.DataFrame): Input dataset
        final_feature_cols (list): List of final feature column names
        
    Returns:
        tuple: Train/validation/test splits for features and targets
    """
    logger.info("\n" + "="*50)
    logger.info("SECTION 7: DATA PREPARATION AND TRAIN-TEST SPLIT")
    logger.info("="*50)
    
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
    
    logger.info(f"Data split complete:")
    logger.info(f"   Training: {len(X_train)} samples")
    logger.info(f"   Validation: {len(X_val)} samples") 
    logger.info(f"   Test: {len(X_test)} samples")
    
    return X_train, X_val, X_test, y_class_train, y_class_val, y_class_test, y_reg_train, y_reg_val, y_reg_test

def scale_features(X_train, X_val, X_test):
    """
    Apply standard scaling to feature sets
    
    Args:
        X_train, X_val, X_test: Feature matrices
        
    Returns:
        tuple: (scaler, X_train_scaled, X_val_scaled, X_test_scaled)
    """
    logger.info("\n" + "="*50)
    logger.info("SECTION 8: SCALING AND PREPROCESSING PIPELINE")
    logger.info("="*50)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrames with feature names
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_val_scaled = pd.DataFrame(X_val_scaled, columns=X_val.columns, index=X_val.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
    
    logger.info("Feature scaling complete!")
    
    return scaler, X_train_scaled, X_val_scaled, X_test_scaled

def run_complete_feature_engineering(df):
    """
    Run the complete feature engineering pipeline
    
    Args:
        df (pd.DataFrame): Input dataset (preprocessed)
        
    Returns:
        tuple: All components needed for ML pipeline
    """
    logger.info("Starting complete feature engineering pipeline...")
    
    # Create all feature types
    df = create_temporal_features(df)
    df = create_operational_features(df)
    df = create_financial_features(df)
    
    # Encode categorical variables
    df, label_encoders = encode_categorical_features(df)
    
    # Feature selection and correlation analysis
    df, final_feature_cols, feature_importance_class = select_features_and_analyze_correlation(df)
    
    # Prepare train-test split
    X_train, X_val, X_test, y_class_train, y_class_val, y_class_test, y_reg_train, y_reg_val, y_reg_test = prepare_train_test_split(df, final_feature_cols)
    
    # Scale features
    scaler, X_train_scaled, X_val_scaled, X_test_scaled = scale_features(X_train, X_val, X_test)
    
    return (X_train_scaled, X_val_scaled, X_test_scaled, 
            y_class_train, y_class_val, y_class_test,
            y_reg_train, y_reg_val, y_reg_test,
            scaler, label_encoders, final_feature_cols, feature_importance_class)