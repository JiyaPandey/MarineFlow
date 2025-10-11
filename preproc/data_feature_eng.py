# =============================================================================
# MARINEFLOW FEATURE FINDER (FEATURE ENGINEERING)
# =============================================================================
# Purpose: Feature engineering and creation functions
# Contains: Temporal features, categorical encoding, feature selection
# =============================================================================

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler, LabelEncoder
from utils import get_logger, quick_data_summary
import config

def engineer_features(df):
    """
    Main feature engineering function
    
    Args:
        df (pd.DataFrame): Cleaned input data
        
    Returns:
        pd.DataFrame: Data with engineered features
    """
    logger = get_logger()
    logger.info("Starting feature engineering process...")
    
    quick_data_summary(df, "Input Data for Feature Engineering")
    
    df_features = df.copy()
    
    # Step 1: Create temporal features
    df_features = create_temporal_features(df_features)
    
    # Step 2: Create operational features
    df_features = create_operational_features(df_features)
    
    # Step 3: Encode categorical features
    df_features = encode_categorical_features(df_features)
    
    # Step 4: Create interaction features
    df_features = create_interaction_features(df_features)
    
    # Step 5: Scale numerical features
    df_features = scale_numerical_features(df_features)
    
    quick_data_summary(df_features, "Final Engineered Features")
    
    logger.info(f"Feature engineering completed: {df.shape} to {df_features.shape}")
    return df_features

def create_temporal_features(df):
    """
    Create temporal features from timestamp columns
    
    Args:
        df (pd.DataFrame): Input data
        
    Returns:
        pd.DataFrame: Data with temporal features
    """
    logger = get_logger()
    df_new = df.copy()
    
    for col in config.TIMESTAMP_COLS:
        if col in df_new.columns:
            logger.info(f"Creating temporal features for {col}")
            
            # Ensure column is datetime
            if df_new[col].dtype != 'datetime64[ns]':
                df_new[col] = pd.to_datetime(df_new[col], errors='coerce')
            
            base_name = col.replace('_ts', '').replace('_time', '').replace('_dt', '')
            
            # Basic temporal features
            df_new[f'{base_name}_year'] = df_new[col].dt.year
            df_new[f'{base_name}_month'] = df_new[col].dt.month
            df_new[f'{base_name}_quarter'] = df_new[col].dt.quarter
            df_new[f'{base_name}_day'] = df_new[col].dt.day
            df_new[f'{base_name}_dayofweek'] = df_new[col].dt.dayofweek
            df_new[f'{base_name}_hour'] = df_new[col].dt.hour
            
            # Binary features
            df_new[f'{base_name}_is_weekend'] = df_new[col].dt.dayofweek.isin([5, 6]).astype(int)
            df_new[f'{base_name}_is_month_start'] = df_new[col].dt.is_month_start.astype(int)
            df_new[f'{base_name}_is_month_end'] = df_new[col].dt.is_month_end.astype(int)
            
            # Cyclical encoding for circular features
            df_new[f'{base_name}_month_sin'] = np.sin(2 * np.pi * df_new[f'{base_name}_month'] / 12)
            df_new[f'{base_name}_month_cos'] = np.cos(2 * np.pi * df_new[f'{base_name}_month'] / 12)
            df_new[f'{base_name}_hour_sin'] = np.sin(2 * np.pi * df_new[f'{base_name}_hour'] / 24)
            df_new[f'{base_name}_hour_cos'] = np.cos(2 * np.pi * df_new[f'{base_name}_hour'] / 24)
            
            logger.info(f"Created {base_name} temporal features")
    
    # Calculate time differences between key events
    df_new = create_time_differences(df_new)
    
    return df_new

def create_time_differences(df):
    """
    Create time difference features between key timestamps
    
    Args:
        df (pd.DataFrame): Data with timestamp columns
        
    Returns:
        pd.DataFrame: Data with time difference features
    """
    logger = get_logger()
    df_new = df.copy()
    
    # Define key timestamp pairs for difference calculation
    timestamp_pairs = [
        ('arrival_ts', 'departure_ts'),
        ('nor_tendered_ts', 'arrival_ts'),
        ('loading_commenced_ts', 'arrival_ts'),
        ('loading_completed_ts', 'loading_commenced_ts')
    ]
    
    for start_col, end_col in timestamp_pairs:
        if start_col in df_new.columns and end_col in df_new.columns:
            # Calculate time difference in hours
            diff_col = f'{start_col.replace("_ts", "")}_{end_col.replace("_ts", "")}_hours'
            df_new[diff_col] = (df_new[end_col] - df_new[start_col]).dt.total_seconds() / 3600
            
            # Handle negative differences (data quality issues)
            negative_count = (df_new[diff_col] < 0).sum()
            if negative_count > 0:
                logger.warning(f"Found {negative_count} negative time differences in {diff_col}")
                df_new[diff_col] = df_new[diff_col].clip(lower=0)
            
            logger.info(f"Created time difference feature: {diff_col}")
    
    return df_new

def create_operational_features(df):
    """
    Create operational and business logic features
    
    Args:
        df (pd.DataFrame): Input data
        
    Returns:
        pd.DataFrame: Data with operational features
    """
    logger = get_logger()
    df_new = df.copy()
    
    # Port efficiency features
    if 'port_name' in df_new.columns:
        port_stats = df_new.groupby('port_name').agg({
            'arrival_departure_hours': ['mean', 'std', 'count'],
            'demurrage_flag': 'mean'
        }).round(2)
        port_stats.columns = ['port_avg_duration', 'port_std_duration', 'port_visits', 'port_demurrage_rate']
        
        # Merge back to main dataset
        df_new = df_new.merge(port_stats, left_on='port_name', right_index=True, how='left')
        logger.info("Created port efficiency features")
    
    # Vessel efficiency features
    if 'vessel_name' in df_new.columns:
        vessel_stats = df_new.groupby('vessel_name').agg({
            'cargo_qty_mt': ['mean', 'std'],
            'demurrage_flag': 'mean',
            'arrival_departure_hours': 'mean'
        }).round(2)
        vessel_stats.columns = ['vessel_avg_cargo', 'vessel_std_cargo', 'vessel_demurrage_rate', 'vessel_avg_duration']
        
        df_new = df_new.merge(vessel_stats, left_on='vessel_name', right_index=True, how='left')
        logger.info("Created vessel efficiency features")
    
    # Cargo efficiency features
    if 'cargo_qty_mt' in df_new.columns and 'arrival_departure_hours' in df_new.columns:
        df_new['cargo_handling_rate'] = df_new['cargo_qty_mt'] / (df_new['arrival_departure_hours'] + 1e-6)
        logger.info("Created cargo handling rate feature")
    
    # Seasonal and trend features
    if 'arrival_ts' in df_new.columns:
        df_new['arrival_ts_dt'] = pd.to_datetime(df_new['arrival_ts'])
        df_new['days_since_epoch'] = (df_new['arrival_ts_dt'] - pd.Timestamp('2020-01-01')).dt.days
        logger.info("Created trend features")
    
    return df_new

def encode_categorical_features(df):
    """
    Encode categorical features using appropriate methods
    
    Args:
        df (pd.DataFrame): Input data
        
    Returns:
        pd.DataFrame: Data with encoded categorical features
    """
    logger = get_logger()
    df_new = df.copy()
    
    categorical_cols = df_new.select_dtypes(include=['object']).columns.tolist()
    
    # Remove timestamp columns from categorical encoding
    categorical_cols = [col for col in categorical_cols if not col.endswith('_ts')]
    
    for col in categorical_cols:
        unique_count = df_new[col].nunique()
        
        if unique_count <= 10:  # One-hot encoding for low cardinality
            logger.info(f"One-hot encoding {col} ({unique_count} categories)")
            dummies = pd.get_dummies(df_new[col], prefix=col, drop_first=True)
            df_new = pd.concat([df_new.drop(col, axis=1), dummies], axis=1)
            
        else:  # Target encoding for high cardinality
            if 'demurrage_flag' in df_new.columns:
                logger.info(f"Target encoding {col} ({unique_count} categories)")
                
                # Calculate target mean for each category
                target_means = df_new.groupby(col)['demurrage_flag'].mean()
                
                # Add smoothing to prevent overfitting
                global_mean = df_new['demurrage_flag'].mean()
                category_counts = df_new[col].value_counts()
                
                smoothed_means = {}
                for category in target_means.index:
                    count = category_counts[category]
                    # Smoothing factor - higher count = more weight to category mean
                    alpha = count / (count + 10)
                    smoothed_means[category] = alpha * target_means[category] + (1 - alpha) * global_mean
                
                df_new[f'{col}_target_encoded'] = df_new[col].map(smoothed_means)
                df_new = df_new.drop(col, axis=1)
            else:
                # Fallback to label encoding if no target available
                logger.info(f"Label encoding {col} ({unique_count} categories)")
                le = LabelEncoder()
                df_new[f'{col}_encoded'] = le.fit_transform(df_new[col].astype(str))
                df_new = df_new.drop(col, axis=1)
    
    return df_new

def create_interaction_features(df):
    """
    Create interaction features between important variables
    
    Args:
        df (pd.DataFrame): Input data
        
    Returns:
        pd.DataFrame: Data with interaction features
    """
    logger = get_logger()
    df_new = df.copy()
    
    # Define important numeric columns for interactions
    numeric_cols = df_new.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove target columns from interaction features
    feature_cols = [col for col in numeric_cols if col not in config.TARGET_COLS + config.DATA_LEAKAGE_COLS]
    
    interaction_count = 0
    
    # Create interactions between key operational features
    key_features = [col for col in feature_cols if any(keyword in col.lower() 
                   for keyword in ['cargo', 'duration', 'rate', 'hours', 'qty'])]
    
    for i, col1 in enumerate(key_features):
        for col2 in key_features[i+1:]:
            if col1 != col2:
                # Multiplicative interaction
                interaction_name = f'{col1}_x_{col2}'
                df_new[interaction_name] = df_new[col1] * df_new[col2]
                interaction_count += 1
                
                # Ratio interaction (if no zeros)
                if (df_new[col2] != 0).all():
                    ratio_name = f'{col1}_div_{col2}'
                    df_new[ratio_name] = df_new[col1] / (df_new[col2] + 1e-6)
                    interaction_count += 1
                
                # Limit number of interactions to prevent explosion
                if interaction_count >= 20:
                    break
        
        if interaction_count >= 20:
            break
    
    logger.info(f"Created {interaction_count} interaction features")
    return df_new

def scale_numerical_features(df):
    """
    Scale numerical features for ML readiness
    
    Args:
        df (pd.DataFrame): Input data
        
    Returns:
        pd.DataFrame: Data with scaled features
    """
    logger = get_logger()
    df_new = df.copy()
    
    # Get numeric columns excluding targets
    numeric_cols = df_new.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [col for col in numeric_cols if col not in config.TARGET_COLS]
    
    if not feature_cols:
        logger.warning("No numeric features found for scaling")
        return df_new
    
    # Apply standard scaling
    scaler = StandardScaler()
    df_new[feature_cols] = scaler.fit_transform(df_new[feature_cols])
    
    logger.info(f"Scaled {len(feature_cols)} numerical features")
    return df_new

def select_top_features(df, target_col='demurrage_flag', max_features=100):
    """
    Select top features based on correlation with target
    
    Args:
        df (pd.DataFrame): Input data with all features
        target_col (str): Target column name
        max_features (int): Maximum number of features to select
        
    Returns:
        pd.DataFrame: Data with selected features
    """
    logger = get_logger()
    
    if target_col not in df.columns:
        logger.warning(f"Target column {target_col} not found, skipping feature selection")
        return df
    
    # Get feature columns
    feature_cols = [col for col in df.columns if col not in config.TARGET_COLS + config.DATA_LEAKAGE_COLS]
    
    if len(feature_cols) <= max_features:
        logger.info(f"Feature count ({len(feature_cols)}) already within limit ({max_features})")
        return df
    
    # Calculate correlations with target
    correlations = {}
    for col in feature_cols:
        if df[col].dtype in [np.number]:
            corr = abs(df[col].corr(df[target_col]))
            if not np.isnan(corr):
                correlations[col] = corr
    
    # Select top features
    top_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)[:max_features]
    selected_features = [feat[0] for feat in top_features]
    
    # Keep target columns
    final_columns = selected_features + config.TARGET_COLS
    df_selected = df[final_columns].copy()
    
    logger.info(f"Selected top {len(selected_features)} features from {len(feature_cols)}")
    
    return df_selected