# =============================================================================
# MARINEFLOW DATA CLEANER
# =============================================================================
# Purpose: Data cleaning and preprocessing functions
# Contains: Missing value handling, outlier removal, data validation
# =============================================================================

import pandas as pd
import numpy as np
from utils import get_logger, validate_dataframe, quick_data_summary
import config

def clean_data(df):
    """
    Main data cleaning function
    
    Args:
        df (pd.DataFrame): Raw input data
        
    Returns:
        pd.DataFrame: Cleaned data
    """
    logger = get_logger()
    logger.info("Starting data cleaning process...")
    
    # Initial data summary
    quick_data_summary(df, "Raw Data")
    
    # Validation
    validation = validate_dataframe(df, min_rows=100)
    if not validation['is_valid']:
        raise ValueError(f"Data validation failed: {validation['errors']}")
    
    # Step 1: Handle missing values
    df_clean = handle_missing_values(df)
    
    # Step 2: Remove duplicates
    df_clean = remove_duplicates(df_clean)
    
    # Step 3: Convert data types
    df_clean = convert_data_types(df_clean)
    
    # Step 4: Handle outliers
    df_clean = handle_outliers(df_clean)
    
    # Step 5: Remove data leakage columns
    df_clean = remove_leakage_columns(df_clean)
    
    # Final summary
    quick_data_summary(df_clean, "Cleaned Data")
    
    logger.info(f"Data cleaning completed: {df.shape} to {df_clean.shape}")
    return df_clean

def handle_missing_values(df):
    """
    Handle missing values in the dataset
    
    Args:
        df (pd.DataFrame): Input data
        
    Returns:
        pd.DataFrame: Data with missing values handled
    """
    logger = get_logger()
    df_clean = df.copy()
    
    initial_missing = df_clean.isnull().sum().sum()
    logger.info(f"Initial missing values: {initial_missing}")
    
    # Handle timestamp columns
    for col in config.TIMESTAMP_COLS:
        if col in df_clean.columns:
            # Convert to datetime
            df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')
            missing_count = df_clean[col].isnull().sum()
            if missing_count > 0:
                logger.warning(f"Found {missing_count} invalid timestamps in {col}")
    
    # Remove rows with missing target variables
    for target_col in config.TARGET_COLS:
        if target_col in df_clean.columns:
            before_count = len(df_clean)
            df_clean = df_clean.dropna(subset=[target_col])
            removed_count = before_count - len(df_clean)
            if removed_count > 0:
                logger.info(f"Removed {removed_count} rows with missing {target_col}")
    
    # Handle numeric columns - fill with median
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col not in config.TARGET_COLS:
            missing_count = df_clean[col].isnull().sum()
            if missing_count > 0:
                median_val = df_clean[col].median()
                df_clean[col].fillna(median_val, inplace=True)
                logger.info(f"Filled {missing_count} missing values in {col} with median: {median_val}")
    
    # Handle categorical columns - fill with mode
    categorical_cols = df_clean.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        missing_count = df_clean[col].isnull().sum()
        if missing_count > 0:
            mode_val = df_clean[col].mode()[0] if not df_clean[col].mode().empty else 'Unknown'
            df_clean[col].fillna(mode_val, inplace=True)
            logger.info(f"Filled {missing_count} missing values in {col} with mode: {mode_val}")
    
    final_missing = df_clean.isnull().sum().sum()
    logger.info(f"Final missing values: {final_missing}")
    
    return df_clean

def remove_duplicates(df):
    """
    Remove duplicate rows
    
    Args:
        df (pd.DataFrame): Input data
        
    Returns:
        pd.DataFrame: Data without duplicates
    """
    logger = get_logger()
    
    initial_count = len(df)
    df_clean = df.drop_duplicates()
    final_count = len(df_clean)
    
    removed_count = initial_count - final_count
    if removed_count > 0:
        logger.info(f"Removed {removed_count} duplicate rows")
    else:
        logger.info("No duplicate rows found")
    
    return df_clean

def convert_data_types(df):
    """
    Convert data types for optimal memory usage
    
    Args:
        df (pd.DataFrame): Input data
        
    Returns:
        pd.DataFrame: Data with optimized types
    """
    logger = get_logger()
    df_clean = df.copy()
    
    # Convert timestamp columns
    for col in config.TIMESTAMP_COLS:
        if col in df_clean.columns and df_clean[col].dtype == 'object':
            df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')
            logger.info(f"Converted {col} to datetime")
    
    # Optimize integer columns
    int_cols = df_clean.select_dtypes(include=['int64']).columns
    for col in int_cols:
        if col not in config.TARGET_COLS:
            # Check if can be downcast
            if df_clean[col].min() >= 0 and df_clean[col].max() <= 255:
                df_clean[col] = df_clean[col].astype('uint8')
                logger.info(f"Optimized {col} to uint8")
            elif df_clean[col].min() >= -32768 and df_clean[col].max() <= 32767:
                df_clean[col] = df_clean[col].astype('int16')
                logger.info(f"Optimized {col} to int16")
    
    # Optimize float columns
    float_cols = df_clean.select_dtypes(include=['float64']).columns
    for col in float_cols:
        if col not in config.TARGET_COLS:
            df_clean[col] = pd.to_numeric(df_clean[col], downcast='float')
            logger.info(f"Optimized {col} to float32")
    
    return df_clean

def handle_outliers(df, method='iqr', threshold=3.0):
    """
    Handle outliers in numeric columns
    
    Args:
        df (pd.DataFrame): Input data
        method (str): 'iqr' or 'zscore'
        threshold (float): Threshold for outlier detection
        
    Returns:
        pd.DataFrame: Data with outliers handled
    """
    logger = get_logger()
    df_clean = df.copy()
    
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    feature_cols = [col for col in numeric_cols if col not in config.TARGET_COLS]
    
    total_outliers = 0
    
    for col in feature_cols:
        if method == 'iqr':
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers_mask = (df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)
            
        elif method == 'zscore':
            z_scores = np.abs((df_clean[col] - df_clean[col].mean()) / df_clean[col].std())
            outliers_mask = z_scores > threshold
        
        outlier_count = outliers_mask.sum()
        if outlier_count > 0:
            # Cap outliers instead of removing them
            if method == 'iqr':
                df_clean.loc[df_clean[col] < lower_bound, col] = lower_bound
                df_clean.loc[df_clean[col] > upper_bound, col] = upper_bound
            elif method == 'zscore':
                median_val = df_clean[col].median()
                df_clean.loc[outliers_mask, col] = median_val
            
            total_outliers += outlier_count
            logger.info(f"Handled {outlier_count} outliers in {col}")
    
    if total_outliers > 0:
        logger.info(f"Total outliers handled: {total_outliers}")
    else:
        logger.info("No outliers detected")
    
    return df_clean

def remove_leakage_columns(df):
    """
    Remove data leakage columns
    
    Args:
        df (pd.DataFrame): Input data
        
    Returns:
        pd.DataFrame: Data without leakage columns
    """
    logger = get_logger()
    
    leakage_cols = [col for col in config.DATA_LEAKAGE_COLS if col in df.columns]
    
    if leakage_cols:
        df_clean = df.drop(columns=leakage_cols)
        logger.info(f"Removed data leakage columns: {leakage_cols}")
    else:
        df_clean = df.copy()
        logger.info("No data leakage columns found")
    
    return df_clean

def validate_cleaned_data(df):
    """
    Validate cleaned data before proceeding
    
    Args:
        df (pd.DataFrame): Cleaned data
        
    Returns:
        bool: True if validation passes
    """
    logger = get_logger()
    
    # Check for missing values
    missing_count = df.isnull().sum().sum()
    if missing_count > 0:
        logger.error(f"Cleaned data still has {missing_count} missing values")
        return False
    
    # Check for target variables
    for target_col in config.TARGET_COLS:
        if target_col not in df.columns:
            logger.error(f"Target column {target_col} missing from cleaned data")
            return False
    
    # Check minimum rows
    if len(df) < 100:
        logger.error(f"Cleaned data has only {len(df)} rows")
        return False
    
    logger.info("Cleaned data validation passed")
    return True