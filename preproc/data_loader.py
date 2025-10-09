# =============================================================================
# MARINEFLOW DATA LOADER - DATA LOADING AND VALIDATION FUNCTIONS
# =============================================================================
# Purpose: Centralized data loading, column validation, and timestamp conversion
# Functions: load_maritime_data, validate_required_columns, convert_timestamps
# =============================================================================

import pandas as pd
import numpy as np
import logging
from config import TIMESTAMP_COLS, NUMERIC_COLS, TARGET_COLS, FILE_PATHS

# Configure logging
logger = logging.getLogger(__name__)

def load_maritime_data(filepath=None):
    """
    Load and perform initial validation of maritime dataset
    
    Args:
        filepath (str): Path to CSV file. If None, uses config default
        
    Returns:
        pd.DataFrame: Loaded dataset with basic validation
    """
    if filepath is None:
        filepath = FILE_PATHS['input']
    
    logger.info("MarineFlow Data Loading Started...")
    logger.info("="*50)
    
    # Load dataset
    logger.info("Loading dataset...")
    df = pd.read_csv(filepath)
    
    logger.info(f"Dataset loaded successfully!")
    logger.info(f"Shape: {df.shape}")
    logger.info(f"Columns: {len(df.columns)} features")
    
    # Basic target variable analysis
    if 'demurrage_flag' in df.columns:
        logger.info("\nTarget Variable Analysis:")
        logger.info("Demurrage Flag Distribution:")
        logger.info(f"{df['demurrage_flag'].value_counts()}")
    
    logger.info("\nBASIC ANALYSIS COMPLETE!")
    return df

def validate_required_columns(df):
    """
    Validate that all required columns are present in the dataset
    
    Args:
        df (pd.DataFrame): Input dataset
        
    Returns:
        pd.DataFrame: Validated dataset
        
    Raises:
        ValueError: If required columns are missing
    """
    logger.info("Validating required columns...")
    
    # Check timestamp columns
    missing_timestamp_cols = [col for col in TIMESTAMP_COLS if col not in df.columns]
    if missing_timestamp_cols:
        raise ValueError(f"Missing required timestamp columns: {missing_timestamp_cols}")
    
    # Check target columns
    missing_target_cols = [col for col in TARGET_COLS if col not in df.columns]
    if missing_target_cols:
        raise ValueError(f"Missing required target columns: {missing_target_cols}")
    
    # Check numeric columns (warn if missing, don't fail)
    missing_numeric_cols = [col for col in NUMERIC_COLS if col not in df.columns]
    if missing_numeric_cols:
        logger.warning(f"Missing expected numeric columns: {missing_numeric_cols}")
    
    logger.info("Column validation complete!")
    return df

def convert_timestamps(df):
    """
    Convert timestamp columns to datetime and remove rows with invalid timestamps
    
    Args:
        df (pd.DataFrame): Input dataset with timestamp columns
        
    Returns:
        pd.DataFrame: Dataset with converted timestamps and invalid rows removed
    """
    logger.info("\n" + "="*50)
    logger.info("Step 4: Data Type Corrections")
    logger.info("="*50)
    
    # First, let's check the data shape before any corrections
    logger.info(f"Dataset shape before corrections: {df.shape}")
    
    # Convert timestamp columns from object to datetime
    logger.info(f"\nConverting timestamp columns to datetime...")
    
    # Convert each timestamp column with error handling
    for col in TIMESTAMP_COLS:
        if col in df.columns:
            logger.info(f"\nProcessing {col}...")
            logger.info(f"   Original dtype: {df[col].dtype}")
            
            # Convert to datetime with error handling
            df_temp = pd.to_datetime(df[col], errors='coerce')
            invalid_count = df_temp.isnull().sum() - df[col].isnull().sum()
            
            if invalid_count > 0:
                logger.warning(f"   WARNING: Found {invalid_count} invalid timestamp values that will become NaN")
                # Show the invalid values
                invalid_mask = df_temp.isnull() & df[col].notnull()
                if invalid_mask.any():
                    logger.warning(f"   Invalid values: {df.loc[invalid_mask, col].unique()}")
            
            df[col] = df_temp
            logger.info(f"   New dtype: {df[col].dtype}")
            logger.info(f"   Non-null values: {df[col].count()}/{len(df)}")
    
    # Now drop rows with any invalid timestamps (NaN in timestamp columns)
    logger.info(f"\nChecking for rows with invalid timestamps...")
    timestamp_nan_mask = df[TIMESTAMP_COLS].isnull().any(axis=1)
    invalid_timestamp_rows = timestamp_nan_mask.sum()
    
    if invalid_timestamp_rows > 0:
        logger.warning(f"Found {invalid_timestamp_rows} rows with invalid timestamps")
        logger.warning(f"This represents {invalid_timestamp_rows/len(df)*100:.2f}% of the data")
        
        # Show which columns have the issues
        for col in TIMESTAMP_COLS:
            if col in df.columns:
                nan_count = df[col].isnull().sum()
                if nan_count > 0:
                    logger.warning(f"   {col}: {nan_count} NaN values")
        
        # Drop the rows with invalid timestamps
        df_original_shape = df.shape
        df = df[~timestamp_nan_mask].copy()
        
        logger.info(f"\nDropped {invalid_timestamp_rows} rows with invalid timestamps")
        logger.info(f"Dataset shape: {df_original_shape} → {df.shape}")
    else:
        logger.info("No invalid timestamps found - all conversions successful!")
    
    logger.info("\nTIMESTAMP CONVERSION COMPLETE!")
    logger.info(f"Final dataset shape: {df.shape}")
    
    # Verify the timestamp conversions
    logger.info(f"\nTimestamp columns summary after conversion:")
    for col in TIMESTAMP_COLS:
        if col in df.columns:
            logger.info(f"   {col}: {df[col].dtype}, {df[col].count()} valid values")
            logger.info(f"      Sample: {df[col].iloc[0]}")
    
    return df

def load_preprocessed_data():
    """
    Load data and apply standard preprocessing (for feature engineering pipeline)
    
    Returns:
        pd.DataFrame: Preprocessed dataset ready for feature engineering
    """
    logger.info("Loading and preprocessing data for feature engineering...")
    
    # Load data
    df = load_maritime_data()
    
    # Validate columns
    df = validate_required_columns(df)
    
    # Convert timestamps
    df = convert_timestamps(df)
    
    # Create basic business metrics (needed for both scripts)
    df['voyage_duration_days'] = (df['arrival_ts'] - df['depart_ts']).dt.total_seconds() / (24 * 3600)
    df['nor_processing_hours'] = (df['nor_accepted_ts'] - df['nor_tender_ts']).dt.total_seconds() / 3600
    df['laytime_efficiency'] = df['used_laytime_h'] / df['allowed_laytime_h']
    
    logger.info("Basic preprocessing complete!")
    return df