# =============================================================================
# MARINEFLOW DATA VALIDATOR - DATA QUALITY AND CONSISTENCY FUNCTIONS
# =============================================================================
# Purpose: Data quality checks, missing value analysis, and business rule validation
# Functions: analyze_missing_values, analyze_data_types, check_data_consistency
# =============================================================================

import pandas as pd
import numpy as np
import logging
from config import TIMESTAMP_COLS, NUMERIC_COLS, VALIDATION_RANGES

# Configure logging
logger = logging.getLogger(__name__)

def analyze_missing_values(df):
    """
    Comprehensive missing values analysis and reporting
    
    Args:
        df (pd.DataFrame): Input dataset
        
    Returns:
        pd.DataFrame: Dataset (unchanged)
    """
    logger.info("\n" + "="*50)
    logger.info("Step 2: Missing Values Analysis")
    logger.info("="*50)
    
    missing_values = df.isnull().sum()
    missing_percent = (missing_values / len(df)) * 100
    
    logger.info(f"Total missing values in dataset: {missing_values.sum()}")
    
    if missing_values.sum() > 0:
        logger.info("\nColumns with missing values:")
        for col in df.columns:
            if missing_values[col] > 0:
                logger.info(f"   {col}: {missing_values[col]} missing ({missing_percent[col]:.1f}%)")
    else:
        logger.info("No missing values found in any column!")
    
    # Show summary of missing data
    missing_summary = pd.DataFrame({
        'Column': missing_values.index,
        'Missing_Count': missing_values.values,
        'Missing_Percent': missing_percent.values
    })
    top_missing = missing_summary[missing_summary['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False).head()
    
    if len(top_missing) > 0:
        logger.info(f"\nTop columns with most missing data:")
        logger.info(f"{top_missing.to_string(index=False)}")
    
    logger.info("\nMISSING VALUES ANALYSIS COMPLETE!")
    return df

def analyze_data_types(df):
    """
    Analyze and report data types for all columns
    
    Args:
        df (pd.DataFrame): Input dataset
        
    Returns:
        pd.DataFrame: Dataset (unchanged)
    """
    logger.info("\n" + "="*50)
    logger.info("Step 3: Data Type Analysis")
    logger.info("="*50)
    
    logger.info("Current data types:")
    dtype_counts = df.dtypes.value_counts()
    for dtype, count in dtype_counts.items():
        logger.info(f"   {dtype}: {count} columns")
    
    logger.info(f"\nDetailed data types:")
    data_types = df.dtypes
    for col in df.columns:
        logger.info(f"   {col}: {data_types[col]}")
    
    # Check timestamp columns
    logger.info(f"\nTimestamp columns analysis:")
    for col in TIMESTAMP_COLS:
        if col in df.columns:
            logger.info(f"   {col}: {df[col].dtype}")
            logger.info(f"      Sample value: {df[col].iloc[0]}")
    
    # Check numeric columns that should be numeric
    logger.info(f"\nNumerical columns check:")
    for col in NUMERIC_COLS:
        if col in df.columns:
            logger.info(f"   {col}: {df[col].dtype}")
            if df[col].dtype == 'object':
                logger.warning(f"      WARNING: Should be numeric but is object type")
    
    logger.info("\nDATA TYPE ANALYSIS COMPLETE!")
    return df

def check_data_consistency(df):
    """
    Perform comprehensive business logic and data consistency checks
    
    Args:
        df (pd.DataFrame): Input dataset with converted timestamps
        
    Returns:
        pd.DataFrame: Dataset (unchanged)
    """
    logger.info("\n" + "="*50)
    logger.info("Step 5: Data Consistency Checks")
    logger.info("="*50)
    
    logger.info("Checking logical relationships and data consistency...")
    
    # Check 1: Arrival should be after departure
    logger.info("\n1. Checking departure vs arrival times...")
    invalid_journey_mask = df['arrival_ts'] <= df['depart_ts']
    invalid_journeys = invalid_journey_mask.sum()
    if invalid_journeys > 0:
        logger.warning(f"   WARNING: Found {invalid_journeys} records where arrival <= departure")
        logger.warning("   This could indicate data quality issues")
    else:
        logger.info("   All arrival times are after departure times")
    
    # Check 2: NOR tender should be after arrival
    logger.info("\n2. Checking arrival vs NOR tender times...")
    invalid_nor_mask = df['nor_tender_ts'] < df['arrival_ts']
    invalid_nor = invalid_nor_mask.sum()
    if invalid_nor > 0:
        logger.warning(f"   WARNING: Found {invalid_nor} records where NOR tender < arrival")
    else:
        logger.info("   All NOR tender times are after arrival times")
    
    # Check 3: NOR accepted should be after NOR tender
    logger.info("\n3. Checking NOR tender vs accepted times...")
    invalid_nor_accept_mask = df['nor_accepted_ts'] < df['nor_tender_ts']
    invalid_nor_accept = invalid_nor_accept_mask.sum()
    if invalid_nor_accept > 0:
        logger.warning(f"   WARNING: Found {invalid_nor_accept} records where NOR accepted < NOR tender")
    else:
        logger.info("   All NOR accepted times are after NOR tender times")
    
    # Check 4: Laytime usage should be positive
    logger.info("\n4. Checking laytime values...")
    negative_laytime = (df['used_laytime_h'] < 0).sum()
    if negative_laytime > 0:
        logger.warning(f"   WARNING: Found {negative_laytime} records with negative used laytime")
    else:
        logger.info("   All used laytime values are non-negative")
    
    # Check 5: Demurrage amount consistency with flag
    logger.info("\n5. Checking demurrage amount vs flag consistency...")
    # When flag=1, amount should be > 0
    flag_1_zero_amount = ((df['demurrage_flag'] == 1) & (df['demurrage_amount_usd'] <= 0)).sum()
    # When flag=0, amount should be 0
    flag_0_nonzero_amount = ((df['demurrage_flag'] == 0) & (df['demurrage_amount_usd'] > 0)).sum()
    
    if flag_1_zero_amount > 0:
        logger.warning(f"   WARNING: {flag_1_zero_amount} records have demurrage_flag=1 but amount <= 0")
    if flag_0_nonzero_amount > 0:
        logger.warning(f"   WARNING: {flag_0_nonzero_amount} records have demurrage_flag=0 but amount > 0")
    if flag_1_zero_amount == 0 and flag_0_nonzero_amount == 0:
        logger.info("   Demurrage flag and amount are consistent")
    
    # Check 6: Port efficiency and congestion indices range
    logger.info("\n6. Checking port indices ranges...")
    efficiency_range_issues = ((df['port_efficiency_index'] < 0) | (df['port_efficiency_index'] > 100)).sum()
    congestion_range_issues = ((df['port_congestion_index'] < 0) | (df['port_congestion_index'] > 100)).sum()
    
    if efficiency_range_issues > 0:
        logger.warning(f"   WARNING: {efficiency_range_issues} records have port efficiency index outside 0-100 range")
    else:
        logger.info("   Port efficiency indices are within expected range (0-100)")
    
    if congestion_range_issues > 0:
        logger.warning(f"   WARNING: {congestion_range_issues} records have port congestion index outside 0-100 range")
    else:
        logger.info("   Port congestion indices are within expected range (0-100)")
    
    logger.info("\nCONSISTENCY CHECKS COMPLETE!")
    return df

def generate_data_quality_summary(df, original_count=600, dropped_count=5):
    """
    Generate final data quality summary report
    
    Args:
        df (pd.DataFrame): Final cleaned dataset
        original_count (int): Original number of records
        dropped_count (int): Number of records dropped
        
    Returns:
        pd.DataFrame: Dataset (unchanged)
    """
    logger.info("\n" + "="*50)
    logger.info("DATA QUALITY SUMMARY")
    logger.info("="*50)
    logger.info(f"Original dataset: {original_count} records")
    logger.info(f"Invalid timestamps removed: {dropped_count} records ({dropped_count/original_count*100:.2f}%)")
    logger.info(f"Final clean dataset: {df.shape[0]} records, {df.shape[1]} features")
    logger.info("Data consistency checks completed")
    logger.info("\nDataset is now ready for Exploratory Data Analysis!")
    return df