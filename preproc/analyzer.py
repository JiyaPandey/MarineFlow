# =============================================================================
# MARINEFLOW ANALYZER - EDA AND STATISTICAL ANALYSIS FUNCTIONS
# =============================================================================
# Purpose: Exploratory data analysis, descriptive statistics, and correlation analysis
# Functions: generate_descriptive_stats, analyze_targets, analyze_categorical, etc.
# =============================================================================

import pandas as pd
import numpy as np
import logging
from config import KEY_FEATURES, TARGET_COLS

# Configure logging
logger = logging.getLogger(__name__)

def generate_descriptive_stats(df):
    """
    Generate comprehensive descriptive statistics for numerical columns
    
    Args:
        df (pd.DataFrame): Input dataset
        
    Returns:
        pd.DataFrame: Dataset (unchanged)
    """
    logger.info("\n" + "="*50)
    logger.info("Step 6: Descriptive Statistics")
    logger.info("="*50)
    
    # Numerical columns summary
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    logger.info(f"\nDescriptive statistics for {len(numerical_cols)} numerical columns:")
    logger.info("-" * 60)
    
    desc_stats = df[numerical_cols].describe()
    logger.info(f"{desc_stats.round(2)}")
    
    logger.info("\nDESCRIPTIVE STATISTICS COMPLETE!")
    return df

def analyze_target_variables(df):
    """
    Analyze target variables (demurrage flag and amount)
    
    Args:
        df (pd.DataFrame): Input dataset
        
    Returns:
        pd.DataFrame: Dataset (unchanged)
    """
    logger.info(f"\n" + "="*40)
    logger.info("TARGET VARIABLES ANALYSIS")
    logger.info("="*40)
    
    # Demurrage Flag Analysis
    logger.info(f"\n1. Demurrage Flag Distribution:")
    flag_counts = df['demurrage_flag'].value_counts().sort_index()
    for flag, count in flag_counts.items():
        percentage = (count/len(df)) * 100
        logger.info(f"   Flag {flag}: {count} records ({percentage:.1f}%)")
    
    # Demurrage Amount Analysis
    logger.info(f"\n2. Demurrage Amount Analysis:")
    demurrage_stats = df['demurrage_amount_usd'].describe()
    logger.info(f"   Mean: ${demurrage_stats['mean']:,.2f}")
    logger.info(f"   Median: ${demurrage_stats['50%']:,.2f}")
    logger.info(f"   Max: ${demurrage_stats['max']:,.2f}")
    logger.info(f"   Records with demurrage > 0: {(df['demurrage_amount_usd'] > 0).sum()}")
    
    return df

def analyze_categorical_variables(df):
    """
    Analyze categorical variables with frequency distributions
    
    Args:
        df (pd.DataFrame): Input dataset
        
    Returns:
        pd.DataFrame: Dataset (unchanged)
    """
    logger.info(f"\n" + "="*40)
    logger.info("CATEGORICAL VARIABLES ANALYSIS")
    logger.info("="*40)
    
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    logger.info(f"\nAnalyzing {len(categorical_cols)} categorical columns:")
    
    for col in categorical_cols:
        logger.info(f"\n{col}:")
        value_counts = df[col].value_counts()
        unique_count = len(value_counts)
        logger.info(f"   Unique values: {unique_count}")
        
        # Show top 5 most frequent values
        top_5 = value_counts.head(5)
        for value, count in top_5.items():
            percentage = (count/len(df)) * 100
            logger.info(f"   '{value}': {count} ({percentage:.1f}%)")
        
        if unique_count > 5:
            logger.info(f"   ... and {unique_count - 5} other values")
    
    return df

def calculate_business_metrics(df):
    """
    Calculate and analyze key maritime business metrics
    
    Args:
        df (pd.DataFrame): Input dataset with timestamp columns converted
        
    Returns:
        pd.DataFrame: Dataset with new business metric columns added
    """
    logger.info("\n" + "="*50)
    logger.info("Step 7: Key Maritime Business Metrics")
    logger.info("="*50)
    
    # Calculate voyage duration (arrival - departure)
    df['voyage_duration_days'] = (df['arrival_ts'] - df['depart_ts']).dt.total_seconds() / (24 * 3600)
    
    # Calculate NOR processing time (accepted - tender)
    df['nor_processing_hours'] = (df['nor_accepted_ts'] - df['nor_tender_ts']).dt.total_seconds() / 3600
    
    # Calculate laytime efficiency (used/allowed ratio)
    df['laytime_efficiency'] = df['used_laytime_h'] / df['allowed_laytime_h']
    
    logger.info(f"Created new business metrics:")
    logger.info(f"   voyage_duration_days: Mean = {df['voyage_duration_days'].mean():.1f} days")
    logger.info(f"   nor_processing_hours: Mean = {df['nor_processing_hours'].mean():.1f} hours")
    logger.info(f"   laytime_efficiency: Mean = {df['laytime_efficiency'].mean():.2f}")
    
    # Laytime performance analysis
    logger.info(f"\nLaytime Performance Analysis:")
    efficient_voyages = (df['laytime_efficiency'] <= 1.0).sum()
    inefficient_voyages = (df['laytime_efficiency'] > 1.0).sum()
    logger.info(f"   Efficient voyages (≤100% laytime used): {efficient_voyages} ({efficient_voyages/len(df)*100:.1f}%)")
    logger.info(f"   Inefficient voyages (>100% laytime used): {inefficient_voyages} ({inefficient_voyages/len(df)*100:.1f}%)")
    
    logger.info(f"\nKEY BUSINESS METRICS COMPLETE!")
    return df

def analyze_port_performance(df):
    """
    Analyze port performance metrics and insights
    
    Args:
        df (pd.DataFrame): Input dataset with port data
        
    Returns:
        tuple: (avg_efficiency_by_port, avg_congestion_by_port) for use in summary
    """
    logger.info(f"\nPort Performance Insights:")
    
    # Port efficiency analysis
    avg_efficiency_by_port = df.groupby('discharge_port')['port_efficiency_index'].mean().sort_values(ascending=False)
    logger.info(f"   Top 3 most efficient ports:")
    for i, (port, efficiency) in enumerate(avg_efficiency_by_port.head(3).items(), 1):
        logger.info(f"   {i}. {port}: {efficiency:.1f}")
    
    # Port congestion analysis
    avg_congestion_by_port = df.groupby('discharge_port')['port_congestion_index'].mean().sort_values()
    logger.info(f"   Top 3 least congested ports:")
    for i, (port, congestion) in enumerate(avg_congestion_by_port.head(3).items(), 1):
        logger.info(f"   {i}. {port}: {congestion:.1f}")
    
    return avg_efficiency_by_port, avg_congestion_by_port

def correlation_analysis(df):
    """
    Perform correlation analysis with target variables
    
    Args:
        df (pd.DataFrame): Input dataset with all features
        
    Returns:
        tuple: (flag_correlations, amount_correlations) for use in summary
    """
    logger.info("\n" + "="*50)
    logger.info("Step 8: Correlation Analysis")
    logger.info("="*50)
    
    # Focus on correlations with our target variables
    logger.info("\nCorrelations with Demurrage Flag:")
    flag_correlations = df[KEY_FEATURES + ['demurrage_flag']].corr()['demurrage_flag'].drop('demurrage_flag').sort_values(key=abs, ascending=False)
    for feature, corr in flag_correlations.head(8).items():
        direction = "Positive" if corr > 0 else "Negative"
        strength = "Strong" if abs(corr) > 0.5 else "Moderate" if abs(corr) > 0.3 else "Weak"
        logger.info(f"   {feature}: {corr:.3f} ({direction}, {strength})")
    
    logger.info("\nCorrelations with Demurrage Amount:")
    amount_correlations = df[KEY_FEATURES + ['demurrage_amount_usd']].corr()['demurrage_amount_usd'].drop('demurrage_amount_usd').sort_values(key=abs, ascending=False)
    for feature, corr in amount_correlations.head(8).items():
        direction = "Positive" if corr > 0 else "Negative"
        strength = "Strong" if abs(corr) > 0.5 else "Moderate" if abs(corr) > 0.3 else "Weak"
        logger.info(f"   {feature}: {corr:.3f} ({direction}, {strength})")
    
    logger.info("CORRELATION ANALYSIS COMPLETE!")
    return flag_correlations, amount_correlations

def generate_key_insights(df, flag_correlations, amount_correlations):
    """
    Generate key insights from correlation analysis
    
    Args:
        df (pd.DataFrame): Input dataset
        flag_correlations (pd.Series): Correlations with demurrage flag
        amount_correlations (pd.Series): Correlations with demurrage amount
        
    Returns:
        pd.DataFrame: Dataset (unchanged)
    """
    logger.info(f"\n" + "="*40)
    logger.info("KEY INSIGHTS FROM CORRELATIONS")
    logger.info("="*40)
    
    top_flag_driver = flag_correlations.abs().idxmax()
    top_amount_driver = amount_correlations.abs().idxmax()
    
    logger.info(f"Top driver for demurrage occurrence: {top_flag_driver} (r={flag_correlations[top_flag_driver]:.3f})")
    logger.info(f"Top driver for demurrage amount: {top_amount_driver} (r={amount_correlations[top_amount_driver]:.3f})")
    
    return df

def generate_final_summary(df, avg_efficiency_by_port, avg_congestion_by_port):
    """
    Generate comprehensive final summary of preprocessing and EDA
    
    Args:
        df (pd.DataFrame): Final processed dataset
        avg_efficiency_by_port (pd.Series): Port efficiency rankings
        avg_congestion_by_port (pd.Series): Port congestion rankings
        
    Returns:
        pd.DataFrame: Dataset (unchanged)
    """
    logger.info("\n" + "="*60)
    logger.info("PREPROCESSING & EDA SUMMARY")
    logger.info("="*60)
    
    logger.info("DATA PREPARATION COMPLETE:")
    logger.info(f"   Clean dataset: {df.shape[0]} records, {df.shape[1]} features")
    logger.info(f"   Target distribution: {(df['demurrage_flag']==1).sum()} with demurrage ({(df['demurrage_flag']==1).sum()/len(df)*100:.1f}%)")
    logger.info(f"   Created business features: voyage_duration_days, nor_processing_hours, laytime_efficiency")
    
    logger.info(f"\nKEY BUSINESS INSIGHTS:")
    logger.info(f"   {(df['laytime_efficiency'] > 1.0).sum()} voyages ({(df['laytime_efficiency'] > 1.0).sum()/len(df)*100:.1f}%) exceed allowed laytime")
    logger.info(f"   Average demurrage cost: ${df[df['demurrage_amount_usd']>0]['demurrage_amount_usd'].mean():,.0f}")
    logger.info(f"   Most efficient port: {avg_efficiency_by_port.index[0]} ({avg_efficiency_by_port.iloc[0]:.1f})")
    logger.info(f"   Least congested port: {avg_congestion_by_port.index[0]} ({avg_congestion_by_port.iloc[0]:.1f})")
    
    logger.info(f"\nProcessing complete.")
    return df

def run_complete_eda_analysis(df):
    """
    Run the complete EDA analysis pipeline
    
    Args:
        df (pd.DataFrame): Input dataset (after data loading and validation)
        
    Returns:
        pd.DataFrame: Dataset with business metrics added
    """
    logger.info("\n" + "="*60)
    logger.info("PHASE 2: EXPLORATORY DATA ANALYSIS (EDA)")
    logger.info("="*60)
    
    # Run all analysis steps
    df = generate_descriptive_stats(df)
    df = analyze_target_variables(df)
    df = analyze_categorical_variables(df)
    df = calculate_business_metrics(df)
    
    # Port analysis
    avg_efficiency_by_port, avg_congestion_by_port = analyze_port_performance(df)
    
    # Correlation analysis
    flag_correlations, amount_correlations = correlation_analysis(df)
    
    # Generate insights
    df = generate_key_insights(df, flag_correlations, amount_correlations)
    
    # Final summary
    df = generate_final_summary(df, avg_efficiency_by_port, avg_congestion_by_port)
    
    return df