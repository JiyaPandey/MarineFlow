# =============================================================================
# MARINEFLOW EDA (EXPLORATORY DATA ANALYSIS)
# =============================================================================
# Purpose: Generate basic EDA analysis and logging (no plots or summary files)
# Contains: Statistical analysis, data quality checks
# =============================================================================

import pandas as pd
import numpy as np
from utils import get_logger, quick_data_summary
import config
import warnings
warnings.filterwarnings('ignore')

def generate_eda_report(df):
    """
    Generate basic EDA analysis (logging only, no plots or summary files)
    
    Args:
        df (pd.DataFrame): Input dataset
        
    Returns:
        dict: EDA report results
    """
    logger = get_logger()
    logger.info("Generating EDA analysis...")
    
    eda_results = {}
    
    # Basic data overview
    eda_results['overview'] = generate_data_overview(df)
    
    # Target variable analysis
    eda_results['target_analysis'] = analyze_target_variables(df)
    
    # Feature analysis
    eda_results['feature_analysis'] = analyze_features(df)
    
    # Correlation analysis
    eda_results['correlation_analysis'] = analyze_correlations(df)
    
    # Data quality analysis
    eda_results['quality_analysis'] = analyze_data_quality(df)
    
    logger.info("EDA analysis completed")
    return eda_results

def generate_data_overview(df):
    """
    Generate basic data overview
    
    Args:
        df (pd.DataFrame): Input dataset
        
    Returns:
        dict: Data overview statistics
    """
    logger = get_logger()
    
    overview = {
        'shape': df.shape,
        'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024 * 1024),
        'column_types': df.dtypes.value_counts().to_dict(),
        'missing_values_total': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum(),
        'unique_values_per_column': df.nunique().to_dict()
    }
    
    # Identify column categories
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    
    overview['column_categories'] = {
        'numeric': len(numeric_cols),
        'categorical': len(categorical_cols),
        'datetime': len(datetime_cols)
    }
    
    logger.info(f"Data overview: {overview['shape']}, {overview['memory_usage_mb']:.2f} MB")
    return overview

def analyze_target_variables(df):
    """
    Analyze target variables in detail
    
    Args:
        df (pd.DataFrame): Input dataset
        
    Returns:
        dict: Target variable analysis
    """
    logger = get_logger()
    target_analysis = {}
    
    for target_col in config.TARGET_COLS:
        if target_col in df.columns:
            logger.info(f"Analyzing target variable: {target_col}")
            
            target_stats = {
                'column_name': target_col,
                'data_type': str(df[target_col].dtype),
                'missing_values': df[target_col].isnull().sum(),
                'unique_values': df[target_col].nunique()
            }
            
            if target_col == 'demurrage_flag':
                # Binary classification analysis
                value_counts = df[target_col].value_counts()
                target_stats['distribution'] = value_counts.to_dict()
                target_stats['positive_rate'] = value_counts.get(1, 0) / len(df) * 100
                target_stats['balance_ratio'] = value_counts.min() / value_counts.max()
                
            elif target_col == 'demurrage_amount_usd':
                # Regression analysis
                target_stats['statistics'] = df[target_col].describe().to_dict()
                target_stats['zeros_count'] = (df[target_col] == 0).sum()
                target_stats['positive_values'] = (df[target_col] > 0).sum()
                target_stats['skewness'] = df[target_col].skew()
                target_stats['kurtosis'] = df[target_col].kurtosis()
            
            target_analysis[target_col] = target_stats
    
    return target_analysis

def analyze_features(df):
    """
    Analyze feature variables
    
    Args:
        df (pd.DataFrame): Input dataset
        
    Returns:
        dict: Feature analysis results
    """
    logger = get_logger()
    
    # Get feature columns (exclude targets and leakage)
    exclude_cols = config.TARGET_COLS + config.DATA_LEAKAGE_COLS
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    feature_analysis = {
        'total_features': len(feature_cols),
        'numeric_features': [],
        'categorical_features': [],
        'high_cardinality_features': [],
        'high_missing_features': []
    }
    
    for col in feature_cols:
        col_info = {
            'name': col,
            'dtype': str(df[col].dtype),
            'missing_count': df[col].isnull().sum(),
            'missing_percentage': df[col].isnull().sum() / len(df) * 100,
            'unique_count': df[col].nunique()
        }
        
        # Categorize features
        if df[col].dtype in [np.number]:
            col_info['statistics'] = df[col].describe().to_dict()
            col_info['outliers'] = detect_outliers(df[col])
            feature_analysis['numeric_features'].append(col_info)
            
        elif df[col].dtype == 'object':
            col_info['top_values'] = df[col].value_counts().head(5).to_dict()
            feature_analysis['categorical_features'].append(col_info)
            
            # High cardinality check
            if col_info['unique_count'] > 50:
                feature_analysis['high_cardinality_features'].append(col)
        
        # High missing values check
        if col_info['missing_percentage'] > 20:
            feature_analysis['high_missing_features'].append({
                'column': col,
                'missing_percentage': col_info['missing_percentage']
            })
    
    logger.info(f"Feature analysis: {len(feature_analysis['numeric_features'])} numeric, "
                f"{len(feature_analysis['categorical_features'])} categorical")
    
    return feature_analysis

def detect_outliers(series, method='iqr'):
    """
    Detect outliers in a numeric series
    
    Args:
        series (pd.Series): Numeric series
        method (str): 'iqr' or 'zscore'
        
    Returns:
        dict: Outlier statistics
    """
    if method == 'iqr':
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = series[(series < lower_bound) | (series > upper_bound)]
        
    elif method == 'zscore':
        z_scores = np.abs((series - series.mean()) / series.std())
        outliers = series[z_scores > 3]
    
    return {
        'count': len(outliers),
        'percentage': len(outliers) / len(series) * 100,
        'values': outliers.tolist()[:10]  # Show first 10 outliers
    }

def analyze_correlations(df):
    """
    Analyze correlations between features and targets
    
    Args:
        df (pd.DataFrame): Input dataset
        
    Returns:
        dict: Correlation analysis results
    """
    logger = get_logger()
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) < 2:
        logger.warning("Not enough numeric columns for correlation analysis")
        return {}
    
    # Calculate correlation matrix
    corr_matrix = df[numeric_cols].corr()
    
    correlation_analysis = {
        'correlation_matrix_shape': corr_matrix.shape,
        'high_correlations': [],
        'target_correlations': {}
    }
    
    # Find highly correlated feature pairs
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            col1 = corr_matrix.columns[i]
            col2 = corr_matrix.columns[j]
            corr_value = corr_matrix.iloc[i, j]
            
            if abs(corr_value) > 0.8 and col1 not in config.TARGET_COLS and col2 not in config.TARGET_COLS:
                correlation_analysis['high_correlations'].append({
                    'feature1': col1,
                    'feature2': col2,
                    'correlation': corr_value
                })
    
    # Analyze correlations with target variables
    for target_col in config.TARGET_COLS:
        if target_col in corr_matrix.columns:
            target_corrs = corr_matrix[target_col].drop(target_col).abs().sort_values(ascending=False)
            correlation_analysis['target_correlations'][target_col] = target_corrs.head(10).to_dict()
    
    logger.info(f"Found {len(correlation_analysis['high_correlations'])} high feature correlations")
    return correlation_analysis

def analyze_data_quality(df):
    """
    Comprehensive data quality analysis
    
    Args:
        df (pd.DataFrame): Input dataset
        
    Returns:
        dict: Data quality report
    """
    logger = get_logger()
    
    quality_analysis = {
        'completeness': {},
        'consistency': {},
        'validity': {},
        'uniqueness': {}
    }
    
    # Completeness analysis
    missing_by_column = df.isnull().sum()
    quality_analysis['completeness'] = {
        'total_missing': missing_by_column.sum(),
        'columns_with_missing': (missing_by_column > 0).sum(),
        'worst_columns': missing_by_column.nlargest(5).to_dict()
    }
    
    # Consistency analysis (data types, formats)
    quality_analysis['consistency'] = {
        'mixed_types': [],
        'date_format_issues': []
    }
    
    # Check for mixed types in object columns
    for col in df.select_dtypes(include=['object']).columns:
        sample_values = df[col].dropna().head(100)
        types = set(type(val).__name__ for val in sample_values)
        if len(types) > 1:
            quality_analysis['consistency']['mixed_types'].append({
                'column': col,
                'types_found': list(types)
            })
    
    # Validity analysis
    quality_analysis['validity'] = {
        'negative_values_in_positive_columns': [],
        'extreme_outliers': []
    }
    
    # Check for negative values in columns that should be positive
    positive_keywords = ['amount', 'qty', 'rate', 'hours', 'duration']
    for col in df.select_dtypes(include=[np.number]).columns:
        if any(keyword in col.lower() for keyword in positive_keywords):
            negative_count = (df[col] < 0).sum()
            if negative_count > 0:
                quality_analysis['validity']['negative_values_in_positive_columns'].append({
                    'column': col,
                    'negative_count': negative_count
                })
    
    # Uniqueness analysis
    quality_analysis['uniqueness'] = {
        'duplicate_rows': df.duplicated().sum(),
        'low_cardinality_columns': []
    }
    
    # Find columns with very low cardinality
    for col in df.columns:
        unique_ratio = df[col].nunique() / len(df)
        if unique_ratio < 0.01 and df[col].nunique() > 1:  # Less than 1% unique values
            quality_analysis['uniqueness']['low_cardinality_columns'].append({
                'column': col,
                'unique_count': df[col].nunique(),
                'unique_ratio': unique_ratio
            })
    
    logger.info("Data quality analysis completed")
    return quality_analysis