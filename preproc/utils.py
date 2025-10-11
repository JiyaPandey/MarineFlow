# =============================================================================
# MARINEFLOW CORE UTILITIES
# =============================================================================
# Purpose: All reusable functions used across the entire project
# Contains: Smart logging, CSV operations, basic validations
# =============================================================================

import pandas as pd
import numpy as np
import logging
import os
from datetime import datetime
import pickle
import json

# =============================================================================
# SMART LOGGING SYSTEM
# =============================================================================

def check_logs_directory():
    """
    Check if logs directory exists, create if not
    
    Returns:
        str: Path to logs directory
    """
    # Always create logs in the same directory as this script (preproc/)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    logs_dir = os.path.join(script_dir, 'logs')
    
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
        print(f"📁 Created logs directory: {logs_dir}")
    else:
        print(f"📁 Logs directory exists: {logs_dir}")
    
    return logs_dir

def setup_logging(log_level='INFO', custom_filename=None):
    """
    Smart logging setup - checks if logs exist, appends or creates
    
    Args:
        log_level (str): Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
        custom_filename (str): Custom log filename (auto-generated if None)
        
    Returns:
        logging.Logger: Configured logger instance
    """
    # 1. Check/create logs directory
    logs_dir = check_logs_directory()
    
    # 2. Generate filename
    if custom_filename:
        log_filename = custom_filename
    else:
        timestamp = datetime.now().strftime("%Y%m%d")
        log_filename = f"marineflow_{timestamp}.log"
    
    log_filepath = os.path.join(logs_dir, log_filename)
    
    # 3. Check if log file exists
    if os.path.exists(log_filepath):
        print(f"📄 Log file exists - will append to: {log_filename}")
        file_mode = 'a'  # Append mode
    else:
        print(f"📄 Creating new log file: {log_filename}")
        file_mode = 'w'  # Write mode
    
    # 4. Configure logging
    # Clear any existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # Setup new logging configuration
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filepath, mode=file_mode),
            logging.StreamHandler()  # Also print to terminal
        ],
        force=True  # Override any existing configuration
    )
    
    logger = logging.getLogger('MarineFlow')
    logger.info(f"=" * 50)
    logger.info(f"LOGGING INITIALIZED - Level: {log_level}")
    logger.info(f"Log file: {log_filepath} (mode: {file_mode})")
    logger.info(f"=" * 50)
    
    return logger

def get_logger(name='MarineFlow'):
    """
    Get logger instance (use after setup_logging)
    
    Args:
        name (str): Logger name
        
    Returns:
        logging.Logger: Logger instance
    """
    return logging.getLogger(name)

# =============================================================================
# CSV OPERATIONS
# =============================================================================

def load_csv(filepath, encoding='utf-8', parse_dates=None, log_details=True):
    """
    Smart CSV loader with error handling and logging
    
    Args:
        filepath (str): Path to CSV file
        encoding (str): File encoding
        parse_dates (list): Columns to parse as dates
        log_details (bool): Whether to log detailed info
        
    Returns:
        pd.DataFrame: Loaded dataset
    """
    logger = get_logger()
    
    try:
        logger.info(f"Loading CSV from: {filepath}")
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        
        # Get file size
        file_size = os.path.getsize(filepath) / (1024 * 1024)  # MB
        logger.info(f"File size: {file_size:.2f} MB")
        
        df = pd.read_csv(filepath, encoding=encoding, parse_dates=parse_dates)
        
        if log_details:
            logger.info(f"CSV loaded successfully: {df.shape} (rows, cols)")
            logger.info(f"Memory usage: {df.memory_usage(deep=True).sum() / (1024*1024):.2f} MB")
            logger.info(f"Columns: {list(df.columns[:5])}{'...' if len(df.columns) > 5 else ''}")
        
        return df
        
    except Exception as e:
        logger.error(f"Error loading CSV {filepath}: {str(e)}")
        raise

def save_csv(df, filepath, index=False, log_details=True):
    """
    Save DataFrame to CSV with logging
    
    Args:
        df (pd.DataFrame): DataFrame to save
        filepath (str): Output file path
        index (bool): Whether to save index
        log_details (bool): Whether to log detailed info
    """
    logger = get_logger()
    
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        df.to_csv(filepath, index=index)
        
        if log_details:
            file_size = os.path.getsize(filepath) / (1024 * 1024)  # MB
            logger.info(f"CSV saved: {filepath}")
            logger.info(f"Saved: {df.shape[0]} rows, {df.shape[1]} cols, {file_size:.2f} MB")
        
    except Exception as e:
        logger.error(f"Error saving CSV {filepath}: {str(e)}")
        raise

def load_multiple_csvs(file_dict, log_summary=True):
    """
    Load multiple CSV files at once
    
    Args:
        file_dict (dict): {'name': 'filepath'} mapping
        log_summary (bool): Whether to log summary
        
    Returns:
        dict: {'name': DataFrame} mapping
    """
    logger = get_logger()
    datasets = {}
    
    for name, filepath in file_dict.items():
        try:
            datasets[name] = load_csv(filepath, log_details=False)
            if log_summary:
                logger.info(f"Loaded {name}: {datasets[name].shape}")
        except Exception as e:
            logger.error(f"Failed to load {name}: {str(e)}")
            datasets[name] = None
    
    if log_summary:
        successful_loads = len([d for d in datasets.values() if d is not None])
        logger.info(f"Loaded {successful_loads}/{len(file_dict)} datasets successfully")
    
    return datasets

# =============================================================================
# BASIC VALIDATION UTILITIES
# =============================================================================

def validate_dataframe(df, required_columns=None, min_rows=1):
    """
    Basic DataFrame validation
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (list): Required column names
        min_rows (int): Minimum number of rows
        
    Returns:
        dict: Validation results
    """
    logger = get_logger()
    
    validation_results = {
        'is_valid': True,
        'errors': [],
        'warnings': []
    }
    
    # Check if DataFrame is empty
    if df is None or df.empty:
        validation_results['is_valid'] = False
        validation_results['errors'].append("DataFrame is empty or None")
        return validation_results
    
    # Check minimum rows
    if len(df) < min_rows:
        validation_results['is_valid'] = False
        validation_results['errors'].append(f"DataFrame has {len(df)} rows, minimum required: {min_rows}")
    
    # Check required columns
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            validation_results['is_valid'] = False
            validation_results['errors'].append(f"Missing required columns: {missing_cols}")
    
    # Check for excessive missing values
    missing_percentage = (df.isnull().sum() / len(df) * 100)
    high_missing = missing_percentage[missing_percentage > 50]
    if not high_missing.empty:
        validation_results['warnings'].append(f"Columns with >50% missing values: {high_missing.to_dict()}")
    
    # Log validation results
    if validation_results['is_valid']:
        logger.info("DataFrame validation passed")
    else:
        logger.error(f"DataFrame validation failed: {validation_results['errors']}")
    
    if validation_results['warnings']:
        logger.warning(f"Validation warnings: {validation_results['warnings']}")
    
    return validation_results

def quick_data_summary(df, name="DataFrame"):
    """
    Generate and log quick data summary
    
    Args:
        df (pd.DataFrame): DataFrame to summarize
        name (str): Name for logging
        
    Returns:
        dict: Summary statistics
    """
    logger = get_logger()
    
    summary = {
        'name': name,
        'shape': df.shape,
        'memory_mb': df.memory_usage(deep=True).sum() / (1024*1024),
        'missing_values': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum(),
        'numeric_columns': len(df.select_dtypes(include=[np.number]).columns),
        'categorical_columns': len(df.select_dtypes(include=['object']).columns),
        'datetime_columns': len(df.select_dtypes(include=['datetime64']).columns)
    }
    
    logger.info(f"{name} Summary:")
    logger.info(f"  Shape: {summary['shape']}")
    logger.info(f"  Memory: {summary['memory_mb']:.2f} MB")
    logger.info(f"  Missing values: {summary['missing_values']}")
    logger.info(f"  Duplicates: {summary['duplicate_rows']}")
    logger.info(f"  Numeric: {summary['numeric_columns']}, Categorical: {summary['categorical_columns']}, DateTime: {summary['datetime_columns']}")
    
    return summary

# =============================================================================
# FILE SYSTEM UTILITIES
# =============================================================================

def ensure_directory(directory_path):
    """
    Ensure directory exists, create if not
    
    Args:
        directory_path (str): Directory path
        
    Returns:
        str: Absolute path to directory
    """
    abs_path = os.path.abspath(directory_path)
    if not os.path.exists(abs_path):
        os.makedirs(abs_path)
        get_logger().info(f"Created directory: {abs_path}")
    
    return abs_path

def save_pickle(obj, filepath):
    """
    Save object to pickle file with logging
    
    Args:
        obj: Object to save
        filepath (str): Output file path
    """
    logger = get_logger()
    
    try:
        ensure_directory(os.path.dirname(filepath))
        with open(filepath, 'wb') as f:
            pickle.dump(obj, f)
        
        file_size = os.path.getsize(filepath) / (1024 * 1024)  # MB
        logger.info(f"Pickle saved: {filepath} ({file_size:.2f} MB)")
        
    except Exception as e:
        logger.error(f"Error saving pickle {filepath}: {str(e)}")
        raise

def load_pickle(filepath):
    """
    Load object from pickle file with logging
    
    Args:
        filepath (str): Pickle file path
        
    Returns:
        object: Loaded object
    """
    logger = get_logger()
    
    try:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Pickle file not found: {filepath}")
        
        with open(filepath, 'rb') as f:
            obj = pickle.load(f)
        
        logger.info(f"Pickle loaded: {filepath}")
        return obj
        
    except Exception as e:
        logger.error(f"Error loading pickle {filepath}: {str(e)}")
        raise

