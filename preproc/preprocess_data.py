# =============================================================================
# MARINEFLOW DATA PREPROCESSING PIPELINE - MODULAR VERSION
# =============================================================================
# Purpose: Comprehensive data preprocessing and EDA for demurrage prediction
# Input: marineflow_demurrage_synth.csv (600 records, 42 features)
# Output: Clean dataset ready for feature engineering and ML modeling
# =============================================================================

import pandas as pd
import numpy as np
import logging

# Import utility modules
from data_loader import load_maritime_data, validate_required_columns, convert_timestamps
from data_validator import analyze_missing_values, analyze_data_types, check_data_consistency, generate_data_quality_summary
from analyzer import run_complete_eda_analysis

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """
    Main preprocessing pipeline - orchestrates the complete data preprocessing workflow
    
    Returns:
        pd.DataFrame: Clean, preprocessed dataset ready for feature engineering
    """
    try:
        # Phase 1: Data Loading and Validation
        df = load_maritime_data()
        df = validate_required_columns(df)
        df = convert_timestamps(df)
        
        # Phase 2: Data Quality Analysis
        df = analyze_missing_values(df)
        df = analyze_data_types(df)
        df = check_data_consistency(df)
        df = generate_data_quality_summary(df)
        
        # Phase 3: Exploratory Data Analysis
        df = run_complete_eda_analysis(df)
        
        return df
        
    except Exception as e:
        logger.error(f"Error in preprocessing pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    # Run the complete preprocessing pipeline
    processed_data = main()