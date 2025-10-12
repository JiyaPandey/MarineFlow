#!/usr/bin/env python3
# =============================================================================
# MARINEFLOW MAIN EXECUTION SCRIPT
# =============================================================================
# Purpose: One-click execution of entire preprocessing pipeline
# Usage: python main.py
# =============================================================================

import os
import sys
from datetime import datetime

# Import our modules
from utils import setup_logging, load_csv, save_csv
from data_cleaner import clean_data
from data_feature_eng import engineer_features
from data_exporter import export_datasets
from data_eda import generate_eda_report
import config

def main():
    """
    Main execution function - runs entire pipeline
    """
    print("=" * 60)
    print(" MARINEFLOW PREPROCESSING PIPELINE STARTED")
    print("=" * 60)
    
    # 1. INITIALIZE LOGGING
    print("\n Step 1: Initializing logging system...")
    logger = setup_logging(log_level='INFO')
    logger.info("MarineFlow preprocessing pipeline started")
    
    # 2. LOAD RAW DATA
    print("\n Step 2: Loading raw data...")
    try:
        raw_data = load_csv(config.RAW_DATA_PATH)
        logger.info(f"Raw data loaded: {raw_data.shape}")
        print(f"    Raw data loaded: {raw_data.shape[0]} rows, {raw_data.shape[1]} columns")
    except Exception as e:
        logger.error(f"Failed to load raw data: {e}")
        print(f"    Error loading data: {e}")
        return False
    
    # 3. DATA CLEANING
    print("\n🧹 Step 3: Cleaning data...")
    try:
        cleaned_data = clean_data(raw_data)
        logger.info(f"Data cleaned: {raw_data.shape} to {cleaned_data.shape}")
        print(f"    Data cleaned: {cleaned_data.shape[0]} rows remaining")
    except Exception as e:
        logger.error(f"Data cleaning failed: {e}")
        print(f"    Data cleaning error: {e}")
        return False
    
    # 4. FEATURE ENGINEERING
    print("\n⚙️ Step 4: Engineering features...")
    try:
        featured_data = engineer_features(cleaned_data)
        logger.info(f"Features engineered: {featured_data.shape[1]} total features")
        print(f"    Features engineered: {featured_data.shape[1]} total features")
    except Exception as e:
        logger.error(f"Feature engineering failed: {e}")
        print(f"    Feature engineering error: {e}")
        return False
    
    # 5. EXPLORATORY DATA ANALYSIS
    print("\n Step 5: Running EDA analysis...")
    try:
        eda_report = generate_eda_report(featured_data)
        logger.info("EDA analysis completed successfully")
        print(f"    EDA analysis completed")
    except Exception as e:
        logger.warning(f"EDA analysis failed: {e}")
        print(f"   EDA analysis warning: {e}")
    
    # 6. EXPORT DATASETS
    print("\n Step 6: Exporting train/validation/test datasets...")
    try:
        export_results = export_datasets(featured_data)
        logger.info("Datasets exported successfully")
        print(f"    Datasets exported:")
        for dataset_name, info in export_results.items():
            if dataset_name != 'artifacts':  # Skip artifacts entry as it's just a path string
                print(f"      - {dataset_name}: {info['shape']} to {info['filepath']}")
        if 'artifacts' in export_results:
            print(f"      - Artifacts saved: {export_results['artifacts']}")
    except Exception as e:
        logger.error(f"Dataset export failed: {e}")
        print(f"   ❌ Export error: {e}")
        return False
    
    # 7. PIPELINE COMPLETE
    print("\n" + "=" * 60)
    print(" MARINEFLOW PREPROCESSING PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    logger.info("MarineFlow preprocessing pipeline completed successfully")
    
    print(f"\n Summary:")
    print(f"   • Raw data: {raw_data.shape}")
    print(f"   • Cleaned data: {cleaned_data.shape}")
    print(f"   • Final features: {featured_data.shape[1]}")
    print(f"   • Datasets exported: {len(export_results)}")
    print(f"   • Log files: Check 'logs/' directory")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        exit_code = 0 if success else 1
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n Unexpected error: {e}")
        sys.exit(1)