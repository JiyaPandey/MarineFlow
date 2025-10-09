# =============================================================================
# MARINEFLOW FEATURE ENGINEERING PIPELINE - MODULAR VERSION
# =============================================================================
# Purpose: Advanced feature engineering for demurrage prediction
# Input: Preprocessed data from preprocess_data.py
# Output: ML-ready datasets with engineered features
# =============================================================================

import pandas as pd
import numpy as np
import logging
import warnings

# Import utility modules
from data_loader import load_preprocessed_data
from feature_builder import run_complete_feature_engineering
from data_exporter import save_complete_pipeline_outputs

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

def main():
    """
    Main feature engineering pipeline - orchestrates the complete feature engineering workflow
    
    Returns:
        dict: Summary of pipeline outputs and saved files
    """
    try:
        logger.info("MarineFlow Feature Engineering Started...")
        logger.info("="*50)
        
        # Phase 1: Load preprocessed data
        logger.info("Loading preprocessed data...")
        df = load_preprocessed_data()
        
        # Phase 2: Complete feature engineering pipeline
        logger.info("Running complete feature engineering pipeline...")
        (X_train_scaled, X_val_scaled, X_test_scaled,
         y_class_train, y_class_val, y_class_test,
         y_reg_train, y_reg_val, y_reg_test,
         scaler, label_encoders, feature_names, feature_importance) = run_complete_feature_engineering(df)
        
        # Phase 3: Save all outputs
        logger.info("Saving pipeline outputs...")
        pipeline_summary = save_complete_pipeline_outputs(
            X_train_scaled, X_val_scaled, X_test_scaled,
            y_class_train, y_class_val, y_class_test,
            y_reg_train, y_reg_val, y_reg_test,
            scaler, label_encoders, feature_names, feature_importance
        )
        
        logger.info("\n" + "="*60)
        logger.info("FEATURE ENGINEERING PIPELINE COMPLETE!")
        logger.info("="*60)
        logger.info(f"Total features: {pipeline_summary['total_features']}")
        logger.info(f"Training samples: {pipeline_summary['datasets']['train_samples']}")
        logger.info(f"Validation samples: {pipeline_summary['datasets']['val_samples']}")
        logger.info(f"Test samples: {pipeline_summary['datasets']['test_samples']}")
        logger.info(f"Files saved: {len(pipeline_summary)} output files")
        
        return pipeline_summary
        
    except Exception as e:
        logger.error(f"Error in feature engineering pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    # Run the complete feature engineering pipeline
    results = main()