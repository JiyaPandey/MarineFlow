# =============================================================================
# MARINEFLOW DATA EXPORTER - SAVE DATASETS AND PIPELINE FUNCTIONS
# =============================================================================
# Purpose: Save train/validation/test datasets, preprocessing pipeline, and documentation
# Functions: save_train_test_splits, save_preprocessing_pipeline, create_feature_documentation
# =============================================================================

import pandas as pd
import pickle
import logging
from config import FILE_PATHS

# Configure logging
logger = logging.getLogger(__name__)

def save_train_test_splits(X_train_scaled, X_val_scaled, X_test_scaled, 
                          y_class_train, y_class_val, y_class_test,
                          y_reg_train, y_reg_val, y_reg_test):
    """
    Save train, validation, and test datasets to CSV files
    
    Args:
        X_train_scaled, X_val_scaled, X_test_scaled: Scaled feature matrices
        y_class_train, y_class_val, y_class_test: Classification targets
        y_reg_train, y_reg_val, y_reg_test: Regression targets
        
    Returns:
        dict: Summary of saved datasets
    """
    logger.info("\n" + "="*50)
    logger.info("SECTION 9: SAVE DATASETS AND PREPROCESSING PIPELINE")
    logger.info("="*50)
    
    # Training data
    train_data = pd.concat([X_train_scaled, y_class_train, y_reg_train], axis=1)
    train_data.to_csv(FILE_PATHS['train'], index=False)
    logger.info(f"Training data saved: {FILE_PATHS['train']} ({len(train_data)} samples)")
    
    # Validation data
    val_data = pd.concat([X_val_scaled, y_class_val, y_reg_val], axis=1)
    val_data.to_csv(FILE_PATHS['validation'], index=False)
    logger.info(f"Validation data saved: {FILE_PATHS['validation']} ({len(val_data)} samples)")
    
    # Test data
    test_data = pd.concat([X_test_scaled, y_class_test, y_reg_test], axis=1)
    test_data.to_csv(FILE_PATHS['test'], index=False)
    logger.info(f"Test data saved: {FILE_PATHS['test']} ({len(test_data)} samples)")
    
    return {
        'train_samples': len(train_data),
        'val_samples': len(val_data),
        'test_samples': len(test_data),
        'total_features': len(X_train_scaled.columns)
    }

def save_preprocessing_pipeline(scaler, label_encoders, feature_names, target_names, feature_importance):
    """
    Save preprocessing pipeline components to pickle file
    
    Args:
        scaler: Fitted StandardScaler
        label_encoders: Dictionary of fitted LabelEncoders
        feature_names: List of feature column names
        target_names: List of target column names
        feature_importance: DataFrame with feature importance scores
        
    Returns:
        str: Path to saved pipeline file
    """
    preprocessing_pipeline = {
        'scaler': scaler,
        'label_encoders': label_encoders,
        'feature_names': list(feature_names),
        'target_names': target_names,
        'feature_importance': feature_importance
    }
    
    with open(FILE_PATHS['pipeline'], 'wb') as f:
        pickle.dump(preprocessing_pipeline, f)
    
    logger.info(f"Preprocessing pipeline saved: {FILE_PATHS['pipeline']}")
    return FILE_PATHS['pipeline']

def create_feature_documentation(dataset_summary, feature_importance, encoding_summary=None):
    """
    Create comprehensive feature engineering documentation
    
    Args:
        dataset_summary: Dictionary with dataset statistics
        feature_importance: DataFrame with feature importance scores
        encoding_summary: Optional dictionary with encoding statistics
        
    Returns:
        str: Path to documentation file
    """
    # Calculate feature statistics
    total_features = dataset_summary.get('total_features', 0)
    original_features = 42  # Based on original dataset
    engineered_features = max(0, total_features - original_features)
    
    # Estimate feature categories (approximate)
    temporal_features = 12  # Based on temporal feature creation
    efficiency_features = 10  # Based on operational features
    financial_features = 6   # Based on financial features
    
    if encoding_summary:
        categorical_encoded = encoding_summary.get('one_hot_count', 0) + encoding_summary.get('label_encode_count', 0)
    else:
        categorical_encoded = max(0, engineered_features - temporal_features - efficiency_features - financial_features)
    
    feature_docs = {
        'total_features': total_features,
        'original_features': original_features,
        'engineered_features': engineered_features,
        'temporal_features': temporal_features,
        'efficiency_features': efficiency_features,
        'financial_features': financial_features,
        'categorical_encoded': categorical_encoded,
        'train_samples': dataset_summary.get('train_samples', 0),
        'val_samples': dataset_summary.get('val_samples', 0),
        'test_samples': dataset_summary.get('test_samples', 0)
    }
    
    with open(FILE_PATHS['documentation'], 'w') as f:
        f.write("MARINEFLOW FEATURE ENGINEERING SUMMARY\n")
        f.write("="*50 + "\n\n")
        for key, value in feature_docs.items():
            f.write(f"{key}: {value}\n")
        
        f.write(f"\nTop 10 Most Important Features:\n")
        for idx, row in feature_importance.head(10).iterrows():
            f.write(f"{row['feature']}: {row['importance_class']:.4f}\n")
    
    logger.info(f"Feature documentation created: {FILE_PATHS['documentation']}")
    return FILE_PATHS['documentation']

def save_complete_pipeline_outputs(X_train_scaled, X_val_scaled, X_test_scaled,
                                 y_class_train, y_class_val, y_class_test,
                                 y_reg_train, y_reg_val, y_reg_test,
                                 scaler, label_encoders, feature_names, feature_importance):
    """
    Save all outputs from the complete feature engineering pipeline
    
    Args:
        All components from feature engineering pipeline
        
    Returns:
        dict: Summary of all saved outputs
    """
    logger.info("Saving complete pipeline outputs...")
    
    # Save datasets
    dataset_summary = save_train_test_splits(
        X_train_scaled, X_val_scaled, X_test_scaled,
        y_class_train, y_class_val, y_class_test,
        y_reg_train, y_reg_val, y_reg_test
    )
    
    # Save preprocessing pipeline
    pipeline_path = save_preprocessing_pipeline(
        scaler, label_encoders, feature_names, 
        ['demurrage_flag', 'demurrage_amount_usd'], feature_importance
    )
    
    # Create documentation
    doc_path = create_feature_documentation(dataset_summary, feature_importance)
    
    logger.info("Feature engineering pipeline complete!")
    
    return {
        'datasets': dataset_summary,
        'pipeline_path': pipeline_path,
        'documentation_path': doc_path,
        'total_features': len(feature_names)
    }