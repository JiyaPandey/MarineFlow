# =============================================================================
# MARINEFLOW DATA EXPORTER - SAVE DATASETS AND PIPELINE FUNCTIONS
# =============================================================================
# Purpose: Save train/validation/test datasets, preprocessing pipeline, and documentation
# Functions: save_train_test_splits, save_preprocessing_pipeline, create_feature_documentation
# =============================================================================

import pandas as pd
import pickle
import logging
from config import FILE_PATHS, DATA_LEAKAGE_COLS

# Configure logging
logger = logging.getLogger(__name__)

def save_train_test_splits(X_train_scaled, X_val_scaled, X_test_scaled, 
                          y_class_train, y_class_val, y_class_test,
                          y_reg_train, y_reg_val, y_reg_test):
    """
    Save train, validation, and test datasets as complete ML-ready CSV files
    
    Args:
        X_train_scaled, X_val_scaled, X_test_scaled: Scaled feature matrices
        y_class_train, y_class_val, y_class_test: Classification targets
        y_reg_train, y_reg_val, y_reg_test: Regression targets
        
    Returns:
        dict: Summary of saved datasets
    """
    logger.info("\n" + "="*50)
    logger.info("SECTION 9: SAVE COMPLETE ML-READY DATASETS")
    logger.info("="*50)
    logger.info("Saving complete datasets with X and Y combined for easy ML training...")
    
    # Clean features by removing any data leakage columns
    def clean_features(X_data):
        clean_X = X_data.copy()
        leakage_found = []
        for col in DATA_LEAKAGE_COLS:
            if col in clean_X.columns:
                clean_X = clean_X.drop(columns=[col])
                leakage_found.append(col)
        
        if leakage_found:
            logger.warning(f"Removed data leakage columns: {leakage_found}")
        
        return clean_X
    
    # Clean all feature sets
    X_train_clean = clean_features(X_train_scaled)
    X_val_clean = clean_features(X_val_scaled)
    X_test_clean = clean_features(X_test_scaled)
    
    # Reset indices to ensure proper alignment
    X_train_clean = X_train_clean.reset_index(drop=True)
    X_val_clean = X_val_clean.reset_index(drop=True)
    X_test_clean = X_test_clean.reset_index(drop=True)
    
    y_class_train = y_class_train.reset_index(drop=True)
    y_class_val = y_class_val.reset_index(drop=True)
    y_class_test = y_class_test.reset_index(drop=True)
    
    y_reg_train = y_reg_train.reset_index(drop=True)
    y_reg_val = y_reg_val.reset_index(drop=True)
    y_reg_test = y_reg_test.reset_index(drop=True)
    
    # Combine features and targets into complete datasets
    train_data = pd.concat([X_train_clean, y_class_train, y_reg_train], axis=1)
    val_data = pd.concat([X_val_clean, y_class_val, y_reg_val], axis=1)
    test_data = pd.concat([X_test_clean, y_class_test, y_reg_test], axis=1)
    
    # Save complete ML-ready datasets
    train_data.to_csv('train_data.csv', index=False)
    val_data.to_csv('validation_data.csv', index=False)
    test_data.to_csv('test_data.csv', index=False)
    
    logger.info(f"✅ Training data saved: train_data.csv ({len(train_data)} samples, {len(X_train_clean.columns)} features + 2 targets)")
    logger.info(f"✅ Validation data saved: validation_data.csv ({len(val_data)} samples, {len(X_val_clean.columns)} features + 2 targets)")
    logger.info(f"✅ Test data saved: test_data.csv ({len(test_data)} samples, {len(X_test_clean.columns)} features + 2 targets)")
    
    # Create a simple usage guide
    usage_guide = f"""
# MarineFlow ML-Ready Datasets Usage Guide

## Quick Start for Training:
```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Load training data
train_df = pd.read_csv('train_data.csv')

# Separate features and targets
feature_cols = train_df.columns[:-2]  # All except last 2 columns
X_train = train_df[feature_cols]
y_classification = train_df['demurrage_flag']  # Binary classification
y_regression = train_df['demurrage_amount_usd']  # Regression

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_classification)
```

## Dataset Information:
- Training samples: {len(train_data)}
- Validation samples: {len(val_data)}
- Test samples: {len(test_data)}
- Features: {len(X_train_clean.columns)} (clean, no data leakage)
- Targets: 2 (demurrage_flag, demurrage_amount_usd)

## Files Generated:
- train_data.csv: Complete training dataset (features + targets)
- validation_data.csv: Complete validation dataset (features + targets)  
- test_data.csv: Complete test dataset (features + targets)

## Target Variables:
- demurrage_flag: Binary (0/1) - Classification task
- demurrage_amount_usd: Continuous - Regression task
"""
    
    with open('ML_USAGE_GUIDE.txt', 'w') as f:
        f.write(usage_guide)
    
    logger.info("📋 Usage guide created: ML_USAGE_GUIDE.txt")
    
    return {
        'train_samples': len(train_data),
        'val_samples': len(val_data),
        'test_samples': len(test_data),
        'total_features': len(X_train_clean.columns),
        'files_created': ['train_data.csv', 'validation_data.csv', 'test_data.csv', 'ML_USAGE_GUIDE.txt']
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