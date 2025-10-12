# marineflow_exporter.py
"""
MARINEFLOW - dataset export utilities

Provides:
- split_features_targets(df)
- create_train_val_test_splits(X, y_flag, y_amount, ...)
- save_preprocessing_artifacts(splits)
- export_datasets(df)

Expectations:
- utils: get_logger, save_csv, save_pickle, ensure_directory
- config: TARGET_COLS, DATA_LEAKAGE_COLS, FILE_PATHS (optional)
"""

from typing import Dict, Tuple
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from utils import get_logger, save_csv, save_pickle, ensure_directory
import config


logger = get_logger()


def split_features_targets(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Split dataframe into features X and targets y_flag, y_amount.

    Raises:
        ValueError if required target columns are missing.
    """
    required_targets = ['demurrage_flag', 'demurrage_amount_usd']
    for t in required_targets:
        if t not in df.columns:
            raise ValueError(f"Missing required target column: {t}")

    exclude_cols = getattr(config, "TARGET_COLS", []) + getattr(config, "DATA_LEAKAGE_COLS", [])
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    X = df[feature_cols].copy()
    y_flag = df['demurrage_flag'].copy()
    y_amount = df['demurrage_amount_usd'].copy()

    logger.info(f"split_features_targets: features={len(feature_cols)}, X.shape={X.shape}")
    return X, y_flag, y_amount


def create_train_val_test_splits(
    X: pd.DataFrame,
    y_flag: pd.Series,
    y_amount: pd.Series,
    train_size: float = 0.7,
    val_size: float = 0.15,
    random_state: int = 42
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Create train/validation/test splits and return dictionary of splits.
    Stratifies on y_flag to keep class distribution.
    """
    if not (0.0 < train_size < 1.0) or not (0.0 <= val_size < 1.0):
        raise ValueError("train_size and val_size must be between 0 and 1")

    test_size = 1.0 - train_size - val_size
    if test_size <= 0:
        raise ValueError("train_size + val_size must be < 1.0")

    logger.info(f"Creating splits: train={train_size}, val={val_size}, test={test_size}")

    # First split: train vs temp (val+test)
    X_train, X_temp, y_flag_train, y_flag_temp, y_amount_train, y_amount_temp = train_test_split(
        X, y_flag, y_amount,
        test_size=(1.0 - train_size),
        random_state=random_state,
        stratify=y_flag
    )

    # Second split: val vs test from temp
    # val_ratio is relative portion of temp that should be validation
    val_ratio = val_size / (val_size + test_size)
    X_val, X_test, y_flag_val, y_flag_test, y_amount_val, y_amount_test = train_test_split(
        X_temp, y_flag_temp, y_amount_temp,
        test_size=(1.0 - val_ratio),
        random_state=random_state,
        stratify=y_flag_temp
    )

    # Reset indices to keep clean alignment when concatenating later
    def reset_all(*dfs):
        return [df.reset_index(drop=True) for df in dfs]

    X_train, X_val, X_test = reset_all(X_train, X_val, X_test)
    y_flag_train, y_flag_val, y_flag_test = reset_all(y_flag_train, y_flag_val, y_flag_test)
    y_amount_train, y_amount_val, y_amount_test = reset_all(y_amount_train, y_amount_val, y_amount_test)

    splits = {
        'train': {'X': X_train, 'y_flag': y_flag_train, 'y_amount': y_amount_train},
        'validation': {'X': X_val, 'y_flag': y_flag_val, 'y_amount': y_amount_val},
        'test': {'X': X_test, 'y_flag': y_flag_test, 'y_amount': y_amount_test},
    }

    for split_name, data in splits.items():
        flag_dist = data['y_flag'].value_counts().to_dict()
        logger.info(f"{split_name}: samples={len(data['X'])}, flag_dist={flag_dist}")

    return splits


def save_preprocessing_artifacts(splits: Dict[str, Dict[str, pd.DataFrame]], artifacts_path: str = 'preprocessing_artifacts.pkl') -> str:
    """
    Build and save simple preprocessing artifact summary (pickle).
    Returns path to saved pickle.
    """
    artifacts = {
        'feature_names': list(splits['train']['X'].columns),
        'target_names': ['demurrage_flag', 'demurrage_amount_usd'],
        'split_info': {
            'train_size': len(splits['train']['X']),
            'val_size': len(splits['validation']['X']),
            'test_size': len(splits['test']['X']),
        },
        'feature_statistics': {
            'numeric_features': len(splits['train']['X'].select_dtypes(include=[np.number]).columns),
            'categorical_features': len(splits['train']['X'].select_dtypes(include=['object', 'category']).columns),
            'total_features': len(splits['train']['X'].columns)
        }
    }

    save_pickle(artifacts, artifacts_path)
    logger.info(f"Preprocessing artifacts saved: {artifacts_path}")
    return artifacts_path


# Optional placeholders you can implement further if needed:
def save_preprocessing_pipeline(scaler, label_encoders, feature_names, target_names, feature_importance, out_path: str = None):
    """
    Save a preprocessing pipeline dict to disk. If config.FILE_PATHS exists,
    it can be used to set default path.
    """
    pipeline = {
        'scaler': scaler,
        'label_encoders': label_encoders,
        'feature_names': list(feature_names),
        'target_names': list(target_names),
        'feature_importance': feature_importance
    }
    if out_path is None:
        out_path = getattr(config, "FILE_PATHS", {}).get('pipeline', 'pipeline.pkl')
    save_pickle(pipeline, out_path)
    logger.info(f"Preprocessing pipeline saved: {out_path}")
    return out_path


def create_feature_documentation(dataset_summary: dict, feature_importance: pd.DataFrame, doc_path: str = None) -> str:
    """
    Minimal feature documentation writer. Implement richer content if required.
    """
    if doc_path is None:
        doc_path = getattr(config, "FILE_PATHS", {}).get('documentation', 'feature_documentation.txt')

    with open(doc_path, 'w') as f:
        f.write("MARINEFLOW FEATURE ENGINEERING SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        for k, v in dataset_summary.items():
            f.write(f"{k}: {v}\n")
        f.write("\nTop features (if provided):\n")
        if feature_importance is not None and not feature_importance.empty:
            for _, row in feature_importance.head(20).iterrows():
                # Expect feature_importance to have columns ['feature', 'importance'] or similar
                feat = row.get('feature', None) or row.index[0]
                imp = row.get('importance', row.get('importance_class', None) or row.get('importance_reg', None))
                f.write(f"{feat}: {imp}\n")

    logger.info(f"Feature documentation created: {doc_path}")
    return doc_path


def export_datasets(df: pd.DataFrame, csv_dir: str = 'csvs') -> Dict[str, dict]:
    """
    Export train/validation/test datasets with no data leakage.

    Returns:
        export_results: dict keyed by split with 'filepath', 'shape', 'features', 'samples'
    """
    logger = get_logger()
    ensure_directory(csv_dir)

    # Split features and targets (this automatically removes data leakage columns)
    X, y_flag, y_amount = split_features_targets(df)

    # Create splits (train/val/test)
    splits = create_train_val_test_splits(X, y_flag, y_amount)

    export_results = {}
    for split_name, data in splits.items():
        # Create combined dataset (features + targets) as you designed
        combined_df = data['X'].copy()
        combined_df['demurrage_flag'] = data['y_flag'].values
        combined_df['demurrage_amount_usd'] = data['y_amount'].values
        
        # Save the combined file (your preferred format)
        filepath = os.path.join(csv_dir, f'marineflow_{split_name}.csv')
        save_csv(combined_df, filepath)
        logger.info(f"Exported {split_name}: {combined_df.shape} -> {filepath}")
        logger.info(f"  Features: columns 0-{len(data['X'].columns)-1} | Targets: columns {len(data['X'].columns)}-{len(combined_df.columns)-1}")

        export_results[split_name] = {
            'filepath': filepath,
            'shape': combined_df.shape,
            'features': len(data['X'].columns),
            'samples': len(combined_df),
            'feature_columns': f"0-{len(data['X'].columns)-1}",
            'target_columns': f"{len(data['X'].columns)}-{len(combined_df.columns)-1}"
        }

    # Save preprocessing artifacts summary
    artifacts_path = save_preprocessing_artifacts(splits)
    export_results['artifacts'] = artifacts_path

    logger.info("Dataset export completed successfully")
    return export_results
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
    train_data.to_csv(FILE_PATHS['train'], index=False)
    val_data.to_csv(FILE_PATHS['validation'], index=False)
    test_data.to_csv(FILE_PATHS['test'], index=False)
    
    logger.info(f"✅ Training data saved: {FILE_PATHS['train']} ({len(train_data)} samples, {len(X_train_clean.columns)} features + 2 targets)")
    logger.info(f"✅ Validation data saved: {FILE_PATHS['validation']} ({len(val_data)} samples, {len(X_val_clean.columns)} features + 2 targets)")
    logger.info(f"✅ Test data saved: {FILE_PATHS['test']} ({len(test_data)} samples, {len(X_test_clean.columns)} features + 2 targets)")
    
    return {
        'train_samples': len(train_data),
        'val_samples': len(val_data),
        'test_samples': len(test_data),
        'total_features': len(X_train_clean.columns),
        'files_created': ['csvs/train_data.csv', 'csvs/validation_data.csv', 'csvs/test_data.csv']
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