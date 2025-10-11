# =============================================================================
# MARINEFLOW CONFIGURATION - CONSTANTS AND SETTINGS
# =============================================================================
# Purpose: Centralized configuration for all preprocessing and feature engineering
# Contains: Column names, file paths, and processing parameters
# =============================================================================

# Raw data file path
RAW_DATA_PATH = 'csvs/marineflow_demurrage_synth.csv'

# Timestamp columns that need datetime conversion
TIMESTAMP_COLS = ['depart_ts', 'arrival_ts', 'nor_tender_ts', 'nor_accepted_ts']

# Numeric columns that should be validated as numeric
NUMERIC_COLS = [
    'port_congestion_index', 'port_efficiency_index', 'allowed_laytime_h', 
    'used_laytime_h', 'demurrage_amount_usd', 'berth_wait_h', 'pilot_delay_h',
    'rain_hours', 'demurrage_rate_usd_per_day', 'sof_count', 'prior_claims_6m'
]

# Target columns for prediction
TARGET_COLS = ['demurrage_flag', 'demurrage_amount_usd']

# Data leakage columns that should NEVER be used as features
DATA_LEAKAGE_COLS = [
    'overage_hours_chargeable',  # Directly derived from target calculation
    'laytime_vs_allowed',        # Ratio that reveals overage status
    'demurrage_rate_usd_per_day', # Used in target calculation
    'pred_risk_score'            # Any prediction-based score
]

# File paths for input and output
FILE_PATHS = {
    'input': 'csvs/marineflow_demurrage_synth.csv',
    'train': 'csvs/marineflow_train.csv',
    'validation': 'csvs/marineflow_validation.csv', 
    'test': 'csvs/marineflow_test.csv',
    'pipeline': 'preprocessing_pipeline.pkl',
    'documentation': 'feature_documentation.txt'
}

# Key features for correlation analysis
KEY_FEATURES = [
    'port_congestion_index', 'port_efficiency_index', 'allowed_laytime_h', 
    'used_laytime_h', 'berth_wait_h', 'pilot_delay_h', 'rain_hours',
    'demurrage_rate_usd_per_day', 'voyage_duration_days', 'nor_processing_hours',
    'laytime_efficiency', 'sof_count', 'prior_claims_6m'
]

# Categorical columns for encoding
CATEGORICAL_COLS = [
    'vessel_type', 'cargo_type', 'loading_port', 'discharge_port', 
    'charter_type', 'weather_conditions'
]

# Validation ranges for business rules
VALIDATION_RANGES = {
    'port_efficiency_index': (0, 100),
    'port_congestion_index': (0, 100),
    'laytime_efficiency_max': 3.0,  # Maximum reasonable laytime efficiency
    'voyage_duration_max': 365  # Maximum reasonable voyage duration in days
}

# Logging configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
}