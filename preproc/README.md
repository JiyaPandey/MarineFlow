# MarineFlow Data Preprocessing

**Complete Guide to Data Cleaning and Feature Engineering**

## Overview

This module transforms raw maritime operational data into machine learning-ready datasets. It handles data quality issues, creates domain-specific features, and prepares train/validation/test splits for demurrage prediction models.

## Quick Start (TL;DR)

```powershell
# 1. Install dependencies
pip install pandas numpy scikit-learn

# 2. Run preprocessing (from main MarineFlow folder)
python .\preproc\preprocess_data.py
python .\preproc\feature_engineering.py

# 3. Check results
type .\preproc\feature_documentation.txt
```

**Result**: Ready-to-use train/validation/test CSV files with 47 features

## Quick Navigation

| What You Want to Do | File to Use                      | Quick Access                                                 |
| ------------------- | -------------------------------- | ------------------------------------------------------------ |
| Clean raw data      | `preprocess_data.py`             | [Click to see cleaning code](preprocess_data.py)             |
| Create new features | `feature_engineering.py`         | [Click to see feature code](feature_engineering.py)          |
| View feature stats  | `feature_documentation.txt`      | [Click to see feature importance](feature_documentation.txt) |
| See original data   | `marineflow_demurrage_synth.csv` | [Click to see raw data](marineflow_demurrage_synth.csv)      |
| Use training data   | `marineflow_train.csv`           | [Click to see training data](marineflow_train.csv)           |
| Use validation data | `marineflow_validation.csv`      | [Click to see validation data](marineflow_validation.csv)    |
| Use test data       | `marineflow_test.csv`            | [Click to see test data](marineflow_test.csv)                |
| Load saved pipeline | `preprocessing_pipeline.pkl`     | [Click to download pipeline](preprocessing_pipeline.pkl)     |

## What This Folder Does

This folder takes messy ship data and turns it into clean, ready-to-use datasets for predicting demurrage costs.

**Input**: Raw maritime data (600 records with timestamp errors and missing values)  
**Output**: Clean datasets split into train/validation/test with 47 engineered features  
**Processing Time**: ~2-3 minutes for complete pipeline

## Prerequisites

Before running the preprocessing scripts, make sure you have:

```powershell
# Install required Python packages
pip install pandas numpy scikit-learn
```

**Python Version**: Python 3.7 or higher  
**Required Libraries**: pandas, numpy, scikit-learn, datetime  
**Disk Space**: ~50MB for all generated files

## How to Run Everything

### Step 1: Clean the Raw Data

```powershell
# From the main MarineFlow folder, run:
python .\preproc\preprocess_data.py
```

**What this does:**

- Loads [marineflow_demurrage_synth.csv](marineflow_demurrage_synth.csv) (600 records)
- Fixes timestamp formats (converts text dates to proper dates)
- Removes 5 records with impossible timestamps (like arrival before departure)
- Creates basic time calculations (voyage duration, processing time)
- Checks all data for errors and inconsistencies
- **Result**: Clean dataset with 595 good records

### Step 2: Create Advanced Features

```powershell
# From the main MarineFlow folder, run:
python .\preproc\feature_engineering.py
```

**What this does:**

- Takes the clean data from Step 1
- Creates 47 total features (42 original + 5 new engineered features)
- Calculates efficiency ratios, risk scores, seasonal patterns
- Splits data into train (416), validation (89), and test (90) samples
- Saves everything as separate CSV files
- **Result**: Ready-to-use datasets for machine learning

### Step 3: View Your Results

```powershell
# See the feature importance and statistics:
type .\preproc\feature_documentation.txt
```

Or [Click to see feature stats online](feature_documentation.txt)

## Files in This Folder

### Scripts (Python Code)

- **[preprocess_data.py](preprocess_data.py)** - Main data cleaning script

  - **What it does**: Cleans raw data, fixes timestamps, removes errors
  - **Run with**: `python .\preproc\preprocess_data.py`
  - **Input**: `marineflow_demurrage_synth.csv`
  - **Output**: Clean data in memory (595 records)

- **[feature_engineering.py](feature_engineering.py)** - Feature creation script
  - **What it does**: Creates efficiency ratios, risk scores, seasonal patterns
  - **Run with**: `python .\preproc\feature_engineering.py`
  - **Input**: Clean data from Step 1
  - **Output**: Train/validation/test CSV files with 47 features

### Data Files

- **[marineflow_demurrage_synth.csv](marineflow_demurrage_synth.csv)** - Original raw data

  - **Size**: 600 records, 42 features
  - **Issues**: Has 5 records with bad timestamps
  - **Use**: Starting point for all processing

- **[marineflow_train.csv](marineflow_train.csv)** - Training dataset

  - **Size**: 416 samples, 47 features
  - **Use**: Train machine learning models
  - **Clean**: Yes, ready to use

- **[marineflow_validation.csv](marineflow_validation.csv)** - Validation dataset

  - **Size**: 89 samples, 47 features
  - **Use**: Tune model parameters
  - **Clean**: Yes, ready to use

- **[marineflow_test.csv](marineflow_test.csv)** - Test dataset
  - **Size**: 90 samples, 47 features
  - **Use**: Final model evaluation
  - **Clean**: Yes, ready to use

### Documentation & Artifacts

- **[feature_documentation.txt](feature_documentation.txt)** - Feature importance stats

  - **Contains**: Top 10 most important features, feature categories, statistics
  - **Use**: Understand which features matter most for predictions

- **[preprocessing_pipeline.pkl](preprocessing_pipeline.pkl)** - Saved pipeline
  - **Contains**: All preprocessing steps saved as Python object
  - **Use**: Apply same preprocessing to new data later

## Key Formulas Used

### Time Calculations

- **Voyage Duration**: `(arrival_time - departure_time) / 86400` (in days)
- **NOR Processing**: `(accepted_time - tender_time) / 3600` (in hours)

### Efficiency Ratios

- **Laytime Efficiency**: `used_time / allowed_time` (>1.0 means overtime)
- **Berth Wait Ratio**: `wait_hours / total_voyage_days`

### Risk Indicators

- **Overage Hours**: `max(0, used_time - allowed_time)` (billable overtime)
- **Port Performance**: `(efficiency_rating - congestion_rating) / 100`

## What We Discovered

### Data Quality Issues Fixed

- **Found**: 5 records where arrival time was before departure time
- **Action**: Removed these impossible records
- **Result**: 595 clean, reliable records

### Most Important Features for Predicting Demurrage

1. **Overage Hours Chargeable** (0.6672 importance) - Direct predictor of charges
2. **Laytime vs Allowed** (0.3465 importance) - Overtime indicator
3. **Predicted Risk Score** (0.1287 importance) - Combined risk rating

### Port Performance Rankings

- **Best Efficiency**: Calculated by averaging port efficiency scores
- **Least Congestion**: Calculated by averaging congestion scores
- **Method**: Grouped all voyages by port, calculated averages

### Feature Engineering Success

- **Created**: 47 total features from 42 original variables
- **Categories**: Temporal (12), Efficiency (10), Financial (6), Categorical (10)
- **Best New Features**: Port performance scores, efficiency ratios, risk indicators

## Performance Metrics

### Data Processing Stats

- **Records Processed**: 600 → 595 (99.2% retention rate)
- **Features Created**: 42 → 47 (12% feature expansion)
- **Data Quality**: 100% clean timestamps, 0% missing values after processing
- **Processing Speed**: ~30 seconds for cleaning, ~90 seconds for feature engineering

### Data Split Distribution

- **Training Set**: 416 samples (70%)
- **Validation Set**: 89 samples (15%)
- **Test Set**: 90 samples (15%)
- **Split Method**: Stratified by demurrage flag to maintain class balance

## Troubleshooting

### Common Issues & Solutions

**1. "File not found" error**

```
Error: FileNotFoundError: marineflow_demurrage_synth.csv
Solution: Make sure you're running from the main MarineFlow folder, not inside preproc/
```

**2. "Module not found" error**

```
Error: ModuleNotFoundError: No module named 'pandas'
Solution: pip install pandas numpy scikit-learn
```

**3. "Permission denied" error**

```
Error: PermissionError when saving files
Solution: Close any CSV files open in Excel, run terminal as administrator
```

**4. Different results each run**

```
Issue: Feature engineering gives different train/test splits
Solution: This is normal - the script uses random splitting. For consistent results, modify the random_state parameter
```

### Performance Issues

- **Slow processing**: Check available RAM (need ~2GB free)
- **Large file sizes**: Normal - CSV files are uncompressed for readability
- **Memory warnings**: Ignore pandas warnings about mixed data types

### Getting Help

- **For data issues**: Check [preprocess_data.py](preprocess_data.py) - it has detailed error checking
- **For feature questions**: See [feature_documentation.txt](feature_documentation.txt)
- **For code questions**: Both Python files have detailed comments explaining each step
- **For maritime domain questions**: Refer to the main project README formulas section

## Next Steps

### For Machine Learning

1. **Load processed data**: Use any of the train/validation/test CSV files

   ```python
   import pandas as pd
   train_data = pd.read_csv('preproc/marineflow_train.csv')
   ```

2. **Build models**: Use the 47 features to predict `demurrage_flag` or `demurrage_amount_usd`

   - **Classification**: Predict if demurrage will occur (binary: 0/1)
   - **Regression**: Predict exact demurrage amount in USD

3. **Feature selection**: Start with top 10 features from [feature_documentation.txt](feature_documentation.txt)

### For New Data

1. **Apply preprocessing**: Use [preprocessing_pipeline.pkl](preprocessing_pipeline.pkl) for consistency

   ```python
   import pickle
   pipeline = pickle.load(open('preproc/preprocessing_pipeline.pkl', 'rb'))
   new_clean_data = pipeline.transform(new_raw_data)
   ```

2. **Maintain same format**: Ensure new data has same 42 original columns as training data

### For Further Development

1. **Add more features**: Modify [feature_engineering.py](feature_engineering.py) to create domain-specific features
2. **Improve data quality**: Enhance [preprocess_data.py](preprocess_data.py) validation rules
3. **Scale up**: Pipeline handles larger datasets - tested up to 10,000 records

---

**Back to Main Project**: [Click to go to main README](../README.md)
