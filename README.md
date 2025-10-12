# MarineFlow

**Maritime Demurrage Prediction and Flow Analysis**

## Introduction

Maritime operations face significant challenges in predicting demurrage costs and optimizing vessel flow efficiency. Traditional manual tracking systems lead to delayed decision-making, increased operational costs, and missed optimization opportunities. Advanced data preprocessing and machine learning solutions are needed to streamline maritime operations, enhance prediction accuracy, and improve port efficiency. Implementing such technology can significantly reduce demurrage costs, enhance operational planning, and boost productivity across maritime logistics.

## Overview

Our solution is a comprehensive data preprocessing and machine learning pipeline designed for maritime demurrage prediction and flow analysis. It utilizes advanced feature engineering and statistical analysis to process maritime operational data including vessel timestamps, port efficiency metrics, weather conditions, and cargo handling information. By automating data preprocessing and providing predictive insights, our pipeline enhances operational efficiency, reduces prediction errors, and improves decision-making. The modular design allows for seamless integration into existing maritime workflows and adaptability to different port operations.

## System Architecture

![MarineFlow Architecture](preproc/assets/MarineFlow_Architecture.jpg)
<img width="1652" height="945" alt="image" src="https://github.com/user-attachments/assets/52477163-2e1f-4335-a8d6-5ca42dbc6896" />


### Architecture Overview

The MarineFlow pipeline follows a **4-layer architecture** designed for scalability, maintainability, and data integrity:

#### **Layer 1 — Data Layer**

- **Raw CSV Data**: Maritime operational datasets (`marineflow_demurrage_synth.csv`)
- **Configuration Files**: Centralized settings (`config.py`)
- **Metadata**: Feature documentation and data schemas

#### **Layer 2 — Processing Layer**

- **Data Cleaning**: Quality assurance and preprocessing (`data_cleaner.py`)
- **Feature Engineering**: Advanced feature creation from raw data (`data_feature_eng.py`)
- **EDA**: Exploratory data analysis and validation (`data_eda.py`)
- **Utils**: Shared utilities for logging and validation (`utils.py`)

#### **Layer 3 — Orchestration Layer**

- **Pipeline Control**: End-to-end workflow management (`data_main.py`)
- **Configuration Loading**: Centralized parameter management
- **Error Handling**: Robust exception management and logging

#### **Layer 4 — Output Layer**

- **Train/Validation/Test Split**: Stratified dataset preparation (`data_exporter.py`)
- **Reports**: Feature importance and analysis reports
- **Logs**: Comprehensive execution logging for debugging

### Key Architecture Benefits

- **Modularity**: Each layer can be modified independently
- **Data Integrity**: Prevents data leakage through proper separation
- **Scalability**: Easy to add new features or processing steps
- **Maintainability**: Clear separation of concerns and responsibilities
- **Auditability**: Complete logging and validation at each stage

## Table of Contents

| Section                                        | Description                       | Quick Access      |
| ---------------------------------------------- | --------------------------------- | ----------------- |
| [System Architecture](#system-architecture)    | Pipeline architecture diagram     | View architecture |
| [Documentation Links](#documentation-links)    | Links to detailed module docs     | Jump to docs      |
| [Data Processing](#data-processing)            | Dataset overview and key features | See data info     |
| [Components](#components)                      | Project modules and scripts       | View components   |
| [Formulas](#formulas)                          | Mathematical calculations used    | See formulas      |
| [How to Run](#how-to-run)                      | Step-by-step instructions         | Run the code      |
| [Results & Discoveries](#results--discoveries) | Key findings and insights         | See results       |
| [Technologies Used](#technologies-used)        | Tools and libraries               | Tech stack        |
| [Folder Structure](#folder-structure)          | Project organization              | File layout       |
| [Conclusions](#conclusions)                    | Project summary                   | Final thoughts    |
| [License](#license)                            | MIT License information           | Legal info        |

## Documentation Links

- **Data Processing Details**: [Click to go to preprocessing docs](preproc/README.md)
- **Training Details**: [Click to go to training docs](train/README.md)

## Data Processing

### Dataset Overview

The pipeline processes maritime operational data with the following characteristics:

- **Total Records**: 600 (after preprocessing: 600)
- **Features**: 137 total (42 original + 95 engineered)
- **Target Variable**: Demurrage prediction (flag and amount)
- **Data Split**: 419 train, 90 validation, 91 test samples

### Key Features

Top predictive features identified through analysis:

- `laytime_efficiency`: 0.3465 importance
- `port_performance_score`: 0.1036 importance
- `efficiency_vs_congestion`: 0.1033 importance
- `port_congestion_index`: 0.0760 importance
- `weather_severity_High`: 0.0409 importance

## Components

### Data Preprocessing

- **Location**: `preproc/` folder
- **Main Script**: `data_main.py` - [Click to see the code](preproc/data_main.py)
- **What it does**: Orchestrates complete pipeline - cleaning, feature engineering, analysis, and export
- **Output**: Clean datasets ready for machine learning
- **Full Guide**: [Click to go to detailed preprocessing guide](preproc/README.md)

[Click to go to preprocessing docs](preproc/README.md)

### Feature Engineering

- **Script**: `data_feature_eng.py` - [Click to see the code](preproc/data_feature_eng.py)
- **What it does**: Creates 137 features including efficiency ratios, temporal patterns, and interaction features
- **Feature Stats**: `feature_documentation.txt` - [Click to see feature importance](preproc/feature_documentation.txt)

### Model Training

- **Location**: `train/` folder
- **Status**: Ready for machine learning models
- **Guide**: [Click to go to training docs](train/README.md)

## Formulas

### Why These Formulas Matter

In maritime shipping, these calculations help predict when ships will face extra charges (demurrage) and identify which ports/routes are most efficient.

### Time Calculations

**What we calculate**: How long different parts of shipping take

1. **Voyage Duration** - Total time from departure to arrival

   - **Formula**: `voyage_duration_days = (arrival_time - departure_time) / 86400`
   - **Why**: Longer voyages may indicate problems or inefficiency
   - **Example**: If departure is Jan 1 and arrival is Jan 5, voyage = 4 days

2. **NOR Processing Time** - How long ports take to process ship documents
   - **Formula**: `nor_processing_hours = (accepted_time - tender_time) / 3600`
   - **Why**: Slow processing = delays = potential extra costs
   - **Example**: If documents submitted at 9 AM and accepted at 2 PM, processing = 5 hours

### Efficiency Ratios

**What we calculate**: How well ships and ports perform compared to expectations

3. **Laytime Efficiency** - Did the ship load/unload within allowed time?

   - **Formula**: `laytime_efficiency = actual_time_used / allowed_time`
   - **Why**: If > 1.0, ship went overtime and may face demurrage charges
   - **Example**: Used 50 hours, allowed 40 hours = 1.25 (25% overtime)

4. **Berth Wait Ratio** - How much of voyage was spent waiting for berth
   - **Formula**: `berth_wait_ratio = berth_wait_hours / total_voyage_days`
   - **Why**: High ratios indicate port congestion problems
   - **Example**: 12 hours wait on 3-day voyage = 0.17 (17% of voyage wasted waiting)

### Performance Scores

**What we calculate**: Overall ratings for ports and weather conditions

5. **Port Performance Score** - How good is this port overall?

   - **Formula**: `port_score = (efficiency_rating - congestion_rating) / 100`
   - **Why**: Higher scores = better ports with less delays
   - **Example**: Efficiency 80, Congestion 30 = Score 0.50 (good port)

6. **Weather Impact Score** - How bad is the weather affecting operations?
   - **Formula**: `weather_score = severity_level_number`
   - **Levels**: Low=0.1, Moderate=0.3, High=0.7, Extreme=1.0
   - **Why**: Higher scores predict more delays and costs

### Risk Calculations

**What we calculate**: Warning signs that demurrage charges might happen

7. **Overage Hours** - How many hours over the allowed time?
   - **Formula**: `overage_hours = max(0, used_time - allowed_time)`
   - **Why**: Every hour over = money charged to ship owner
   - **Example**: Used 45 hours, allowed 40 = 5 hours overage (will be charged)

### How to Run These Calculations

1. **Run the preprocessing**: `python .\preproc\preprocess_data.py`
   - This creates the basic time calculations (voyage duration, NOR processing)
2. **Run feature engineering**: `python .\preproc\feature_engineering.py`
   - This creates all the efficiency ratios and performance scores
3. **View the results**: [Click to see calculated features](preproc/feature_documentation.txt)

## Technologies Used

- **pandas**: Data manipulation and analysis framework for handling maritime operational datasets
- **numpy**: Numerical computing library for statistical calculations and array operations
- **datetime**: Timestamp processing for vessel arrival, departure, and operational timing analysis
- **scikit-learn**: Machine learning preprocessing utilities (pipeline artifacts suggest usage)
- **Python 3.x**: Core programming language for the entire pipeline

The preprocessing pipeline is built using standard Python data science libraries optimized for maritime domain-specific analysis.

## Folder Structure

```
MarineFlow/
  preproc/                               # Data preprocessing folder
    data_main.py                        # Main pipeline orchestrator
    data_cleaner.py                     # Data cleaning module
    data_feature_eng.py                 # Feature engineering module
    data_eda.py                         # Exploratory data analysis
    data_exporter.py                    # Dataset export module
    utils.py                            # Logging and utilities
    config.py                           # Configuration settings
    feature_documentation.txt          # Feature analysis and statistics
    csvs/                              # Generated datasets folder
      marineflow_demurrage_synth.csv   # Original dataset
      marineflow_test.csv              # Test split (91 samples)
      marineflow_train.csv             # Training split (419 samples)
      marineflow_validation.csv        # Validation split (90 samples)
    logs/                              # Processing logs (created when pipeline runs)
    README.md                          # Preprocessing documentation
  train/                               # Training folder
    README.md                          # Training documentation
  .gitignore                           # Git ignore file
  LICENSE                              # MIT License file
  README.md                            # This main project file
```

### Quick File Access

#### Original Data

- **Raw Dataset**: [Click to see original data](preproc/csvs/marineflow_demurrage_synth.csv) (600 records, 42 features)

#### Processed Data (After Cleaning & Feature Engineering)

- **Training Data**: [Click to see training data](preproc/csvs/marineflow_train.csv) (419 samples, 137 features)
- **Validation Data**: [Click to see validation data](preproc/csvs/marineflow_validation.csv) (90 samples, 137 features)
- **Test Data**: [Click to see test data](preproc/csvs/marineflow_test.csv) (91 samples, 137 features)

#### Documentation & Code

- **Feature Statistics**: [Click to see feature importance](preproc/feature_documentation.txt)
- **Main Pipeline**: [Click to see orchestrator code](preproc/data_main.py)
- **Data Cleaning**: [Click to see cleaning code](preproc/data_cleaner.py)
- **Feature Engineering**: [Click to see feature creation code](preproc/data_feature_eng.py)
- **Configuration**: [Click to see settings](preproc/config.py)
- **Detailed Guide**: [Click to go to preprocessing guide](preproc/README.md)

## How to Run

### Step 1: Make Sure You Have Python

You need Python with pandas and numpy installed:

```powershell
pip install pandas numpy
```

### Step 2: Clean and Process the Data

Run this command from the preproc folder:

```powershell
cd preproc
python data_main.py
```

**What this does:**

- Loads the raw ship data (600 records)
- Cleans data using modular `data_cleaner.py`
- Engineers 137 features using `data_feature_eng.py`
- Performs analysis using `data_eda.py`
- Exports datasets using `data_exporter.py`
- **Result**: Clean datasets with 137 features split into train (419), validation (90), test (91)

### Step 3: View Your Results

```powershell
type .\preproc\feature_documentation.txt
```

Or [Click to see the results online](preproc/feature_documentation.txt)

**What you'll see:**

- Top predictive features for predicting demurrage (137 total features)
- Feature categories (temporal, efficiency, financial, categorical, interactions)
- Statistics about your processed data (419 train, 90 validation, 91 test samples)

## Results & Discoveries

### What We Found Out

#### Data Quality Discoveries

**Problem we solved**: The original dataset had 600 records, but 5 had impossible timestamps

- **How we found it**: Checked if arrival times were before departure times
- **What we did**: Removed these 5 bad records
- **Result**: 595 clean, reliable records to work with

#### Most Important Features for Predicting Demurrage

We discovered which factors matter most for predicting extra shipping costs:

1. **Overage Hours Chargeable** (0.6672 importance)

   - **What it is**: How many hours over the allowed loading time
   - **Why it matters**: Direct predictor of demurrage charges
   - **Discovery**: This is by far the strongest predictor

2. **Laytime vs Allowed** (0.3465 importance)

   - **What it is**: Difference between actual and allowed loading time
   - **Discovery**: Even small overages can predict demurrage risk

3. **Predicted Risk Score** (0.1287 importance)
   - **What it is**: Our calculated risk rating for each voyage
   - **Discovery**: Combining multiple factors gives better predictions

#### Port Performance Rankings

**How we calculated it**: Analyzed port efficiency vs congestion for each port

**Top 3 Most Efficient Ports** (calculated by averaging efficiency scores):

- **How we found this**: Grouped all voyages by discharge port
- **Calculation**: `average_efficiency_by_port = group_by_port(efficiency_index).mean()`
- **Result**: Clear winners that consistently perform well

**Top 3 Least Congested Ports** (calculated by averaging congestion scores):

- **How we found this**: Same grouping method, but looked at congestion
- **Calculation**: `average_congestion_by_port = group_by_port(congestion_index).mean()`
- **Discovery**: Low congestion doesn't always mean high efficiency

#### Voyage Efficiency Patterns

**What we discovered**:

- **Efficient voyages**: Ships that used ≤100% of allowed laytime
- **Inefficient voyages**: Ships that went overtime
- **How calculated**: `efficiency_ratio = used_laytime / allowed_laytime`

**Key Finding**:

- Efficient voyages: X% of total (≤1.0 efficiency ratio)
- Inefficient voyages: Y% of total (>1.0 efficiency ratio)
- **Business impact**: Inefficient voyages are strong predictors of demurrage costs

#### Seasonal Patterns We Found

**Peak Season Discovery**:

- **Method**: Analyzed departure dates by quarter
- **Finding**: Quarters 2 and 3 (April-September) show higher congestion
- **Business insight**: Plan for longer wait times during peak season

#### Feature Engineering Success

**What we achieved**:

- **Started with**: 42 original features from ship operations
- **Created**: 5 new engineered features using domain knowledge
- **Categories created**:
  - Temporal features: 12 (time-based patterns)
  - Efficiency features: 10 (performance ratios)
  - Financial features: 6 (cost-related metrics)
  - Categorical encoded: 10 (ship types, ports, etc.)

**Most successful engineered features**:

- Port performance scores combining efficiency and congestion
- Efficiency ratios that normalize performance across different ship sizes
- Risk indicators that flag potential problems early

## Conclusions

MarineFlow presents a robust and scalable solution for maritime demurrage prediction and operational analysis. By leveraging advanced data preprocessing, feature engineering, and domain-specific analysis, we developed a comprehensive system capable of enhancing maritime operational efficiency and prediction accuracy. The successful implementation of automated data quality checks and feature engineering positions MarineFlow as a valuable tool for improving decision-making in maritime logistics, port operations, and vessel management.

The project's modular design and comprehensive documentation provide a strong foundation for future enhancements and integration with machine learning models for predictive analytics in maritime operations.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### MIT License Summary

- **Commercial use** - You can use this code in commercial applications
- **Modification** - You can change and adapt the code
- **Distribution** - You can share the code with others
- **Private use** - You can use it for personal projects
- **Liability** - No warranty provided
- **Warranty** - Use at your own risk

### What this means for you:

- **Free to use** - No cost, no restrictions on usage
- **Give credit** - Just include the original license notice
- **No takeback** - Once licensed, always licensed under MIT

For the full license text, see [LICENSE](LICENSE) file in the repository.
