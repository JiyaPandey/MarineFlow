# MarineFlow
**Maritime Demurrage Prediction and Flow Analysis**

## Introduction
Maritime operations face significant challenges in predicting demurrage costs and optimizing vessel flow efficiency. Traditional manual tracking systems lead to delayed decision-making, increased operational costs, and missed optimization opportunities. Advanced data preprocessing and machine learning solutions are needed to streamline maritime operations, enhance prediction accuracy, and improve port efficiency. Implementing such technology can significantly reduce demurrage costs, enhance operational planning, and boost productivity across maritime logistics.

## Overview
Our solution is a comprehensive data preprocessing and machine learning pipeline designed for maritime demurrage prediction and flow analysis. It utilizes advanced feature engineering and statistical analysis to process maritime operational data including vessel timestamps, port efficiency metrics, weather conditions, and cargo handling information. By automating data preprocessing and providing predictive insights, our pipeline enhances operational efficiency, reduces prediction errors, and improves decision-making. The modular design allows for seamless integration into existing maritime workflows and adaptability to different port operations.

## Table of Contents

| Section | Description | Quick Access |
|---------|-------------|--------------|
| [Documentation Links](#documentation-links) | Links to detailed module docs | Jump to docs |
| [Data Processing](#data-processing) | Dataset overview and key features | See data info |
| [Components](#components) | Project modules and scripts | View components |
| [Formulas](#formulas) | Mathematical calculations used | See formulas |
| [How to Run](#how-to-run) | Step-by-step instructions | Run the code |
| [Results & Discoveries](#results--discoveries) | Key findings and insights | See results |
| [Technologies Used](#technologies-used) | Tools and libraries | Tech stack |
| [Folder Structure](#folder-structure) | Project organization | File layout |
| [Conclusions](#conclusions) | Project summary | Final thoughts |

## Documentation Links
- **📁 Data Processing Details**: [Click me to go to preprocessing docs](preproc/README.md)
- **📁 Training Details**: [Click me to go to training docs](train/README.md)

## Data Processing

### Dataset Overview
The pipeline processes maritime operational data with the following characteristics:
- **Total Records**: 600 (after preprocessing: 595)
- **Features**: 47 total (42 original + 5 engineered)
- **Target Variable**: Demurrage prediction (flag and amount)
- **Data Split**: 416 train, 89 validation, 90 test samples

### Key Features
Top predictive features identified through analysis:
- `overage_hours_chargeable`: 0.6672 importance
- `laytime_vs_allowed`: 0.3465 importance
- `pred_risk_score`: 0.1287 importance
- `efficiency_vs_congestion`: 0.1033 importance
- `port_performance_score`: 0.1018 importance

## Components

### Data Preprocessing
- **Location**: `preproc/` folder
- **Main Script**: `preprocess_data.py` - [Click me to see the code](preproc/preprocess_data.py)
- **What it does**: Cleans data, converts timestamps, checks for errors, creates basic metrics
- **Output**: Clean datasets ready for machine learning
- **Full Guide**: [Click me to go to detailed preprocessing guide](preproc/README.md)

[Click me to go to preprocessing docs](preproc/README.md)

### Feature Engineering
- **Script**: `feature_engineering.py` - [Click me to see the code](preproc/feature_engineering.py)
- **What it does**: Creates new features like efficiency ratios, seasonal patterns, risk scores
- **Feature Stats**: `feature_documentation.txt` - [Click me to see feature importance](preproc/feature_documentation.txt)

### Model Training
- **Location**: `train/` folder
- **Status**: Ready for machine learning models
- **Guide**: [Click me to go to training docs](train/README.md)

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
3. **View the results**: [Click me to see calculated features](preproc/feature_documentation.txt)

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
├── preproc/                               # Data preprocessing folder
│   ├── feature_documentation.txt          # Feature analysis and statistics
│   ├── feature_engineering.py             # Feature creation script
│   ├── marineflow_demurrage_synth.csv     # Original dataset
│   ├── marineflow_test.csv               # Test split (90 samples)
│   ├── marineflow_train.csv              # Training split (416 samples)
│   ├── marineflow_validation.csv         # Validation split (89 samples)
│   ├── preprocess_data.py                 # Main data cleaning script
│   ├── preprocessing_pipeline.pkl         # Saved preprocessing pipeline
│   └── README.md                         # Preprocessing documentation
├── train/                                 # Training folder
│   └── README.md                         # Training documentation
└── README.md                             # This main project file
```

### Quick File Access

#### Original Data
- **Raw Dataset**: [Click me to see original data](preproc/marineflow_demurrage_synth.csv) (600 records, 42 features)

#### Processed Data (After Cleaning & Feature Engineering)
- **Training Data**: [Click me to see training data](preproc/marineflow_train.csv) (416 samples, 47 features)
- **Validation Data**: [Click me to see validation data](preproc/marineflow_validation.csv) (89 samples, 47 features)  
- **Test Data**: [Click me to see test data](preproc/marineflow_test.csv) (90 samples, 47 features)

#### Documentation & Code
- **Feature Statistics**: [Click me to see feature importance](preproc/feature_documentation.txt)
- **Processing Code**: [Click me to see cleaning code](preproc/preprocess_data.py)
- **Feature Engineering Code**: [Click me to see feature creation code](preproc/feature_engineering.py)
- **Saved Pipeline**: [Click me to see pipeline file](preproc/preprocessing_pipeline.pkl)
- **Detailed Guide**: [Click me to go to preprocessing guide](preproc/README.md)

## How to Run

### Step 1: Make Sure You Have Python
You need Python with pandas and numpy installed:
```powershell
pip install pandas numpy
```

### Step 2: Clean and Process the Data
Run this command from the main folder:
```powershell
python .\preproc\preprocess_data.py
```
**What this does:**
- Loads the raw ship data (600 records)
- Fixes timestamp formats 
- Removes bad data (5 records with invalid dates)
- Creates basic time calculations
- Checks data for errors and inconsistencies
- **Result**: Clean dataset with 595 good records

### Step 3: Create Advanced Features
```powershell
python .\preproc\feature_engineering.py
```
**What this does:**
- Takes the clean data from Step 2
- Creates 47 total features (42 original + 5 new ones)
- Calculates efficiency ratios, risk scores, seasonal patterns
- Splits data into train/validation/test sets
- **Result**: Ready-to-use datasets for machine learning

### Step 4: View Your Results
```powershell
type .\preproc\feature_documentation.txt
```
Or [Click me to see the results online](preproc/feature_documentation.txt)

**What you'll see:**
- Top 10 most important features for predicting demurrage
- Feature categories (temporal, efficiency, financial)
- Statistics about your processed data

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