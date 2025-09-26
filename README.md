# sWARm: (Sid) Wins Above Replacement Metric

Welcome to **sWARm**, a personalized reimplementation of the Wins Above Replacement (WAR) metric commonly used throughout baseball analytics and discussions. This project aims to provide a fresh perspective on evaluating player value by calculating the number of wins a player contributes to their team through a variety of machine learning algorithms and features.

## Table of Contents

- [sWARm: (Sid) Wins Above Replacement Metric](#swarm-sid-wins-above-replacement-metric)
  - [Table of Contents](#table-of-contents)
  - [What is WAR?](#what-is-war)
  - [About sWARm](#about-swarm)
  - [Installation \& System Requirements](#installation--system-requirements)
    - [**System Requirements**](#system-requirements)
    - [**Hardware Acceleration**](#hardware-acceleration)
    - [**Dependencies**](#dependencies)
    - [**Installation Process**](#installation-process)
  - [Project Structure](#project-structure)
    - [**Key Architecture Features**](#key-architecture-features)
    - [**Module Interdependencies**](#module-interdependencies)
  - [Usage](#usage)
    - [**Quick Start**](#quick-start)
    - [**System Capabilities**](#system-capabilities)
  - [Key Features](#key-features)
    - [**Machine Learning Models**](#machine-learning-models)
    - [**Strategic Feature Selection**](#strategic-feature-selection)
    - [**Data Quality Innovations**](#data-quality-innovations)
    - [**Quality Over Quantity Philosophy**](#quality-over-quantity-philosophy)
    - [**Complete Feature Catalog**](#complete-feature-catalog)
  - [Methodology](#methodology)
    - [**Feature Engineering**](#feature-engineering)
    - [**Model Architecture**](#model-architecture)
    - [**Validation Framework**](#validation-framework)
    - [**Performance Optimization**](#performance-optimization)
  - [System Performance \& Features](#system-performance--features)
    - [**Performance Benchmarks**](#performance-benchmarks)
    - [**Current Model Performance Metrics**](#current-model-performance-metrics)
    - [**Intelligent Caching System**](#intelligent-caching-system)
    - [**Interactive Features**](#interactive-features)
    - [**Troubleshooting Guide**](#troubleshooting-guide)
    - [**Quick Start Guide**](#quick-start-guide)
  - [Contributing](#contributing)
    - [**Welcome Contributors**](#welcome-contributors)
    - [**Contribution Guidelines**](#contribution-guidelines)
    - [**Areas for Contribution**](#areas-for-contribution)
    - [**Communication**](#communication)
    - [**Recognition**](#recognition)
  - [License](#license)
  - [Contact](#contact)
  - [Roadmap](#roadmap)
    - [**High Priority Features**](#high-priority-features)
    - [**Advanced Analytics**](#advanced-analytics)
    - [**Visualization \& User Experience**](#visualization--user-experience)
    - [**Future Integrations**](#future-integrations)
    - [**Development Priorities**](#development-priorities)
  - [Citations \& Attribution](#citations--attribution)
    - [**Data Sources \& Statistical Foundations**](#data-sources--statistical-foundations)
    - [**Methodological Contributions**](#methodological-contributions)
    - [**Copyright \& Intellectual Property**](#copyright--intellectual-property)
  - [License Information](#license-information)
    - [**Mozilla Public License 2.0**](#mozilla-public-license-20)
    - [**What This License Allows**](#what-this-license-allows)
  - [Version History](#version-history)
      - [**v2.1.0** (September 2025) - *Feature Complete v1* ](#v210-september-2025---feature-complete-v1-)
      - [**v2.0.3** (September 2025) - *Data Quality Fixes*](#v203-september-2025---data-quality-fixes)
      - [**v2.0.2** (September 2025) - *Code Stability*](#v202-september-2025---code-stability)
      - [**v2.0.1** (September 2025) - *Organization \& Documentation*](#v201-september-2025---organization--documentation)
      - [**v2.0.0** (September 2025) - *Modular Architecture*](#v200-september-2025---modular-architecture)
      - [**v1.3.0** (September 2025) - *Enhanced Data \& Visualization*](#v130-september-2025---enhanced-data--visualization)
      - [**v1.2.0** (September 2025) - *Expanded Dataset*](#v120-september-2025---expanded-dataset)
      - [**v1.1.0** (September 2025) - *Performance Crisis \& Recovery*](#v110-september-2025---performance-crisis--recovery)
      - [**v1.0.0** (September 2025) - *Machine Learning Foundation*](#v100-september-2025---machine-learning-foundation)
      - [**v0.2.0** (September 2025) - *Advanced ML Integration*](#v020-september-2025---advanced-ml-integration)
      - [**v0.1.1** (September 2025) - *Initial Cleanup*](#v011-september-2025---initial-cleanup)
      - [**v0.1.0** (September 2025) - *Project Genesis*](#v010-september-2025---project-genesis)
  - [Data Sources](#data-sources)

## What is WAR?

**Wins Above Replacement (WAR)** is a sabermetric statistic that quantifies a player's total contributions to their team in terms of wins. It combines various aspects of a player's performance, like batting, baserunning, fielding, and pitching, to estimate how many more wins a player is worth compared to a replacement-level player.

- Note that replacement-level is that ambiguous term but is typically defined at about [^1]0 WAR[^2] , (average AAAA player at any position should get that amount by playing a full MLB season).

## About sWARm

**sWARm** stands for **Sid Wins Above Replace Metric**, reflecting my personal approach to calculating WAR. While it draws inspiration from existing models like FanGraphs' fWAR and Baseball Prospectus' WARP, sWARm attempts to use different combinations of commonly available statistics and the adjustments that these precursors have helpfully made available to simplify the capturing of player value.

Key features of sWARm include:

- **Personalized Adjustments**: Tailored modifications to standard WAR calculations based on my analysis and insights.
- **Comprehensive Metrics**: Integration of various performance metrics to provide a holistic view of player contributions.
- **Open Source**: Transparent and accessible codebase for collaboration and further development. All data I used is available in the github repo as well, I paid for it where appropriate and it encompasses 2016-2024 (soon to be 2025).

## Installation & System Requirements

### **System Requirements**

**Minimum Requirements:**

- **Python**: 3.9+ (Recommended: 3.11 or 3.13+)
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 2GB free space (includes data cache)
- **OS**: Windows 10+, macOS 10.15+, Linux (Ubuntu 20.04+)

**Recommended for Optimal Performance:**

- **RAM**: 32GB for large dataset processing
- **CPU**: Multi-core processor (8+ cores recommended)
- **Storage**: SSD with 5GB+ free space for caching
- **GPU**: NVIDIA GPU with CUDA support (optional, for neural network acceleration)

### **Hardware Acceleration**

**GPU Support (Optional):**

- **TensorFlow GPU**: Automatically detects and uses NVIDIA GPUs
- **Memory**: 4GB+ VRAM recommended for neural network training
- **CUDA**: Compatible with CUDA 11.8+ and cuDNN 8.6+
- **Performance**: ~3-5x speedup for neural network models

**Memory Usage:**

- **Dataset Loading**: ~500MB RAM for full 2016-2024 data
- **Model Training**: 2-8GB RAM depending on model complexity
- **Caching**: ~195MB disk space for preprocessed data
- **Jupyter**: Additional 1-2GB for notebook execution

### **Dependencies**

**Core Libraries:**

```txt
pandas>=1.5.0          # Data manipulation
numpy>=1.24.0           # Numerical computing
scikit-learn>=1.2.0     # Traditional ML algorithms
tensorflow>=2.13.0      # Neural networks & GPU acceleration
xgboost>=1.7.0          # Gradient boosting
rapidfuzz>=2.0.0        # Fast string matching
plotly>=5.14.0          # Interactive visualizations
```

**Development Tools:**

```txt
jupyter>=1.0.0          # Notebook environment
ipykernel>=6.0.0        # Jupyter kernel support
```

### **Installation Process**

**1. Clone Repository:**

``` bash
git clone https://github.com/NairSiddharth/sWARm.git
cd sWARm
```

**2. Set Up Environment:**

``` bash
# Create virtual environment
python3 -m venv venv

# Activate environment
# Linux/Mac:
source venv/bin/activate
# Windows:
venv\Scripts\activate
```

**3. Install Dependencies:**

``` bash
# Install all requirements
pip install -r requirements.txt

# Optional: Verify TensorFlow GPU detection
python -c "import tensorflow as tf; print('GPU Available:', len(tf.config.list_physical_devices('GPU')) > 0)"
```

**4. Launch System:**

``` bash
# Start Jupyter notebook
jupyter notebook sWARm.ipynb

# Or run data parser directly
python modularized_data_parser.py
```

## Project Structure

```txt
sWARm/
├──  sWARm.ipynb                    # Main analysis notebook (Feature Complete v1)
├──  modularized_data_parser.py     # Core data processing pipeline
├──  requirements.txt               # Python dependencies
├──  README.md                      # Project documentation
├──  TODO.md                        # Development roadmap
├──  PLAN.md                        # Comprehensive development plan
├──  LICENSE                        # MIT License
│
├──  modules/                       # Modular architecture (24 specialized modules)
│   ├──  __init__.py               # Package initialization
│   ├──  animated_analysis_clean.py # Advanced animated visualizations
│   ├──  baserunning_analytics.py  # Enhanced baserunning metrics
│   ├──  basic_cleaners.py         # Data cleaning utilities
│   ├──  bp_derived_stats.py       # Baseball Prospectus statistics
│   ├──  catcher_framing.py        # Catcher framing metrics
│   ├──  data_loading.py           # Core data loading functions
│   ├──  data_validation.py        # Data quality validation
│   ├──  data_visualization.py     # Interactive plotly visualizations
│   ├──  defensive_metrics.py      # Advanced defensive statistics
│   ├──  duplicate_names.py        # Player name conflict resolution
│   ├──  enhanced_data_loading.py  # Advanced data integration
│   ├──  fangraphs_integration.py  # FanGraphs API integration
│   ├──  modeling.py               # ML models & ensemble methods
│   ├──  name_mapping_caching.py   # Name mapping optimization
│   ├──  name_mapping_optimization.py # Advanced name matching
│   ├──  park_factors.py           # Ballpark adjustment factors
│   ├──  stadium_operations.py     # Stadium-specific operations
│   ├──  temporal_modeling.py      # Time-series prediction
│   ├──  test_modules.py          # Module testing framework
│   ├──  two_way_players.py        # Two-way player handling
│   ├──  war_processing.py         # WAR metric processing
│   └──  warp_processing.py        # WARP metric processing
│
├──  cache/                        # Preprocessed data cache (~195MB)
│   ├──  comprehensive_fangraphs_data.json      # FanGraphs integration
│   ├──  comprehensive_fangraphs_war_cleaned.json # Clean WAR data
│   ├──  enhanced_baserunning_values.json       # Baserunning analytics
│   ├──  fielding_oaa_values_v4_seasonal.json   # Defensive metrics
│   ├──  yearly_catcher_framing_data.json       # Catcher framing
│   ├──  yearly_warp_hitter_cleaned.json        # Clean hitter WARP
│   └──  yearly_warp_pitcher_cleaned_v2.json    # Clean pitcher WARP
│
├──  MLB Player Data/              # Raw baseball datasets (~183MB)
│   └──  awards/                   # Player awards and recognition
│
├──  old/                          # Legacy/deprecated files
│   ├──  cleanedCode_orig.ipynb    # Original monolithic notebook
│   ├──  cleanedDataParser_orig.py # Original data parser
│   └──  main_loader_visualizer.ipynb # Legacy visualization
│
└──  test_scope_fix.py             # Testing utilities
```

### **Key Architecture Features**

**Modular Design:**

- **24 specialized modules** for maintainability and scalability
- **Separation of concerns**: Each module handles specific functionality
- **Dependency injection**: Modules can be easily swapped or updated

**Performance Optimizations:**

- **Intelligent caching**: Preprocessed data stored for rapid access
- **Lazy loading**: Data loaded only when needed
- **GPU acceleration**: TensorFlow automatically detects and uses available GPUs

**Data Pipeline:**

- **Raw data** → **Processing modules** → **Cache** → **Analysis**
- **Comprehensive validation** at each stage
- **Automatic cache invalidation** when source data changes

**Development Workflow:**

- **Main notebook** (`sWARm.ipynb`) for complete analysis
- **Modular components** for targeted development
- **Legacy preservation** in `/old` directory

### **Module Interdependencies**

**Core Data Flow:**

```txt
modularized_data_parser.py
├── data_loading.py              # Raw data ingestion
├── enhanced_data_loading.py     # FanGraphs integration
├── bp_derived_stats.py          # Manual calculations
├── name_mapping_optimization.py # Player matching
└── data_validation.py           # Quality assurance

Enhanced Analytics Pipeline:
├── baserunning_analytics.py     # Run expectancy calculations
├── defensive_metrics.py         # Advanced fielding stats
├── catcher_framing.py           # Specialized catching metrics
├── park_factors.py              # Ballpark adjustments
└── two_way_players.py           # Dual-position handling

Modeling & Analysis:
├── modeling.py                  # Core ML algorithms
├── temporal_modeling.py         # Time-series prediction
├── war_processing.py            # WAR calculations
├── warp_processing.py           # WARP calculations
└── data_visualization.py       # Interactive plots
```

**Key Dependencies:**

- **Core Pipeline**: `data_loading.py` → `enhanced_data_loading.py` → `modeling.py`
- **Feature Engineering**: `bp_derived_stats.py` + `baserunning_analytics.py` + `defensive_metrics.py`
- **Name Resolution**: `name_mapping_optimization.py` ← Used by all data integration modules
- **Visualization**: `data_visualization.py` ← Consumes all modeling outputs

**Interface Contracts:**

- **Data Modules**: Return pandas DataFrames with standardized column names
- **Analytics Modules**: Accept player names, return value dictionaries
- **Modeling Modules**: Use ModelResults class for consistent output format
- **Visualization Modules**: Accept ModelResults objects and configuration parameters

## Usage

### **Quick Start**

**1. Launch the Complete System:**

``` bash
jupyter notebook sWARm.ipynb
```

The main notebook contains a feature-complete analysis pipeline with cache control and comprehensive modeling.

**2. Modular Development:**

```python
# Import core functionality
from modularized_data_parser import *
from modules.modeling import run_comprehensive_modeling
from modules.data_visualization import plot_consolidated_model_comparison

# Run specific components
results = run_comprehensive_analysis()
model_results = run_comprehensive_modeling()
```

### **System Capabilities**

**Automated Pipeline:**

- **Data Integration**: 5 FanGraphs datasets + Baseball Prospectus (2016-2024)
- **Advanced Name Matching**: Fuzzy matching with duplicate resolution
- **Cache Management**: Intelligent rebuild system for fresh/fast execution
- **Model Training**: 6 ML algorithms with automated selection

**Enhanced Analytics:**

- **50+ Features**: Comprehensive player metrics vs ~8 basic features previously
- **Multi-Metric Prediction**: WAR, WARP, and component predictions
- **Interactive Visualizations**: Plotly-based animated analysis
- **Player Lookup**: Instant analysis for any player (2016-2024)

**Current Performance:**

- **RandomForest**: 64.7% high accuracy rate, 0.621 avg combined error
- **Neural Networks**: GPU-accelerated with early stopping
- **Ensemble Methods**: Automatic best-model selection by category
- **Temporal Analysis**: Chronological progression visualization

## Key Features

### **Machine Learning Models**

**Current Active Models:**

- **Ridge Regression**: Baseline linear model with L2 regularization
- **K-Nearest Neighbors (KNN)**: Instance-based learning with distance weighting
- **Random Forest**: Ensemble tree-based method (best overall performer)
- **XGBoost**: Gradient boosting with advanced regularization
- **Support Vector Regression (SVR)**: Kernel-based non-linear modeling
- **Neural Networks**: Deep learning with AdamW optimizer, early stopping, and GPU acceleration

**Deprecated Models** *(Removed for poor performance)*:

- AdaBoost, Gaussian Process Regression (v1.x)
- Linear, Lasso, ElasticNet Regression (kept Ridge only)

### **Strategic Feature Selection**

sWARm employs a **focused, curated approach** to feature engineering, utilizing 7 carefully selected features for both hitters and pitchers rather than a kitchen-sink methodology. This design prioritizes **interpretability, statistical significance, and predictive power** over feature quantity.

**Hitter Features (7 Core Metrics):**

- **Plate Discipline**: K% (strikeout rate), BB% (walk rate)
- **Offensive Production**: AVG (batting average), OBP (on-base percentage), SLG (slugging percentage)
- **Enhanced Analytics**: Enhanced_Baserunning, Enhanced_Defense

**Pitcher Features (6 Core Metrics):**

- **Workload & Control**: IP (innings pitched), BB% (walk rate), K% (strikeout rate)
- **Run Prevention**: ERA (earned run average), HR% (home run rate)
- **Enhanced Analytics**: Enhanced_Defense

### **Data Quality Innovations**

**Manual Calculations:**

- **Pre-2020 K%/BB% Resolution**: Manually calculated strikeout and walk percentages for Baseball Prospectus data (2016-2019) using SO/PA and BB/PA ratios, ensuring 100% coverage across all years
- **Statistical Consistency**: Eliminated the correlation gap between pre-2020 and post-2020 BP data that was impacting model performance

**Enhanced Metrics Integration:**

- **Baserunning Analytics**: Run expectancy matrix-based calculations for situational baserunning value
- **Defensive Metrics**: Advanced fielding statistics including OAA (Outs Above Average) and catcher framing data

**Advanced Data Processing:**

- **Fuzzy Name Matching**: Sophisticated player matching between datasets with duplicate resolution
- **Two-Way Player Handling**: MLB-standard definition (20+ innings pitched AND 20+ games as position player)
- **Automatic Data Cleaning**: Infinite/NaN value handling and spring training data removal

### **Quality Over Quantity Philosophy**

**Why 7 Features?**

- **Interpretability**: Each feature has clear baseball meaning and can be explained to any fan
- **Statistical Robustness**: Focused on metrics with strong correlation to winning
- **Model Stability**: Reduces overfitting and improves generalization
- **Data Availability**: Ensures consistent coverage across 2016-2024 timeframe

**Comprehensive Data Access:**

While sWARm accesses 50+ features from comprehensive FanGraphs integration, the modeling pipeline strategically selects the most predictive subset. This approach combines the **breadth of available data** with the **precision of targeted feature engineering**.

### **Complete Feature Catalog**

**Offensive Features - Hitters:**

- **K%** (Strikeout Rate): Plate discipline and contact ability
- **BB%** (Walk Rate): Selectivity and eye for the strike zone
- **AVG** (Batting Average): Raw contact and hit quality
- **OBP** (On-Base Percentage): Overall offensive value creation
- **SLG** (Slugging Percentage): Power and extra-base hit production

**Pitching Features - Pitchers:**

- **IP** (Innings Pitched): Workload and durability indicator
- **BB%** (Walk Rate): Command and control measurement
- **K%** (Strikeout Rate): Swing-and-miss stuff and dominance
- **ERA** (Earned Run Average): Run prevention effectiveness
- **HR%** (Home Run Rate): Power suppression ability

**Enhanced Analytics - Both:**

- **Enhanced_Baserunning**: Situational baserunning value using run expectancy matrix
- **Enhanced_Defense**: Advanced fielding metrics including OAA and positional adjustments

**Contextual Adjustments:**

- **Park Factors**: Ballpark environment effects (1.2x multiplier)
- **Positional Scaling**: Defensive importance weighting by position
- **Era Normalization**: Cross-year statistical consistency (2016-2024)

**Feature Engineering Process:**

1. **Raw Stat Calculation**: Manual derivation for pre-2020 BP data
2. **Normalization**: Cross-dataset statistical harmonization
3. **Enhancement**: Advanced metric integration (baserunning, defense)
4. **Validation**: Feature correlation and significance testing
5. **Selection**: Strategic subset for optimal interpretability

**Visualization & Analysis:**

- Quadrant analysis showing prediction error patterns
- Delta-1 margin analysis comparing to official MLB accuracy standards
- Cross-shaped visualization for WAR≤1 OR WARP≤1 official margins
- Model performance comparison across different categories
- Interactive Plotly visualizations

**Performance Metrics:**

- R-squared and RMSE for model evaluation
- Individual metric accuracy (WAR-only and WARP-only predictions)
- Cross-validation and intersection analysis for delta-1 margins
- Auto-selection of best performing models by category

## Methodology

sWARm employs a comprehensive machine learning approach built on strategic feature selection and advanced analytics:

### **Feature Engineering**

**7-Feature Strategic Approach:**

- **Hitters**: K%, BB%, AVG, OBP, SLG + Enhanced_Baserunning + Enhanced_Defense
- **Pitchers**: IP, BB%, K%, ERA, HR% + Enhanced_Baserunning + Enhanced_Defense

**Data Quality Innovations:**

- **Manual K%/BB% Calculations**: Pre-2020 Baseball Prospectus data manually calculated for 100% coverage
- **Advanced Name Matching**: Fuzzy matching with duplicate player resolution
- **Enhanced Metrics**: Run expectancy-based baserunning, OAA defensive metrics

### **Model Architecture**

**Ensemble Approach:**

- **6 ML Algorithms**: Ridge, KNN, RandomForest, XGBoost, SVR, Neural Networks
- **Automatic Selection**: Best-performing models chosen by category
- **GPU Acceleration**: TensorFlow neural networks with AdamW optimizer

**Training Pipeline:**

- **Cross-Validation**: Robust model evaluation and selection
- **Early Stopping**: Prevents overfitting in neural networks
- **Ensemble Methods**: Combined predictions for superior accuracy

### **Validation Framework**

**Accuracy Standards:**

- **Official Margins**: ±1 WAR and ±1 WARP accuracy zones
- **Quadrant Analysis**: Error pattern identification across prediction ranges
- **Temporal Analysis**: Chronological performance evolution (2016-2024)

**Performance Metrics:**

- **Primary**: R² and RMSE for model evaluation
- **Advanced**: Delta-1 margin analysis, cross-validation intersection
- **Category-Specific**: Separate hitter/pitcher evaluation

### **Performance Optimization**

**System Efficiency:**

- **Intelligent Caching**: ~195MB preprocessed data cache
- **Lazy Loading**: Data loaded only when needed
- **GPU Acceleration**: Automatic detection and utilization
- **Memory Management**: Efficient processing of 2016-2024 datasets

## System Performance & Features

### **Performance Benchmarks**

**Model Accuracy Comparison:**

| Model | Hitter R² | Pitcher R² | Combined Accuracy | Speed |
|-------|-----------|------------|-------------------|-------|
| **RandomForest** | 0.853 | 0.940 | **64.7%** | Fast |
| **Neural Network** | 0.438 | 0.755 | 48.1% | GPU Accelerated |
| **SVR** | 0.482 | 0.908 | 56.0% | Medium |
| **Ridge** | 0.420 | 0.899 | 47.9% | Very Fast |

**vs Industry Standards:**

- **sWARm vs fWAR**: ±1 WAR accuracy: 64.7% (RandomForest)
- **sWARm vs WARP**: ±1 WARP accuracy: 56.0% (SVR)
- **Processing Speed**: 2016-2024 dataset processed in 5-15 minutes

### **Current Model Performance Metrics**

**Best Performing Models (Auto-Selected):**

- **KERAS** (Neural Networks)
- **RIDGE** (Linear Regression)
- **SVR** (Support Vector Regression)
- **RANDOMFOREST** (Ensemble Tree Method)

**Detailed Performance Results:**

**Hitter Performance:**

| Model | WARP R² | WARP RMSE | WAR R² | WAR RMSE |
|-------|---------|-----------|--------|----------|
| **RandomForest** | 0.210 | 1.375 | **0.853** | **0.690** |
| **SVR** | **0.323** | **1.273** | 0.482 | 1.294 |
| **Keras** | 0.279 | 1.313 | 0.438 | 1.347 |
| **Ridge** | 0.299 | 1.295 | 0.420 | 1.369 |
| KNN | -0.078 | 1.606 | 0.936 | 0.455 |
| XGBoost | 0.186 | 1.395 | 0.764 | 0.874 |

**Pitcher Performance:**

| Model | WARP R² | WARP RMSE | WAR R² | WAR RMSE |
|-------|---------|-----------|--------|----------|
| **RandomForest** | **0.825** | **0.686** | **0.940** | **0.322** |
| **SVR** | 0.804 | 0.724 | 0.908 | 0.401 |
| **Keras** | 0.793 | 0.745 | 0.755 | 0.654 |
| **Ridge** | 0.670 | 0.941 | 0.899 | 0.419 |
| KNN | 0.751 | 0.818 | 0.917 | 0.380 |
| XGBoost | 0.818 | 0.698 | 0.930 | 0.349 |

**Key Performance Insights:**

- **RandomForest** dominates pitcher predictions (R² > 0.82 for both metrics)
- **Hitter predictions** more challenging across all models (lower R² values)
- **Best overall**: RandomForest for comprehensive player evaluation
- **Specialized use**: SVR excels at hitter WARP prediction (R² = 0.323)

**Model Selection Strategy:**
The auto-selection algorithm identified the four most balanced performers across different prediction scenarios, ensuring robust performance for diverse analytical needs.

### **Intelligent Caching System**

**What It Is:**
sWARm implements a sophisticated caching system that stores preprocessed data in JSON format (~195MB total) to dramatically improve performance and user experience.

**Why I Implemented It:**

- **Performance**: Initial data loading from raw sources takes 15-30 minutes; cached data loads in 30-60 seconds
- **Reliability**: Eliminates repeated network calls to FanGraphs and data processing errors
- **Development Efficiency**: Enables rapid iteration and testing without waiting for full data pipeline
- **User Experience**: Provides near-instant analysis for subsequent runs

**Cache Structure:**

```txt
cache/
├── comprehensive_fangraphs_data.json           # FanGraphs integration (~80MB)
├── comprehensive_fangraphs_war_cleaned.json    # Clean WAR data (~60MB)
├── enhanced_baserunning_values.json           # Baserunning analytics (~20MB)
├── fielding_oaa_values_v4_seasonal.json       # Defensive metrics (~15MB)
├── yearly_catcher_framing_data.json           # Catcher framing (~10MB)
├── yearly_warp_hitter_cleaned.json            # Clean hitter WARP (~5MB)
└── yearly_warp_pitcher_cleaned_v2.json        # Clean pitcher WARP (~5MB)
```

**Cache Management Commands:**

```python
# Force complete cache rebuild (in sWARm.ipynb)
FORCE_CACHE_REBUILD = True

# Clear specific cache files
import os
os.remove('cache/comprehensive_fangraphs_data.json')  # Clear FanGraphs cache
os.remove('cache/enhanced_baserunning_values.json')   # Clear baserunning cache
```

```bash
# Clear entire cache directory
rm -rf cache/

# Clear cache and restart (Linux/Mac)
rm -rf cache/ && jupyter notebook sWARm.ipynb

# Clear cache and restart (Windows)
rmdir /s cache && jupyter notebook sWARm.ipynb
```

**Automatic Cache Invalidation:**

- Cache automatically rebuilds when source data files are modified
- Manual rebuild option available for data updates or troubleshooting
- Individual cache files can be selectively cleared without affecting others

### **Interactive Features**

**Animated Temporal Analysis:**

- **Chronological Progression**: 2016-2024 performance evolution
- **Cinematic Visualizations**: 3D surfaces, gradient heatmaps
- **Model Comparison**: Side-by-side algorithm performance tracking

**Advanced Analytics Tools:**

- **Quadrant Analysis**: Prediction accuracy zones with ±1 WAR/WARP margins
- **Player Lookup**: Instant analysis for any player (`quick_player_lookup("Mike Trout")`)
- **Residual Analysis**: Comprehensive error pattern identification
- **Confidence Intervals**: Prediction uncertainty visualization

**Interactive Controls:**

- **Model Toggles**: Click legends to show/hide specific algorithms
- **Temporal Navigation**: Animate through seasons or jump to specific years
- **Player Filtering**: Searchable player selection and comparison

### **Troubleshooting Guide**

**Common Issues & Solutions:**

**Cache Problems:**

```python
# Force cache rebuild
FORCE_CACHE_REBUILD = True  # In sWARm.ipynb
```

**GPU Detection Issues:**

```bash
# Verify TensorFlow GPU
python -c "import tensorflow as tf; print('GPU Available:', len(tf.config.list_physical_devices('GPU')) > 0)"
```

**Memory Optimization:**

- **Reduce dataset size**: Filter by years or players
- **Clear cache**: Delete `/cache` directory if corrupted
- **Increase RAM**: 16GB+ recommended for full dataset

**Performance Issues:**

- **Enable GPU**: Automatic TensorFlow detection
- **Use SSD storage**: Significantly improves cache performance
- **Parallel processing**: Multiple CPU cores utilized automatically

**Data Loading Errors:**

- **Check data integrity**: Verify MLB Player Data directory exists
- **Network connectivity**: Required for initial FanGraphs integration
- **File permissions**: Ensure read/write access to cache directory

### **Quick Start Guide**

**30-Second Demo:**

```bash
# 1. Install and activate
git clone https://github.com/NairSiddharth/sWARm.git
cd sWARm && python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Launch analysis
jupyter notebook sWARm.ipynb

# 3. Run complete analysis (in notebook)
results = run_comprehensive_analysis()
model_results = run_comprehensive_modeling()
```

**Immediate Results:**

- **Player Analysis**: `quick_player_lookup("Shohei Ohtani")`
- **Model Performance**: Automatic best-model selection and metrics
- **Interactive Visualizations**: Plotly charts with animation controls

**Next Steps:**

- **Explore modules**: Check `/modules/API.md` for detailed function reference
- **Customize analysis**: Modify feature selection or model parameters
- **Add players**: Update dataset with new player analysis

## Contributing

### **Welcome Contributors**

sWARm welcomes contributions from the baseball analytics and machine learning communities. Whether you're interested in improving models, adding features, or enhancing documentation, your contributions are valued.

### **Contribution Guidelines**

**Before Contributing:**

1. **Review the codebase**: Familiarize yourself with the modular architecture
2. **Check existing issues**: Look for open issues or feature requests
3. **Read the roadmap**: Understand planned development priorities

**Development Process:**

1. **Fork & Clone**: Fork the repository and clone your fork locally
2. **Create Branch**: Use descriptive branch names (`feature/ensemble-stacking`, `fix/cache-invalidation`)
3. **Set Up Environment**: Follow installation instructions in README
4. **Make Changes**: Implement your improvements following project conventions

**Code Standards:**

- **Integration**: Ensure new code integrates seamlessly with existing modules
- **Documentation**: Add clear docstrings and comments explaining functionality
- **Testing**: Verify your changes don't break existing functionality
- **Style**: Follow existing code patterns and naming conventions

**Submission Process:**

1. **Test Thoroughly**: Run the complete pipeline to ensure stability
2. **Update Documentation**: Modify README/API docs if needed
3. **Commit Messages**: Use clear, descriptive commit messages
4. **Pull Request**: Submit PR with detailed description of changes

### **Areas for Contribution**

**High-Impact Opportunities:**

- **Feature Engineering**: Additional defensive metrics (DRS, UZR)
- **Model Enhancement**: Ensemble meta-modeling, LSTM networks
- **Visualization**: Interactive dashboards, prediction confidence intervals
- **Documentation**: Usage examples, troubleshooting guides

**Technical Improvements:**

- **Performance**: Caching optimization, GPU utilization
- **Data Integration**: New data sources, real-time updates
- **Testing**: Unit tests, integration tests, performance benchmarks

### **Communication**

- **Issues**: Use GitHub issues for bug reports and feature requests
- **Discussions**: Engage in issue discussions before major changes
- **Questions**: Feel free to ask questions about implementation details

### **Recognition**

Contributors will be recognized in project documentation and release notes. Significant contributions may lead to collaborator status with enhanced project access.

## License

[License](LICENSE)

## Contact

For questions or feedback, feel free to open an [issue](https://github.com/NairSiddharth/sWARm/issues) on the GitHub repository.

## Roadmap

### **High Priority Features**

**Enhanced Feature Engineering:**

- **Advanced Defensive Metrics**: Integration of DRS (Defensive Runs Saved) and UZR (Ultimate Zone Rating) for superior positional evaluation
- **Situational Performance**: RISP batting average and leverage index for clutch situation analysis
- **Expanded Catching Metrics**: Catcher blocking and caught stealing statistics

**Predictive Analytics:**

- **Future Performance Modeling**: 3-year player projections with expected vs actual stat blending (70%/30% ratio)
- **Injury Risk Integration**: Historical injury frequency and severity impact on performance decline
- **Workload Analysis**: Usage pattern tracking to identify overuse risk factors

### **Advanced Analytics**

**Model Enhancements:**

- **Ensemble Meta-Modeling**: Stacking ensemble combining RandomForest, XGBoost, and Neural Networks
- **Time-Aware Models**: LSTM networks and seasonal decomposition for career arc modeling
- **Model Interpretability**: SHAP values and LIME for prediction explanation

**Interactive Analysis:**

- **Player Comparison Dashboard**: Side-by-side multi-player analysis tool
- **Prediction Confidence Intervals**: Uncertainty visualization for prediction reliability
- **Career Trajectory Matching**: Find players with similar projected development paths

### **Visualization & User Experience**

**Advanced Visualizations:**

- **Prediction Tracking System**: Monitor prediction changes over time for model validation
- **Cross-Validation Graphs**: Enhanced model performance visualization
- **Interactive Player Search**: Name-based lookup with 3-year projections

**Performance Metrics:**

- **MAE Implementation**: Alternative error metric evaluation for outlier handling
- **Model Stability Monitoring**: Automated retraining triggers based on data drift

### **Future Integrations**

**Data Expansion:**

- **2025 Season Data**: Complete integration when season/postseason concludes
- **Additional Defensive Features**: Catch probability, outfield jump metrics
- **Pitcher Velocity Analysis**: Fastball speed delta to league average

**System Enhancements:**

- **Real-Time Updates**: Live season data integration capabilities
- **API Development**: External access for third-party applications
- **Performance Optimization**: Enhanced caching and GPU utilization

### **Development Priorities**

1. **Short-Term (v2.2.x)**: Enhanced defensive metrics, situational performance analysis
2. **Medium-Term (v2.3.x)**: Future performance modeling, injury risk integration
3. **Long-Term (v3.x.x)**: Advanced ensemble methods, real-time data integration

For detailed technical specifications and progress tracking, see [TODO.md](TODO.md).

## Citations & Attribution

### **Data Sources & Statistical Foundations**

- **FanGraphs**: Sullivan, Jeff, et al. "FanGraphs Sabermetrics Library." *FanGraphs*, 2002-2025. [https://www.fangraphs.com/](https://www.fangraphs.com/)
- **Baseball Prospectus**: Silver, Nate, et al. "WARP (Wins Above Replacement Player) Methodology." *Baseball Prospectus*, 2003-2025. [https://www.baseballprospectus.com/](https://www.baseballprospectus.com/)
- **MLB Statcast**: "Statcast Search." *Baseball Savant*, Major League Baseball, 2015-2025. [https://baseballsavant.mlb.com/](https://baseballsavant.mlb.com/)

### **Methodological Contributions**

**Original Research & Analysis:**

- **Manual K%/BB% Calculations**: Novel methodology for calculating pre-2020 Baseball Prospectus derived statistics, ensuring 100% feature coverage across all years (2016-2024)
- **Strategic Feature Selection**: 7-feature focused approach prioritizing interpretability and statistical significance over quantity
- **Enhanced Baserunning Analytics**: Run expectancy matrix-based calculations for situational baserunning value assessment

**Data Integration Innovations:**

- **Multi-Source Harmonization**: Comprehensive integration of FanGraphs, Baseball Prospectus, and MLB Statcast data with advanced name matching algorithms
- **Temporal Consistency**: Standardized feature engineering across 9-year dataset spanning significant rule and measurement changes in baseball

### **Copyright & Intellectual Property**

**Original Work:**

- All code, analysis, and documentation: © 2025 Siddharth Nair
- Original algorithms and methodological improvements: © 2025 Siddharth Nair

**Data Acknowledgments:**

- Baseball statistics used under fair use provisions for research and analysis
- All commercial data sources properly licensed and attributed

## License Information

### **Mozilla Public License 2.0**

This project is licensed under the **Mozilla Public License 2.0 (MPL-2.0)**, a copyleft license that balances open source collaboration with commercial flexibility.

### **What This License Allows**

For complete license terms, see [LICENSE](LICENSE) file.

## Version History

#### **v2.1.0** (September 2025) - *Feature Complete v1* <!-- markdownlint-disable MD001 -->

- **Major**: Comprehensive planning and documentation overhaul
- **Features**: Added LICENSE, comprehensive PLAN.md, feature-complete sWARm.ipynb
- **Status**: All planned features implemented and working

#### **v2.0.3** (September 2025) - *Data Quality Fixes*

- **Fix**: Resolved pre-2020 BP data statistical mismatches
- **Enhancement**: Fixed animated visualizations
- **Data**: Standardized FanGraphs vs Baseball Prospectus feature alignment

#### **v2.0.2** (September 2025) - *Code Stability*

- **Fix**: General code fixes and bug resolution
- **Stability**: Improved system reliability

#### **v2.0.1** (September 2025) - *Organization & Documentation*

- **Organization**: Created TODO.md, renamed files for consistency
- **Cleanup**: Deprecated old files, improved naming conventions
- **Documentation**: Enhanced project structure

#### **v2.0.0** (September 2025) - *Modular Architecture*

- **Major**: Complete modularization from monolithic structure
- **Architecture**: 24 specialized modules for better maintainability
- **Data**: Expanded coverage to 2016-2024 (vs single year)
- **Models**: Removed poorly performing algorithms (AdaBoost, Gaussian Process)
- **Features**: Enhanced duplicate name handling, improved park factors
- **Cleanup**: Removed spring league data contamination

#### **v1.3.0** (September 2025) - *Enhanced Data & Visualization*

- **Data**: Added WARP data for 2016-2020, catcher framing metrics
- **Architecture**: Began modularization process from 2000+ line files
- **Documentation**: Improved README, added TODO tracking
- **Cleanup**: Removed deprecated code for clarity

#### **v1.2.0** (September 2025) - *Expanded Dataset*

- **Data**: Added Baseball Prospectus data (2016-2020, 2022-2024)
- **Accuracy**: Massively improved correlation calculations
- **Features**: More parameters for model selection
- **Tuning**: Park factor adjustments (1.5 → 1.2)

#### **v1.1.0** (September 2025) - *Performance Crisis & Recovery*

- **Challenge**: Major performance issues identified
- **Strategy**: Increased training data to address model weaknesses
- **Cleanup**: Improved data quality, removed spring training contamination
- **Analysis**: Feature re-evaluation and data source review

#### **v1.0.0** (September 2025) - *Machine Learning Foundation*

- **ML**: Added Keras/TensorFlow neural networks
- **Algorithms**: Integrated XGBoost and traditional ML methods
- **Optimization**: Cached data mapping (many-to-one relationships)
- **Visualization**: Enhanced graphs, fWAR/WARP comparisons
- **Performance**: Significant speed improvements

#### **v0.2.0** (September 2025) - *Advanced ML Integration*

- **ML**: Keras/TensorFlow integration, XGBoost implementation
- **Data**: Improved player mapping and data cleaning
- **Performance**: Major optimizations for computational efficiency

#### **v0.1.1** (September 2025) - *Initial Cleanup*

- **Data**: Uploaded cleaned code and datasets
- **Foundation**: Established baseline functionality

#### **v0.1.0** (September 2025) - *Project Genesis*

- **Foundation**: Initial project structure and concept
- **Core**: Basic WAR calculation framework

---

## Data Sources

- Data from [Baseball Savant](https://baseballsavant.mlb.com/) (MLB Statcast data)
- Data from [FanGraphs](https://www.fangraphs.com/) (advanced baseball statistics and personally my source of baseball news!)
- Data from [Baseball Prospectus](https://www.baseballprospectus.com/) (player and team analysis)
- Zip files w/ currently unused data from [Retrosheets](https://www.retrosheets.org/) – Historical game logs and play-by-play data

[^2]: Roughly 1000 WAR exists per season, so depending on a given player performs compared to the average, their share changes year-over-year[^3].

[^3]: An example we can look at to illustrate this is say a 10 WAR player decides to suddenly retire, with no new players coming in to the MLB. One simple assumption to make is that their team would be projected to win 10 fewer games in the upcoming season, but that's not far enough. The next, and most important assumption you must make is that the wins from the retired player would get redistributed across the league to the teams that his team would be playing (so if we looked at all of the projected win totals of the different teams that this team was playing, the sum of the deltas between the projection of their win totals before and after this player retired should equal approximately ten). Some value may go to his teammates, but most doesn't.

[^1]: This is something I have some minor quibbles with due to the disparity between hitting and pitching talent in baseball (i.e. I think that batters should have a slightly higher level set for their replacement level when compared to pitchers, because the average AAA batter called up will probably do better than the average AAA pitcher because I think that better pitchers get called up faster than better batters so the general level of batters stuck in the AAA is higher), but the people who came up with this are much smarter than me so consider this young man yelling at clouds.
