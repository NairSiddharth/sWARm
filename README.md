# sWARm: (Sid) Wins Above Replacement Metric

Welcome to **sWARm**, a personalized reimplementation of the Wins Above Replacement (WAR) metric commonly used throughout baseball analytics and discussions. This project aims to provide a fresh perspective on evaluating player value by calculating the number of wins a player contributes to their team through a variety of machine learning algorithms and features.

## Table of Contents

- [sWARm: (Sid) Wins Above Replacement Metric](#swarm-sid-wins-above-replacement-metric)
  - [Table of Contents](#table-of-contents)
  - [What is WAR?](#what-is-war)
  - [About sWARm](#about-swarm)
  - [Installation](#installation)
    - [**Dependencies**](#dependencies)
    - [**Installation Process**](#installation-process)
  - [Project Structure](#project-structure)
    - [**Architecture Overview**](#architecture-overview)
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
  - [Documentation](#documentation)
    - [Quick Summary](#quick-summary)
  - [License Information](#license-information)
    - [**Mozilla Public License 2.0**](#mozilla-public-license-20)
    - [**What This License Allows**](#what-this-license-allows)
  - [Data Sources](#data-sources)

## What is WAR?

**Wins Above Replacement (WAR)** is a sabermetric statistic that quantifies a player's total contributions to their team in terms of wins. It combines various aspects of a player's performance, like batting, baserunning, fielding, and pitching, to estimate how many more wins a player is worth compared to a replacement-level player.

- Note that replacement-level is that ambiguous term but is typically defined at about [^1]0 WAR[^2] , (average AAAA player at any position should get that amount by playing a full MLB season).

## About sWARm

**sWARm** stands for **Sid Wins Above Replace Metric**, reflecting my personal approach to calculating WAR. While it draws inspiration from existing models like FanGraphs' fWAR and Baseball Prospectus' WARP, sWARm attempts to use different combinations of commonly available statistics and the adjustments that these precursors have helpfully made available to simplify the capturing of player value.

Key features of sWARm include:

- **Personalized Adjustments**: Tailored modifications to standard WAR calculations based on my analysis and insights.
- **Comprehensive Metrics**: Integration of various performance metrics to provide a holistic view of player contributions.
- **Open Source**: Transparent and accessible codebase for collaboration and further development. All data I used is available in the github repo as well, I paid for it where appropriate and it encompasses 2016-2025 (and the in-progress 2026 season).

## Installation

### **Dependencies**

**Core Runtime Libraries** (requirements-core.txt):

```txt
darts==0.38.0           # Time series forecasting
keras==3.11.3           # Neural network API
lifelines==0.30.0       # Survival analysis
numpy==2.3.4            # Numerical computing
pandas==2.3.3           # Data manipulation
plotly==6.3.0           # Interactive visualizations
pybaseball==2.2.7       # Baseball data API
pytorch_lightning==2.5.2 # PyTorch training framework
scikit_learn==1.7.2     # Traditional ML algorithms
scipy==1.16.2           # Scientific computing
shap==0.48.0            # Model explainability
sktime==0.39.0          # Time series ML
tensorflow==2.20.0      # Neural networks & GPU acceleration
torch==2.8.0+cu128      # PyTorch with CUDA support
torchmetrics==1.8.2     # PyTorch metrics
```

**Development Tools** (requirements-dev.txt):

```txt
jupyter==1.1.1          # Notebook environment
jupyterlab==4.4.7       # Advanced notebook IDE
matplotlib==3.10.6      # Static visualizations
notebook==7.4.5         # Classic notebook interface
prettytable==3.16.0     # Table formatting
pytest==8.4.2           # Testing framework
```

**Dependency Philosophy:**

sWARm now uses a split requirements approach for cleaner dependency management:

- **Only 15 core packages** for production runtime (vs 154 in old requirements.txt)
- **6 development packages** for notebooks and testing
- **89% reduction** in explicit dependencies - transitive dependencies auto-installed by pip

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
# Production: Core dependencies only (15 packages)
pip install -r requirements-core.txt

# Development: Core + dev tools (21 packages total)
pip install -r requirements-core.txt -r requirements-dev.txt

# Optional: Verify TensorFlow GPU detection
python -c "import tensorflow as tf; print('GPU Available:', len(tf.config.list_physical_devices('GPU')) > 0)"
```

**4. Launch System:**

``` bash
# Main analysis notebook (recommended entry point)
jupyter notebook new_pipeline/notebooks/sWARm_overview.ipynb

# Detailed analysis with visualizations
jupyter notebook new_pipeline/notebooks/sWARm_deep_dive.ipynb

# Hitter-specific analysis
jupyter notebook new_pipeline/notebooks/hitters/hitter_pipeline_main.ipynb

# Pitcher-specific analysis
jupyter notebook new_pipeline/notebooks/pitchers/pitcher_pipeline_main.ipynb

# Future projections (multi-year WAR predictions)
jupyter notebook new_pipeline/notebooks/sWARm_future_overview.ipynb

# Or explore modular components directly
python -c "from new_pipeline.models.current_season import MultiQuantileHistGBRegressor; print('System ready')"
```

## Project Structure

```txt
sWARm/
├──  requirements-core.txt          # Core runtime dependencies (15 packages)
├──  requirements-dev.txt           # Development tools (6 packages)
├──  README.md                      # Project documentation
├──  TODO.md                        # Development roadmap
├──  LICENSE                        # MPL 2.0 License
│
├──  .claude/                       # Claude Code configuration
│   └──  CLAUDE.md                 # Development guidelines
│
├──  new_pipeline/                  # Modern modular architecture (v4.0+)
│   │
│   ├──  common/                   # Shared functionality across all projections
│   │   ├──  loaders/             # Data loading from FanGraphs, BP, Statcast
│   │   │   ├──  hitter_loaders.py
│   │   │   └──  pitcher_loaders.py
│   │   ├──  features/            # Feature engineering modules
│   │   │   ├──  confidence_scorer.py      # Confidence scoring system
│   │   │   ├──  injury_recovery.py       # Injury impact features
│   │   │   ├──  elite_detection.py       # Elite player identification
│   │   │   ├──  rookie_detection.py      # Rookie classification
│   │   │   └──  age_curves.py            # Age adjustment curves
│   │   ├──  transformers/        # sklearn-style feature transformers
│   │   │   ├──  hitter_transformer.py
│   │   │   └──  pitcher_transformer.py
│   │   ├──  data_preparation/    # Data preprocessing and cleaning
│   │   └──  projections/         # Projection utilities and helpers
│   │
│   ├──  models/                   # Model implementations
│   │   ├──  current_season/      # Current season WAR modeling
│   │   │   ├──  multi_quantile_histgb.py  # Multi-quantile ensemble
│   │   │   └──  keras_utils.py            # Neural network utilities
│   │   ├──  ros/                 # Rest-of-season projections
│   │   │   ├──  hitter_ros_model.py
│   │   │   └──  pitcher_ros_model.py
│   │   └──  future_season/       # Multi-year projections (1-3 years)
│   │       ├──  data_preparation.py       # Historical data loading
│   │       ├──  longitudinal_model.py     # Year-to-year modeling
│   │       ├──  survival_model.py         # Retirement probability
│   │       ├──  age_curves.py             # Position-specific aging
│   │       ├──  joint_projection.py       # Combined projections
│   │       ├──  elite_player_adjuster.py  # Elite protection system
│   │       ├──  elite_adjustments.py      # Adjustment wrappers
│   │       ├──  injury_recovery.py        # Injury impact modeling
│   │       ├──  constraint_optimizer.py   # Constraint enforcement
│   │       ├──  expected_stats.py         # Expected statistics
│   │       ├──  temporal_validation.py    # Validation framework
│   │       └──  future_projection_pipeline.py  # End-to-end pipeline
│   │
│   ├──  notebooks/                # Analysis notebooks
│   │   ├──  sWARm_overview.ipynb          # Main entry point
│   │   ├──  sWARm_deep_dive.ipynb         # Detailed analysis
│   │   ├──  sWARm_future_overview.ipynb   # Future projections overview
│   │   ├──  sWARm_future_deep_dive.ipynb  # Future detailed analysis
│   │   ├──  hitters/                      # Hitter-specific notebooks
│   │   │   ├──  hitter_pipeline_main.ipynb
│   │   │   ├──  hitter_ros_training.ipynb
│   │   │   ├──  hitter_deep_dive.ipynb
│   │   │   └──  hitter_future_projections.ipynb
│   │   ├──  pitchers/                     # Pitcher-specific notebooks
│   │   │   ├──  pitcher_pipeline_main.ipynb
│   │   │   ├──  pitcher_ros_training.ipynb
│   │   │   ├──  pitcher_deep_dive.ipynb
│   │   │   └──  pitcher_future_projections.ipynb
│   │   └──  shared/                       # Shared notebook utilities
│   │       ├──  pipeline_runner.py
│   │       └──  table_utils.py
│   │
│   └──  tests/                    # Comprehensive test suite (18 modules)
│       ├──  test_ensemble_model.py
│       ├──  test_ensemble_model_pitcher.py
│       ├──  test_hitter_features.py
│       ├──  test_pitcher_features.py
│       ├──  test_integration.py
│       ├──  test_temporal_validation.py
│       └──  ... (12 more test files)
│
├──  common_modules/               # DEPRECATED - Legacy modules (v3.x)
├──  current_season_modules/       # DEPRECATED - Use new_pipeline/models/current_season/
├──  future_season_modules/        # DEPRECATED - Use new_pipeline/models/future_season/
│
├──  predictions/                  # Model output storage
│   ├──  future_projections_hitter_2025.csv
│   └──  future_projections_pitcher_2025.csv
│
├──  cache/                        # Preprocessed data cache (~195MB)
│   ├──  comprehensive_fangraphs_data.json
│   ├──  enhanced_baserunning_values.json
│   ├──  fielding_oaa_values_v4_seasonal.json
│   └──  ... (more cache files)
│
├──  MLB Player Data/              # Raw baseball datasets (~183MB)
│   ├──  FanGraphs_Data/           # 2016-2024 FanGraphs data
│   │   └──  injuries/             # Injury data
│   ├──  BP_Data/                  # Baseball Prospectus WARP data
│   └──  Statcast_Data/            # MLB Statcast metrics
│
└──  models/                       # Saved model artifacts
    └──  ... (serialized models)
```

### **Architecture Overview**

The new_pipeline architecture (v4.0+) follows a layered design:

1. **Data Loading** → FanGraphs, Baseball Prospectus, Statcast integration
2. **Feature Engineering** → Confidence scoring, injury detection, elite identification
3. **Transformation** → sklearn-compatible feature pipelines
4. **Modeling** → Multi-quantile current season, ROS, and future projections
5. **Post-Processing** → Elite adjustments, constraint optimization
6. **Visualization** → Interactive notebooks and analysis

**Key Features:**

- Modular design with 24 specialized modules
- Intelligent caching (~195MB preprocessed data)
- GPU acceleration via TensorFlow
- Comprehensive test suite (18 pytest modules)

For detailed architecture documentation, data flow diagrams, and interface contracts, see **[ARCHITECTURE.md](documents/ARCHITECTURE.md)**.

## Usage

### **Quick Start**

**1. Main Analysis (Recommended Entry Point):**

``` bash
jupyter notebook new_pipeline/notebooks/sWARm_overview.ipynb
```

Complete sWARm analysis with current season and future projections overview.

**2. Current Season Deep Dive:**

``` bash
# Detailed current season analysis
jupyter notebook new_pipeline/notebooks/sWARm_deep_dive.ipynb

# Hitter-specific pipeline
jupyter notebook new_pipeline/notebooks/hitters/hitter_pipeline_main.ipynb

# Pitcher-specific pipeline
jupyter notebook new_pipeline/notebooks/pitchers/pitcher_pipeline_main.ipynb
```

**3. Future Projections:**

``` bash
# Overview of 1-3 year projections
jupyter notebook new_pipeline/notebooks/sWARm_future_overview.ipynb

# Detailed future projections analysis
jupyter notebook new_pipeline/notebooks/sWARm_future_deep_dive.ipynb
```

**4. Modular Development:**

```python
# Import current season models
from new_pipeline.models.current_season import MultiQuantileHistGBRegressor
from new_pipeline.common.loaders.hitter_loaders import load_k_pct_all_years

# Import rest-of-season models
from new_pipeline.models.ros import HitterROSModel, PitcherROSModel

# Import future projection system
from new_pipeline.models.future_season import (
    JointProjectionModel,
    FutureProjectionPipeline,
    apply_elite_adjustments
)

# Load data and run projections
from new_pipeline.notebooks.shared.pipeline_runner import load_historical_data
data = load_historical_data('hitter', years=range(2016, 2025))

# Initialize models
model = MultiQuantileHistGBRegressor(quantiles=[0.1, 0.5, 0.9])
future_pipeline = FutureProjectionPipeline(player_type='hitter', base_year=2024)
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

**Hitter Features (Enhanced Analytics):**

- **Plate Discipline**: K% (strikeout rate), BB% (walk rate)
- **Offensive Production**: AVG (batting average), OBP (on-base percentage), SLG (slugging percentage)
- **Enhanced Analytics**: Enhanced_Baserunning, Enhanced_Defense, plate appearances (PA)
- **Statcast Integration**: Exit velocity, launch angle, contact quality metrics

**Pitcher Features (Advanced Metrics):**

- **Workload & Control**: IP (innings pitched), BB% (walk rate), K% (strikeout rate)
- **Run Prevention**: ERA (earned run average), LOB% (left on base percentage)
- **Contact Quality**: Hard%, Med%, Soft% contact rates, Contact Quality Index
- **Advanced Analytics**: Damage control ratio, opportunity success, HBP%, WP
- **Statcast Features**: Statcast Launch Quality Index

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
| ------- | ----------- | ------------ | ------------------- | ------- |
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
| ------- | --------- | ----------- | -------- | ---------- |
| **RandomForest** | 0.210 | 1.375 | **0.853** | **0.690** |
| **SVR** | **0.323** | **1.273** | 0.482 | 1.294 |
| **Keras** | 0.279 | 1.313 | 0.438 | 1.347 |
| **Ridge** | 0.299 | 1.295 | 0.420 | 1.369 |
| KNN | -0.078 | 1.606 | 0.936 | 0.455 |
| XGBoost | 0.186 | 1.395 | 0.764 | 0.874 |

**Pitcher Performance:**

| Model | WARP R² | WARP RMSE | WAR R² | WAR RMSE |
| ------- | --------- | ----------- | -------- | ---------- |
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

## Documentation

For detailed information about the project:

- **[METHODOLOGY.md](documents/METHODOLOGY.md)** - Research methodology, data sources, and methodological contributions
- **[ARCHITECTURE.md](documents/ARCHITECTURE.md)** - Technical architecture, module dependencies, and development workflow
- **[CHANGELOG.md](documents/CHANGELOG.md)** - Complete version history from v0.1.0 to v4.0.0

### Quick Summary

**Data Sources**: FanGraphs, Baseball Prospectus, MLB Statcast (2016-2024)

**Original Contributions**: Multi-quantile uncertainty quantification, future projection framework with survival modeling, elite player adjustment system, temporal validation methodology

**Copyright**: All code, analysis, and documentation © 2025 Siddharth Nair. Baseball statistics used under fair use provisions.

## License Information

### **Mozilla Public License 2.0**

This project is licensed under the **Mozilla Public License 2.0 (MPL-2.0)**, a copyleft license that balances open source collaboration with commercial flexibility.

### **What This License Allows**

For complete license terms, see [LICENSE](LICENSE) file.

## Data Sources

- Data from [Baseball Savant](https://baseballsavant.mlb.com/) (MLB Statcast data)
- Data from [FanGraphs](https://www.fangraphs.com/) (advanced baseball statistics and personally my source of baseball news!)
- Data from [Baseball Prospectus](https://www.baseballprospectus.com/) (player and team analysis)
- Zip files w/ currently unused data from [Retrosheets](https://www.retrosheets.org/) – Historical game logs and play-by-play data

[^2]: Roughly 1000 WAR exists per season, so depending on a given player performs compared to the average, their share changes year-over-year[^3].

[^3]: An example we can look at to illustrate this is say a 10 WAR player decides to suddenly retire, with no new players coming in to the MLB. One simple assumption to make is that their team would be projected to win 10 fewer games in the upcoming season, but that's not far enough. The next, and most important assumption you must make is that the wins from the retired player would get redistributed across the league to the teams that his team would be playing (so if we looked at all of the projected win totals of the different teams that this team was playing, the sum of the deltas between the projection of their win totals before and after this player retired should equal approximately ten). Some value may go to his teammates, but most doesn't.

[^1]: This is something I have some minor quibbles with due to the disparity between hitting and pitching talent in baseball (i.e. I think that batters should have a slightly higher level set for their replacement level when compared to pitchers, because the average AAA batter called up will probably do better than the average AAA pitcher because I think that better pitchers get called up faster than better batters so the general level of batters stuck in the AAA is higher), but the people who came up with this are much smarter than me so consider this young man yelling at clouds.
