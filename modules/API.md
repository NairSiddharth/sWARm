# sWARm API Reference

This document provides a comprehensive reference for the sWARm modular architecture, covering essential functions and integration patterns for developers working with the codebase.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Core Data Pipeline](#core-data-pipeline)
3. [Machine Learning & Modeling](#machine-learning--modeling)
4. [Data Processing & Enhancement](#data-processing--enhancement)
5. [Visualization & Analysis](#visualization--analysis)
6. [Utility Modules](#utility-modules)
7. [Integration Patterns](#integration-patterns)
8. [Common Workflows](#common-workflows)

## Architecture Overview

### **Modular Design Philosophy**

sWARm employs a **24-module architecture** with clear separation of concerns:

* **Data Pipeline**: Loading, cleaning, validation
* **Feature Engineering**: Advanced metrics and calculations
* **Machine Learning**: Model training and evaluation
* **Visualization**: Interactive analysis and reporting
* **Utilities**: Helper functions and optimizations

### **Import Conventions**

```python
# Main entry point
from modularized_data_parser import *

# Core modules
from modules.modeling import run_comprehensive_modeling
from modules.data_visualization import plot_consolidated_model_comparison

# Specific functionality
from modules.baserunning_analytics import calculate_enhanced_baserunning_values
from modules.defensive_metrics import clean_defensive_players
```

## Core Data Pipeline

### **modularized_data_parser.py** - Main Entry Point

#### `run_comprehensive_analysis()`

**Purpose**: Execute the complete data loading and analysis pipeline

**Returns**: Dictionary containing processed datasets and analysis results

**Example**:

```python
results = run_comprehensive_analysis()
# Returns comprehensive data with hitter/pitcher stats, enhanced metrics
```

**Cache**: Uses multiple cache files for performance optimization

---

#### `quick_player_lookup(player_name)`

**Purpose**: Fast lookup of player statistics across all datasets

**Parameters**:

* `player_name` (str): Player name to search for

**Returns**: Prints player statistics summary

**Example**:

```python
quick_player_lookup("Mike Trout")
# Outputs: WAR, WARP, position, and FanGraphs data
```

---

#### `get_all_player_stats(player_name, enhanced=True)`

**Purpose**: Retrieve comprehensive player statistics

**Parameters**:

* `player_name` (str): Player name
* `enhanced` (bool): Include enhanced baserunning/defensive metrics

**Returns**: Dictionary with complete player data

**Example**:

```python
stats = get_all_player_stats("Shohei Ohtani", enhanced=True)
# Returns WAR, WARP, enhanced metrics, historical data
```

### **data_loading.py** - Core Data Functions

#### `clean_yearly_warp_hitter()` / `clean_yearly_warp_pitcher()`

**Purpose**: Load and clean Baseball Prospectus WARP data with manual K%/BB% calculations

**Returns**: DataFrame with cleaned WARP statistics (2016-2024)

**Cache**: `yearly_warp_hitter_cleaned.json`, `yearly_warp_pitcher_cleaned_v2.json`

---

#### `clean_comprehensive_fangraphs_war()`

**Purpose**: Load comprehensive FanGraphs WAR data with 50+ features

**Returns**: DataFrame with extensive player metrics

**Cache**: `comprehensive_fangraphs_war_cleaned.json`

### **enhanced_data_loading.py** - Advanced Integration

#### `load_comprehensive_fangraphs_data()`

**Purpose**: Load and combine 5 FanGraphs dataset types

**Returns**: Dictionary with hitter, pitcher, and defensive datasets

**Features**: Combines basic, advanced, and standard statistics across all years

## Machine Learning & Modeling

### **modeling.py** - Core ML Functions

#### `run_comprehensive_modeling()`

**Purpose**: Execute complete machine learning pipeline with all models

**Returns**: ModelResults object containing predictions and metrics

**Models**: Ridge, KNN, RandomForest, XGBoost, SVR, Neural Networks

**Example**:

```python
model_results = run_comprehensive_modeling()
# Trains all models and returns performance metrics
```

---

#### `run_basic_regressions(train_test_splits, model_results, print_fn, plot_fn)`

**Purpose**: Train linear regression models

**Parameters**:

* `train_test_splits`: Prepared training/testing data
* `model_results`: ModelResults container
* `print_fn`: Function for metrics output
* `plot_fn`: Function for visualization

---

#### `run_neural_network(train_test_splits, model_results, print_fn, plot_fn, history_fn)`

**Purpose**: Train Keras neural networks with GPU acceleration

**Features**: AdamW optimizer, early stopping, dropout regularization

**GPU**: Automatically detects and uses available GPUs

---

#### `select_best_models_by_category(model_results)`

**Purpose**: Automatically select best-performing models

**Returns**: List of model names with highest performance

**Categories**: Overall performance, hitter prediction, pitcher prediction

### **temporal_modeling.py** - Time Series Analysis

#### Functions for future season prediction and trend analysis

**Features**: Chronological progression, confidence intervals, age curve adjustments

## Data Processing & Enhancement

### **bp_derived_stats.py** - Manual Calculations

#### `load_fixed_bp_data(data_dir=None)`

**Purpose**: Load Baseball Prospectus data with manually calculated K%/BB% for pre-2020

**Returns**: Tuple of (hitter_df, pitcher_df) with 100% feature coverage

**Key Feature**: Resolves correlation gaps in pre-2020 data

---

#### `fix_bp_derived_statistics(df, year)`

**Purpose**: Calculate K% and BB% from raw counting stats

**Parameters**:

* `df`: DataFrame with SO, BB, PA columns
* `year`: Year for context

**Returns**: DataFrame with calculated percentage statistics

### **baserunning_analytics.py** - Enhanced Baserunning

#### `calculate_enhanced_baserunning_values()`

**Purpose**: Calculate run expectancy-based baserunning values

**Returns**: Dictionary mapping player names to baserunning value

**Method**: Uses situational run expectancy matrix

**Cache**: `enhanced_baserunning_values.json`

---

#### `get_baserunning_for_player(player_name, enhanced=True)`

**Purpose**: Get individual player baserunning statistics

**Parameters**:

* `player_name` (str): Player name
* `enhanced` (bool): Use enhanced calculations vs basic

**Returns**: Baserunning value for the player

### **defensive_metrics.py** - Advanced Defense

#### `clean_defensive_players()`

**Purpose**: Load and combine OAA and catcher framing data

**Returns**: Dictionary mapping player names to defensive values

**Features**: Position-specific weights, advanced fielding metrics

---

#### `get_positional_defensive_weights()`

**Purpose**: Get defensive importance weights by position

**Returns**: Dictionary with position adjustment factors

### **park_factors.py** - Ballpark Adjustments

#### `calculate_park_factors()` / `apply_enhanced_hitter_park_adjustments()`

**Purpose**: Calculate and apply ballpark environment adjustments

**Factor**: Currently using 1.2x multiplier (reduced from 1.5x in v1.x)

### **name_mapping_optimization.py** - Player Matching

#### `create_optimized_name_mapping_with_indices(source_df, target_df)`

**Purpose**: Create fuzzy name matching between datasets

**Parameters**:

* `source_df`: Source dataset DataFrame
* `target_df`: Target dataset DataFrame

**Returns**: Dictionary mapping source indices to target indices

**Features**: Handles duplicate names, fuzzy matching with rapidfuzz

## Visualization & Analysis

### **data_visualization.py** - Interactive Plots

#### `plot_consolidated_model_comparison(model_results, model_names=None, show_residuals=True, show_metrics=True)`

**Purpose**: Create unified model comparison visualizations

**Parameters**:

* `model_results`: ModelResults from training
* `model_names`: List of models to include
* `show_residuals`: Include residual analysis
* `show_metrics`: Display performance metrics

**Returns**: Plotly interactive visualization

---

#### `plot_quadrant_analysis_px_toggle(model_results, season_col="Season", model_names=None, show_hitters=True, show_pitchers=True)`

**Purpose**: Create quadrant analysis with accuracy zones

**Features**: Interactive toggles, dual accuracy standards (±1 WAR/WARP)

---

#### `plot_war_warp_animated(model_results, season_col="Season", model_names=None, show_hitters=True, show_pitchers=True)`

**Purpose**: Generate animated temporal analysis

**Features**: Chronological progression (2016-2024), smooth transitions

### **animated_analysis_clean.py** - Advanced Animations

#### `create_animated_model_comparison()`

**Purpose**: Create sophisticated animated visualizations

**Features**: Cinematic themes, 3D surfaces, performance heatmaps

## Utility Modules

### **duplicate_names.py** - Name Conflict Resolution

#### Functions for handling players with identical names

**Features**: Position-based disambiguation, team-based separation

### **two_way_players.py** - Dual Position Handling

#### `get_cleaned_two_way_data()`

**Purpose**: Identify and process two-way players

**Definition**: MLB standard (20+ innings pitched AND 20+ games as position player)

### **test_modules.py** - Testing Framework

#### Functions for module validation and testing

**Features**: Unit tests, integration tests, performance benchmarks

## Integration Patterns

### **Standard Import Pattern**

```python
# Main functionality
from modularized_data_parser import run_comprehensive_analysis

# Modeling pipeline
from modules.modeling import run_comprehensive_modeling, select_best_models_by_category

# Visualization
from modules.data_visualization import plot_consolidated_model_comparison

# Enhanced metrics
from modules.baserunning_analytics import calculate_enhanced_baserunning_values
from modules.defensive_metrics import clean_defensive_players
```

### **Data Flow Pipeline**

```python
# 1. Load comprehensive data
results = run_comprehensive_analysis()

# 2. Prepare training data
train_test_splits = prepare_train_test_splits()

# 3. Train models
model_results = run_comprehensive_modeling()

# 4. Select best models
best_models = select_best_models_by_category(model_results)

# 5. Visualize results
plot_consolidated_model_comparison(model_results, best_models)
```

### **Player Analysis Workflow**

```python
# Quick lookup
quick_player_lookup("Mike Trout")

# Comprehensive analysis
stats = get_all_player_stats("Mike Trout", enhanced=True)

# Enhanced metrics
baserunning = get_baserunning_for_player("Mike Trout")
```

## Common Workflows

### **Complete Analysis Pipeline**

```python
# Cache control (optional)
FORCE_CACHE_REBUILD = True  # Set in notebook

# Full system analysis
comprehensive_data = run_comprehensive_analysis()
model_results = run_comprehensive_modeling()

# Visualization and results
best_models = select_best_models_by_category(model_results)
plot_consolidated_model_comparison(model_results, best_models)
```

### **Custom Model Training**

```python
# Prepare data
train_test_splits = prepare_train_test_splits()

# Train specific models
model_results = ModelResults()
run_basic_regressions(train_test_splits, model_results, print_metrics, plot_results)
run_neural_network(train_test_splits, model_results, print_metrics, plot_results, plot_training_history)
```

### **Player Comparison Analysis**

```python
players = ["Mike Trout", "Mookie Betts", "Ronald Acuña Jr."]

for player in players:
    stats = get_all_player_stats(player, enhanced=True)
    quick_player_lookup(player)
```

### **Performance Optimization**

```python
# Use caching for repeated operations
enhanced_baserunning = calculate_enhanced_baserunning_values()  # Cached
defensive_metrics = clean_defensive_players()  # Cached

# GPU acceleration (automatic)
# TensorFlow automatically detects and uses available GPUs
model_results = run_neural_network(...)  # GPU accelerated if available
```

---

## Key Performance Notes

* **Caching**: Most functions use intelligent caching for performance
* **GPU Support**: Neural networks automatically use GPU if available
* **Memory Usage**: ~500MB for full dataset, 2-8GB during model training
* **Processing Time**: Complete analysis ~5-15 minutes (varies by hardware)

## Error Handling

* **Missing Dependencies**: Graceful degradation (XGBoost, TensorFlow optional)
* **Data Validation**: Automatic infinite/NaN value handling
* **Cache Management**: Automatic cache rebuild when source data changes
* **GPU Detection**: Automatic fallback to CPU if GPU unavailable

This API reference covers the essential functions needed for working with the sWARm system. For complete function signatures and parameters, refer to the individual module source files.
