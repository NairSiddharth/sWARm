# sWARm Architecture Documentation

Technical architecture details for the sWARm (Sid Wins Above Replacement Metric) project.

---

## Table of Contents

- [Overview](#overview)
- [Key Architecture Features](#key-architecture-features)
- [Module Interdependencies](#module-interdependencies)
- [Data Flow](#data-flow)
- [Interface Contracts](#interface-contracts)
- [Performance Optimizations](#performance-optimizations)
- [Development Workflow](#development-workflow)

---

## Overview

sWARm uses a modular, layered architecture designed for:
- **Maintainability**: Clear separation of concerns across specialized modules
- **Scalability**: Easy addition of new features and models
- **Performance**: Intelligent caching and GPU acceleration
- **Testability**: Comprehensive test coverage with pytest

The architecture follows a three-tier projection system:
1. **Current Season**: Real-time WAR calculations with uncertainty quantification
2. **Rest-of-Season (ROS)**: Projections for remainder of current season
3. **Future (1-3 years)**: Multi-year projections with survival modeling

---

## Key Architecture Features

### Modular Design

- **Specialized modules** for maintainability and scalability
- **Separation of concerns**: Each module handles specific functionality
- **Dependency injection**: Modules can be easily swapped or updated
- **sklearn-compatible**: Transformers follow scikit-learn API conventions

### Performance Optimizations

- **Intelligent caching**: Preprocessed data stored for rapid access (~195MB cache)
- **Lazy loading**: Data loaded only when needed
- **GPU acceleration**: TensorFlow automatically detects and uses available GPUs
- **Multi-quantile prediction**: Simultaneous 10th, 50th, 90th percentile forecasts

### Data Pipeline

```
Raw Data → Processing Modules → Cache → Analysis
```

- **Comprehensive validation** at each stage
- **Automatic cache invalidation** when source data changes
- **Multi-source integration**: FanGraphs + Baseball Prospectus + Statcast

### Development Workflow

- **Main notebooks** for complete analysis (`sWARm_overview.ipynb`, `sWARm_deep_dive.ipynb`)
- **Modular components** for targeted development
- **Hitter/Pitcher separation**: Separate pipelines for position-specific modeling
- **Comprehensive testing**: 18 pytest modules covering all components

---

## Module Interdependencies

### New Pipeline Architecture (v4.0+)

```
new_pipeline/
│
├── common/                                 # Shared functionality
│   ├── loaders/                           # Data loading layer
│   │   ├── hitter_loaders.py              # FanGraphs + BP hitter data
│   │   └── pitcher_loaders.py             # FanGraphs + BP pitcher data
│   │
│   ├── features/                          # Feature engineering layer
│   │   ├── confidence_scorer.py           # Performance confidence scoring
│   │   ├── injury_recovery.py             # Injury impact features
│   │   ├── elite_detection.py             # Elite player identification
│   │   ├── rookie_detection.py            # Rookie classification
│   │   └── age_curves.py                  # Age adjustment factors
│   │
│   ├── transformers/                      # Transformation layer
│   │   ├── hitter_transformer.py          # sklearn-style hitter pipeline
│   │   └── pitcher_transformer.py         # sklearn-style pitcher pipeline
│   │
│   ├── data_preparation/                  # Preprocessing utilities
│   └── projections/                       # Projection helpers
│
├── models/                                 # Modeling layer
│   ├── current_season/                    # Current season models
│   │   ├── multi_quantile_histgb.py       # Multi-quantile ensemble
│   │   └── keras_utils.py                 # Neural network utilities
│   │
│   ├── ros/                               # Rest-of-season models
│   │   ├── hitter_ros_model.py            # Hitter ROS projections
│   │   └── pitcher_ros_model.py           # Pitcher ROS projections
│   │
│   └── future_season/                     # Future projection models
│       ├── data_preparation.py            # Historical data loading
│       ├── longitudinal_model.py          # Year-to-year WAR modeling
│       ├── survival_model.py              # Cox PH retirement probability
│       ├── age_curves.py                  # Position-specific aging
│       ├── joint_projection.py            # Combined WAR + survival + aging
│       ├── elite_player_adjuster.py       # Elite protection system
│       ├── elite_adjustments.py           # Adjustment wrappers
│       ├── injury_recovery.py             # Injury impact modeling
│       ├── constraint_optimizer.py        # Constraint enforcement
│       ├── expected_stats.py              # Expected statistics
│       ├── temporal_validation.py         # Validation framework
│       └── future_projection_pipeline.py  # End-to-end orchestration
│
├── notebooks/                              # Analysis notebooks
│   ├── sWARm_overview.ipynb               # Main entry point
│   ├── sWARm_deep_dive.ipynb              # Detailed analysis
│   ├── sWARm_future_overview.ipynb        # Future projections
│   ├── sWARm_future_deep_dive.ipynb       # Future detailed analysis
│   ├── hitters/                           # Hitter-specific
│   ├── pitchers/                          # Pitcher-specific
│   └── shared/                            # Shared utilities
│       ├── pipeline_runner.py
│       └── table_utils.py
│
└── tests/                                  # Test suite
    ├── test_ensemble_model.py
    ├── test_ensemble_model_pitcher.py
    ├── test_hitter_features.py
    ├── test_pitcher_features.py
    ├── test_integration.py
    ├── test_temporal_validation.py
    └── ... (12 more test modules)
```

---

## Data Flow

### 1. Data Loading
```
FanGraphs CSV → Loaders → Dict[(playerid, year)] → value
Baseball Prospectus CSV → Loaders → Dict[(playerid, year)] → value
Statcast Data → Loaders → Dict[(playerid, year)] → value
```

### 2. Feature Engineering
```
Raw Stats → Feature Engineering Layer → Enhanced Features
├── Injury features (games missed, severity)
├── Elite detection (6+ WAR identification)
├── Rookie classification (MLB debut season)
├── Age curves (position-specific adjustments)
└── Confidence scores (0-8 scale)
```

### 3. Transformation
```
Enhanced Features → sklearn Transformers → Model-Ready DataFrames
├── Standardization
├── Missing value imputation
├── Feature selection
└── Composite feature creation
```

### 4. Modeling
```
Transformed Data → Models → Predictions with Uncertainty
├── Current Season: MultiQuantileHistGB (10th, 50th, 90th percentiles)
├── ROS: Position-specific models with time decay
└── Future: Joint projection (WAR + survival + aging)
```

### 5. Post-Processing
```
Raw Predictions → Adjustments → Final Projections
├── Elite player protection (MVP/Superstar/All-Star tiers)
├── Constraint optimization (min/max bounds)
├── Injury recovery adjustments
└── Expected statistics integration
```

### 6. Output
```
Final Projections → Notebooks → Visualization & Analysis
├── Interactive Plotly charts
├── Comparison tables
├── Uncertainty bands
└── Export to CSV
```

---

## Interface Contracts

### Loaders
- **Input**: Years (range or list), player type ('hitter' or 'pitcher')
- **Output**: `Dict[(playerid: int, year: int)] → value: float`
- **Example**: `{(123456, 2024): 0.285}` for batting average

### Transformers
- **Interface**: sklearn-compatible `BaseEstimator` and `TransformerMixin`
- **Methods**:
  - `fit(X, y=None)`: Learn transformation parameters
  - `transform(X)`: Apply transformation
  - `fit_transform(X, y=None)`: Fit and transform in one step
- **Input**: pandas DataFrame with MLBAMID and Year columns
- **Output**: pandas DataFrame with added feature columns

### Models
- **Input**: pandas DataFrame with feature columns
- **Output**:
  - Current Season: DataFrame with `war_q10`, `war_q50`, `war_q90` columns
  - ROS: DataFrame with projected WAR for remainder of season
  - Future: DataFrame with `war_year_1`, `war_year_2`, `war_year_3`, `survival_prob`
- **Methods**:
  - `fit(X, y)`: Train model
  - `predict(X)`: Generate predictions

### Pipelines
- **Interface**: End-to-end orchestration classes
- **Methods**:
  - `__init__(player_type, base_year, **kwargs)`: Initialize pipeline
  - `run()` or `project()`: Execute full pipeline
  - `validate()`: Run temporal validation
- **Output**: Complete projection DataFrame with all features and predictions

---

## Performance Optimizations

### Caching Strategy

```
cache/
├── comprehensive_fangraphs_data.json      # ~80MB
├── enhanced_baserunning_values.json       # ~20MB
├── fielding_oaa_values_v4_seasonal.json   # ~15MB
└── ... (more cache files)
```

- **Cache invalidation**: Automatic based on source file modification times
- **Load time**: 30-60 seconds (vs 15-30 minutes without cache)

### GPU Acceleration

- **TensorFlow**: Automatic GPU detection and utilization
- **Speedup**: ~3-5x for neural network components
- **Requirements**: NVIDIA GPU with CUDA 11.8+ and cuDNN 8.6+

### Memory Management

- **Lazy loading**: Data loaded only when accessed
- **Generator patterns**: Used for large dataset iteration
- **Chunked processing**: Large CSV files processed in chunks

---

## Development Workflow

### Adding a New Feature

1. **Create loader** in `new_pipeline/common/loaders/`
2. **Add to transformer** in `new_pipeline/common/transformers/`
3. **Update model** to use new feature
4. **Add tests** in `new_pipeline/tests/`
5. **Update notebooks** to visualize new feature

### Adding a New Model

1. **Create model class** in appropriate `new_pipeline/models/` subdirectory
2. **Implement sklearn-compatible interface** (fit, predict)
3. **Add to pipeline** orchestration
4. **Create tests** for model
5. **Document** in model docstrings and this file

### Running Tests

```bash
# All tests
pytest new_pipeline/tests/

# Specific test module
pytest new_pipeline/tests/test_ensemble_model.py

# With coverage
pytest --cov=new_pipeline new_pipeline/tests/
```

---

## Migration from v3.x to v4.0

### Old → New Module Mapping

| Old Module | New Location |
|------------|--------------|
| `common_modules/ensemble_modeling.py` | `new_pipeline/models/current_season/multi_quantile_histgb.py` |
| `current_season_modules/predictive_modeling.py` | `new_pipeline/models/current_season/` |
| `future_season_modules/future_projections.py` | `new_pipeline/models/future_season/joint_projection.py` |
| `common_modules/elite_adjustment.py` | `new_pipeline/models/future_season/elite_player_adjuster.py` |
| `common_modules/confidence_scorer.py` | `new_pipeline/common/features/confidence_scorer.py` |

### Import Changes

```python
# OLD (v3.x)
from common_modules.ensemble_modeling import EnsembleWARPredictor
from future_season_modules.expected_stats import ExpectedStatsCalculator

# NEW (v4.0+)
from new_pipeline.models.current_season import MultiQuantileHistGBRegressor
from new_pipeline.models.future_season import JointProjectionModel
```

---

## See Also

- [README.md](../README.md) - Project overview and quick start
- [CHANGELOG.md](CHANGELOG.md) - Version history
- [METHODOLOGY.md](METHODOLOGY.md) - Research methodology and citations
