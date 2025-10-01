# Common Modules Directory

This directory contains shared utility modules and functions used across the sWARm (simplified WAR model) analytics system.

## Purpose
Houses reusable components that provide core functionality for data processing, feature engineering, statistical calculations, and modeling operations used by multiple parts of the sWARm system.

## Module Categories

### Data Processing & Feature Engineering
- **`enhanced_features.py`** - Advanced feature creation (contact quality, launch angle adjustments, batted ball metrics)
- **`historical_feature_preparation.py`** - Feature preparation for historical analysis and backtesting
- **`derived_stats.py`** - Calculation of derived baseball statistics from raw data
- **`fix_feature_compatibility.py`** - Feature compatibility and standardization utilities

### Statistical Calculations & Adjustments
- **`park_factors.py`** - Ballpark adjustment calculations and application
- **`positional_adjustments.py`** - Position-based WAR adjustments and defensive value calculations
- **`warp_calculator.py`** - Baseball Prospectus WARP (Wins Above Replacement Player) calculations
- **`elite_adjustment.py`** - Adjustments for elite player performance patterns
- **`confidence_scorer.py`** - Confidence scoring for projections and predictions

### Modeling & Machine Learning
- **`ensemble_modeling.py`** - Ensemble model with SEPARATE WAR/WARP training (improved accuracy)

### Specialized Calculations
- **`game_progress_calculator.py`** - In-season performance tracking and game-by-game analysis
- **`pitcher_workload_calculator.py`** - Pitcher projections using team-games-based participation rates
- **`scenario_projections.py`** - Alternative scenario modeling and what-if analysis
- **`position_normalizer.py`** - Position classification and normalization utilities

## Key Features

### Data Processing
- **Multi-Source Integration**: Handles FanGraphs, Baseball Prospectus, and Statcast data
- **Feature Engineering**: Creates advanced metrics from raw statistics
- **Data Validation**: Ensures data quality and consistency across sources
- **Missing Data Handling**: Robust imputation and error handling strategies

### Statistical Methods
- **Park Adjustments**: Neutralizes ballpark effects on player statistics
- **Position Adjustments**: Applies positional value corrections to WAR calculations
- **Age Adjustments**: Incorporates aging curves for future projections
- **Sample Size Adjustments**: Handles small sample size reliability issues

### Modeling Infrastructure
- **Ensemble Methods**: Combines multiple modeling approaches for improved accuracy
- **Cross-Validation**: Temporal validation for time-series baseball data
- **Model Persistence**: Saves and loads trained models with version control
- **Feature Selection**: Automated and manual feature importance analysis

## Usage Patterns

### Import Structure
```python
# Feature engineering
from common_modules.enhanced_features import create_contact_quality_features
from common_modules.derived_stats import calculate_advanced_metrics

# Adjustments
from common_modules.park_factors import apply_park_factors
from common_modules.positional_adjustments import calculate_positional_value

# Modeling
from common_modules.ensemble_modeling import create_ensemble_model
```

### Integration Points
- **Current Season Analysis**: Used by `current_season_modules/`
- **Future Projections**: Core components for `future_season_modules/`
- **Research**: Supporting functions for `research_notebooks/`
- **Main Notebooks**: Direct integration with `sWARm_CS.ipynb` and `sWARm_FutureProjections.ipynb`

## Data Flow

### Input Sources
- **Raw Data**: MLB Player Data repository
- **Processed Data**: Output from data loading modules
- **Model Inputs**: Features prepared by other common modules

### Output Destinations
- **Feature Sets**: Enhanced datasets for modeling
- **Adjusted Statistics**: Park/position/age adjusted metrics
- **Model Objects**: Trained ensemble models saved to `models/` directory
- **Calculated Metrics**: WAR, WARP, and derived statistics

## Quality Standards

### Code Requirements
- **Modular Design**: Single responsibility principle for each module
- **Error Handling**: Robust handling of edge cases and missing data
- **Documentation**: Comprehensive docstrings and inline comments
- **Testing**: Unit tests in `testing/` directory for critical functions

### Performance Considerations
- **Efficiency**: Optimized for large dataset processing
- **Memory Management**: Careful handling of memory-intensive operations
- **Caching**: Strategic caching of expensive calculations
- **Vectorization**: NumPy/Pandas optimized operations where possible

## Dependencies

### Core Libraries
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning algorithms
- **scipy**: Statistical functions
- **joblib**: Model serialization

### Baseball-Specific
- **Custom Logic**: Baseball-specific calculations and rules
- **Domain Knowledge**: Incorporation of baseball analytics best practices
- **Validation**: Cross-validation against known baseball metrics

## Future Development

### Planned Enhancements
- **Real-Time Processing**: Live data integration capabilities
- **Advanced Modeling**: Deep learning and neural network implementations
- **Uncertainty Quantification**: Confidence intervals and prediction ranges
- **API Integration**: RESTful interfaces for external system integration

### Maintenance Notes
- **Version Control**: Models saved with timestamps and version tracking
- **Backward Compatibility**: Maintained for historical analysis workflows
- **Documentation**: Keep module documentation current with code changes
- **Testing**: Expand test coverage for new features and edge cases

This directory forms the computational backbone of the sWARm system, providing reliable, tested, and reusable components for baseball analytics and WAR calculations.