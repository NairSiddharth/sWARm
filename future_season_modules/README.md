# Future Season Modules Directory

This directory contains specialized modules for multi-year player projections and future season analysis in the sWARm (simplified WAR model) system.

## Purpose
Houses components specifically designed for projecting player performance 1-3 years into the future, incorporating aging curves, expected statistics regression, injury impact, and constraint optimization to ensure realistic projections.

## Core Architecture

### Data Pipeline Modules
- **`data_loader.py`** - Loads and preprocesses data for future projections
- **`data_integration.py`** - Integrates multiple data sources and handles missing data
- **`integration.py`** - Master integration module coordinating data flow and processing

### Statistical Projection Modules
- **`future_projections.py`** - Core projection algorithms and aging curve application
- **`expected_stats.py`** - Expected statistics calculations (xBA, xSLG, xwOBA) for regression analysis
- **`constraint_optimizer.py`** - Ensures zero-sum WAR constraints and realistic league totals

### Specialized Analysis Modules
- **`injury_impact_analyzer.py`** - Quantifies injury impact on future performance
- **`player_profile_classifier.py`** - Classifies players into performance archetypes
- **`two_way_player_model.py`** - Specialized modeling for pitcher/hitter combinations
- **`player_role_validator.py`** - Validates player role assignments and eligibility

### System Coordination
- **`pipeline_orchestrator.py`** - Coordinates the full projection pipeline
- **`validation.py`** - Cross-validation and model performance assessment

## Key Features

### Projection Methodology
- **Aging Curves**: Position-specific performance decline modeling
- **Regression to Mean**: Expected statistics integration for sustainable performance
- **Sample Size Adjustments**: Reliability weighting based on playing time
- **Multi-Year Projections**: 1, 2, and 3-year forward projections

### Advanced Analytics
- **Expected Statistics**: Statcast-based performance expectations
- **Injury Modeling**: Recovery timelines and performance impact
- **Role Transitions**: Player position and role change predictions
- **Market Constraints**: Realistic playing time and opportunity distribution

### Quality Assurance
- **Zero-Sum Constraints**: League WAR totals maintained at ~1000
- **Cross-Validation**: Temporal validation using historical data
- **Outlier Detection**: Identification and handling of extreme projections
- **Uncertainty Quantification**: Confidence ranges for projections

## Module Integration

### Data Flow Pipeline
```
data_loader.py → data_integration.py → integration.py
                                          ↓
expected_stats.py → future_projections.py → constraint_optimizer.py
                                          ↓
player_profile_classifier.py → two_way_player_model.py → validation.py
                                          ↓
pipeline_orchestrator.py → results/future_projections_YYYY.csv
```

### Dependencies
- **Common Modules**: Extensive use of `common_modules/` for core calculations
- **MLB Data**: Primary data source from `MLB Player Data/` repository
- **Models**: Trained ensemble models from `models/` directory
- **Results**: Output generation to `results/` directory

## Projection Components

### Core Algorithms
- **Longitudinal Modeling**: Multi-year performance tracking
- **Survival Analysis**: Player career length and retirement probability
- **Ensemble Methods**: Multiple model combination for robust projections
- **Bayesian Updating**: Prior performance integration with recent data

### Adjustment Factors
- **Age Curves**: Position-specific aging patterns
- **Park Factors**: Ballpark neutralization for fair comparison
- **Position Adjustments**: Positional value incorporation
- **Usage Patterns**: Playing time and role-based adjustments

### Constraint Management
- **League Totals**: WAR sum constraints at league level
- **Position Scarcity**: Realistic position player distributions
- **Playing Time**: Opportunity-based projection scaling
- **Market Reality**: Contract and roster construction considerations

## Specialized Features

### Injury Impact Analysis
- **Recovery Modeling**: Performance recovery timelines post-injury
- **Risk Assessment**: Injury probability based on player history
- **Performance Impact**: Quantified performance decline from injuries
- **Career Longevity**: Injury impact on career length projections

### Two-Way Player Handling
- **Dual Role Projections**: Separate hitting and pitching projections
- **Usage Optimization**: Optimal role allocation for two-way players
- **Development Tracking**: Evolution of two-way player capabilities
- **Market Value**: Combined hitting/pitching value calculations

### Player Classification
- **Performance Archetypes**: Star, regular, replacement level classification
- **Development Curves**: Young player progression modeling
- **Decline Patterns**: Veteran player aging and decline trajectories
- **Role Specialization**: Position and role-specific projections

## Output and Results

### Projection Files
- **`results/future_projections_YYYY.csv`** - Annual projection outputs
- **Multi-Year Projections**: 1, 2, and 3-year forward projections
- **WARP Equivalents**: Baseball Prospectus metric conversions
- **Confidence Metrics**: Projection reliability indicators

### Validation Reports
- **`results/validation_report.csv`** - Model performance metrics
- **Historical Accuracy**: Backtesting against known outcomes
- **Cross-Validation**: Temporal validation results
- **Model Comparison**: Alternative methodology benchmarking

### Diagnostic Outputs
- **`results/issues/dropped_players_log.txt`** - Data quality issues
- **Player Exclusions**: Insufficient data or eligibility issues
- **Processing Warnings**: Data quality and edge case handling
- **Performance Flags**: Unusual projection results requiring review

## Usage in Main System

### Notebook Integration
- **Primary**: `sWARm_FutureProjections.ipynb` - Main projection notebook
- **Research**: `research_notebooks/` - Experimental projection approaches
- **Validation**: Historical backtesting and accuracy assessment

### API and Workflow
```python
# Core projection workflow
from future_season_modules.pipeline_orchestrator import run_full_projection_pipeline
from future_season_modules.validation import validate_projections

# Data integration
from future_season_modules.data_integration import load_integrated_data
from future_season_modules.integration import process_all_data

# Specialized components
from future_season_modules.two_way_player_model import project_two_way_player
from future_season_modules.constraint_optimizer import apply_zero_sum_constraints
```

## Quality Standards

### Validation Requirements
- **Historical Accuracy**: R² > 0.3 for 1-year projections
- **Constraint Compliance**: Zero-sum WAR within ±5 of 1000
- **Outlier Management**: < 1% of projections beyond realistic bounds
- **Data Coverage**: > 95% of eligible players included

### Performance Metrics
- **Processing Speed**: Full projection pipeline < 10 minutes
- **Memory Efficiency**: Handles 2000+ player datasets
- **Error Handling**: Graceful degradation for missing data
- **Reproducibility**: Consistent results across runs

## Future Development

### Planned Enhancements
- **Real-Time Updates**: Live projection updates during season
- **Advanced ML**: Deep learning integration for complex patterns
- **Market Integration**: Contract and salary projection components
- **International Players**: NPB, KBO, and other league integration

### Research Areas
- **Uncertainty Quantification**: Bayesian projection confidence intervals
- **Causal Inference**: Treatment effect analysis for player development
- **Multi-Modal Learning**: Video and tracking data integration
- **Dynamic Aging**: Personalized aging curve development

This directory represents the cutting edge of baseball projection methodology, combining traditional sabermetrics with modern analytics to provide comprehensive, reliable future performance estimates for baseball decision-making.