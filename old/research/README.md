# Research Notebooks Directory

This directory contains exploratory analysis and research notebooks for the sWARm (simplified WAR model) system.

## Purpose
Houses experimental and research-focused Jupyter notebooks used for developing modeling approaches, testing hypotheses, and conducting deep-dive analyses that inform the main sWARm system.

## Current Notebooks

### `sWARm_CS_Historical.ipynb` - Historical Current Season Analysis
- **Purpose**: Historical backtesting and validation of current season modeling approaches
- **Focus**: Analysis of how well the current season model performs on historical data
- **Output**: Validation metrics and performance insights for current season predictions
- **Usage**: Research validation for current season modeling methodology

### `sWARm_AgeCurve.ipynb` - Age Curve Research
- **Purpose**: Analysis of player performance aging patterns across different positions and player types
- **Focus**: Developing age adjustment factors for future projections
- **Output**: Age curve coefficients and aging pattern insights
- **Usage**: Informs aging adjustments in future season projections

## Research Workflow

### Development Process
1. **Hypothesis Formation**: Initial research questions and approaches
2. **Exploratory Analysis**: Data exploration and pattern identification
3. **Model Development**: Prototype modeling approaches
4. **Validation**: Testing against historical data
5. **Integration**: Successful approaches migrate to main system modules

### Data Sources
- **Primary**: MLB Player Data repository (`../MLB Player Data/`)
- **Processed**: Outputs from main system modules
- **Historical**: Multi-season datasets for temporal analysis

## Integration with Main System

### Research to Production Pipeline
- **Successful Research** → Integration into `common_modules/` or `future_season_modules/`
- **Validation Results** → Updates to main notebooks (`sWARm_FutureProjections.ipynb`, `sWARm_CS.ipynb`)
- **Model Improvements** → Version updates in production system

### Dependencies
- **Data Loading**: Utilizes `future_season_modules/data_integration.py`
- **Feature Engineering**: May use functions from `common_modules/`
- **Model Components**: Accesses ensemble models from `models/` directory

## Research Standards

### Documentation Requirements
- **Clear Hypothesis**: Each notebook should state research questions
- **Methodology**: Document analytical approach and assumptions
- **Results Summary**: Key findings and implications
- **Next Steps**: Recommendations for further research or implementation

### Code Quality
- **Reproducible**: All analyses should be reproducible with available data
- **Modular**: Extract reusable functions for potential integration
- **Commented**: Extensive documentation of research decisions and findings

## Future Research Areas

### Potential Investigations
- **Contact Quality Modeling**: Advanced batted ball analysis
- **Injury Impact Modeling**: Quantifying performance impact of injuries
- **Two-Way Player Analysis**: Specialized modeling for pitcher/hitter combinations
- **Park Factor Evolution**: Temporal changes in ballpark effects
- **Positional Value**: Market-based positional adjustment refinements

### Experimental Features
- **Machine Learning**: Advanced ML approaches beyond current ensemble models
- **Real-Time Integration**: Live data incorporation methodologies
- **Uncertainty Quantification**: Confidence intervals and prediction ranges
- **Comparative Analysis**: Alternative WAR methodologies and benchmarking

## Output and Results

### Research Outputs
- **Findings**: Documented insights and discoveries
- **Model Prototypes**: Experimental modeling approaches
- **Validation Results**: Performance metrics and comparisons
- **Data Insights**: Previously unknown patterns or relationships

### Integration Path
- **Successful Research**: Migrate to appropriate system modules
- **Failed Experiments**: Document lessons learned for future reference
- **Partial Success**: Refine and iterate in subsequent research cycles

This directory serves as the innovation hub for the sWARm system, where new ideas are tested and validated before integration into the production analytics pipeline.