# oWAR System Architecture Plan

## Executive Summary

This document outlines the comprehensive plan for building two complementary WAR systems:
1. **Current Season WAR Calculator** - Real-time WAR calculation during the season
2. **Future Performance Projections** - ZIPS-style prediction system for future seasons

## System Architecture Overview

### System 1: Current Season WAR Calculator
**Purpose:** Calculate WAR for players within the current season
**Method:** Within-Year K-Fold Cross-Validation
**Output:** Real-time WAR values with confidence intervals

### System 2: Future Performance Projections (COMPLETED)
**Purpose:** Predict future season performance
**Method:** Temporal GroupKFold Cross-Validation
**Output:** Projected stats and WAR for upcoming seasons
**Status:** ✅ Implemented in sWARM_test.ipynb

## Current Implementation Status

### System 2 - Completed Features:
- **Temporal validation**: Predicts future years using historical data
- **Comprehensive coverage**: 90,236 predictions across 5,248 player-seasons
- **Multiple models**: Ridge, RandomForest, SVR, Keras
- **Performance metrics**: R² = 0.69 (Keras WAR), R² = 0.73 (RandomForest WARP)
- **Year-by-year analysis**: 2016-2024 temporal validation
- **Data organization**: Clean folder structure with BP, FanGraphs, Statcast data
- **Advanced features**: MLBID matching, positional adjustments, enhanced metrics

### System 1 - To Be Implemented:
**Current Gap:** No within-season WAR calculation capability

## Implementation Plan

### Phase 1: System 1 Development (Within-Year WAR Calculator)

#### Features to Include:
```
Hitting: K%, BB%, AVG, OBP, SLG, wRC+
Defense: Positional_WAR, Enhanced_Defense, fielding metrics
Baserunning: Enhanced_Baserunning, SB%, baserunning runs
Park Factors: Park-adjusted offensive stats (year-specific)
Contextual: Position, PA (for playing time weighting)
```

#### Training Strategy:
```python
# For each year 2016-2024:
for year in years:
    year_data = get_year_data(year)

    # Split players into 5 folds
    player_folds = split_players_into_folds(year_data, n_folds=5)

    for fold in range(5):
        # Train on 80% of year's players
        train_players = get_folds_except(player_folds, fold)
        test_players = player_folds[fold]

        # Train model and predict
        model.fit(train_players)
        predictions = model.predict(test_players)

    # Ensemble 4-5 predictions per player
    final_war = average_predictions_per_player(all_predictions)
```

#### Benefits:
- **Contextual accuracy**: Uses same-year replacement level
- **Park factor precision**: Year-specific ballpark effects
- **Ensemble robustness**: Multiple predictions reduce error
- **Real-time applicable**: How you'd actually use it mid-season
- **Uncertainty quantification**: Confidence intervals per player

### Phase 2: System Integration

#### Unified Pipeline:
1. **Data Preparation**: Use existing organized folder structure
2. **Feature Engineering**: Leverage current positional adjustments and enhanced metrics
3. **Model Training**:
   - System 1: Within-year folds for current season calculation
   - System 2: Temporal folds for future projection
4. **Validation**: Cross-validate both systems independently
5. **Production**: Deploy both systems for comprehensive player analysis

#### Enhanced Features to Add:
- **Park factors**: Year-specific ballpark adjustments
- **Age curves**: Position-specific aging patterns
- **Injury tracking**: Health impact on performance
- **Team context**: Lineup protection, ballpark familiarity
- **Trend analysis**: Hot/cold streaks, seasonal patterns

### Phase 3: Advanced Capabilities

#### System 1 Enhancements:
- **Real-time updates**: Incorporate daily game results
- **Confidence intervals**: Prediction uncertainty bands
- **Player comparison**: Head-to-head WAR evaluations
- **Position adjustments**: Dynamic positional value

#### System 2 Enhancements:
- **Multi-year projections**: 2-3 year forecasts
- **Breakout detection**: Identify emerging talents
- **Decline prediction**: Age-related performance drops
- **Contract analysis**: Value vs salary projections

## Technical Specifications

### Current Data Pipeline:
```
MLB Player Data/
├── BP_Data/hitters/          → WARP calculation
├── FanGraphs_Data/hitters/   → WAR calculation
├── Statcast_Data/            → Advanced metrics
└── Original_Data/            → Historical context
```

### Feature Sets:

#### System 1 (Current Season):
```python
features = [
    'K%', 'BB%', 'AVG', 'OBP', 'SLG',           # Offensive core
    'Enhanced_Baserunning', 'Enhanced_Defense',   # Advanced metrics
    'Positional_WAR',                             # Position adjustment
    'Park_Factor_Offense', 'Park_Factor_Defense', # Park effects
    'PA',                                         # Playing time
]
```

#### System 2 (Future Projections):
```python
features = [
    'Three_Year_Weighted_AVG', 'Career_Trend',    # Historical patterns
    'Age_Curve_Adjustment', 'Peak_Distance',      # Age factors
    'Park_Factor_Transition', 'Team_Change',      # Context changes
    'Injury_History', 'Workload_Pattern',         # Health factors
    'Regression_to_Mean',                         # Statistical adjustment
]
```

### Model Architecture:

#### Current Performance (System 2):
- **Best WAR Model**: Keras Neural Network (R² = 0.69)
- **Best WARP Model**: RandomForest (R² = 0.73)
- **Validation Method**: Temporal GroupKFold (5 folds)
- **Data Coverage**: 11,495 WARP + 11,064 WAR records

#### Target Performance (System 1):
- **Target Accuracy**: R² > 0.80 (within-year should be higher)
- **Ensemble Method**: Average 4-5 predictions per player
- **Uncertainty**: Standard deviation of ensemble predictions
- **Coverage**: All players with meaningful PA (>50 PA target)

## Validation Strategy

### System 1 Validation:
1. **Historical backtesting**: Apply to 2016-2023 seasons
2. **Known player validation**: Test against established stars
3. **Extreme case testing**: Validate on MVP/Cy Young winners
4. **Consistency checks**: Ensure reasonable WAR totals per year

### System 2 Validation:
1. **✅ Temporal robustness**: Already validated across years
2. **✅ Player coverage**: 5,248 unique player-seasons tested
3. **✅ Model comparison**: Multiple algorithms benchmarked
4. **Future enhancement**: Compare to existing systems (ZIPS, Steamer)

## Production Deployment

### System 1 Use Cases:
- **Mid-season analysis**: "What's Player X's current WAR?"
- **Trade evaluation**: "How much value are we getting?"
- **Award voting**: "Who are the MVP candidates?"
- **Roster decisions**: "Should we call up this prospect?"

### System 2 Use Cases:
- **Free agent evaluation**: "What will Player X do next year?"
- **Contract negotiations**: "Is this player worth $20M/year?"
- **Draft analysis**: "How will this prospect develop?"
- **Long-term planning**: "What does our team look like in 3 years?"

## Success Metrics

### System 1 (Current Season):
- **Accuracy**: R² > 0.80 for within-year predictions
- **Coverage**: >95% of qualified players (>50 PA)
- **Uncertainty**: Mean confidence interval < ±0.5 WAR
- **Consistency**: Total WAR ~1000 per year (570 hitters, 430 pitchers)

### System 2 (Future Projections):
- **✅ Accuracy**: R² = 0.69-0.73 achieved
- **✅ Robustness**: Validated across 9 years
- **Future target**: Compare favorably to ZIPS/Steamer
- **Enhancement**: Add multi-year projection capability

## Risk Assessment

### Technical Risks:
- **Data quality**: Missing PA or position data
- **Model overfitting**: Within-year folds may overfit to year-specific patterns
- **Feature correlation**: High correlation between offensive metrics

### Mitigation Strategies:
- **Data validation**: Comprehensive checks before model training
- **Cross-validation**: Rigorous fold validation for both systems
- **Feature selection**: Regularization to handle multicollinearity
- **Ensemble methods**: Multiple models to reduce single-model bias

## Timeline

### Phase 1 (4-6 weeks):
- Week 1-2: Implement within-year K-fold framework
- Week 3-4: Add park factors and enhanced features
- Week 5-6: System 1 validation and testing

### Phase 2 (2-3 weeks):
- Week 1: Integrate System 1 with existing System 2
- Week 2: Unified pipeline and validation
- Week 3: Documentation and production setup

### Phase 3 (Ongoing):
- Monthly: Model updates with new data
- Quarterly: Feature engineering improvements
- Annually: Comprehensive system evaluation

## Conclusion

This architecture provides:
1. **Comprehensive coverage**: Both current evaluation and future projection
2. **Production ready**: Real-world applicable systems
3. **Scientifically rigorous**: Proper validation methodology
4. **Scalable**: Framework for continuous improvement

The combination of within-year calculation (System 1) and temporal prediction (System 2) creates a complete player evaluation ecosystem comparable to professional systems used by MLB teams.

**Current Status**: System 2 complete and validated. System 1 development ready to begin.

**Next Steps**: Implement within-year K-fold cross-validation for System 1, starting with feature engineering and model architecture design.