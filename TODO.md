# Todo List for Tasks in Repo

## Additional Features for Current Year Performance

Note - **Hitting** has features: K%, BB%, average, onbase percentage, slugging, plate appearances (PA).
Note - **Pitching** has features: innings pitched, walks, strikeouts, homeruns given up, earned runs average, LOB%, damage control ratio, contact quality metrics.
Note - **Defense (Positional)** has features: double_plays, assists, errors, catch_probability (outfielders), enhanced defensive metrics
Note - **Defense (Catcher)** has features: framing_runs, thrown_out, blocking, arm strength metrics
Note - **Baserunning** has dynamically allocated values for stealing 1st, 2nd, and 3rd in different situations (baseline for success is 75%, below is negative value added above is positive value added)

- [X] Add in more features for **defense** like catch_probability and outfield_jump
- [X] For **catching** could add in catcher blocking and caught stealing
- [X] For **hitters** potential features to add: plate appearances
- [X] For **pitchers** potential features to add: LOB
- [X] **Enhanced pitcher features** - added contact quality metrics (Hard%, Med%, Soft%), opportunity success, damage control ratio
- [X] **Statcast integration** - integrated exit velocity, launch angle, and catch probability data
- [X] **Percentage standardization** - converted pitcher features to consistent percentage scaling (BB%, K%, HBP%, etc.)
- [ ] Add **situational performance metrics** leverage index performance for clutch situations (ON HOLD)

## Existing Features for Current Year Performance

- [x] Decrease effects of park factors, currently 1.5 reduce to maybe 1.2

## Features for Future Performance

- [X] Hitters: blend of expected stats vs. actual from past 3 years (lets put 70% on actual and 30% on expected, i.e. if they've consistently underperformed their expected stats they probably won't magically fix it, but its fair to potentially expect a bit higher than what their actual stats would indicate) **for all stats currently used in current year performance**, age
- [ ] Pitchers: blend of expected stats vs. actual from past 3 years (lets put 70% on actual and 30% on expected) **for all stats currently used in current year performance**, age, LOB_delta(find left on base delta to average)
- [X] Add **injury history integration** - track player injury frequency and severity over past 3 years to adjust for higher risk of future performance decline
- [x] Add **workload/usage pattern analysis** - incorporate innings pitched trends, plate appearance patterns to identify players at risk of overuse-related decline (partially incorporated through base IP/PA in features, could potentially add a rolling window delta but brings risk of double counting and overfitting)

## Visualizations

- [x] Make sure selectable filters apply to all graphs, maybe make a searchable filter?
- [x] Condense the different graphs for the different methods into one graph with different selectable traces, can potentially be a good way to compare them on one graph and will get rid of clutter in output
- [~] Create **interactive player comparison dashboard** - side-by-side comparison tool allowing users to select multiple players and compare predictions, actual performance, and key metrics (somewhat implemented in notebook, just have to enter in different set of players for comp. and table will display)

## Analysis

- [ ] Implement feature where user can enter a player name and get predictions of 3 future years of player performance?
- [x] Implement MAE for model evaluation (curious to see if trying to minimize RMSE vs. MAE is better for this dataset as I don't necessarily want to minimize ALL outliers, only the negative ones really and at that point it might be better to just try adjusting everything the same)
- [ ] Implement [cross-validation graphs](https://scikit-learn.org/stable/modules/cross_validation.html)
- [x] Implement residual graphs so that we can see the error difference between the actual and prediction in a comparative way between ML algo's
- [ ] Add **model interpretability features** - implement SHAP values or LIME to explain individual predictions and show which features most influenced each player's projected WAR (ON HOLD)
- [ ] Create **prediction tracking system** - monitor how predictions change over time as new data becomes available, helping validate model stability and identify when retraining is needed (ON HOLD)

## Models

- [x] TODO - Deprecate due to poor performance Linear Methods: Linear, Lasso
- [x] TODO - Deprecate due to poor performance Ensemble Methods: AdaBoost
- [x] TODO - Deprecate due to poor performance Non-linear Methods: Gaussian Process
- [X] Implement **ensemble meta-modeling** - create a stacking ensemble that combines predictions from the best-performing individual models (Random Forest, Neural Networks) for superior accuracy
- [X] Add **time-aware modeling approaches** - implement models specifically designed for temporal baseball data patterns, such as LSTM networks or seasonal decomposition methods that account for career arcs
- [X] **Advanced ensemble system** - RandomForest + Keras with metric-specific weighting and overfitting prevention
- [X] **Backend feature improvements** - enhanced PA, positional adjustments, GDP rate integration with R² improvements
- [ ] **Sample weighting for elite pitchers** - Explore RandomForest sample_weight parameter (start conservative at 1.5x for 5+ WAR pitchers) to improve elite pitcher predictions without sacrificing overall model accuracy. Balance elite MAE vs. overall MAE through grid search [1.0, 1.5, 2.0, 2.5]. Risk: May become whack-a-mole optimization between competing objectives.
- [ ] **Uncertainty quantification with TensorFlow Probability** - Add probabilistic layers to provide prediction intervals (e.g., "3.2 WAR ± 0.8") instead of point estimates. Benefits: Honest uncertainty for rookies/small samples, out-of-distribution detection. Start with simple DistributionLambda wrapper around existing Keras model. Primary use case: Flag unreliable predictions for manual review.
- [X] **Quantile stacking ensemble** - Implemented QuantileStackingEnsemble addressing elite pitcher underprediction via asymmetric loss functions. Architecture: RandomForest (MSE) + Multi-Quantile Keras (q50/q75/q90) + XGBoost (q90) base learners with XGBoost (q75) meta-estimator. Validated via unit tests (test_multi_quantile_keras.py, test_xgboost_quantile.py). Module: common_modules/multi_quantile_keras.py. Implementation: current_season_modules/modeling/pitcher_roles_ensemble_standalone.py lines 1253-1659. Next: Test in sWARm_CS_pitching.ipynb on 2025 elite pitchers to validate MAE improvement from 2.0 → 0.7 (-65%) and bias reduction from -37% → -5%.

## Testing and Infrastructure

- [X] **Testing framework** - comprehensive data quality analysis and model validation tools
- [X] **Feature comparison tracking** - historical performance baseline monitoring
- [X] **Data quality diagnostics** - MLBAID matching analysis and feature coverage evaluation
- [X] **Model validation system** - backend improvement verification and R² tracking
- [X] **Repository organization** - cleaned up root directory, moved test files to proper testing structure
- [X] **Model version control** - organized ensemble models with proper versioning in models/history/
- [X] **Code modularization** - relocated utility scripts to appropriate module directories
- [X] **Testing documentation** - updated TESTING.md framework and testing/README.md implementation
- [X] **new_pipeline architecture** - sklearn-compatible pipeline system with modular transformers
- [X] **Integration testing** - end-to-end validation from raw data to predictions (new_pipeline/tests/test_integration.py)
- [ ] **Model deployment pipeline** - automate promotion of models from history to production
- [ ] **Feature engineering pipeline** - systematic approach for adding and validating new baseball metrics
- [ ] **Data pipeline optimization** - streamline multi-source data integration (FanGraphs, Statcast, Baseball Reference)
- [ ] **Documentation maintenance** - keep module READMEs current as system evolves

## new_pipeline Integration Test Results (2025-10-06)

**Status**: ALL 6 TESTS PASSED

### Test Summary

1. **TEST 1: Data Loading** - PASS
   - Loaded 840 pitchers from 2024 (training)
   - Loaded 651 hitters from 2024 (training)
   - Loaded 754 pitchers from 2025 (predictions)
   - Loaded 606 hitters from 2025 (predictions)

2. **TEST 2: Pipeline Execution** - PASS
   - Pitcher pipeline: 655 qualified pitchers (filtered 185 position players)
   - Hitter pipeline: 526 qualified hitters (filtered 125 low-PA players)
   - FeatureSelector: 13 pitcher features + 9 metadata columns
   - FeatureSelector: 10 hitter features + 8 metadata columns

3. **TEST 3: Pitcher Model Training** - PASS
   - Training shape: (655, 13)
   - Role distribution: 326 relievers, 236 starters, 93 swing
   - Training MAE: 1.096, RMSE: 1.642, R²: 0.592
   - Models trained from scratch (RandomForest + Keras + MultiQuantileHistGB)

4. **TEST 4: Hitter Model Training** - PASS
   - Training shape: (526, 10)
   - Training MAE: 1.096, RMSE: 1.404, R²: 0.714
   - Models trained from scratch (RandomForest + Keras + MultiQuantileHistGB)

5. **TEST 5: Prediction Generation** - PASS
   - Generated predictions for 597 pitchers (2025)
   - Generated predictions for 510 hitters (2025)
   - ROS (Rest of Season) projections calculated
   - Total projected WAR = Current WAR + ROS WAR

6. **TEST 6: Utilities** - PASS
   - Combined leaderboard creation successful
   - Top projected player: Aaron Judge (8.29 WAR)
   - Two-way player handling verified (0 detected in 2025 firsthalf)

### Key Fixes Applied

1. **FeatureSelector Transformer** - Added to filter pipeline output to exact modeling features
2. **Feature Constants** - Added PITCHER_MODEL_FEATURES (13) and HITTER_MODEL_FEATURES (10) to constants.py
3. **Data Loader Fix** - Changed partial season file pattern from `*_advanced.csv` to base files (includes IP, GS, G)
4. **Prediction Function** - Updated generate_predictions to use feature constants instead of dynamic exclusion

### Architecture Validation

- Sklearn pipeline pattern working correctly
- Feature order preserved (critical for monotonic constraints)
- Metadata properly separated from modeling features
- WAR normalization (WAR_per_162, WAR_per_600) functioning
- Role-based pitcher ensembles (starter/reliever/swing) working
- Two-way player detection framework in place

---

## new_pipeline Phase 6: Notebook Integration (2025-10-06)

**Status**: PHASE 6 COMPLETE - All 3 core notebooks created

### Notebooks Created

**1. oWAR_overview.ipynb** (Dashboard)
- **Location**: `new_pipeline/notebooks/oWAR_overview.ipynb`
- **Size**: 6 cells (~5k characters)
- **Purpose**: Quick current season dashboard
- **Features**:
  - Load 2025 pitcher + hitter data
  - Interactive scatter plots (WAR vs IP/PA)
  - Featured player tables with rankings
  - Two-way player support (Shohei Ohtani)
  - ROS (Rest of Season) projections

**2. pitcher_pipeline_main.ipynb** (Full Training Workflow)
- **Location**: `new_pipeline/notebooks/pitchers/pitcher_pipeline_main.ipynb`
- **Size**: 14 cells (~12k characters)
- **Purpose**: Complete pitcher training and validation
- **Features**:
  - Load historical data (2016-2024)
  - Run full sklearn pipeline
  - Split by role (starter/reliever/swing)
  - Train role-based ensemble models
  - Generate 2025 predictions
  - Performance validation (actual vs predicted)
  - Residual analysis by role
  - Feature importance visualization
  - Error analysis by role
  - Save models and predictions

**3. hitter_pipeline_main.ipynb** (Full Training Workflow)
- **Location**: `new_pipeline/notebooks/hitters/hitter_pipeline_main.ipynb`
- **Size**: 15 cells (~12k characters)
- **Purpose**: Complete hitter training and validation
- **Features**:
  - Load historical data (2016-2024)
  - Run full sklearn pipeline
  - Train unified ensemble model (single model for all positions)
  - Position distribution analysis
  - Generate 2025 predictions
  - Performance validation (actual vs predicted)
  - Residual analysis by position
  - Feature importance visualization
  - Enhanced feature analysis (Baserunning, Defense)
  - Save model and predictions

### Key Design Decisions

**Pitcher vs Hitter Models**:
- **Pitchers**: 3 separate models (starter/reliever/swing) due to different usage patterns
- **Hitters**: 1 unified model for all positions - positional differences handled by Positional_WAR feature

**Notebook Architecture**:
- All logic in shared utilities (`pipeline_runner.py`, `plotting_utils.py`, `table_utils.py`, `analysis_utils.py`)
- Notebooks call functions, don't define them (maintainability)
- Character counts kept under targets for Claude Code editability
- Interactive Plotly visualizations with scattergl for performance

### Integration Points

All notebooks use:
- `new_pipeline.common.transformers.pipeline_builder` - Build sklearn pipelines
- `new_pipeline.models` - PitcherRoleEnsemble, HitterEnsemble
- `new_pipeline.notebooks.shared.pipeline_runner` - Data loading, pipeline execution, predictions
- `new_pipeline.notebooks.shared.plotting_utils` - Interactive plots
- `new_pipeline.notebooks.shared.table_utils` - Featured player tables
- `new_pipeline.notebooks.shared.analysis_utils` - Metrics, error analysis

### Next Steps

- [ ] User testing of all 5 notebooks
- [ ] Validate predictions match integration test results
- [ ] Deprecate old notebooks after validation period

---

## new_pipeline Phase 6 Completion: Deep-Dive Notebooks (2025-10-06)

**Status**: PHASE 6 COMPLETE - All 5 notebooks created and utility functions updated

### Notebooks Created (Deep-Dive Analysis)

**1. pitcher_deep_dive.ipynb** (Advanced Pitcher Analysis)
- **Location**: `new_pipeline/notebooks/pitchers/pitcher_deep_dive.ipynb`
- **Size**: 13 cells
- **Purpose**: Deep analysis of pitcher predictions
- **Features**:
  - Elite pitcher analysis (>5 WAR/162)
  - Replacement level analysis (<0 WAR/162)
  - Feature correlation heatmap (13 pitcher features)
  - Partial dependence plots (K%, BB%, damage_control_ratio)
  - SHAP values for top 20 pitchers
  - Error analysis by year and team
  - Prediction interval analysis (quantile predictions)
  - Model component comparison (RF/Keras/XGBoost)
  - Outlier investigation (>2 sigma residuals)

**2. hitter_deep_dive.ipynb** (Advanced Hitter Analysis)
- **Location**: `new_pipeline/notebooks/hitters/hitter_deep_dive.ipynb`
- **Size**: 13 cells
- **Purpose**: Deep analysis of hitter predictions
- **Features**:
  - Elite hitter analysis (>5 WAR/600)
  - Position-specific performance analysis
  - Enhanced feature impact (Baserunning, Defense, Positional_WAR)
  - Positional adjustment validation
  - Feature correlation heatmap (10 hitter features)
  - Partial dependence plots (AVG, OBP, SLG)
  - SHAP values for top 20 hitters
  - Error analysis by year and team
  - Model component comparison
  - Outlier investigation

### Utility Function Updates

**1. analysis_utils.py**
- Updated `calculate_elite_performance()` to return dict with 'elite_MAE' and 'elite_count' keys
- Updated `analyze_errors_by_group()` to include 'count', 'MAE', 'RMSE', 'mean_error', 'std_error' keys
- All functions match notebook usage patterns

**2. plotting_utils.py**
- Updated `create_partial_dependence()` to handle both DataFrames and numpy arrays
- Added `feature_names` parameter for numpy array support
- All deep-dive notebooks now pass feature_names correctly

### Complete Notebook Suite (Phase 3-6)

**Main Workflows** (Phase 6):
1. `oWAR_overview.ipynb` - Quick dashboard (6 cells)
2. `pitcher_pipeline_main.ipynb` - Complete pitcher training (14 cells)
3. `hitter_pipeline_main.ipynb` - Complete hitter training (15 cells)

**Deep-Dive Analysis** (Phase 6):
4. `pitcher_deep_dive.ipynb` - Advanced pitcher analysis (13 cells)
5. `hitter_deep_dive.ipynb` - Advanced hitter analysis (13 cells)

**Shared Utilities** (Phase 5):
- `pipeline_runner.py` - Data loading and pipeline execution (7 functions)
- `plotting_utils.py` - Interactive visualizations (7 functions)
- `table_utils.py` - Featured player tables (2 functions)
- `analysis_utils.py` - Advanced analysis (6 functions)

### Architecture Validation

- All notebooks use sklearn pipeline pattern
- Lean notebooks (<15k chars) call utilities, don't define them
- Feature order preserved through FeatureSelector transformer
- Role-based pitcher ensembles vs unified hitter model working correctly
- Two-way player support (Shohei Ohtani) framework in place
- SHAP integration for model interpretability
- Partial dependence plots for feature relationships
- Quantile prediction support for uncertainty quantification

### Key Implementation Details

**No Unicode Characters**: All notebooks avoid unicode (git bash compatibility per Claude.md)
**Numpy Array Support**: All plotting/analysis functions handle both DataFrame and numpy array inputs
**Graceful Degradation**: Notebooks check for data availability before analysis (handles current season vs historical)
**Feature Constants**: PITCHER_MODEL_FEATURES and HITTER_MODEL_FEATURES from constants.py ensure consistency

---
