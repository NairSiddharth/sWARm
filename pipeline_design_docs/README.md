# Pipeline Design Documentation

**Date Created:** 2025-10-05
**Purpose:** Clean slate sklearn-native data pipelines for pitcher and hitter WAR prediction

This folder contains all planning documentation for the new feature pipeline implementations.

---

## Documents Overview

### 📋 **pitcher_feature_pipeline_design.md** (MAIN DOCUMENT)
**The complete design specification for the new pipeline.**

**Contents:**
- System architecture diagram (CSV → predictions)
- 12 active features with sources
- Data loading strategy (no caching, direct CSV)
- Train/validation/test split (2016-2023 / 2024 / 2025)
- WAR normalization (WAR/162 IP)
- Replacement level imputation (25th percentile)
- Feature scale consistency (decimal → percentage conversion)
- Park adjustment strategy (ERA, GB%, HR/FB%)
- Complete transformer implementations
- End-to-end workflow examples
- Transformer list (NoIDFilter, TwoWayPlayerFilter, etc.)

**Start here** for understanding the full system.

---

### ⚠️ **CRITICAL_SCALE_MISMATCH_ISSUE.md**
**Critical bug analysis: FanGraphs decimal vs percentage format.**

**Problem Identified:**
- FanGraphs CSVs store percentages as decimals (0.232 = 23.2%)
- Composite formulas assume percentage format (23.2)
- Results in 5-10,000x magnitude errors in composite features

**Features Affected:**
- ✅ `interaction_features.py` - Correctly converts decimals → percentages
- ❌ `realtime_composite_calculator.py` - Doesn't convert (BUG!)
- Impact: `Opportunity_Success` 10,000x too small, `damage_control_ratio` 5x too small

**Solution:**
- ALL loaders must convert decimals → percentages immediately
- FeatureValidator checks for scale mismatches
- Validation tests for composite feature ranges

**This must be fixed** or the model will train on corrupted features.

---

### 🏟️ **park_adjustment_recommendations.md**
**Analysis of which stats should be park-adjusted and why.**

**Philosophy:** Adjust PARK-INFLUENCED stats, keep SKILL stats raw.

**Should Adjust:**
- ✅ ERA - 3yr park factor (run environment)
- ✅ GB% - GB park factor (dimensions affect batted ball distribution)
- ⚠️ HR/FB% - HR park factor (**CRITICAL - Currently MISSING**)

**Should NOT Adjust:**
- ❌ K%, BB%, SwStr% - Pitcher controls (no ball in play)
- ❌ Hard%, Soft%, Med% - Contact quality is skill
- ❌ WPA/LI - Context stat
- ❌ LOB% - Primarily pitcher skill

**Impact Example:**
- Without HR/FB% adjustment: Coors vs Oracle pitcher = 47% difference (mostly park)
- With HR/FB% adjustment: Coors vs Oracle pitcher = 32% difference (mostly skill)

**FanGraphs Methodology:**
- 50% blend: `effective_park = (park_factor + 100) / 2`
- Accounts for ~50% home games

---

### 🔧 **hr_fb_park_adjustment_integration.md**
**Step-by-step guide for adding HR/FB% park adjustment to the pipeline.**

**Integration Steps:**

1. **Create loader function:** `load_hr_fb_pct_park_adjusted()` in `pitcher_feature_loaders.py`

2. **Update PitcherFeatureTransformer:** Change one line in `fit()` method:
   ```python
   'HR/FB%': load_hr_fb_pct_park_adjusted(self.years)  # ← NEW
   ```

3. **No other changes needed!** The rest of the pipeline automatically uses park-adjusted values.

**Data Flow:**
```
CSV → load_hr_fb_pct_park_adjusted() → intermediate_dicts_
  → transform() → damage_control_ratio (now correct!)
```

**Testing:**
- Test cases for Coors Field (HR factor 108)
- Test cases for Oracle Park (HR factor 85)
- Validation that composite features use adjusted values

---

### 🧪 **testing_and_migration_guide.md**
**Testing strategy and migration path from old to new pipeline.**

**Contents:**
- **Comprehensive test suite** - pytest examples for all transformers
- **Scale consistency tests** - Validate decimal → percentage conversion
- **Park adjustment tests** - Verify HR/FB% adjustment works
- **A/B comparison strategy** - Compare old vs new pipeline performance
- **5-phase migration plan** - Parallel development → Cleanup
- **Advanced sklearn patterns** - ColumnTransformer, FunctionTransformer

**Key sections:**
- Unit tests for WARNormalizer, RoleClassifier, FeatureValidator
- Integration tests for full pipeline
- A/B comparison script with decision criteria
- Gradual migration strategy (non-destructive)
- Production pipeline with column-specific scaling

**Use this** for implementation testing and safe migration.

---

### 🏏 **hitter_feature_pipeline_design.md**
**Complete design specification for the hitter pipeline (parallel to pitcher pipeline).**

**Contents:**
- 10 hitter features: K%, BB%, AVG, OBP, SLG, PA, Positional_WAR, GDP_rate, Enhanced_Baserunning, Enhanced_Defense
- Enhanced_Baserunning calculation from SB, CS, SB%, XBT%, sprint speed (BP + Statcast)
- Enhanced_Defense calculation from fielding%, DPs, catch probability
- Positional adjustments per 600 PA (C: +1.25, DH: -1.50)
- WAR/600 PA normalization (different from pitchers)
- Position-based splits (IF/OF/C/DH)
- Park adjustment for batting stats (AVG, OBP, SLG)
- HitterFeatureTransformer implementation
- Data sources: FanGraphs + Baseball Prospectus + Statcast
- Complete pipeline example

**Use this** for implementing hitter WAR predictions with the same clean slate approach as pitchers.

---

### 📓 **notebook_architecture.md**
**Overall notebook structure and design philosophy.**

**Contents:**
- Notebook hierarchy (overview/main/deep-dive)
- Design principles (lean notebooks, logic in modules)
- Complete folder structure (notebooks/pitchers/, notebooks/hitters/, notebooks/shared/)
- Specifications overview for all 5 notebooks
- User workflows (quick check / full pipeline / deep analysis)
- Integration with pipeline design docs
- Testing strategy for notebooks

**Use this** to understand the complete notebook ecosystem.

---

### 📊 **notebook_overview_spec.md**
**Detailed specification for oWAR_overview.ipynb (all-in-one dashboard).**

**Contents:**
- Cell-by-cell implementation (~5k chars, 6 cells)
- Pitcher scatter plot (cumulative WAR vs IP)
- Hitter scatter plot (cumulative WAR vs PA)
- Featured pitchers table with type grouping and rank
- Featured hitters table with position grouping and rank
- Two-way player handling (Shohei Ohtani)
- Customization guide (changing featured players, reference lines)

**Use this** to implement the main dashboard notebook.

---

### ⚾ **notebook_pitcher_main_spec.md**
**Detailed specification for pitcher_pipeline_main.ipynb (full pitcher workflow).**

**Contents:**
- Cell-by-cell implementation (~12k chars, 12-15 cells)
- Data loading and pipeline execution
- Role-based model training (starters/relievers/swing)
- Performance metrics by role
- Actual vs predicted scatter plots
- Residual analysis with marginals
- Feature importance visualization
- Error analysis by role
- Model and prediction saving

**Use this** to implement the pitcher pipeline notebook.

---

### 🏏 **notebook_hitter_main_spec.md**
**Detailed specification for hitter_pipeline_main.ipynb (full hitter workflow).**

**Contents:**
- Cell-by-cell implementation (~12k chars, 12-15 cells)
- Data loading and pipeline execution
- Position-based model training (IF/OF/C/DH)
- Performance metrics by position
- Actual vs predicted scatter plots
- Enhanced_Baserunning and Enhanced_Defense analysis
- Feature importance visualization
- Error analysis by position
- Model and prediction saving

**Use this** to implement the hitter pipeline notebook.

---

### 🔬 **notebook_deepdive_spec.md**
**Detailed specification for deep-dive analysis notebooks (both pitcher and hitter).**

**Contents:**
- Elite player analysis (>5 WAR)
- Replacement level analysis (<0 WAR)
- Feature correlation heatmaps
- Partial dependence plots (K%, BB%, ERA, damage_control_ratio, etc.)
- SHAP value analysis for top players
- Error analysis by year and team
- Prediction interval analysis
- Model component comparison
- Outlier investigation

**Use this** to implement optional deep-dive notebooks for detailed analysis.

---

### 🛠️ **shared_utilities_spec.md**
**Complete API documentation for all shared utility functions.**

**Contents:**
- `pipeline_runner.py` - Data loading, pipeline execution, splitting, metrics
- `plotting_utils.py` - Plotly scatter plots, residual plots, feature importance, PDP
- `table_utils.py` - PrettyTable formatting, type/position classification, rank calculation
- `analysis_utils.py` - Elite performance, error analysis, SHAP, outliers
- Function signatures with parameters and return types
- Usage examples for each function
- Testing recommendations

**Use this** as reference when implementing or calling utility functions.

---

### 📖 **IMPLEMENTATION GUIDE (Parts 1-4)** ⭐ **START HERE FOR IMPLEMENTATION**

**Complete step-by-step implementation guide with full code examples.**

#### **implementation_guide.md (Part 1)**
**Covers Phase 0 and Phase 1 (partial).**

**Contents:**
- Phase 0: Setup & Foundation (directory structure, inventory)
- Phase 1: Core Loaders (beginning)
  - Helper functions with complete implementations
  - Pitcher base loaders (BB%, K%, SwStr%, WPA/LI, LOB%, Hard%)
  - All with test scripts and checklists

#### **implementation_guide_part2.md (Part 2)**
**Covers Phase 1 (complete) and Phase 2 (complete).**

**Contents:**
- Phase 1: Core Loaders (continued)
  - Pitcher park-adjusted loaders (ERA, GB%, HR/FB%)
  - Statcast loader with fallback
  - All 10 hitter loaders (K%, BB%, PA, GDP, AVG/OBP/SLG park-adjusted, Positional_WAR, Enhanced_Baserunning, Enhanced_Defense)
  - Data loading helpers (load_pitcher_data, load_hitter_data, replacement levels)
- Phase 2: Transformers (complete)
  - Base transformers (NoIDFilter, IPFilter, PAFilter, TwoWayPlayerFilter)
  - WARNormalizer (WAR/162 for pitchers, WAR/600 for hitters)
  - FeatureValidator and MissingValueImputer
  - PitcherFeatureTransformer (combines all 10 pitcher features)
  - HitterFeatureTransformer (combines all 10 hitter features)
  - Pipeline builder helpers

#### **implementation_guide_part3.md (Part 3)**
**Covers Phase 3, Phase 4, and Phase 5 (partial).**

**Contents:**
- Phase 3: Testing Framework (complete)
  - Loader tests (pitcher and hitter)
  - Transformer tests (filters, normalizers, validators)
  - Integration tests (end-to-end pipelines)
- Phase 4: Models (complete)
  - QuantileStackingEnsemble base class
  - PitcherRoleEnsemble (3 role-specific models: starter/reliever/swing)
  - HitterEnsemble (single unified model)
  - Save/load functionality
- Phase 5: Shared Utilities (partial)
  - Pipeline runner (run_pitcher_training, run_hitter_training)

#### **implementation_guide_part4.md (Part 4)**
**Covers Phase 5 (complete), Phase 6 (complete), and Phases 7-8 (high-level).**

**Contents:**
- Phase 5: Shared Utilities (continued)
  - Plotting utilities (create_war_scatter with reference lines, prediction vs actual)
  - Table utilities (create_war_leaderboard with dividers and groups)
- Phase 6: Notebooks (complete)
  - oWAR_overview.ipynb (all-in-one dashboard, 9 cells)
  - oWAR_pitcher_main.ipynb (detailed pitcher analysis)
  - oWAR_hitter_main.ipynb (detailed hitter analysis)
- Phase 7: Validation & Comparison (high-level)
  - A/B testing, performance validation, prediction quality checks
- Phase 8: Migration & Cleanup (high-level)
  - Backup old code, move to production, final testing

**Total Implementation Time: ~41 hours** (roughly 1 week of full-time work)

**How to Use:**
1. Follow phases in order (0→1→2→3→4→5→6→7→8)
2. Complete all tasks within each phase before moving to next
3. Run test scripts after each section
4. Check off checklist items as you go
5. Verify expected outputs match

**This guide contains EVERYTHING needed** to implement the entire pipeline from scratch - either you or I can follow it independently.

---

### 🗂️ **feature_consolidation_guide.md**
**Maps existing scattered features to new consolidated structure.**

**Contents:**
- Current state inventory (10+ old files)
- Feature-by-feature migration map
- Critical changes needed (decimal→percentage conversion)
- Missing HR/FB% park adjustment identified
- Complete file structure examples for pitcher_loaders.py and hitter_loaders.py

**Use this** to understand what to pull from old files into new loaders.

---

## Implementation Checklist

### Phase 1: Core Infrastructure

#### Pitcher Loaders
- [ ] Create `common_modules/pitcher_feature_loaders.py`
  - [ ] `load_bb_pct_all_years()` - Convert decimal → percentage
  - [ ] `load_k_pct_all_years()` - Convert decimal → percentage
  - [ ] `load_era_park_adjusted()` - 3yr park factor adjustment
  - [ ] `load_gb_pct_park_adjusted()` - GB park factor adjustment
  - [ ] `load_swstr_all_years()` - Convert decimal → percentage
  - [ ] `load_wpa_li_all_years()` - Keep raw scale
  - [ ] `load_lob_pct_all_years()` - Convert decimal → percentage
  - [ ] `load_hr_fb_pct_park_adjusted()` - **HR park factor (CRITICAL!)**
  - [ ] `load_hard_pct_all_years()` - Convert decimal → percentage
  - [ ] `load_statcast_launch_quality()` - Keep 0-100 scale

#### Hitter Loaders
- [ ] Create `common_modules/hitter_feature_loaders.py`
  - [ ] `load_k_pct_all_years()` - Convert decimal → percentage
  - [ ] `load_bb_pct_all_years()` - Convert decimal → percentage
  - [ ] `load_avg_park_adjusted()` - Basic park factor adjustment
  - [ ] `load_obp_park_adjusted()` - Basic park factor adjustment
  - [ ] `load_slg_park_adjusted()` - HR + XBH park factors
  - [ ] `load_pa_all_years()` - Keep raw count
  - [ ] `load_gdp_all_years()` - Keep raw count
  - [ ] `load_baserunning_data()` - Enhanced_Baserunning composite
  - [ ] `load_defense_data()` - Enhanced_Defense composite

### Phase 2: Transformers

#### Shared Transformers
- [ ] Create `common_modules/transformers.py`
  - [ ] `NoIDFilter` - Remove players without MLBAMID
  - [ ] `TwoWayPlayerFilter` - Remove position players (keep Ohtani)
  - [ ] `IPFilter` - Remove < 10 IP (pitchers)
  - [ ] `PAFilter` - Remove < 100 PA (hitters)
  - [ ] `WARNormalizer` - WAR → WAR/162 IP or WAR/600 PA
  - [ ] `FeatureValidator` - **Validate scale consistency**

#### Pitcher Transformers
- [ ] Create `common_modules/pitcher_feature_transformer.py`
  - [ ] `PitcherFeatureTransformer` - Add all 12 features
  - [ ] Load base features in `fit()`
  - [ ] Load intermediate features (including HR/FB%)
  - [ ] Calculate composites in `transform()`
  - [ ] Calculate interactions in `transform()`
  - [ ] Replacement level imputation (25th percentile)

#### Hitter Transformers
- [ ] Create `common_modules/hitter_feature_transformer.py`
  - [ ] `HitterFeatureTransformer` - Add all 10 features
  - [ ] Load base batting stats in `fit()`
  - [ ] Load Enhanced_Baserunning and Enhanced_Defense
  - [ ] Calculate GDP_rate in `transform()`
  - [ ] Add Positional_WAR adjustments
  - [ ] Replacement level imputation (25th percentile)

### Phase 3: Helper Functions
- [ ] Create `common_modules/data_loading.py`
  - [ ] `load_pitcher_data()` - Load pitcher CSVs with WAR targets
  - [ ] `load_hitter_data()` - Load hitter CSVs with WAR targets
  - [ ] `calculate_league_averages_by_year()` - Year-specific averages (pitchers & hitters)
  - [ ] `calculate_replacement_levels_by_year()` - 25th percentile (pitchers & hitters)

### Phase 4: Testing

#### Pitcher Tests
- [ ] Test scale conversions (decimal → percentage)
- [ ] Test park adjustments (Coors, Oracle pitchers; HR/FB% adjustment)
- [ ] Test composite features (damage_control_ratio ranges)
- [ ] Test interaction features (strikeout_efficiency values)
- [ ] Test FeatureValidator catches scale errors
- [ ] Test TwoWayPlayerFilter (Ohtani passes, position players removed)
- [ ] Test WARNormalizer (WAR/162 IP calculation)
- [ ] Test replacement level imputation

#### Hitter Tests
- [ ] Test scale conversions (decimal → percentage)
- [ ] Test park adjustments for batting stats (AVG, OBP, SLG)
- [ ] Test Enhanced_Baserunning calculation
- [ ] Test Enhanced_Defense calculation
- [ ] Test Positional_WAR adjustments
- [ ] Test GDP_rate calculation
- [ ] Test WARNormalizer (WAR/600 PA calculation)
- [ ] Test PAFilter (>100 PA threshold)

### Phase 5: Integration

#### Pitcher Integration
- [ ] Update `current_season_modules/modeling/data_loading_v2.py` (pitchers)
- [ ] Update `current_season_modules/modeling/pitcher_roles_ensemble_standalone.py`
- [ ] Test end-to-end pitcher pipeline (CSV → predictions)
- [ ] Validate against old pitcher pipeline results
- [ ] Compare pitcher model performance

#### Hitter Integration
- [ ] Create `current_season_modules/modeling/hitter_data_loading.py`
- [ ] Create `current_season_modules/modeling/hitter_ensemble.py` (single unified model)
- [ ] Test end-to-end hitter pipeline (CSV → predictions)
- [ ] Validate against existing hitter model results
- [ ] Compare hitter model performance
- [ ] Verify Positional_WAR feature correctly handles position differences

### Phase 6: Notebooks

#### Shared Utilities
- [ ] Create `notebooks/shared/__init__.py`
- [ ] Create `notebooks/shared/pipeline_runner.py`
  - [ ] `load_current_season_data()`
  - [ ] `load_historical_data()`
  - [ ] `run_data_pipeline()`
  - [ ] `generate_predictions()`
  - [ ] `split_by_role()` and `split_by_position()`
  - [ ] `calculate_metrics()`
- [ ] Create `notebooks/shared/plotting_utils.py`
  - [ ] `create_war_scatter()` with reference lines (0/3/6 WAR)
  - [ ] `create_actual_vs_predicted()`
  - [ ] `create_residual_plot()` with marginals
  - [ ] `create_feature_importance()`
  - [ ] `create_correlation_heatmap()`
  - [ ] `create_partial_dependence()`
- [ ] Create `notebooks/shared/table_utils.py`
  - [ ] `create_featured_table()` with dividers and rank
  - [ ] `create_metrics_table()`
  - [ ] `get_pitcher_type()` and `get_hitter_position()`
  - [ ] `get_rank_within_type()`
  - [ ] `handle_two_way_player()`
- [ ] Create `notebooks/shared/analysis_utils.py`
  - [ ] `calculate_elite_performance()`
  - [ ] `calculate_replacement_performance()`
  - [ ] `analyze_errors_by_group()`
  - [ ] `calculate_shap_values()`
  - [ ] `find_outliers()`
  - [ ] `compare_models()`

#### Main Notebooks
- [ ] Create `notebooks/oWAR_overview.ipynb` (~5k chars, 6 cells)
  - [ ] Pitcher scatter (cumulative WAR vs IP)
  - [ ] Hitter scatter (cumulative WAR vs PA)
  - [ ] Featured pitchers table (with type dividers)
  - [ ] Featured hitters table (with position dividers)
  - [ ] Two-way player handling (Shohei Ohtani)
- [ ] Create `notebooks/pitchers/pitcher_pipeline_main.ipynb` (~12k chars)
  - [ ] Full pipeline execution
  - [ ] Role-specific model training
  - [ ] Validation and test predictions
  - [ ] Performance visualizations
- [ ] Create `notebooks/hitters/hitter_pipeline_main.ipynb` (~12k chars)
  - [ ] Full pipeline execution
  - [ ] Position-specific model training
  - [ ] Validation and test predictions
  - [ ] Performance visualizations

#### Deep Dive Notebooks (Optional)
- [ ] Create `notebooks/pitchers/pitcher_deep_dive.ipynb` (~15k chars)
  - [ ] Elite pitcher analysis
  - [ ] Feature correlation and PDP
  - [ ] SHAP analysis
  - [ ] Outlier investigation
- [ ] Create `notebooks/hitters/hitter_deep_dive.ipynb` (~15k chars)
  - [ ] Elite hitter analysis
  - [ ] Enhanced feature analysis
  - [ ] Feature correlation and PDP
  - [ ] Outlier investigation

### Phase 7: Cleanup
- [ ] Mark old notebooks as deprecated (sWARm_CS_new.ipynb, sWARm_CS_pitching.ipynb)
- [ ] Update documentation
- [ ] Remove old imports
- [ ] Create migration guide from old to new notebooks

---

## Key Design Decisions

### ✅ **Feature Scale: PERCENTAGE (0-100)**
All percentage features converted from FanGraphs decimals (0.232 → 23.2)

### ✅ **Park Adjustment: 50% BLEND**
Accounts for ~50% home games: `effective_park = (park_factor + 100) / 2`

### ✅ **Imputation: REPLACEMENT LEVEL (25th percentile)**
Missing data filled with 25th percentile (not league average) - assumes missing = lower quality

### ✅ **WAR Normalization**
- **Pitchers:** WAR/162 IP (industry standard)
- **Hitters:** WAR/600 PA (industry standard)
Both provide interpretable scales (0-8 WAR) and easy prorating

### ✅ **Train/Test Split: TEMPORAL**
- Train: 2016-2023 (8 years)
- Validation: 2024 (holdout)
- Test: 2025 first half (unseen future)

### ✅ **Model Strategy: ROLE-BASED vs UNIFIED**

**Pitchers: 3 SEPARATE MODELS (by role)**
- Starters: GS/G > 0.7 → Starter model
- Relievers: GS/G < 0.1 → Reliever model
- Mixed: 0.1 ≤ GS/G ≤ 0.7 → Swing model
- **Reason:** Fundamentally different usage patterns (180 IP vs 60 IP)
- **Benefit:** Prevents reliever WAR ceiling from dragging down elite starter predictions

**Hitters: 1 UNIFIED MODEL (all positions)**
- All positions train on same model
- Position differences handled by Positional_WAR feature (-1.50 to +1.25)
- **Reason:** Similar PA patterns across positions (500-600 PA if starter)
- **Benefit:** More training data = better elite hitter predictions

**Analysis Splits (for visualization only):**
- Pitchers: By role (same as models)
- Hitters: By position group (IF/OF/C/DH) despite unified model

### ✅ **No Caching: DIRECT CSV LOADING**
Eliminates potential error sources, ensures reproducibility

---

## Questions to Resolve

1. **Caching strategy:** Should we cache feature dicts as JSON for faster loading?
2. **Error handling:** Strict (raise errors) or permissive (fill with defaults)?
3. **Multi-year lookups:** Most recent value or average last N years?
4. **Two-way filter window:** Check same 3-year window for pitcher IP + hitter stats?

---

## References

- **FanGraphs Park Factors:** https://library.fangraphs.com/principles/park-factors/
- **Existing Implementation:** `common_modules/park_factors.py`
- **Feature Sets:** `common_modules/feature_sets.py` (outdated, needs replacement)
- **Current Ensemble:** `current_season_modules/modeling/pitcher_roles_ensemble_standalone.py`

---

## Document History

| Date | Document | Changes |
|------|----------|---------|
| 2025-10-05 | pitcher_feature_pipeline_design.md | Initial creation - full pitcher design spec |
| 2025-10-05 | CRITICAL_SCALE_MISMATCH_ISSUE.md | Identified decimal vs percentage bug |
| 2025-10-05 | park_adjustment_recommendations.md | Analyzed which stats need park adjustment (pitchers) |
| 2025-10-05 | hr_fb_park_adjustment_integration.md | Integration guide for HR/FB% fix |
| 2025-10-05 | testing_and_migration_guide.md | Testing strategy and migration path |
| 2025-10-05 | hitter_feature_pipeline_design.md | Full hitter design spec (parallel to pitchers) |
| 2025-10-05 | notebook_architecture.md | Overall notebook structure and design philosophy |
| 2025-10-05 | notebook_overview_spec.md | oWAR_overview.ipynb specification |
| 2025-10-05 | notebook_pitcher_main_spec.md | pitcher_pipeline_main.ipynb specification |
| 2025-10-05 | notebook_hitter_main_spec.md | hitter_pipeline_main.ipynb specification |
| 2025-10-05 | notebook_deepdive_spec.md | Deep dive notebooks specification |
| 2025-10-05 | shared_utilities_spec.md | Complete API docs for shared utilities |
| 2025-10-05 | feature_consolidation_guide.md | Maps old scattered features to new consolidated structure |
| 2025-10-05 | implementation_guide.md (Part 1) | Phase 0 and Phase 1 (partial) - Setup and pitcher base loaders |
| 2025-10-05 | implementation_guide_part2.md (Part 2) | Phase 1 (complete) and Phase 2 - All loaders and transformers |
| 2025-10-05 | implementation_guide_part3.md (Part 3) | Phase 3, 4, and 5 (partial) - Testing, models, utilities |
| 2025-10-05 | implementation_guide_part4.md (Part 4) | Phase 5 (complete), 6, 7, 8 - Utilities, notebooks, validation, migration |
| 2025-10-05 | README.md | This index document (updated for implementation guide) |

---

## Next Steps

1. ✅ Complete design documentation (pipelines + notebooks)
2. ✅ **Complete implementation guide** (Parts 1-4 with full code examples)
3. ⏳ **Phase 0:** Setup new_pipeline/ directory structure
4. ⏳ **Phase 1:** Implement loader functions with scale conversion
5. ⏳ **Phase 2:** Implement transformers (especially FeatureValidator)
6. ⏳ **Phase 3:** Write comprehensive tests (scale, park adjustments, features)
7. ⏳ **Phase 4:** Implement models (QuantileStackingEnsemble, PitcherRoleEnsemble, HitterEnsemble)
8. ⏳ **Phase 5:** Implement shared utilities (pipeline_runner, plotting, tables)
9. ⏳ **Phase 6:** Implement notebooks (overview, pitcher main, hitter main)
10. ⏳ **Phase 7:** Validate and compare performance vs old pipeline
11. ⏳ **Phase 8:** Cleanup, deprecate old code, and deploy to production

**📖 Follow the implementation guide (Parts 1-4) for step-by-step instructions with complete code examples.**
