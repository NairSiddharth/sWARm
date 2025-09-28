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

## Testing and Infrastructure

- [X] **Testing framework** - comprehensive data quality analysis and model validation tools
- [X] **Feature comparison tracking** - historical performance baseline monitoring
- [X] **Data quality diagnostics** - MLBAID matching analysis and feature coverage evaluation
- [X] **Model validation system** - backend improvement verification and R² tracking
- [X] **Repository organization** - cleaned up root directory, moved test files to proper testing structure
- [X] **Model version control** - organized ensemble models with proper versioning in models/history/
- [X] **Code modularization** - relocated utility scripts to appropriate module directories
- [X] **Testing documentation** - updated TESTING.md framework and testing/README.md implementation
- [ ] **Model deployment pipeline** - automate promotion of models from history to production
- [ ] **Feature engineering pipeline** - systematic approach for adding and validating new baseball metrics
- [ ] **Data pipeline optimization** - streamline multi-source data integration (FanGraphs, Statcast, Baseball Reference)
- [ ] **Documentation maintenance** - keep module READMEs current as system evolves
