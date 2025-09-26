# Todo List for Tasks in Repo

## Additional Features for Current Year Performance

Note - **Hitting** has features: k%, bb%, average, onbase percentage, and slugging.
Note - **Pitching** has features: innings pitched, walks, strikeouts, homeruns given up, and earned runs average.
Note - **Defense (Positional)** has features: double_plays, assists, errors
Note - **Defense (Catcher)** has features: framing_runs, thrown_out
Note - **Baserunning** has dyanmically allocated values for stealing 1st, 2nd, and 3rd in different situations (baseline for success is 75%, below is negative value added above is positive value added)

- [X] Add in more features for **defense** like catch_probability and outfield_jump
- [X] For **catching** could add in catcher blocking and caught stealing
- [ ] For **hitters** potential features to add: total bases, plate appearances,
- [ ] For **pitchers** potential features to add: total bases, LOB
- [ ] Add **situational performance metrics** like RISP (runners in scoring position) batting average and leverage index performance for clutch situations

## Existing Features for Current Year Performance

- [x] Decrease effects of park factors, currently 1.5 reduce to maybe 1.2

## Features for Future Performance

- [X] Hitters: blend of expected stats vs. actual from past 3 years (lets put 70% on actual and 30% on expected, i.e. if they've consistently underperformed their expected stats they probably won't magically fix it, but its fair to potentially expect a bit higher than what their actual stats would indicate) **for all stats currently used in current year performance**, age
- [ ] Pitchers: blend of expected stats vs. actual from past 3 years (lets put 70% on actual and 30% on expected) **for all stats currently used in current year performance**, vFA_delta_to_avg(find avg fastball speed, calculate difference between pitchers fastball speed to avg.), age, LOB_delta(find left on base delta to average)
- [X] Add **injury history integration** - track player injury frequency and severity over past 3 years to adjust for higher risk of future performance decline
- [x] Add **workload/usage pattern analysis** - incorporate innings pitched trends, plate appearance patterns to identify players at risk of overuse-related decline (partially incorporated through base IP/PA in features, could potentially add a rolling window delta but brings risk of double counting and overfitting)

## Existing Features for Future Performance

TBD

## Visualizations

- [x] Make sure selectable filters apply to all graphs, maybe make a searchable filter?
- [x] Condense the different graphs for the different methods into one graph with different selectable traces, can potentially be a good way to compare them on one graph and will get rid of clutter in output
- [ ] Create **interactive player comparison dashboard** - side-by-side comparison tool allowing users to select multiple players and compare predictions, actual performance, and key metrics
- [ ] Add **prediction confidence intervals** - visualize uncertainty ranges around predictions rather than just point estimates, helping users understand prediction reliability

## Analysis

- [ ] Implement feature where user can enter a player name and get predictions of 3 future years of player performance?
- [ ] Implement feature where user can enter a player name and get 5 players who's career project in the same way?
- [x] Implement MAE for model evaluation (curious to see if trying to minimize RMSE vs. MAE is better for this dataset as I don't necessarily want to minimize ALL outliers, only the negative ones really and at that point it might be better to just try adjusting everything the same)
- [ ] Implement [cross-validation graphs](https://scikit-learn.org/stable/modules/cross_validation.html)
- [x] Implement residual graphs so that we can see the error difference between the actual and prediction in a comparative way between ML algo's
- [ ] Add **model interpretability features** - implement SHAP values or LIME to explain individual predictions and show which features most influenced each player's projected WAR
- [ ] Create **prediction tracking system** - monitor how predictions change over time as new data becomes available, helping validate model stability and identify when retraining is needed

## Models

- [x] TODO - Deprecate due to poor performance Linear Methods: Linear, Lasso
- [x] TODO - Deprecate due to poor performance Ensemble Methods: AdaBoost
- [x] TODO - Deprecate due to poor performance Non-linear Methods: Gaussian Process
- [X] Implement **ensemble meta-modeling** - create a stacking ensemble that combines predictions from the best-performing individual models (Random Forest, Neural Networks) for superior accuracy
- [X] Add **time-aware modeling approaches** - implement models specifically designed for temporal baseball data patterns, such as LSTM networks or seasonal decomposition methods that account for career arcs
