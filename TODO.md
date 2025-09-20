# Todo List for Tasks in Repo

## Additional Featurea

- [ ] Add in more features for defense like catch_probability and outfield_jump
- [ ] Potentially add in catcher blocking
- [ ] Add in more features for both hitters and pitchers, have data for it just need to make sure features aren't overlapping (i.e. are the base stats OR if an advanced stat, don't include an already included base stat) Potential features to add:
  - [ ] Hitters: intentional walks (how feared of a hitter you are in comparison to the avg. joe hitting in your spot), plate appearances, look at batted_ball_profile to see if we can pick out any features to test. Get rid of walks because double counting with OBP.
  - [ ] Pitchers: TB, HBP

## Existing Features

- [x] Decrease effects of park factors, currently 1.5 reduce to maybe 1.2

## Visualizations

- [ ] Make sure selectable filters apply to all graphs, maybe make a searchable filter?
- [ ] Condense the different graphs for the different methods into one graph with different selectable traces, can potentially be a good way to compare them on one graph and will get rid of clutter in output

## Analysis

- [ ] Implement feature where user can enter a player name and get predictions of 3 future years of player performance?
- [ ] Implement feature where user can enter a player name and get 5 players who's career project in the same way?
- [ ] Implement MAE for model evaluation (curious to see if trying to minimize RMSE vs. MAE is better for this dataset as I don't necessarily want to minimize ALL outliers, only the negative ones really and at that point it might be better to just try adjusting everything the same)
- [ ] Implement [cross-validation graphs](https://scikit-learn.org/stable/modules/cross_validation.html)
- [ ] Implement residual graphs so that we can see the error difference between the actual and prediction in a comparative way between ML algo's

## Models

- [x] TODO - Deprecate due to poor performance Linear Methods: Linear, Lasso
- [x] TODO - Deprecate due to poor performance Ensemble Methods: AdaBoost
- [x] TODO - Deprecate due to poor performance Non-linear Methods: Gaussian Process
