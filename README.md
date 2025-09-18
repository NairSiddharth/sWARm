# sWARm: (Sid) Wins Above Replacement Metric

Welcome to **sWARm**, a personalized reimplementation of the Wins Above Replacement (WAR) metric commonly used throughout baseball analytics and discussions. This project aims to provide a fresh perspective on evaluating player value by calculating the number of wins a player contributes to their team through a variety of machine learning algorithms and features.

## ⚾ What is WAR?

**Wins Above Replacement (WAR)** is a sabermetric statistic that quantifies a player's total contributions to their team in terms of wins. It combines various aspects of a player's performance, including batting, baserunning, fielding, and pitching, to estimate how many more wins a player is worth compared to a replacement-level player.

* Note that replacement-level is that ambigious term but is typically defined at about 0 WAR, (average AAA player at any position should get that amount by playing a full MLB season)

* This is something I have some minor quibbles with due to the disparity between hitting and pitching talent in baseball (i.e. I think that batters should have a slightly higher level set for their replacement level when compared to pitchers, because the average AAA batter called up will probably do better than the average AAA pitcher because I think that better pitchers get called up faster than better batters so the general level of batters stuck in the AAA is higher), but the people who came up with this are much smarter than me so consider this young man yelling at clouds.

## 🧪 About sWARm

**sWARm** stands for **Sid Wins Above Replace Metric**, reflecting my personal approach to calculating WAR. While it draws inspiration from existing models like FanGraphs' fWAR and Baseball Prospectus' WARP, sWARm attempts to use different combinations of commonly available statistics and the adjustments that these precursors have helpfully made available to simplify the capturing of player value.

Key features of sWARm include:

* **Personalized Adjustments**: Tailored modifications to standard WAR calculations based on my analysis and insights.
* **Comprehensive Metrics**: Integration of various performance metrics to provide a holistic view of player contributions.
* **Open Source**: Transparent and accessible codebase for collaboration and further development.

## 🔧 Installation

To get started with sWARm, clone this repository to your local machine:

``` bash
git clone [https://github.com/NairSiddharth/sWARm.git](https://github.com/NairSiddharth/sWARm.git)
cd sWARm
```

Ensure you have the necessary dependencies installed. You can set up a virtual environment and install the required packages as follows:

python3 -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
pip install -r requirements.txt

## 📊 Usage

The main notebook `cleanedCode.ipynb` contains a complete machine learning pipeline for predicting WAR and WARP values:

```python
# Run the complete pipeline
# Execute all cells in cleanedCode.ipynb
```

The pipeline automatically:

* Loads and cleans baseball statistics data
* Applies fuzzy name matching to align datasets
* Trains multiple ML models (Linear, Random Forest, XGBoost, Neural Networks)
* Generates comprehensive analysis and visualizations

## 📈 Key Features

Machine Learning Models:

* Linear Models: Linear, Lasso, Ridge, ElasticNet
* Instance-based Learning: K-Nearest Neighbors (KNN)
* Tree-based Methods: Random Forest, XGBoost
* Kernel / Non-linear Methods: SVR , Gaussian Process Regression
* Deep Learning: Keras Neural Networks (with AdamW optimizer and early stopping)

- [ ] (TODO - Deprecate due to poor performance) Ensemble Methods: AdaBoost 

**Data Processing:**

* Fuzzy name matching between datasets (names are in different formats so an easier way of matching players between the datasets when compared to manually cleaning the datasets, and subjectively better than permanently overwriting the files with one format or the other)
* Automatic data cleaning for infinite/NaN values
* Integration of advanced metrics including catcher framing data
* Enhanced defensive and baserunning statistics
* Hitters/Pitchers with same name are properly handled
* Made decision to ignore hitting stats/pitching stats for players not considered "2-way players" by MLB definition (i.e. a player must have pitched at least 20 Major League innings and started at least 20 games as a position player or designated hitter with at least three plate appearances in each of those games)

**Visualization & Analysis:**

* Quadrant analysis showing prediction error patterns
* Delta-1 margin analysis comparing to official MLB accuracy standards
* Cross-shaped visualization for WAR≤1 OR WARP≤1 official margins
* Model performance comparison across different categories
* Interactive Plotly visualizations with player-specific hover tooltip

- [ ] (TODO: make sure selectable filters apply to all graphs, maybe make a searchable filter?)
- [ ] TODO: Implement feature where user can enter a player name and get predictions of 3 years of player performance?
- [ ] TODO: Implement feature where user can enter a player name and get 5 players who's career project in the same way?

**Performance Metrics:**

* R-squared and RMSE for model evaluation
* Individual metric accuracy (WAR-only and WARP-only predictions)
* Cross-validation and intersection analysis for delta-1 margins
* Auto-selection of best performing models by category

- [ ] TODO: Implement MAE for model evaluation (curious to see if trying to minimize RMSE vs. MAE is better for this dataset as I don't necessarily want to minimize ALL outliers, only the negative ones really and at that point it might be better to just try adjusting everything the same)

## 🧠 Methodology

sWARm employs a comprehensive machine learning approach that includes:

**Data Integration:**

* Advanced defensive metrics including catcher framing contributions
* Baserunning value calculations with caching optimization
* Contextual adjustments for ballpark effects
* Position value scaling based on their relative defensive importance (i.e. a shortstop will be weighted more than a first baseman due to the position being more difficult to play)

- [ ] TODO: potentially add in catcher blocking
- [ ] TODO - decrease effects of park factors, currently 1.5 reduce to maybe 1.2

**Model Architecture:**

* Ensemble approach using multiple algorithm types
* Automatic model selection based on cross-validation performance
* Feature engineering with 7-dimensional input for batters(5 hitting specific + baserunning, defense) and 5-dimensional input for pitchers
  * Note - Hitting has features strikeouts, walks, average, onbase percentage, and slugging. Pitching has innings pitched, walks, strikeouts, homeruns given up, and earned runs average. 
* Neural network architecture with dropout regularization

- [ ] TODO: add in more features for both hitters and pitchers, have data for it just need to make sure features aren't overlapping (i.e. are the base stats OR if an advanced stat, don't include an already included base stat) Potential features to add Hitters: intentional walks (how feared of a hitter you are in comparison to the avg. joe hitting in your spot), plate appearances. Get rid of walks because double counting with OBP. /Pitchers: TB, HBP

**Validation Framework:**

* "Official" fWAR/WARP/bWAR accuracy standards (±1 WAR and ±1 WARP margins)
* Cross-shaped region analysis for statistical significance
* Quadrant-based error pattern identification
* Separate evaluation for hitters and pitchers

**Performance Optimization:**

* Cached computations for repeated operations
* Parallel model training where possible
* Memory-efficient data processing
* GPU acceleration for neural network training

## 🤝 Contributing

Contributions are welcome! If you have suggestions for improvements, bug fixes, or new features, please fork the repository and submit a pull request. Ensure that your code adheres to the project's coding standards and has been tested.

## 📄 License

TBA

## 📬 Contact

For questions or feedback, feel free to open an issue on the GitHub repository.

## Data Sources

* Data from [Baseball Savant](https://baseballsavant.mlb.com/) (MLB Statcast data)
* Data from [FanGraphs](https://www.fangraphs.com/) (advanced baseball statistics and personally my source of baseball news!)
* Data from [Baseball Prospectus](https://www.baseballprospectus.com/) (player and team analysis)
* Zip files w/ currently unused data from [Retrosheets](https://www.retrosheets.org/) – Historical game logs and play-by-play data
