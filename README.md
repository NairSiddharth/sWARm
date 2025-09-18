# sWARm: (Sid) Wins Above Replacement Metric

Welcome to **sWARm**, a personalized reimplementation of the Wins Above Replacement (WAR) metric in baseball analytics. This project aims to provide a fresh perspective on evaluating player value by calculating the number of wins a player contributes to their team above a replacement-level player.

## ⚾ What is WAR?

**Wins Above Replacement (WAR)** is a sabermetric statistic that quantifies a player's total contributions to their team in terms of wins. It combines various aspects of a player's performance, including batting, baserunning, fielding, and pitching, to estimate how many more wins a player is worth compared to a replacement-level player—someone readily available to any team at minimal cost.

WAR is widely used for player evaluation and comparison across different eras and positions. However, different organizations have developed their own versions of WAR, leading to variations in calculations and interpretations.

## 🧪 About sWARm

**sWARm** stands for **Sid Wins Above Replace Metric**, reflecting my personal approach to calculating WAR. While it draws inspiration from existing models like FanGraphs' fWAR and Baseball Reference's bWAR, sWARm introduces unique methodologies and adjustments to simplify the capturing of player value.

Key features of sWARm include:

* **Personalized Adjustments**: Tailored modifications to standard WAR calculations based on my analysis and insights.
* **Comprehensive Metrics**: Integration of various performance metrics to provide a holistic view of player contributions.
* **Open Source**: Transparent and accessible codebase for collaboration and further development.

## 🔧 Installation

To get started with sWARm, clone this repository to your local machine:

git clone [https://github.com/NairSiddharth/sWARm.git](https://github.com/NairSiddharth/sWARm.git)
cd sWARm

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

**Machine Learning Models:**

* Linear Regression variants (Linear, Lasso, ElasticNet)
* Tree-based models (Random Forest, XGBoost)
* Instance-based learning (K-Nearest Neighbors)
* Deep learning (Keras Neural Networks with early stopping)

**Data Processing:**

* Fuzzy name matching between datasets (460+ players vs original 5)
* Automatic data cleaning for infinite/NaN values
* Integration of advanced metrics including catcher framing data
* Enhanced defensive and baserunning statistics

**Visualization & Analysis:**

* Interactive Plotly visualizations with player-specific hover tooltips
* Quadrant analysis showing prediction error patterns
* Delta-1 margin analysis comparing to official MLB accuracy standards
* Cross-shaped visualization for WAR≤1 OR WARP≤1 official margins
* Model performance comparison across different categories

**Performance Metrics:**

* R-squared and RMSE for model evaluation
* Individual metric accuracy (WAR-only and WARP-only predictions)
* Cross-validation and intersection analysis for delta-1 margins
* Auto-selection of best performing models by category

## 🧠 Methodology

sWARm employs a comprehensive machine learning approach that includes:

**Data Integration:**

* Advanced defensive metrics including catcher framing contributions
* Baserunning value calculations with caching optimization
* Contextual adjustments for ballpark and era effects
* Position value scaling based on defensive importance

**Model Architecture:**

* Ensemble approach using multiple algorithm types
* Automatic model selection based on cross-validation performance
* Feature engineering with 7-dimensional input (5 hitting + baserunning defense)
* Neural network architecture with dropout regularization

**Validation Framework:**

* Official fWAR/WARP accuracy standards (±1 WAR and ±1 WARP margins)
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
