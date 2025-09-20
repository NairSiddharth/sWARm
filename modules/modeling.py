"""
Modeling Module for sWARm

This module contains all machine learning model training functions including:
- Basic regression models (Linear, Lasso, Ridge, ElasticNet)
- Advanced tree-based models (RandomForest, XGBoost, KNN)
- Non-linear models (SVR)
- Neural networks (Keras with AdamW)
- Model evaluation and comparison utilities
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import SVR
# from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("Warning: XGBoost not available. XGBoost models will be skipped.")
try:
    import tensorflow as tf
    from keras.models import Sequential
    from keras.layers import Dense, Activation, Dropout, Input
    from keras.callbacks import EarlyStopping
    from keras.optimizers import AdamW
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False
    print("Warning: TensorFlow/Keras not available. Neural network models will be skipped.")

# Import plotting functions from parent module
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

__all__ = [
    'create_keras_model',
    'run_basic_regressions',
    'run_advanced_models',
    # 'run_ensemble_models',
    'run_nonlinear_models',
    'run_neural_network',
    'ModelResults',
    'select_best_models_by_category',
    'apply_proper_war_adjustments',
    'load_position_data',
    'get_positional_adjustment',
    'get_replacement_level_adjustment'
]

# ===== MODEL RESULTS CLASS =====
class ModelResults:
    """Class to store and manage model prediction results for analysis"""

    def __init__(self):
        self.results = {}

    def store_results(self, model_name, player_type, metric_type, y_true, y_pred, player_names, seasons=None):
        """Store model results for later analysis with optional season data"""
        key = f"{model_name}_{player_type}_{metric_type}"
        self.results[key] = {
            'y_true': y_true,
            'y_pred': y_pred,
            'player_names': player_names,
            'Season': seasons if seasons is not None else ['2021'] * len(player_names)
        }

    def get_results(self, model_name, player_type, metric_type):
        """Retrieve stored results"""
        key = f"{model_name}_{player_type}_{metric_type}"
        return self.results.get(key, None)

    def list_available_results(self):
        """List all available result keys"""
        return list(self.results.keys())

# ===== UTILITY FUNCTIONS =====
def print_metrics(name, y_true, y_pred):
    """Print R² and RMSE metrics for model evaluation"""
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"{name} - R2: {r2:.4f}, RMSE: {rmse:.4f}")

def create_keras_model(input_dim, name="model"):
    """
    Create an optimized Keras neural network with improved architecture and hyperparameters

    FIXES:
    - Adaptive architecture based on problem difficulty (WAR vs WARP)
    - BatchNormalization for training stability
    - Better hyperparameters (lower LR for WAR, Huber loss)
    - Deeper architecture for complex WAR prediction
    - Reduced retracing through consistent layer naming
    """
    if not HAS_TENSORFLOW:
        raise ImportError("TensorFlow/Keras not available. Cannot create neural network model.")

    # Adaptive architecture based on prediction difficulty
    if "war" in name.lower():
        # MORE COMPLEX architecture for WAR prediction (harder problem)
        model = tf.keras.Sequential([ # type: ignore
            tf.keras.layers.Input(shape=(input_dim,), name=f'{name}_input'), # type: ignore
            tf.keras.layers.BatchNormalization(name=f'{name}_bn_input'), # type: ignore
            tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001), name=f'{name}_dense1'), # type: ignore
            tf.keras.layers.Dropout(0.3, name=f'{name}_dropout1'), # type: ignore
            tf.keras.layers.BatchNormalization(name=f'{name}_bn1'), # type: ignore
            tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001), name=f'{name}_dense2'), # type: ignore
            tf.keras.layers.Dropout(0.2, name=f'{name}_dropout2'), # type: ignore
            tf.keras.layers.Dense(32, activation='relu', name=f'{name}_dense3'), # type: ignore
            tf.keras.layers.Dropout(0.1, name=f'{name}_dropout3'), # type: ignore
            tf.keras.layers.Dense(1, activation='linear', name=f'{name}_output') # type: ignore
        ], name=name)
        learning_rate = 0.0003  # LOWER LR for stability
        weight_decay = 0.02     # Higher weight decay for regularization
    else:
        # SIMPLER architecture for WARP prediction (easier problem)
        model = tf.keras.Sequential([ # type: ignore
            tf.keras.layers.Input(shape=(input_dim,), name=f'{name}_input'), # type: ignore
            tf.keras.layers.BatchNormalization(name=f'{name}_bn_input'), # type: ignore
            tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001), name=f'{name}_dense1'), # type: ignore
            tf.keras.layers.Dropout(0.2, name=f'{name}_dropout1'), # type: ignore
            tf.keras.layers.Dense(32, activation='relu', name=f'{name}_dense2'), # type: ignore
            tf.keras.layers.Dropout(0.1, name=f'{name}_dropout2'), # type: ignore
            tf.keras.layers.Dense(1, activation='linear', name=f'{name}_output') # type: ignore
        ], name=name)
        learning_rate = 0.001   # Standard LR for easier problem
        weight_decay = 0.01

    # CHOICE: Use log-cosh for WAR (balanced elite emphasis + noise robustness) vs Huber for WARP
    if "warp" in name.lower():
        # Huber for WARP data which might have more noise
        loss_function = tf.keras.losses.Huber()  # FIXED: Use actual loss class, not string
        print(f"   Using Huber loss for {name} - robust to WARP data noise")
    else:
        # Log-cosh: smooth transition from MSE-like (small errors) to MAE-like (large errors)
        # Perfect for WAR: emphasizes elite accuracy while robust to measurement noise
        loss_function = tf.keras.losses.LogCosh()  # FIXED: Use actual loss class, not string
        print(f"   Using log-cosh loss for {name} - balanced elite emphasis + noise robustness")

    # Compile the model with optimized hyperparameters
    optimizer = tf.keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=weight_decay) # type: ignore

    model.compile(
        optimizer=optimizer,
        loss=loss_function,
        metrics=['mae', 'mse']
    )

    print(f"   Model compiled: {model.count_params()} parameters, LR={learning_rate}, Loss={loss_function}")
    return model

# ===== DIAGNOSTIC FUNCTIONS =====
def analyze_feature_target_correlations(data_splits):
    """
    DIAGNOSTIC: Analyze feature-target correlations to understand why hitter WARP performs poorly

    This function helps investigate:
    - Whether BP hitter features correlate well with WARP targets
    - Whether FanGraphs hitter features correlate well with WAR targets
    - Data quality differences between BP and FanGraphs datasets
    """
    print("\nDIAGNOSTIC: Feature-Target Correlation Analysis")
    print("=" * 70)

    (x_warp_train, x_warp_test, y_warp_train, y_warp_test,
     x_war_train, x_war_test, y_war_train, y_war_test,
     a_warp_train, a_warp_test, b_warp_train, b_warp_test,
     a_war_train, a_war_test, b_war_train, b_war_test,
     h_names_warp_test, h_names_war_test, p_names_warp_test, p_names_war_test,
     h_seasons_warp_test, h_seasons_war_test, p_seasons_warp_test, p_seasons_war_test) = data_splits

    datasets = [
        ('Hitter WARP (BP)', x_warp_train, y_warp_train),
        ('Hitter WAR (FanGraphs)', x_war_train, y_war_train),
        ('Pitcher WARP (BP)', a_warp_train, b_warp_train),
        ('Pitcher WAR (FanGraphs)', a_war_train, b_war_train)
    ]

    correlation_summary = {}

    for dataset_name, X, y in datasets:
        if len(X) == 0:
            print(f"\nX {dataset_name}: No data available")
            continue

        print(f"\n{dataset_name}:")
        print(f"   Sample size: {len(X)} records")
        print(f"   Target range: {y.min():.2f} to {y.max():.2f}")
        print(f"   Target std: {y.std():.3f}")

        # Calculate feature-target correlations
        feature_correlations = {}
        for i, col in enumerate(X.columns):
            if X.iloc[:, i].std() > 0:  # Avoid division by zero
                corr = np.corrcoef(X.iloc[:, i], y)[0, 1]
                if not np.isnan(corr):
                    feature_correlations[col] = corr

        # Sort by absolute correlation
        sorted_correlations = sorted(feature_correlations.items(), key=lambda x: abs(x[1]), reverse=True)

        print(f"   Feature correlations with target:")
        for feature, corr in sorted_correlations[:7]:  # Show top 7 features
            strength = "STRONG" if abs(corr) > 0.7 else "MODERATE" if abs(corr) > 0.4 else "WEAK"
            print(f"      {feature}: {corr:+.3f} ({strength})")

        # Store summary statistics
        max_corr = max([abs(corr) for corr in feature_correlations.values()]) if feature_correlations else 0
        avg_corr = np.mean([abs(corr) for corr in feature_correlations.values()]) if feature_correlations else 0

        correlation_summary[dataset_name] = {
            'max_correlation': max_corr,
            'avg_correlation': avg_corr,
            'feature_count': len(feature_correlations),
            'target_variance': y.var(),
            'sample_size': len(X)
        }

        print(f"   Max |correlation|: {max_corr:.3f}")
        print(f"   Avg |correlation|: {avg_corr:.3f}")

        # Data quality checks
        missing_features = X.isnull().sum().sum()
        zero_variance_features = (X.std() == 0).sum()

        if missing_features > 0:
            print(f"   WARNING: Missing values: {missing_features} total")
        if zero_variance_features > 0:
            print(f"   WARNING: Zero variance features: {zero_variance_features}")

    print(f"\nCORRELATION ANALYSIS SUMMARY:")
    print("=" * 50)

    # Compare hitter datasets
    if 'Hitter WARP (BP)' in correlation_summary and 'Hitter WAR (FanGraphs)' in correlation_summary:
        warp_stats = correlation_summary['Hitter WARP (BP)']
        war_stats = correlation_summary['Hitter WAR (FanGraphs)']

        print(f"HITTER ANALYSIS:")
        print(f"   BP WARP - Max corr: {warp_stats['max_correlation']:.3f}, Avg corr: {warp_stats['avg_correlation']:.3f}")
        print(f"   FG WAR  - Max corr: {war_stats['max_correlation']:.3f}, Avg corr: {war_stats['avg_correlation']:.3f}")

        if war_stats['max_correlation'] > warp_stats['max_correlation'] * 1.5:
            print(f"   INSIGHT: FanGraphs features show {war_stats['max_correlation']/warp_stats['max_correlation']:.1f}x stronger correlation!")
            print(f"      This explains poor WARP prediction performance.")

        if war_stats['target_variance'] > warp_stats['target_variance'] * 1.5:
            print(f"   INSIGHT: FanGraphs WAR has {war_stats['target_variance']/warp_stats['target_variance']:.1f}x more target variance")
            print(f"      Higher variance targets are often easier to predict.")

    # Compare pitcher datasets
    if 'Pitcher WARP (BP)' in correlation_summary and 'Pitcher WAR (FanGraphs)' in correlation_summary:
        p_warp_stats = correlation_summary['Pitcher WARP (BP)']
        p_war_stats = correlation_summary['Pitcher WAR (FanGraphs)']

        print(f"\nPITCHER ANALYSIS:")
        print(f"   BP WARP - Max corr: {p_warp_stats['max_correlation']:.3f}, Avg corr: {p_warp_stats['avg_correlation']:.3f}")
        print(f"   FG WAR  - Max corr: {p_war_stats['max_correlation']:.3f}, Avg corr: {p_war_stats['avg_correlation']:.3f}")

        if abs(p_warp_stats['max_correlation'] - p_war_stats['max_correlation']) < 0.1:
            print(f"   GOOD: Both pitcher datasets show similar correlation strength")
        else:
            print(f"   WARNING: Pitcher datasets show different correlation patterns")

    print(f"\nRECOMMENDATIONS:")
    if 'Hitter WARP (BP)' in correlation_summary:
        if correlation_summary['Hitter WARP (BP)']['max_correlation'] < 0.5:
            print(f"   * Consider adding more predictive features to BP hitter data")
            print(f"   * Investigate data quality issues in BP hitter dataset")
            print(f"   * Consider using FanGraphs features for WARP prediction as well")

    return correlation_summary

# ===== CORE MODELING FUNCTIONS =====
def run_basic_regressions(data_splits, model_results, print_metrics_func, plot_results_func):
    """
    Run basic regression models: Linear, Lasso, Ridge, ElasticNet

    Args:
        data_splits: Tuple of train/test splits from prepare_train_test_splits() with season data
        model_results: ModelResults instance to store results
        print_metrics_func: Function to print metrics
        plot_results_func: Function to plot results
    """
    (x_warp_train, x_warp_test, y_warp_train, y_warp_test,
     x_war_train, x_war_test, y_war_train, y_war_test,
     a_warp_train, a_warp_test, b_warp_train, b_warp_test,
     a_war_train, a_war_test, b_war_train, b_war_test,
     h_names_warp_test, h_names_war_test, p_names_warp_test, p_names_war_test,
     h_seasons_warp_test, h_seasons_war_test, p_seasons_warp_test, p_seasons_war_test) = data_splits

    models = [
        ('ridge', Ridge()),
        ('elasticnet', ElasticNet())
    ]

    for name, model in models:
        print(f"=== {name.upper()} REGRESSION ===")

        datasets = [
            ('hitter', 'warp', x_warp_train, x_warp_test, y_warp_train, y_warp_test, h_names_warp_test, h_seasons_warp_test),
            ('hitter', 'war', x_war_train, x_war_test, y_war_train, y_war_test, h_names_war_test, h_seasons_war_test),
            ('pitcher', 'warp', a_warp_train, a_warp_test, b_warp_train, b_warp_test, p_names_warp_test, p_seasons_warp_test),
            ('pitcher', 'war', a_war_train, a_war_test, b_war_train, b_war_test, p_names_war_test, p_seasons_war_test)
        ]

        for player_type, metric, X_train, X_test, y_train, y_test, names_test, seasons_test in datasets:
            if len(X_train) > 0:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                print_metrics_func(f"{name} {player_type} {metric}", y_test, y_pred)
                plot_results_func(f"{player_type} {metric} ({name})", y_test, y_pred, names_test)
                model_results.store_results(name, player_type, metric, y_test, y_pred, names_test, seasons_test)

def run_advanced_models(data_splits, model_results, print_metrics_func, plot_results_func):
    """
    Run advanced tree-based models: KNN, RandomForest, XGBoost

    Args:
        data_splits: Tuple of train/test splits from prepare_train_test_splits()
        model_results: ModelResults instance to store results
        print_metrics_func: Function to print metrics
        plot_results_func: Function to plot results
    """
    (x_warp_train, x_warp_test, y_warp_train, y_warp_test,
     x_war_train, x_war_test, y_war_train, y_war_test,
     a_warp_train, a_warp_test, b_warp_train, b_warp_test,
     a_war_train, a_war_test, b_war_train, b_war_test,
     h_names_warp_test, h_names_war_test, p_names_warp_test, p_names_war_test,
     h_seasons_warp_test, h_seasons_war_test, p_seasons_warp_test, p_seasons_war_test) = data_splits

    models = [
        ('knn', KNeighborsRegressor(
            n_neighbors=5,           # More neighbors for stability
            weights='distance',      # Weight by distance, not uniform
            algorithm='ball_tree',   # Better for higher dimensions
            metric='minkowski',      # Euclidean distance (p=2)
            p=2,
            n_jobs=-1
        )),
        ('randomforest', RandomForestRegressor(n_estimators=50, max_depth=8, random_state=1, n_jobs=-1))
    ]

    # Add XGBoost if available
    if HAS_XGBOOST:
        models.append(('xgboost', xgb.XGBRegressor(n_estimators=50, max_depth=4, learning_rate=0.1, random_state=1, n_jobs=-1)))
    else:
        print("Skipping XGBoost (not installed)")

    for name, model in models:
        print(f"=== {name.upper()} ===")

        datasets = [
            ('hitter', 'warp', x_warp_train, x_warp_test, y_warp_train, y_warp_test, h_names_warp_test, h_seasons_warp_test),
            ('hitter', 'war', x_war_train, x_war_test, y_war_train, y_war_test, h_names_war_test, h_seasons_war_test),
            ('pitcher', 'warp', a_warp_train, a_warp_test, b_warp_train, b_warp_test, p_names_warp_test, p_seasons_warp_test),
            ('pitcher', 'war', a_war_train, a_war_test, b_war_train, b_war_test, p_names_war_test, p_seasons_war_test)
        ]

        for player_type, metric, X_train, X_test, y_train, y_test, names_test, seasons_test in datasets:
            if len(X_train) > 0:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                print_metrics_func(f"{name} {player_type} {metric}", y_test, y_pred)
                plot_results_func(f"{player_type} {metric} ({name})", y_test, y_pred, names_test)
                model_results.store_results(name, player_type, metric, y_test, y_pred, names_test, seasons_test)

# def run_ensemble_models(data_splits, model_results, print_metrics_func, plot_results_func):
#     """
#     Run ensemble models: AdaBoost

#     Args:
#         data_splits: Tuple of train/test splits from prepare_train_test_splits()
#         model_results: ModelResults instance to store results
#         print_metrics_func: Function to print metrics
#         plot_results_func: Function to plot results
#     """
#     (x_warp_train, x_warp_test, y_warp_train, y_warp_test,
#      x_war_train, x_war_test, y_war_train, y_war_test,
#      a_warp_train, a_warp_test, b_warp_train, b_warp_test,
#      a_war_train, a_war_test, b_war_train, b_war_test,
#      h_names_warp_test, h_names_war_test, p_names_warp_test, p_names_war_test,
#      h_seasons_warp_test, h_seasons_war_test, p_seasons_warp_test, p_seasons_war_test) = data_splits

#     models = [
#         ('adaboost', AdaBoostRegressor(n_estimators=50, learning_rate=1.0, random_state=1))
#     ]

#     for name, model in models:
#         print(f"=== {name.upper()} ===")

#         datasets = [
#             ('hitter', 'warp', x_warp_train, x_warp_test, y_warp_train, y_warp_test, h_names_warp_test, h_seasons_warp_test),
#             ('hitter', 'war', x_war_train, x_war_test, y_war_train, y_war_test, h_names_war_test, h_seasons_war_test),
#             ('pitcher', 'warp', a_warp_train, a_warp_test, b_warp_train, b_warp_test, p_names_warp_test, p_seasons_warp_test),
#             ('pitcher', 'war', a_war_train, a_war_test, b_war_train, b_war_test, p_names_war_test, p_seasons_war_test)
#         ]

#         for player_type, metric, X_train, X_test, y_train, y_test, names_test, seasons_test in datasets:
#             if len(X_train) > 0:
#                 model.fit(X_train, y_train)
#                 y_pred = model.predict(X_test)
#                 print_metrics_func(f"{name} {player_type} {metric}", y_test, y_pred)
#                 plot_results_func(f"{player_type} {metric} ({name})", y_test, y_pred, names_test)
#                 model_results.store_results(name, player_type, metric, y_test, y_pred, names_test, seasons_test)

def run_nonlinear_models(data_splits, model_results, print_metrics_func, plot_results_func):
    """
    Run non-linear models: SVR, Gaussian Process

    Args:
        data_splits: Tuple of train/test splits from prepare_train_test_splits()
        model_results: ModelResults instance to store results
        print_metrics_func: Function to print metrics
        plot_results_func: Function to plot results
    """
    (x_warp_train, x_warp_test, y_warp_train, y_warp_test,
     x_war_train, x_war_test, y_war_train, y_war_test,
     a_warp_train, a_warp_test, b_warp_train, b_warp_test,
     a_war_train, a_war_test, b_war_train, b_war_test,
     h_names_warp_test, h_names_war_test, p_names_warp_test, p_names_war_test,
     h_seasons_warp_test, h_seasons_war_test, p_seasons_warp_test, p_seasons_war_test) = data_splits

    # Need to scale data for SVR and GP
    #Could potentially use RobustScaler instead
    scaler = StandardScaler()

    # Note: GP can be computationally expensive, so using simpler kernel
    # kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))

    models = [
        ('svr', SVR(kernel='rbf', gamma='scale', C=1.0)),
        # ('gaussianprocess', GaussianProcessRegressor(kernel=kernel, random_state=1, n_restarts_optimizer=2))
    ]

    for name, model in models:
        print(f"=== {name.upper()} ===")

        datasets = [
            ('hitter', 'warp', x_warp_train, x_warp_test, y_warp_train, y_warp_test, h_names_warp_test, h_seasons_warp_test),
            ('hitter', 'war', x_war_train, x_war_test, y_war_train, y_war_test, h_names_war_test, h_seasons_war_test),
            ('pitcher', 'warp', a_warp_train, a_warp_test, b_warp_train, b_warp_test, p_names_warp_test, p_seasons_warp_test),
            ('pitcher', 'war', a_war_train, a_war_test, b_war_train, b_war_test, p_names_war_test, p_seasons_war_test)
        ]

        for player_type, metric, X_train, X_test, y_train, y_test, names_test, seasons_test in datasets:
            if len(X_train) > 0:
                print(f"Training {name} for {player_type} {metric}...")

                # Scale the data for these algorithms
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)

                print_metrics_func(f"{name} {player_type} {metric}", y_test, y_pred)
                plot_results_func(f"{player_type} {metric} ({name})", y_test, y_pred, names_test)
                model_results.store_results(name, player_type, metric, y_test, y_pred, names_test, seasons_test)

def run_neural_network(data_splits, model_results, print_metrics_func, plot_results_func, plot_training_history_func):
    """
    IMPROVED Keras neural network with enhanced stability and performance

    FIXES:
    - Separate scalers for each dataset to avoid distribution mismatch
    - Better early stopping with patience based on problem difficulty
    - Robust outlier handling and data validation
    - Reduced batch size for better convergence
    - Additional callbacks for training stability
    - Model compilation outside of training loop to reduce retracing
    """
    (x_warp_train, x_warp_test, y_warp_train, y_warp_test,
     x_war_train, x_war_test, y_war_train, y_war_test,
     a_warp_train, a_warp_test, b_warp_train, b_warp_test,
     a_war_train, a_war_test, b_war_train, b_war_test,
     h_names_warp_test, h_names_war_test, p_names_warp_test, p_names_war_test,
     h_seasons_warp_test, h_seasons_war_test, p_seasons_warp_test, p_seasons_war_test) = data_splits

    print("=== IMPROVED KERAS NEURAL NETWORK WITH ENHANCED STABILITY ===")

    datasets = [
        ('hitter', 'warp', x_warp_train, x_warp_test, y_warp_train, y_warp_test, h_names_warp_test, h_seasons_warp_test),
        ('hitter', 'war', x_war_train, x_war_test, y_war_train, y_war_test, h_names_war_test, h_seasons_war_test),
        ('pitcher', 'warp', a_warp_train, a_warp_test, b_warp_train, b_warp_test, p_names_warp_test, p_seasons_warp_test),
        ('pitcher', 'war', a_war_train, a_war_test, b_war_train, b_war_test, p_names_war_test, p_seasons_war_test)
    ]

    for player_type, metric, X_train, X_test, y_train, y_test, names_test, seasons_test in datasets:
        if len(X_train) == 0:
            continue

        print(f"Training IMPROVED Neural Network for {player_type} {metric}...")

        # IMPROVED: Separate scaler for each dataset to handle distribution differences
        scaler = StandardScaler()

        # ROBUST: Convert to numpy and handle potential data issues
        try:
            X_train_np = np.array(X_train, dtype=np.float32)
            X_test_np = np.array(X_test, dtype=np.float32)
            y_train_np = np.array(y_train, dtype=np.float32)
            y_test_np = np.array(y_test, dtype=np.float32)

            # Check for NaN/Inf values
            if np.any(np.isnan(X_train_np)) or np.any(np.isinf(X_train_np)):
                print(f"   ⚠️  Warning: NaN/Inf detected in {player_type} {metric} features, filling with 0")
                X_train_np = np.nan_to_num(X_train_np, 0)
                X_test_np = np.nan_to_num(X_test_np, 0)

            if np.any(np.isnan(y_train_np)) or np.any(np.isinf(y_train_np)):
                print(f"   ⚠️  Warning: NaN/Inf detected in {player_type} {metric} targets, skipping...")
                continue

        except Exception as e:
            print(f"   ❌ Data conversion failed for {player_type} {metric}: {e}")
            continue

        # CONSERVATIVE: Only remove obvious data errors, not elite performance
        # Baseball WAR can legitimately range from -3 to +12, so be very conservative
        data_error_mask = (y_train_np >= -5.0) & (y_train_np <= 15.0)  # Only remove clear data errors
        removed_count = np.sum(~data_error_mask)

        if removed_count > 0:
            print(f"   📊 Removed {removed_count} data errors (WAR < -5 or > 15) from {len(y_train_np)} samples")
            X_train_np = X_train_np[data_error_mask]
            y_train_np = y_train_np[data_error_mask]

        # Report data range to verify we're keeping elite seasons
        print(f"   📈 WAR range: {y_train_np.min():.2f} to {y_train_np.max():.2f} (keeping elite seasons!)")

        # IMPROVED: Scale data with better handling
        X_train_scaled = scaler.fit_transform(X_train_np)
        X_test_scaled = scaler.transform(X_test_np)

        # ADAPTIVE: Early stopping based on problem difficulty
        if metric == 'war':
            patience = 20  # More patience for harder WAR problem
            min_delta = 0.001
        else:
            patience = 15  # Standard patience for WARP
            min_delta = 0.0005

        # ENHANCED: Multiple callbacks for better training
        callbacks = [
            tf.keras.callbacks.EarlyStopping( # type: ignore
                monitor='val_loss',
                patience=patience,
                min_delta=min_delta,
                restore_best_weights=True,
                verbose=0
            ),
            tf.keras.callbacks.ReduceLROnPlateau( # type: ignore
                monitor='val_loss',
                factor=0.5,
                patience=max(5, patience//3),
                min_lr=1e-6,
                verbose=0
            )
        ]

        # CREATE MODEL (this will use the improved architecture)
        model = create_keras_model(input_dim=X_train_scaled.shape[1], name=f"{player_type}_{metric}")

        # IMPROVED: Create validation split
        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
            X_train_scaled, y_train_np, test_size=0.2, random_state=42
        )

        # IMPROVED: Training with better parameters
        try:
            history = model.fit(
                X_train_split,
                y_train_split,
                validation_data=(X_val_split, y_val_split),
                epochs=100,  # More epochs but early stopping will handle it
                batch_size=16,  # Smaller batch size for better convergence
                callbacks=callbacks,
                verbose=0
            )

            # PREDICTION with proper error handling
            y_pred = model.predict(X_test_scaled, verbose=0).flatten()

            # Ensure predictions are reasonable
            if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
                print(f"   ❌ Model produced invalid predictions for {player_type} {metric}")
                continue

            print_metrics_func(f"IMPROVED Keras {player_type} {metric}", y_test_np, y_pred)
            plot_results_func(f"{player_type} {metric} (IMPROVED Keras NN)", y_test_np, y_pred, names_test)
            plot_training_history_func(history)
            model_results.store_results("keras", player_type, metric, y_test_np, y_pred, names_test, seasons_test)

            # DIAGNOSTIC: Print training info
            final_epoch = len(history.history['loss'])
            final_loss = history.history['val_loss'][-1] if history.history['val_loss'] else 'N/A'
            print(f"   📈 Training stopped at epoch {final_epoch}, final val_loss: {final_loss}")

        except Exception as e:
            print(f"   ❌ Training failed for {player_type} {metric}: {e}")
            continue

# ===== WAR ADJUSTMENT FUNCTIONS =====
def load_position_data():
    """Load position data from FanGraphs Leaderboard"""
    from modularized_data_parser import clean_war

    war_values = clean_war()
    position_mapping = {}

    print(f"WAR data columns: {list(war_values.columns)}")

    # Check if position data is available
    if 'Pos' not in war_values.columns:
        print("⚠️  No position data available in WAR dataset")
        print("   Available columns:", list(war_values.columns))
        print("   Skipping positional adjustments")
        return {}

    print("✅ Position data found! Processing positions...")

    for _, row in war_values.iterrows():
        name = row['Name']
        pos = row['Pos']
        pa = row.get('PA', '')
        ip = row.get('IP', '')

        # Only map hitters (those with PA but no IP) and exclude pitchers
        if pd.notna(pa) and pa != '' and (pd.isna(ip) or ip == '') and pos != 'P':
            # Handle multi-position players (e.g., "2B/SS", "1B-LF", "RF/LF")
            if '/' in str(pos):
                primary_pos = str(pos).split('/')[0]
            elif '-' in str(pos):
                primary_pos = str(pos).split('-')[0]
            else:
                primary_pos = str(pos)

            position_mapping[name] = primary_pos

    print(f"Loaded position data for {len(position_mapping)} hitters")

    # Show position distribution
    pos_counts = {}
    for pos in position_mapping.values():
        pos_counts[pos] = pos_counts.get(pos, 0) + 1
    print("Position distribution:", dict(sorted(pos_counts.items())))

    return position_mapping

def get_positional_adjustment(position):
    """Get FanGraphs positional adjustment in WAR"""
    POSITIONAL_ADJUSTMENTS = {
        'C': +1.25,   # +12.5 runs = +1.25 WAR
        '1B': -1.25,  # -12.5 runs = -1.25 WAR
        '2B': +0.25,  # +2.5 runs = +0.25 WAR
        '3B': +0.25,  # +2.5 runs = +0.25 WAR
        'SS': +0.75,  # +7.5 runs = +0.75 WAR
        'LF': -0.75,  # -7.5 runs = -0.75 WAR
        'CF': +0.25,  # +2.5 runs = +0.25 WAR
        'RF': -0.75,  # -7.5 runs = -0.75 WAR
        'DH': -1.75,  # -17.5 runs = -1.75 WAR
        'P': 0.0      # Pitchers get no positional adjustment
    }
    return POSITIONAL_ADJUSTMENTS.get(position, 0.0)

def get_replacement_level_adjustment(player_type, playing_time_estimate=1.0):
    """Get replacement level adjustment scaled by playing time"""
    if player_type == 'hitter':
        return -2.0 * playing_time_estimate  # -2.0 WAR per 600 PA season
    elif player_type == 'pitcher':
        return -1.0 * playing_time_estimate  # -1.0 WAR per 200 IP season
    else:
        return 0.0

def select_best_models_by_category(model_results):
    """Select best performing model from each category for comparison"""
    # Calculate R² scores for all models
    model_scores = {}

    for key, data in model_results.results.items():
        model_name, player_type, metric_type = key.split('_')
        r2 = r2_score(data['y_true'], data['y_pred'])

        category_key = f"{player_type}_{metric_type}"
        if category_key not in model_scores:
            model_scores[category_key] = {}

        model_scores[category_key][model_name] = r2

    # Select best model from each major category
    selected_models = set()

    # Linear methods: pick best of linear, lasso, ridge, elasticnet
    linear_models = [ 'ridge', 'elasticnet']#'linear', 'lasso',
    # Tree/Ensemble: pick best of knn, randomforest, xgboost, adaboost
    ensemble_models = ['knn', 'randomforest', 'xgboost']#, 'adaboost'
    # Non-linear: pick best of svr, gaussianprocess, keras
    nonlinear_models = ['svr', 'keras']#, 'gaussianprocess'

    for category_models, category_name in [(linear_models, 'linear'),
                                          (ensemble_models, 'ensemble'),
                                          (nonlinear_models, 'nonlinear')]:
        best_model = None
        best_score = -float('inf')

        # Average R² across all player_type/metric combinations
        for model in category_models:
            avg_score = 0
            count = 0
            for category_key, scores in model_scores.items():
                if model in scores:
                    avg_score += scores[model]
                    count += 1

            if count > 0:
                avg_score /= count
                if avg_score > best_score:
                    best_score = avg_score
                    best_model = model

        if best_model:
            selected_models.add(best_model)

    # Always include keras if available
    if 'keras' in [key.split('_')[0] for key in model_results.results.keys()]:
        selected_models.add('keras')

    result = list(selected_models)
    print(f"Auto-selected best models: {result}")
    return result

def apply_proper_war_adjustments(model_results):
    """Apply proper positional and replacement level adjustments"""
    print("\n=== APPLYING PROPER WAR ADJUSTMENTS ===")

    # Load position data
    position_mapping = load_position_data()
    has_position_data = len(position_mapping) > 0

    if has_position_data:
        print("Using real position data from FanGraphs Leaderboard")
    else:
        print("No position data available - applying replacement level adjustments only")

    adjusted_results = ModelResults()

    for key, data in model_results.results.items():
        model_name, player_type, metric_type = key.split('_')

        # Only apply adjustments to WAR predictions, not WARP
        if metric_type != 'war':
            adjusted_results.results[key] = data.copy()
            continue

        print(f"  Adjusting {model_name} {player_type} {metric_type}...")

        adjusted_predictions = []

        for i, (y_pred, y_true, player_name) in enumerate(zip(data['y_pred'], data['y_true'], data['player_names'])):
            adjusted_war = y_pred

            if player_type == 'hitter':
                # Apply replacement level adjustment
                replacement_adj = get_replacement_level_adjustment('hitter', 0.8)
                adjusted_war += replacement_adj

                # Apply positional adjustment if we have position data
                if has_position_data and player_name in position_mapping:
                    position = position_mapping[player_name]
                    positional_adj = get_positional_adjustment(position)
                    adjusted_war += positional_adj

                    if i < 3:  # Debug first few players
                        print(f"    {player_name} ({position}): {y_pred:.2f} -> {adjusted_war:.2f} "
                              f"(pos: {positional_adj:+.2f}, repl: {replacement_adj:+.2f})")
                elif i < 3:  # Debug without position data
                    print(f"    {player_name}: {y_pred:.2f} -> {adjusted_war:.2f} "
                          f"(repl: {replacement_adj:+.2f})")

            elif player_type == 'pitcher':
                replacement_adj = get_replacement_level_adjustment('pitcher', 0.8)
                adjusted_war += replacement_adj

            adjusted_predictions.append(adjusted_war)

        adjusted_results.results[key] = {
            'y_true': data['y_true'],
            'y_pred': adjusted_predictions,
            'player_names': data['player_names']
        }

    return adjusted_results