"""
Modeling Module for sWARm

This module contains all machine learning model training functions including:
- Basic regression models (Linear, Lasso, Ridge, ElasticNet)
- Advanced tree-based models (RandomForest, XGBoost, KNN)
- Ensemble models (AdaBoost)
- Non-linear models (SVR, Gaussian Process)
- Neural networks (Keras with AdamW)
- Model evaluation and comparison utilities
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

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
    'run_ensemble_models',
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

    def store_results(self, model_name, player_type, metric_type, y_true, y_pred, player_names):
        """Store model results for later analysis"""
        key = f"{model_name}_{player_type}_{metric_type}"
        self.results[key] = {
            'y_true': y_true,
            'y_pred': y_pred,
            'player_names': player_names
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
    """Create an optimized Keras neural network with AdamW optimizer"""
    if not HAS_TENSORFLOW:
        raise ImportError("TensorFlow/Keras not available. Cannot create neural network model.")

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),  # Fixed: Use Input layer instead of input_dim
        tf.keras.layers.Dense(32, activation='relu', name=f'{name}_dense1'),
        tf.keras.layers.Dropout(0.3, name=f'{name}_dropout1'),
        tf.keras.layers.Dense(16, activation='relu', name=f'{name}_dense2'),
        tf.keras.layers.Dropout(0.2, name=f'{name}_dropout2'),
        tf.keras.layers.Dense(1, activation='linear', name=f'{name}_output')
    ], name=name)

    # FIXED: Use AdamW optimizer with decoupled weight decay instead of 'adam'
    model.compile(
        optimizer=AdamW(learning_rate=0.001, weight_decay=0.01),
        loss='mse',
        metrics=['mae']
    )
    return model

# ===== CORE MODELING FUNCTIONS =====
def run_basic_regressions(data_splits, model_results, print_metrics_func, plot_results_func):
    """
    Run basic regression models: Linear, Lasso, Ridge, ElasticNet

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
     h_names_warp_test, h_names_war_test, p_names_warp_test, p_names_war_test) = data_splits

    # EXPANDED: Added Ridge to complete regularization suite
    models = [
        ('linear', LinearRegression()),
        ('lasso', Lasso()),
        ('ridge', Ridge()),  # NEW: L2 regularization
        ('elasticnet', ElasticNet())
    ]

    for name, model in models:
        print(f"=== {name.upper()} REGRESSION ===")

        datasets = [
            ('hitter', 'warp', x_warp_train, x_warp_test, y_warp_train, y_warp_test, h_names_warp_test),
            ('hitter', 'war', x_war_train, x_war_test, y_war_train, y_war_test, h_names_war_test),
            ('pitcher', 'warp', a_warp_train, a_warp_test, b_warp_train, b_warp_test, p_names_warp_test),
            ('pitcher', 'war', a_war_train, a_war_test, b_war_train, b_war_test, p_names_war_test)
        ]

        for player_type, metric, X_train, X_test, y_train, y_test, names_test in datasets:
            if len(X_train) > 0:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                print_metrics_func(f"{name} {player_type} {metric}", y_test, y_pred)
                plot_results_func(f"{player_type} {metric} ({name})", y_test, y_pred, names_test)
                model_results.store_results(name, player_type, metric, y_test, y_pred, names_test)

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
     h_names_warp_test, h_names_war_test, p_names_warp_test, p_names_war_test) = data_splits

    models = [
        ('knn', KNeighborsRegressor(n_neighbors=3, n_jobs=-1)),
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
            ('hitter', 'warp', x_warp_train, x_warp_test, y_warp_train, y_warp_test, h_names_warp_test),
            ('hitter', 'war', x_war_train, x_war_test, y_war_train, y_war_test, h_names_war_test),
            ('pitcher', 'warp', a_warp_train, a_warp_test, b_warp_train, b_warp_test, p_names_warp_test),
            ('pitcher', 'war', a_war_train, a_war_test, b_war_train, b_war_test, p_names_war_test)
        ]

        for player_type, metric, X_train, X_test, y_train, y_test, names_test in datasets:
            if len(X_train) > 0:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                print_metrics_func(f"{name} {player_type} {metric}", y_test, y_pred)
                plot_results_func(f"{player_type} {metric} ({name})", y_test, y_pred, names_test)
                model_results.store_results(name, player_type, metric, y_test, y_pred, names_test)

def run_ensemble_models(data_splits, model_results, print_metrics_func, plot_results_func):
    """
    Run ensemble models: AdaBoost

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
     h_names_warp_test, h_names_war_test, p_names_warp_test, p_names_war_test) = data_splits

    models = [
        ('adaboost', AdaBoostRegressor(n_estimators=50, learning_rate=1.0, random_state=1))
    ]

    for name, model in models:
        print(f"=== {name.upper()} ===")

        datasets = [
            ('hitter', 'warp', x_warp_train, x_warp_test, y_warp_train, y_warp_test, h_names_warp_test),
            ('hitter', 'war', x_war_train, x_war_test, y_war_train, y_war_test, h_names_war_test),
            ('pitcher', 'warp', a_warp_train, a_warp_test, b_warp_train, b_warp_test, p_names_warp_test),
            ('pitcher', 'war', a_war_train, a_war_test, b_war_train, b_war_test, p_names_war_test)
        ]

        for player_type, metric, X_train, X_test, y_train, y_test, names_test in datasets:
            if len(X_train) > 0:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                print_metrics_func(f"{name} {player_type} {metric}", y_test, y_pred)
                plot_results_func(f"{player_type} {metric} ({name})", y_test, y_pred, names_test)
                model_results.store_results(name, player_type, metric, y_test, y_pred, names_test)

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
     h_names_warp_test, h_names_war_test, p_names_warp_test, p_names_war_test) = data_splits

    # Need to scale data for SVR and GP
    scaler = StandardScaler()

    # Note: GP can be computationally expensive, so using simpler kernel
    kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))

    models = [
        ('svr', SVR(kernel='rbf', gamma='scale', C=1.0)),
        ('gaussianprocess', GaussianProcessRegressor(kernel=kernel, random_state=1, n_restarts_optimizer=2))
    ]

    for name, model in models:
        print(f"=== {name.upper()} ===")

        datasets = [
            ('hitter', 'warp', x_warp_train, x_warp_test, y_warp_train, y_warp_test, h_names_warp_test),
            ('hitter', 'war', x_war_train, x_war_test, y_war_train, y_war_test, h_names_war_test),
            ('pitcher', 'warp', a_warp_train, a_warp_test, b_warp_train, b_warp_test, p_names_warp_test),
            ('pitcher', 'war', a_war_train, a_war_test, b_war_train, b_war_test, p_names_war_test)
        ]

        for player_type, metric, X_train, X_test, y_train, y_test, names_test in datasets:
            if len(X_train) > 0:
                print(f"Training {name} for {player_type} {metric}...")

                # Scale the data for these algorithms
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)

                print_metrics_func(f"{name} {player_type} {metric}", y_test, y_pred)
                plot_results_func(f"{player_type} {metric} ({name})", y_test, y_pred, names_test)
                model_results.store_results(name, player_type, metric, y_test, y_pred, names_test)

def run_neural_network(data_splits, model_results, print_metrics_func, plot_results_func, plot_training_history_func):
    """
    Run Keras neural network with AdamW optimizer

    Args:
        data_splits: Tuple of train/test splits from prepare_train_test_splits()
        model_results: ModelResults instance to store results
        print_metrics_func: Function to print metrics
        plot_results_func: Function to plot results
        plot_training_history_func: Function to plot training history
    """
    (x_warp_train, x_warp_test, y_warp_train, y_warp_test,
     x_war_train, x_war_test, y_war_train, y_war_test,
     a_warp_train, a_warp_test, b_warp_train, b_warp_test,
     a_war_train, a_war_test, b_war_train, b_war_test,
     h_names_warp_test, h_names_war_test, p_names_warp_test, p_names_war_test) = data_splits

    scaler = StandardScaler()
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True, verbose=0
    )

    print("=== KERAS NEURAL NETWORK WITH ADAMW ===")

    datasets = [
        ('hitter', 'warp', x_warp_train, x_warp_test, y_warp_train, y_warp_test, h_names_warp_test),
        ('hitter', 'war', x_war_train, x_war_test, y_war_train, y_war_test, h_names_war_test),
        ('pitcher', 'warp', a_warp_train, a_warp_test, b_warp_train, b_warp_test, p_names_warp_test),
        ('pitcher', 'war', a_war_train, a_war_test, b_war_train, b_war_test, p_names_war_test)
    ]

    for player_type, metric, X_train, X_test, y_train, y_test, names_test in datasets:
        if len(X_train) > 0:
            print(f"Training Neural Network with AdamW for {player_type} {metric}...")

            # Convert to numpy arrays for scaling
            X_train_np = np.array(X_train)
            X_test_np = np.array(X_test)
            y_train_np = np.array(y_train)

            X_train_scaled = scaler.fit_transform(X_train_np)
            X_test_scaled = scaler.transform(X_test_np)

            model = create_keras_model(input_dim=X_train_scaled.shape[1], name=f"{player_type}_{metric}")

            X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
                X_train_scaled, y_train_np, test_size=0.2, random_state=1
            )

            # Ensure data is in the right format for Keras
            history = model.fit(
                X_train_split.astype(np.float32),
                y_train_split.astype(np.float32),
                validation_data=(X_val_split.astype(np.float32), y_val_split.astype(np.float32)),
                epochs=50, batch_size=32, callbacks=[early_stopping], verbose=0
            )

            y_pred = model.predict(X_test_scaled.astype(np.float32), verbose=0).flatten()
            print_metrics_func(f"Keras {player_type} {metric}", y_test, y_pred)
            plot_results_func(f"{player_type} {metric} (Keras Neural Network + AdamW)", y_test, y_pred, names_test)
            plot_training_history_func(history)
            model_results.store_results("keras", player_type, metric, y_test, y_pred, names_test)

# ===== WAR ADJUSTMENT FUNCTIONS =====
def load_position_data():
    """Load position data from FanGraphs Leaderboard"""
    from cleanedDataParser import clean_war

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
    linear_models = ['linear', 'lasso', 'ridge', 'elasticnet']
    # Tree/Ensemble: pick best of knn, randomforest, xgboost, adaboost
    ensemble_models = ['knn', 'randomforest', 'xgboost', 'adaboost']
    # Non-linear: pick best of svr, gaussianprocess, keras
    nonlinear_models = ['svr', 'gaussianprocess', 'keras']

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