"""
Predictive Modeling Module for Current Season Analysis
Evolved from temp_modeling.py to support real-time WAR/WARP calculation and scenario projections
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.svm import SVR
import sys
import os

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

# Import data loading functions
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from current_season_modules.data_loading import *
from current_season_modules.fangraphs_integration import *


class CrossValidationResults:
    """Class to store K-fold cross-validation results with year information"""

    def __init__(self):
        self.results = {}

    def store_cv_results(self, model_name, player_type, metric_type, y_true, y_pred, player_names, years):
        """Store cross-validation results with year information"""
        key = f"{model_name}_{player_type}_{metric_type}"
        self.results[key] = {
            'y_true': y_true,
            'y_pred': y_pred,
            'player_names': player_names,
            'years': years
        }

    def get_year_data(self, year):
        """Get all predictions for a specific year"""
        year_data = {}

        for key, data in self.results.items():
            year_indices = [i for i, y in enumerate(data['years']) if str(y) == str(year)]

            if year_indices:
                year_data[key] = {
                    'y_true': [data['y_true'][i] for i in year_indices],
                    'y_pred': [data['y_pred'][i] for i in year_indices],
                    'player_names': [data['player_names'][i] for i in year_indices],
                    'years': [data['years'][i] for i in year_indices]
                }

        return year_data

    def get_available_years(self):
        """Get list of all years with prediction data"""
        all_years = set()
        for data in self.results.values():
            all_years.update(str(y) for y in data['years'])
        return sorted(all_years)

    def get_performance_metrics(self, key):
        """Get performance metrics for a specific model/type combination"""
        if key not in self.results:
            return None

        data = self.results[key]
        r2 = r2_score(data['y_true'], data['y_pred'])
        rmse = np.sqrt(mean_squared_error(data['y_true'], data['y_pred']))
        mae = mean_absolute_error(data['y_true'], data['y_pred'])

        return {
            'r2': r2,
            'rmse': rmse,
            'mae': mae,
            'n_predictions': len(data['y_true'])
        }


class CurrentSeasonPredictor:
    """Class for real-time current season WAR/WARP prediction"""

    def __init__(self):
        self.trained_models = {}
        self.scalers = {}
        self.feature_columns = {}
        self.is_trained = False

    def load_historical_data_for_training(self):
        """Load and prepare historical data (2016-2024) for model training"""
        print("Loading historical training data (2016-2024)...")

        # Load data using existing functions
        hitter_data, pitcher_data = prepare_data_for_kfold()

        if not hitter_data or not pitcher_data:
            raise ValueError("Failed to load historical training data")

        return hitter_data, pitcher_data

    def train_ensemble_models(self, holdout_year=2024):
        """Train RandomForest + Keras ensemble models on historical data"""
        print(f"Training ensemble models (holdout year: {holdout_year})...")

        hitter_data, pitcher_data = self.load_historical_data_for_training()

        # Filter out holdout year for training
        for player_type, data in [('hitter', hitter_data), ('pitcher', pitcher_data)]:
            if data:
                for metric in ['warp', 'war']:
                    if metric in data:
                        train_indices = [i for i, year in enumerate(data[metric]['years'])
                                       if year != holdout_year]

                        X_train = np.array(data[metric]['X'])[train_indices]
                        y_train = np.array(data[metric]['y'])[train_indices]

                        # Train RandomForest
                        rf_model = RandomForestRegressor(
                            n_estimators=100,
                            random_state=42,
                            n_jobs=-1
                        )
                        rf_model.fit(X_train, y_train)

                        # Train Keras model if available
                        keras_model = None
                        if HAS_TENSORFLOW:
                            keras_model = self._build_keras_model(X_train.shape[1])
                            keras_model.fit(
                                X_train, y_train,
                                epochs=100,
                                batch_size=32,
                                validation_split=0.2,
                                callbacks=[EarlyStopping(patience=10, restore_best_weights=True)],
                                verbose=0
                            )

                        # Store trained models
                        key = f"{player_type}_{metric}"
                        self.trained_models[f"rf_{key}"] = rf_model
                        if keras_model:
                            self.trained_models[f"keras_{key}"] = keras_model

                        # Store feature information
                        self.feature_columns[key] = list(range(X_train.shape[1]))

                        print(f"  Trained models for {key}: RF + {'Keras' if keras_model else 'RF only'}")

        self.is_trained = True
        print("Ensemble model training completed!")

    def _build_keras_model(self, input_dim):
        """Build Keras neural network model"""
        model = Sequential([
            Dense(64, activation='relu', input_dim=input_dim),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1)
        ])

        model.compile(
            optimizer=AdamW(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )

        return model

    def predict_current_season(self, player_features, player_type, metric_type):
        """Generate ensemble prediction for current season player"""
        if not self.is_trained:
            raise ValueError("Models must be trained before making predictions")

        key = f"{player_type}_{metric_type}"

        # Get model predictions
        rf_key = f"rf_{key}"
        keras_key = f"keras_{key}"

        predictions = {}

        if rf_key in self.trained_models:
            rf_pred = self.trained_models[rf_key].predict([player_features])[0]
            predictions['randomforest'] = rf_pred

        if keras_key in self.trained_models:
            keras_pred = self.trained_models[keras_key].predict([player_features], verbose=0)[0][0]
            predictions['keras'] = keras_pred

        # Apply ensemble weighting based on validation results
        if metric_type == 'warp':
            # RandomForest stronger for WARP
            ensemble_pred = (0.8 * predictions.get('randomforest', 0) +
                           0.2 * predictions.get('keras', 0))
        else:  # WAR
            # Keras stronger for WAR
            ensemble_pred = (0.2 * predictions.get('randomforest', 0) +
                           0.8 * predictions.get('keras', 0))

        return ensemble_pred, predictions


def prepare_data_for_kfold():
    """
    Prepare comprehensive dataset for K-fold cross-validation
    Loads all available data from 2016-2024 for both hitters and pitchers
    """
    print("Preparing comprehensive dataset for K-fold cross-validation...")

    try:
        # Load hitter data
        print("Loading hitter data...")
        hitter_warp_data = load_and_prepare_hitter_data()
        hitter_war_data = load_and_prepare_hitter_war_data()

        hitter_data = None
        if hitter_warp_data or hitter_war_data:
            hitter_data = {
                'warp': hitter_warp_data,
                'war': hitter_war_data
            }

        # Load pitcher data
        print("Loading pitcher data...")
        pitcher_warp_data = load_and_prepare_pitcher_data()
        pitcher_war_data = load_and_prepare_pitcher_war_data()

        pitcher_data = None
        if pitcher_warp_data or pitcher_war_data:
            pitcher_data = {
                'warp': pitcher_warp_data,
                'war': pitcher_war_data
            }

        return hitter_data, pitcher_data

    except Exception as e:
        print(f"Error preparing data: {e}")
        return None, None


def run_kfold_cross_validation(hitter_data, pitcher_data, n_splits=5):
    """
    Run K-fold cross-validation with GroupKFold to prevent data leakage
    """
    print(f"Running {n_splits}-fold cross-validation...")

    cv_results = CrossValidationResults()

    # Define models to test
    models = {
        'ridge': Ridge(alpha=1.0),
        'randomforest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        'svr': SVR(kernel='rbf', C=1.0, gamma='scale')
    }

    # Add Keras if available
    if HAS_TENSORFLOW:
        models['keras'] = 'keras'  # Special handling for Keras

    # Process each dataset
    for player_type, data_dict in [('hitter', hitter_data), ('pitcher', pitcher_data)]:
        if not data_dict:
            continue

        for metric_type, data in data_dict.items():
            if not data:
                continue

            print(f"Running CV for {player_type} {metric_type}: {len(data['X'])} samples")

            X = np.array(data['X'])
            y = np.array(data['y'])
            years = np.array(data['years'])
            player_names = data['player_names']

            # Use GroupKFold with years as groups
            gkf = GroupKFold(n_splits=n_splits)

            # Store all predictions for each model
            all_predictions = {model_name: {'y_true': [], 'y_pred': [], 'years': [], 'names': []}
                             for model_name in models.keys()}

            for train_idx, test_idx in gkf.split(X, y, groups=years):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                test_years = years[test_idx]
                test_names = [player_names[i] for i in test_idx]

                for model_name, model in models.items():
                    if model_name == 'keras' and HAS_TENSORFLOW:
                        # Special handling for Keras
                        keras_model = Sequential([
                            Dense(64, activation='relu', input_dim=X_train.shape[1]),
                            Dropout(0.2),
                            Dense(32, activation='relu'),
                            Dropout(0.2),
                            Dense(16, activation='relu'),
                            Dense(1)
                        ])

                        keras_model.compile(
                            optimizer=AdamW(learning_rate=0.001),
                            loss='mse',
                            metrics=['mae']
                        )

                        keras_model.fit(
                            X_train, y_train,
                            epochs=100,
                            batch_size=32,
                            validation_split=0.2,
                            callbacks=[EarlyStopping(patience=10, restore_best_weights=True)],
                            verbose=0
                        )

                        y_pred = keras_model.predict(X_test, verbose=0).flatten()

                    else:
                        # Standard sklearn models
                        model_copy = type(model)(**model.get_params()) if hasattr(model, 'get_params') else model
                        model_copy.fit(X_train, y_train)
                        y_pred = model_copy.predict(X_test)

                    # Store predictions
                    all_predictions[model_name]['y_true'].extend(y_test)
                    all_predictions[model_name]['y_pred'].extend(y_pred)
                    all_predictions[model_name]['years'].extend(test_years)
                    all_predictions[model_name]['names'].extend(test_names)

            # Store results and print performance
            for model_name, preds in all_predictions.items():
                if preds['y_true']:  # Only store if we have predictions
                    cv_results.store_cv_results(
                        model_name, player_type, metric_type,
                        preds['y_true'], preds['y_pred'],
                        preds['names'], preds['years']
                    )

                    # Calculate and print metrics
                    r2 = r2_score(preds['y_true'], preds['y_pred'])
                    rmse = np.sqrt(mean_squared_error(preds['y_true'], preds['y_pred']))
                    print(f"  {model_name} {player_type} {metric_type}: R2={r2:.4f}, RMSE={rmse:.4f}")

    return cv_results


def print_cv_summary(cv_results):
    """Print comprehensive summary of cross-validation results"""
    print("\nCROSS-VALIDATION SUMMARY")
    print("=" * 50)

    available_years = cv_results.get_available_years()
    print(f"Years with predictions: {available_years}")
    print(f"Total years: {len(available_years)}")
    print()

    for key, data in cv_results.results.items():
        metrics = cv_results.get_performance_metrics(key)
        if metrics:
            print(f"{key}:")
            print(f"  Total predictions: {metrics['n_predictions']}")
            print(f"  R2: {metrics['r2']:.4f}")
            print(f"  RMSE: {metrics['rmse']:.4f}")
            print(f"  MAE: {metrics['mae']:.4f}")

            # Print year-by-year breakdown
            year_breakdown = {}
            for i, year in enumerate(data['years']):
                year_str = str(year)
                if year_str not in year_breakdown:
                    year_breakdown[year_str] = {'y_true': [], 'y_pred': []}
                year_breakdown[year_str]['y_true'].append(data['y_true'][i])
                year_breakdown[year_str]['y_pred'].append(data['y_pred'][i])

            for year in sorted(year_breakdown.keys()):
                year_data = year_breakdown[year]
                if year_data['y_true']:
                    year_r2 = r2_score(year_data['y_true'], year_data['y_pred'])
                    print(f"    {year}: {len(year_data['y_true'])} predictions, R2={year_r2:.4f}")
            print()


# Backward compatibility functions
def load_and_prepare_hitter_data():
    """Load and prepare hitter WARP data for modeling"""
    try:
        return load_comprehensive_warp_hitter_data()
    except Exception as e:
        print(f"Error loading hitter WARP data: {e}")
        return None

def load_and_prepare_hitter_war_data():
    """Load and prepare hitter WAR data for modeling"""
    try:
        return load_comprehensive_fangraphs_hitter_data()
    except Exception as e:
        print(f"Error loading hitter WAR data: {e}")
        return None

def load_and_prepare_pitcher_data():
    """Load and prepare pitcher WARP data for modeling"""
    try:
        return load_comprehensive_warp_pitcher_data()
    except Exception as e:
        print(f"Error loading pitcher WARP data: {e}")
        return None

def load_and_prepare_pitcher_war_data():
    """Load and prepare pitcher WAR data for modeling"""
    try:
        return load_comprehensive_fangraphs_pitcher_data()
    except Exception as e:
        print(f"Error loading pitcher WAR data: {e}")
        return None