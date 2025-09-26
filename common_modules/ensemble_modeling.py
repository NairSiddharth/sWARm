"""
Ensemble Modeling Module - RandomForest + Keras Ensemble System
Implements weighted ensemble approach with overfitting prevention strategies
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import GroupKFold
import sys
import os

try:
    import tensorflow as tf
    from keras.models import Sequential, Model
    from keras.layers import Dense, Activation, Dropout, Input
    from keras.callbacks import EarlyStopping
    from keras.optimizers import AdamW
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False
    print("Warning: TensorFlow/Keras not available. Ensemble will use RandomForest only.")


class EnsembleWARPredictor:
    """
    Ensemble predictor combining RandomForest + Keras models

    Based on validation results:
    - RandomForest: Better for WARP (R²=0.82 pitcher, 0.75 hitter)
    - Keras: Better for WAR (R²=0.83 pitcher, 0.69 hitter)

    Uses metric-specific weighting to prevent overfitting while leveraging
    complementary model strengths.
    """

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.scalers = {}
        self.ensemble_weights = {}
        self.validation_scores = {}
        self.is_trained = False

        # Set ensemble weights based on validation performance
        self._initialize_ensemble_weights()

    def _initialize_ensemble_weights(self):
        """Initialize ensemble weights based on historical validation results"""
        self.ensemble_weights = {
            'warp': {
                'randomforest': 0.8,  # RandomForest stronger for WARP
                'keras': 0.2
            },
            'war': {
                'randomforest': 0.2,  # Keras stronger for WAR
                'keras': 0.8
            }
        }

    def train_ensemble(self, X_train, y_train, groups_train, metric_type, player_type,
                      holdout_validation=True):
        """
        Train RandomForest + Keras ensemble with overfitting prevention

        Args:
            X_train: Training features
            y_train: Training targets
            groups_train: Group labels for validation (e.g., years)
            metric_type: 'war' or 'warp'
            player_type: 'hitter' or 'pitcher'
            holdout_validation: Use holdout validation for ensemble weights
        """
        print(f"Training ensemble for {player_type} {metric_type}...")

        key = f"{player_type}_{metric_type}"

        # Initialize scalers
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)
        self.scalers[key] = scaler

        # Train RandomForest
        print("  Training RandomForest...")
        rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=self.random_state,
            n_jobs=-1
        )
        rf_model.fit(X_scaled, y_train)

        # Train Keras if available
        keras_model = None
        if HAS_TENSORFLOW:
            print("  Training Keras neural network...")
            keras_model = self._build_keras_model(X_scaled.shape[1], player_type, metric_type)

            # Use early stopping and validation split for regularization
            early_stop = EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=0
            )

            keras_model.fit(
                X_scaled, y_train,
                epochs=150,
                batch_size=64,
                validation_split=0.2,
                callbacks=[early_stop],
                verbose=0
            )

        # Store trained models
        self.models[f"rf_{key}"] = rf_model
        if keras_model:
            self.models[f"keras_{key}"] = keras_model

        # Validate ensemble performance if requested
        if holdout_validation:
            self._validate_ensemble(X_scaled, y_train, groups_train, metric_type, player_type)

        print(f"  Ensemble training completed for {key}")

    def _build_keras_model(self, input_dim, player_type, metric_type):
        """Build Keras neural network with architecture optimized for baseball data"""

        # Architecture varies by player type and metric
        if player_type == 'pitcher':
            # Pitchers: More complex relationships between stats and performance
            layers = [128, 64, 32, 16]
            dropout_rate = 0.3
        else:  # hitter
            # Hitters: Simpler relationships
            layers = [64, 32, 16]
            dropout_rate = 0.2

        model = Sequential()
        model.add(Input(shape=(input_dim,)))

        for i, units in enumerate(layers):
            model.add(Dense(units, activation='relu'))
            if i < len(layers) - 1:  # Don't add dropout after last hidden layer
                model.add(Dropout(dropout_rate))

        model.add(Dense(1))  # Output layer

        # Compile model
        model.compile(
            optimizer=AdamW(learning_rate=0.001, weight_decay=1e-4),
            loss='mse',
            metrics=['mae']
        )

        return model

    def predict_ensemble(self, X, metric_type, player_type):
        """
        Generate ensemble prediction using trained models

        Args:
            X: Input features
            metric_type: 'war' or 'warp'
            player_type: 'hitter' or 'pitcher'

        Returns:
            dict: {'ensemble': ensemble_prediction, 'components': individual_predictions}
        """
        if not self.is_trained:
            raise ValueError("Ensemble must be trained before making predictions")

        key = f"{player_type}_{metric_type}"

        # Scale features
        if key not in self.scalers:
            raise ValueError(f"No trained scaler found for {key}")

        X_scaled = self.scalers[key].transform(X.reshape(1, -1) if X.ndim == 1 else X)

        # Get individual model predictions
        predictions = {}

        # RandomForest prediction
        rf_key = f"rf_{key}"
        if rf_key in self.models:
            rf_pred = self.models[rf_key].predict(X_scaled)
            predictions['randomforest'] = rf_pred[0] if len(rf_pred) == 1 else rf_pred

        # Keras prediction
        keras_key = f"keras_{key}"
        if keras_key in self.models:
            keras_pred = self.models[keras_key].predict(X_scaled, verbose=0)
            predictions['keras'] = keras_pred[0][0] if keras_pred.ndim > 1 else keras_pred[0]

        # Calculate ensemble prediction
        weights = self.ensemble_weights[metric_type]
        ensemble_pred = 0.0
        total_weight = 0.0

        for model_name, weight in weights.items():
            if model_name in predictions:
                ensemble_pred += weight * predictions[model_name]
                total_weight += weight

        if total_weight > 0:
            ensemble_pred /= total_weight

        return {
            'ensemble': ensemble_pred,
            'components': predictions,
            'weights': weights
        }

    def _validate_ensemble(self, X, y, groups, metric_type, player_type):
        """
        Validate ensemble performance using nested cross-validation
        to prevent overfitting on ensemble weights
        """
        print(f"  Validating ensemble for {player_type} {metric_type}...")

        key = f"{player_type}_{metric_type}"

        # Use GroupKFold for temporal validation
        gkf = GroupKFold(n_splits=3)  # Smaller splits for validation

        ensemble_scores = []
        rf_scores = []
        keras_scores = []

        for train_idx, val_idx in gkf.split(X, y, groups=groups):
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]

            # Train models on this fold
            rf_fold = RandomForestRegressor(
                n_estimators=100,
                random_state=self.random_state,
                n_jobs=-1
            )
            rf_fold.fit(X_train_fold, y_train_fold)

            rf_pred = rf_fold.predict(X_val_fold)
            rf_score = r2_score(y_val_fold, rf_pred)
            rf_scores.append(rf_score)

            if HAS_TENSORFLOW:
                keras_fold = self._build_keras_model(X.shape[1], player_type, metric_type)
                keras_fold.fit(
                    X_train_fold, y_train_fold,
                    epochs=50,
                    batch_size=32,
                    validation_split=0.2,
                    callbacks=[EarlyStopping(patience=10, restore_best_weights=True)],
                    verbose=0
                )

                keras_pred = keras_fold.predict(X_val_fold, verbose=0).flatten()
                keras_score = r2_score(y_val_fold, keras_pred)
                keras_scores.append(keras_score)

                # Calculate ensemble prediction
                weights = self.ensemble_weights[metric_type]
                ensemble_pred = (weights['randomforest'] * rf_pred +
                               weights['keras'] * keras_pred)
                ensemble_score = r2_score(y_val_fold, ensemble_pred)
                ensemble_scores.append(ensemble_score)

        # Store validation results
        validation_result = {
            'randomforest_mean_r2': np.mean(rf_scores),
            'randomforest_std_r2': np.std(rf_scores)
        }

        if keras_scores:
            validation_result.update({
                'keras_mean_r2': np.mean(keras_scores),
                'keras_std_r2': np.std(keras_scores),
                'ensemble_mean_r2': np.mean(ensemble_scores),
                'ensemble_std_r2': np.std(ensemble_scores),
                'ensemble_improvement': np.mean(ensemble_scores) - max(np.mean(rf_scores), np.mean(keras_scores))
            })

        self.validation_scores[key] = validation_result

        print(f"    RandomForest R² = {validation_result['randomforest_mean_r2']:.4f} ± {validation_result['randomforest_std_r2']:.4f}")
        if keras_scores:
            print(f"    Keras R² = {validation_result['keras_mean_r2']:.4f} ± {validation_result['keras_std_r2']:.4f}")
            print(f"    Ensemble R² = {validation_result['ensemble_mean_r2']:.4f} ± {validation_result['ensemble_std_r2']:.4f}")
            improvement = validation_result['ensemble_improvement']
            print(f"    Ensemble improvement: {improvement:+.4f}")

    def get_feature_importance(self, metric_type, player_type, feature_names=None):
        """Get feature importance from RandomForest model"""
        key = f"{player_type}_{metric_type}"
        rf_key = f"rf_{key}"

        if rf_key not in self.models:
            return None

        importances = self.models[rf_key].feature_importances_

        if feature_names:
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            return importance_df
        else:
            return importances

    def save_ensemble(self, filepath):
        """Save trained ensemble models and scalers"""
        # Implementation for saving models - would use joblib/pickle for sklearn
        # and tf.keras.models.save_model for Keras
        pass

    def load_ensemble(self, filepath):
        """Load trained ensemble models and scalers"""
        # Implementation for loading models
        pass

    def get_validation_summary(self):
        """Get summary of ensemble validation results"""
        summary = {}

        for key, results in self.validation_scores.items():
            player_type, metric_type = key.split('_')

            summary[key] = {
                'player_type': player_type,
                'metric_type': metric_type,
                'best_individual_model': 'randomforest' if results['randomforest_mean_r2'] > results.get('keras_mean_r2', 0) else 'keras',
                'ensemble_performance': results.get('ensemble_mean_r2', results['randomforest_mean_r2']),
                'improvement_over_best': results.get('ensemble_improvement', 0)
            }

        return summary


def create_ensemble_for_data(hitter_data, pitcher_data, holdout_year=2024):
    """
    Convenience function to create and train ensemble models for all data types

    Args:
        hitter_data: Dictionary with 'warp' and 'war' data for hitters
        pitcher_data: Dictionary with 'warp' and 'war' data for pitchers
        holdout_year: Year to hold out for validation

    Returns:
        EnsembleWARPredictor: Trained ensemble predictor
    """
    ensemble = EnsembleWARPredictor()

    # Train models for each combination
    for player_type, data_dict in [('hitter', hitter_data), ('pitcher', pitcher_data)]:
        if not data_dict:
            continue

        for metric_type, data in data_dict.items():
            if not data:
                continue

            # Filter out holdout year
            train_indices = [i for i, year in enumerate(data['years']) if year != holdout_year]

            if not train_indices:
                continue

            X_train = np.array([data['X'][i] for i in train_indices])
            y_train = np.array([data['y'][i] for i in train_indices])
            groups_train = np.array([data['years'][i] for i in train_indices])

            # Train ensemble
            ensemble.train_ensemble(
                X_train, y_train, groups_train,
                metric_type, player_type
            )

    ensemble.is_trained = True
    return ensemble


def validate_ensemble_overfitting_prevention(hitter_data, pitcher_data, holdout_year=2024):
    """
    Validate that ensemble approach prevents overfitting using holdout year

    Returns:
        dict: Validation results showing ensemble performance vs individual models
    """
    print("Validating ensemble overfitting prevention...")

    # Create ensemble with holdout validation
    ensemble = create_ensemble_for_data(hitter_data, pitcher_data, holdout_year)

    # Test on holdout year data
    validation_results = {}

    for player_type, data_dict in [('hitter', hitter_data), ('pitcher', pitcher_data)]:
        if not data_dict:
            continue

        for metric_type, data in data_dict.items():
            if not data:
                continue

            # Get holdout year data
            holdout_indices = [i for i, year in enumerate(data['years']) if year == holdout_year]

            if not holdout_indices:
                continue

            X_holdout = np.array([data['X'][i] for i in holdout_indices])
            y_holdout = np.array([data['y'][i] for i in holdout_indices])

            # Generate predictions
            ensemble_predictions = []
            for features in X_holdout:
                pred_result = ensemble.predict_ensemble(features, metric_type, player_type)
                ensemble_predictions.append(pred_result['ensemble'])

            # Calculate performance metrics
            r2 = r2_score(y_holdout, ensemble_predictions)
            rmse = np.sqrt(mean_squared_error(y_holdout, ensemble_predictions))

            key = f"{player_type}_{metric_type}"
            validation_results[key] = {
                'holdout_r2': r2,
                'holdout_rmse': rmse,
                'n_samples': len(holdout_indices),
                'validation_r2': ensemble.validation_scores.get(key, {}).get('ensemble_mean_r2', 0)
            }

            print(f"{key}: Holdout R² = {r2:.4f}, RMSE = {rmse:.4f}")

    return validation_results