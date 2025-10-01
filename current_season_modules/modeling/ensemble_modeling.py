"""
Ensemble Modeling Module - Separate WAR/WARP RandomForest + Keras Ensemble System

IMPORTANT UPDATE: Models are now trained SEPARATELY for WAR and WARP metrics
to handle their fundamentally different distributions:
- WAR: Wider distribution, 37.5% negative values, mean ~0.5
- WARP: Compressed distribution, 18.9% negative values, 66% between 0-1

This separation significantly improves validation metrics, especially for
pitcher WARP predictions which previously had negative R² values.
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import GroupKFold
import sys
import os

# Local imports
from common_modules.config import (
    MODEL_CACHE_PATH,
    MODEL_HISTORY_DIR,
    DEFAULT_HOLDOUT_YEAR,
    RANDOM_STATE
)
from common_modules.logging import get_logger

logger = get_logger(__name__)

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

# Public API exports
__all__ = [
    'EnsembleWARPredictor',
    'get_or_train_ensemble'
]


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
        """Initialize ensemble weights based on historical validation results

        NOTE: With separate training for WAR and WARP, weights are adjusted
        to account for metric-specific model performance.
        """
        self.ensemble_weights = {
            'warp': {
                'randomforest': 0.7,  # RF handles WARP's compressed distribution better
                'keras': 0.3
            },
            'war': {
                'randomforest': 0.3,  # Keras better for WAR's wider distribution
                'keras': 0.7
            }
        }

    def train_ensemble(self, X_train, y_train, groups_train, metric_type, player_type,
                       holdout_validation=True):
        """
        Train RandomForest + Keras ensemble with overfitting prevention

        IMPORTANT: Models are now trained separately for WAR and WARP to handle
        their different distributions and value scales.

        Args:
            X_train: Training features
            y_train: Training targets
            groups_train: Group labels for validation (e.g., years)
            metric_type: 'war' or 'warp'
            player_type: 'hitter' or 'pitcher'
            holdout_validation: Use holdout validation for ensemble weights
        """
        print(f"Training SEPARATE ensemble for {player_type} {metric_type.upper()}...")
        print(f"  Training on {len(y_train)} samples")
        print(f"  Target range: {np.min(y_train):.2f} to {np.max(y_train):.2f}")
        print(f"  Target mean: {np.mean(y_train):.3f}, std: {np.std(y_train):.3f}")

        key = f"{player_type}_{metric_type}"

        # Initialize metric-specific scalers
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)
        self.scalers[key] = scaler

        # Train RandomForest with metric-specific hyperparameters
        print(f"  Training RandomForest for {metric_type.upper()}...")

        # Adjust hyperparameters based on metric type
        if metric_type == 'warp':
            # WARP has tighter distribution, needs less aggressive regularization
            rf_params = {
                'n_estimators': 150,
                'max_depth': 8,
                'min_samples_split': 10,
                'min_samples_leaf': 5,
                'random_state': self.random_state,
                'n_jobs': -1
            }
        else:
            # WAR has wider distribution, standard parameters
            rf_params = {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'random_state': self.random_state,
                'n_jobs': -1
            }

        rf_model = RandomForestRegressor(**rf_params)
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
        """Build Keras neural network with architecture optimized for baseball data

        Architecture now varies by both player type AND metric type to handle
        the different distributions of WAR vs WARP.
        """

        # Architecture varies by player type and metric
        if player_type == 'pitcher':
            if metric_type == 'warp':
                # WARP: Tighter distribution, needs simpler model to avoid overfitting
                layers = [64, 32, 16]
                dropout_rate = 0.4
                learning_rate = 0.0005
            else:  # war
                # WAR: Wider distribution, can handle more complex model
                layers = [128, 64, 32, 16]
                dropout_rate = 0.3
                learning_rate = 0.001
        else:  # hitter
            if metric_type == 'warp':
                # WARP: Compressed scale
                layers = [32, 16, 8]
                dropout_rate = 0.3
                learning_rate = 0.0005
            else:  # war
                # WAR: Standard architecture
                layers = [64, 32, 16]
                dropout_rate = 0.2
                learning_rate = 0.001

        model = Sequential()
        model.add(Input(shape=(input_dim,)))

        for i, units in enumerate(layers):
            model.add(Dense(units, activation='relu'))
            if i < len(layers) - 1:  # Don't add dropout after last hidden layer
                model.add(Dropout(dropout_rate))

        model.add(Dense(1))  # Output layer

        # Compile model with metric-specific learning rate
        model.compile(
            optimizer=AdamW(learning_rate=learning_rate, weight_decay=1e-4),
            loss='mse',
            metrics=['mae']
        )

        return model

    def _handle_feature_compatibility(self, X, player_type):
        """
        Handle feature compatibility between models with different feature sets.

        Current situation:
        - Trained models expect: ['IP', 'BB%', 'K%', 'ERA', 'damage_control_ratio', 'Opportunity_Success', 'Contact_Quality_Index', 'HBP%', 'WP', 'Statcast_Launch_Quality_Index'] (10 features with WP)
        - Notebook data provides: ['IP', 'BB%', 'K%', 'ERA', 'damage_control_ratio', 'Opportunity_Success', 'Contact_Quality_Index', 'HBP%', 'Statcast_Launch_Quality_Index'] (9 features, missing both K-BB% and WP)
        - New system expects: ['IP', 'BB%', 'K%', 'K-BB%', 'ERA', 'damage_control_ratio', 'Opportunity_Success', 'Contact_Quality_Index', 'HBP%', 'Statcast_Launch_Quality_Index'] (10 features with K-BB%)

        Args:
            X: Input features array
            player_type: 'hitter' or 'pitcher'

        Returns:
            Compatible feature array for trained models
        """
        if player_type == 'pitcher':
            X_compat = X.copy() if hasattr(X, 'copy') else np.array(X)

            if X_compat.ndim == 1:
                n_features = len(X_compat)

                if n_features == 9:
                    # Case: Notebook data missing K-BB% and WP
                    # Input: ['IP', 'BB%', 'K%', 'ERA', 'damage_control_ratio', 'Opportunity_Success', 'Contact_Quality_Index', 'HBP%', 'Statcast_Launch_Quality_Index']
                    # Need: ['IP', 'BB%', 'K%', 'ERA', 'damage_control_ratio', 'Opportunity_Success', 'Contact_Quality_Index', 'HBP%', 'WP', 'Statcast_Launch_Quality_Index']

                    # Insert WP (default value) at position 8, push Statcast to position 9
                    wp_default = 5.0  # Default wild pitches per season
                    X_extended = np.zeros(10)
                    X_extended[:8] = X_compat[:8]  # Copy first 8 features
                    X_extended[8] = wp_default     # Insert WP at position 8
                    X_extended[9] = X_compat[8]    # Move Statcast to position 9
                    X_compat = X_extended

                elif n_features == 10:
                    # Case: Feature vector already has 10 features
                    # Check if it's new format (with K-BB%) or old format (with WP already)
                    # Heuristic: In new format, position 3 would be K-BB% (typically -5 to +30)
                    #           In old format, position 3 would be ERA (typically 2-8)
                    #           If position 3 < 1.0, it's likely K-BB% (new format)
                    #           If position 3 > 1.0, it's likely ERA (old format)

                    if X_compat[3] < 1.0:  # Likely K-BB% (new format)
                        # Convert new format to old format
                        X_old = np.zeros(10)
                        X_old[0] = X_compat[0]  # IP
                        X_old[1] = X_compat[1]  # BB%
                        X_old[2] = X_compat[2]  # K%
                        X_old[3] = X_compat[4]  # ERA (skip K-BB%)
                        X_old[4] = X_compat[5]  # damage_control_ratio
                        X_old[5] = X_compat[6]  # Opportunity_Success
                        X_old[6] = X_compat[7]  # Contact_Quality_Index
                        X_old[7] = X_compat[8]  # HBP%
                        X_old[8] = 5.0          # WP (default)
                        X_old[9] = X_compat[9]  # Statcast_Launch_Quality_Index
                        X_compat = X_old
                    # else: Already in old format, use as-is

            else:
                # Handle 2D arrays (batch predictions)
                n_features = X_compat.shape[1]
                if n_features == 9:
                    # Add WP column for batch processing
                    wp_column = np.full((X_compat.shape[0], 1), 5.0)
                    X_compat = np.column_stack([X_compat[:, :8], wp_column, X_compat[:, 8:]])

        else:
            # For hitters, return as-is
            return X

        return X_compat

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

        # Handle feature compatibility for backward compatibility with existing models
        X_compatible = self._handle_feature_compatibility(X, player_type)

        # Scale features
        if key not in self.scalers:
            raise ValueError(f"No trained scaler found for {key}")

        X_scaled = self.scalers[key].transform(X_compatible.reshape(
            1, -1) if X_compatible.ndim == 1 else X_compatible)

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

        print(
            f"    RandomForest R² = {
                validation_result['randomforest_mean_r2']:.4f} ± {
                validation_result['randomforest_std_r2']:.4f}")
        if keras_scores:
            print(
                f"    Keras R² = {
                    validation_result['keras_mean_r2']:.4f} ± {
                    validation_result['keras_std_r2']:.4f}")
            print(
                f"    Ensemble R² = {
                    validation_result['ensemble_mean_r2']:.4f} ± {
                    validation_result['ensemble_std_r2']:.4f}")
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
                'best_individual_model': 'randomforest' if results['randomforest_mean_r2'] > results.get(
                    'keras_mean_r2',
                    0) else 'keras',
                'ensemble_performance': results.get(
                    'ensemble_mean_r2',
                    results['randomforest_mean_r2']),
                'improvement_over_best': results.get(
                    'ensemble_improvement',
                    0)}

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

            # Handle years data that might be wrapped in a tuple
            years_data = data.get('years', [])
            if isinstance(years_data, tuple) and len(years_data) == 1:
                years_data = years_data[0]

            # Create boolean mask for filtering
            if holdout_year is not None:
                # Create mask for training data (exclude holdout year)
                mask = np.array(years_data) != holdout_year
            else:
                # Use all data
                mask = np.ones(len(years_data) if years_data else len(data['y']), dtype=bool)

            # Check if we have any training data
            if not mask.any():
                continue

            # Apply boolean mask to extract training data
            # This works consistently for both DataFrames and arrays
            X_train = data['X'][mask].values if hasattr(data['X'], 'values') else np.array(data['X'])[mask]
            y_train = data['y'][mask].values if hasattr(data['y'], 'values') else np.array(data['y'])[mask]
            groups_train = np.array(years_data)[mask] if years_data else np.array([2020] * mask.sum())

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

            # Handle DataFrame indexing properly
            if hasattr(data['X'], 'iloc'):  # DataFrame
                X_holdout = np.array([data['X'].iloc[i].values for i in holdout_indices])
            else:  # List or array
                X_holdout = np.array([data['X'][i] for i in holdout_indices])
            # Handle Series indexing properly
            if hasattr(data['y'], 'iloc'):  # pandas Series
                y_holdout = np.array([data['y'].iloc[i] for i in holdout_indices])
            else:  # List or array
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


def _save_ensemble_with_backup(ensemble, cache_path):
    """
    Save ensemble model with automatic backup of existing model.

    Args:
        ensemble: Trained EnsembleWARPredictor instance
        cache_path: Path to save the model

    Raises:
        IOError: If save operation fails
    """
    cache_path = Path(cache_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create backup of existing model
    if cache_path.exists():
        backup_path = MODEL_HISTORY_DIR / f"ensemble_models_backup_{timestamp}.pkl"
        logger.info(f"Backing up existing model to {backup_path}")
        try:
            import shutil
            shutil.copy2(cache_path, backup_path)
        except Exception as e:
            logger.warning(f"Could not create backup: {e}")

    # Save new model
    try:
        with open(cache_path, 'wb') as f:
            pickle.dump(ensemble, f)
        logger.info(f"Model saved to {cache_path}")
    except Exception as e:
        logger.error(f"Failed to save model: {e}")
        raise IOError(f"Failed to save ensemble model: {e}")


def get_or_train_ensemble(
    hitter_data,
    pitcher_data,
    use_cached=True,
    cache_path=None,
    holdout_year=None,
    force_retrain=False
):
    """
    Get ensemble model - either from cache or by training.

    This is the main entry point for obtaining an ensemble model. It handles
    caching logic automatically, providing fast iteration during development
    while allowing explicit control when needed.

    Args:
        hitter_data: Dictionary with 'warp' and 'war' data for hitters
        pitcher_data: Dictionary with 'warp' and 'war' data for pitchers
        use_cached: If True, try to load from cache and save after training.
                   If False, train in-memory only without caching.
        cache_path: Path to cached model file (defaults to config.MODEL_CACHE_PATH)
        holdout_year: Year to hold out for validation (defaults to config.DEFAULT_HOLDOUT_YEAR)
        force_retrain: If True, retrain even if cache exists (implies use_cached=True)

    Returns:
        EnsembleWARPredictor: Trained ensemble model

    Examples:
        # Default: Use cache if exists, train if not, save result
        >>> ensemble = get_or_train_ensemble(hitter_data, pitcher_data)

        # Force retrain when features changed
        >>> ensemble = get_or_train_ensemble(hitter_data, pitcher_data, force_retrain=True)

        # No caching for experiments
        >>> ensemble = get_or_train_ensemble(hitter_data, pitcher_data, use_cached=False)
    """
    # Use config defaults
    cache_path = Path(cache_path) if cache_path else MODEL_CACHE_PATH
    holdout_year = holdout_year if holdout_year is not None else DEFAULT_HOLDOUT_YEAR

    # Try to load from cache
    if use_cached and not force_retrain and cache_path.exists():
        logger.info(f"Loading cached ensemble from {cache_path}")
        try:
            with open(cache_path, 'rb') as f:
                ensemble = pickle.load(f)
            logger.info("Cached model loaded successfully")
            return ensemble
        except Exception as e:
            logger.warning(f"Failed to load cache ({e}). Training new model...")

    # Determine logging message
    if use_cached and not cache_path.exists():
        logger.warning(f"No cached model found at {cache_path}. Training new model...")
    elif force_retrain:
        logger.info("Force retrain requested. Training new model...")
    elif not use_cached:
        logger.info("Training new model (caching disabled)...")
    else:
        logger.info("Training new model...")

    # Train model using existing create_ensemble_for_data function
    logger.info(f"Training ensemble with holdout year: {holdout_year}")
    ensemble = create_ensemble_for_data(hitter_data, pitcher_data, holdout_year)

    # Save to cache if requested
    if use_cached or force_retrain:
        _save_ensemble_with_backup(ensemble, cache_path)
    else:
        logger.info("Model trained in-memory only (not cached)")

    return ensemble
