"""
Ensemble Modeling Module - Three-Tree Ensemble (RF + XGBoost + LightGBM)

This replaces the RF + Keras ensemble with a pure tree-based ensemble that better
handles the IP discontinuity problem for pitchers (relievers vs starters).

Key improvements:
- RandomForest: Bagging, robust to outliers, good baseline
- XGBoost: Level-wise boosting, captures subtle interactions
- LightGBM: Leaf-wise boosting, faster training, better categorical handling

All three are tree-based models that naturally handle the non-linear relationship
between IP and WAR for different pitcher roles.
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

# Gradient boosting libraries
try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("Warning: XGBoost not available")

try:
    from lightgbm import LGBMRegressor
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    print("Warning: LightGBM not available")

# Local imports
from common_modules.config import (
    MODEL_CACHE_PATH,
    MODEL_HISTORY_DIR,
    DEFAULT_HOLDOUT_YEAR,
    RANDOM_STATE
)
from common_modules.logging import get_logger

logger = get_logger(__name__)

# Public API exports
__all__ = [
    'EnsembleTreePredictor',
    'get_or_train_ensemble'
]


class EnsembleTreePredictor:
    """
    Ensemble predictor combining RandomForest + XGBoost + LightGBM

    Three complementary tree-based models:
    - RandomForest: Bagging (parallel trees, variance reduction)
    - XGBoost: Gradient boosting with level-wise growth
    - LightGBM: Gradient boosting with leaf-wise growth

    Equal weighting (0.33 each) to start, can be optimized via validation.
    """

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.scalers = {}
        self.ensemble_weights = {}
        self.validation_scores = {}
        self.is_trained = False

        # Set ensemble weights
        self._initialize_ensemble_weights()

    def _initialize_ensemble_weights(self):
        """Initialize ensemble weights for three-tree ensemble

        Using equal weights to start - let validation determine if adjustments needed.
        """
        # Check which models are available
        n_models = 1 + (1 if HAS_XGBOOST else 0) + (1 if HAS_LIGHTGBM else 0)
        base_weight = 1.0 / n_models

        weights = {'randomforest': base_weight}
        if HAS_XGBOOST:
            weights['xgboost'] = base_weight
        if HAS_LIGHTGBM:
            weights['lightgbm'] = base_weight

        # Same weights for both WAR and WARP (tree models handle both well)
        self.ensemble_weights = {
            'warp': weights.copy(),
            'war': weights.copy()
        }

    def train_ensemble(self, X_train, y_train, groups_train, metric_type, player_type,
                       holdout_validation=True):
        """
        Train RF + XGBoost + LightGBM ensemble

        Args:
            X_train: Training features
            y_train: Training targets
            groups_train: Group labels for validation (e.g., years)
            metric_type: 'war' or 'warp'
            player_type: 'hitter' or 'pitcher'
            holdout_validation: Use holdout validation for ensemble weights
        """
        print(f"Training THREE-TREE ensemble for {player_type} {metric_type.upper()}...")
        print(f"  Training on {len(y_train)} samples")
        print(f"  Target range: {np.min(y_train):.2f} to {np.max(y_train):.2f}")
        print(f"  Target mean: {np.mean(y_train):.3f}, std: {np.std(y_train):.3f}")

        key = f"{player_type}_{metric_type}"

        # Initialize scaler (trees don't strictly need it, but keep for consistency)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)
        self.scalers[key] = scaler

        # Train RandomForest
        print(f"  Training RandomForest for {metric_type.upper()}...")
        rf_params = self._get_rf_params(metric_type)
        rf_model = RandomForestRegressor(**rf_params)
        rf_model.fit(X_scaled, y_train)
        self.models[f"rf_{key}"] = rf_model

        # Train XGBoost
        if HAS_XGBOOST:
            print(f"  Training XGBoost for {metric_type.upper()}...")
            xgb_params = self._get_xgb_params(metric_type)
            xgb_model = XGBRegressor(**xgb_params)
            xgb_model.fit(X_scaled, y_train, verbose=False)
            self.models[f"xgboost_{key}"] = xgb_model

        # Train LightGBM
        if HAS_LIGHTGBM:
            print(f"  Training LightGBM for {metric_type.upper()}...")
            lgbm_params = self._get_lgbm_params(metric_type)
            lgbm_model = LGBMRegressor(**lgbm_params)
            lgbm_model.fit(X_scaled, y_train)
            self.models[f"lightgbm_{key}"] = lgbm_model

        # Validate ensemble performance if requested
        if holdout_validation:
            self._validate_ensemble(X_scaled, y_train, groups_train, metric_type, player_type)

        print(f"  Ensemble training completed for {key}")

    def _get_rf_params(self, metric_type):
        """Get RandomForest hyperparameters based on metric type"""
        if metric_type == 'warp':
            # WARP has tighter distribution
            return {
                'n_estimators': 150,
                'max_depth': 8,
                'min_samples_split': 10,
                'min_samples_leaf': 5,
                'random_state': self.random_state,
                'n_jobs': -1
            }
        else:
            # WAR has wider distribution
            return {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'random_state': self.random_state,
                'n_jobs': -1
            }

    def _get_xgb_params(self, metric_type):
        """Get XGBoost hyperparameters based on metric type"""
        base_params = {
            'n_estimators': 200,
            'max_depth': 6,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,  # L1 regularization
            'reg_lambda': 1.0,  # L2 regularization
            'random_state': self.random_state,
            'n_jobs': -1,
        }

        if metric_type == 'warp':
            # WARP: More regularization to prevent overfitting
            base_params['reg_lambda'] = 2.0
            base_params['min_child_weight'] = 3

        return base_params

    def _get_lgbm_params(self, metric_type):
        """Get LightGBM hyperparameters based on metric type"""
        base_params = {
            'n_estimators': 200,
            'max_depth': 6,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': self.random_state,
            'n_jobs': -1,
            'verbose': -1,
        }

        if metric_type == 'warp':
            # WARP: More regularization
            base_params['reg_lambda'] = 2.0
            base_params['min_child_samples'] = 20

        return base_params

    def _handle_feature_compatibility(self, X, player_type):
        """
        Handle feature compatibility between models with different feature sets.

        Current situation:
        - Trained models expect: ['IP', 'BB%', 'K%', 'ERA', 'damage_control_ratio',
          'Opportunity_Success', 'Contact_Quality_Index', 'HBP%', 'WP',
          'Statcast_Launch_Quality_Index'] (10 features with WP)
        - Notebook data provides: ['IP', 'BB%', 'K%', 'ERA', 'damage_control_ratio',
          'Opportunity_Success', 'Contact_Quality_Index', 'HBP%',
          'Statcast_Launch_Quality_Index'] (9 features, missing both K-BB% and WP)

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
                    # Insert WP (default value) at position 8
                    wp_default = 5.0
                    X_extended = np.zeros(10)
                    X_extended[:8] = X_compat[:8]
                    X_extended[8] = wp_default
                    X_extended[9] = X_compat[8]
                    X_compat = X_extended

                elif n_features == 10:
                    # Check if it's new format (with K-BB%) or old format (with WP)
                    if X_compat[3] < 1.0:  # Likely K-BB%
                        X_old = np.zeros(10)
                        X_old[0] = X_compat[0]  # IP
                        X_old[1] = X_compat[1]  # BB%
                        X_old[2] = X_compat[2]  # K%
                        X_old[3] = X_compat[4]  # ERA
                        X_old[4] = X_compat[5]  # damage_control_ratio
                        X_old[5] = X_compat[6]  # Opportunity_Success
                        X_old[6] = X_compat[7]  # Contact_Quality_Index
                        X_old[7] = X_compat[8]  # HBP%
                        X_old[8] = 5.0          # WP
                        X_old[9] = X_compat[9]  # Statcast_Launch_Quality_Index
                        X_compat = X_old
            else:
                # Handle 2D arrays
                n_features = X_compat.shape[1]
                if n_features == 9:
                    wp_column = np.full((X_compat.shape[0], 1), 5.0)
                    X_compat = np.column_stack([X_compat[:, :8], wp_column, X_compat[:, 8:]])
        else:
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

        # Handle feature compatibility
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

        # XGBoost prediction
        xgb_key = f"xgboost_{key}"
        if xgb_key in self.models:
            xgb_pred = self.models[xgb_key].predict(X_scaled)
            predictions['xgboost'] = xgb_pred[0] if len(xgb_pred) == 1 else xgb_pred

        # LightGBM prediction
        lgbm_key = f"lightgbm_{key}"
        if lgbm_key in self.models:
            lgbm_pred = self.models[lgbm_key].predict(X_scaled)
            predictions['lightgbm'] = lgbm_pred[0] if len(lgbm_pred) == 1 else lgbm_pred

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
        """
        print(f"  Validating ensemble for {player_type} {metric_type}...")

        key = f"{player_type}_{metric_type}"

        # Use GroupKFold for temporal validation
        gkf = GroupKFold(n_splits=3)

        ensemble_scores = []
        rf_scores = []
        xgb_scores = []
        lgbm_scores = []

        for train_idx, val_idx in gkf.split(X, y, groups=groups):
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]

            fold_predictions = {}

            # Train and evaluate RandomForest
            rf_fold = RandomForestRegressor(
                n_estimators=100,
                random_state=self.random_state,
                n_jobs=-1
            )
            rf_fold.fit(X_train_fold, y_train_fold)
            rf_pred = rf_fold.predict(X_val_fold)
            rf_score = r2_score(y_val_fold, rf_pred)
            rf_scores.append(rf_score)
            fold_predictions['randomforest'] = rf_pred

            # Train and evaluate XGBoost
            if HAS_XGBOOST:
                xgb_fold = XGBRegressor(
                    n_estimators=100,
                    random_state=self.random_state,
                    n_jobs=-1
                )
                xgb_fold.fit(X_train_fold, y_train_fold, verbose=False)
                xgb_pred = xgb_fold.predict(X_val_fold)
                xgb_score = r2_score(y_val_fold, xgb_pred)
                xgb_scores.append(xgb_score)
                fold_predictions['xgboost'] = xgb_pred

            # Train and evaluate LightGBM
            if HAS_LIGHTGBM:
                lgbm_fold = LGBMRegressor(
                    n_estimators=100,
                    random_state=self.random_state,
                    n_jobs=-1,
                    verbose=-1
                )
                lgbm_fold.fit(X_train_fold, y_train_fold)
                lgbm_pred = lgbm_fold.predict(X_val_fold)
                lgbm_score = r2_score(y_val_fold, lgbm_pred)
                lgbm_scores.append(lgbm_score)
                fold_predictions['lightgbm'] = lgbm_pred

            # Calculate ensemble prediction
            weights = self.ensemble_weights[metric_type]
            ensemble_pred = np.zeros(len(y_val_fold))
            total_weight = 0.0

            for model_name, weight in weights.items():
                if model_name in fold_predictions:
                    ensemble_pred += weight * fold_predictions[model_name]
                    total_weight += weight

            if total_weight > 0:
                ensemble_pred /= total_weight

            ensemble_score = r2_score(y_val_fold, ensemble_pred)
            ensemble_scores.append(ensemble_score)

        # Store validation results
        validation_result = {
            'randomforest_mean_r2': np.mean(rf_scores),
            'randomforest_std_r2': np.std(rf_scores)
        }

        if xgb_scores:
            validation_result.update({
                'xgboost_mean_r2': np.mean(xgb_scores),
                'xgboost_std_r2': np.std(xgb_scores)
            })

        if lgbm_scores:
            validation_result.update({
                'lightgbm_mean_r2': np.mean(lgbm_scores),
                'lightgbm_std_r2': np.std(lgbm_scores)
            })

        if ensemble_scores:
            best_individual = max(
                np.mean(rf_scores),
                np.mean(xgb_scores) if xgb_scores else 0,
                np.mean(lgbm_scores) if lgbm_scores else 0
            )
            validation_result.update({
                'ensemble_mean_r2': np.mean(ensemble_scores),
                'ensemble_std_r2': np.std(ensemble_scores),
                'ensemble_improvement': np.mean(ensemble_scores) - best_individual
            })

        self.validation_scores[key] = validation_result

        # Print results
        print(f"    RandomForest R² = {validation_result['randomforest_mean_r2']:.4f} ± {validation_result['randomforest_std_r2']:.4f}")
        if xgb_scores:
            print(f"    XGBoost R² = {validation_result['xgboost_mean_r2']:.4f} ± {validation_result['xgboost_std_r2']:.4f}")
        if lgbm_scores:
            print(f"    LightGBM R² = {validation_result['lightgbm_mean_r2']:.4f} ± {validation_result['lightgbm_std_r2']:.4f}")
        if ensemble_scores:
            print(f"    Ensemble R² = {validation_result['ensemble_mean_r2']:.4f} ± {validation_result['ensemble_std_r2']:.4f}")
            print(f"    Ensemble improvement: {validation_result['ensemble_improvement']:+.4f}")

    def get_feature_importance(self, metric_type, player_type, feature_names=None):
        """Get feature importance from all tree models"""
        key = f"{player_type}_{metric_type}"

        importances = {}

        # Get RF importance
        rf_key = f"rf_{key}"
        if rf_key in self.models:
            importances['randomforest'] = self.models[rf_key].feature_importances_

        # Get XGBoost importance
        xgb_key = f"xgboost_{key}"
        if xgb_key in self.models:
            importances['xgboost'] = self.models[xgb_key].feature_importances_

        # Get LightGBM importance
        lgbm_key = f"lightgbm_{key}"
        if lgbm_key in self.models:
            importances['lightgbm'] = self.models[lgbm_key].feature_importances_

        if feature_names and importances:
            # Average importance across models
            avg_importance = np.mean([imp for imp in importances.values()], axis=0)
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': avg_importance
            }).sort_values('importance', ascending=False)

            # Add individual model importances
            for model_name, imp in importances.items():
                importance_df[f'{model_name}_importance'] = imp

            return importance_df
        else:
            return importances

    def get_validation_summary(self):
        """Get summary of ensemble validation results"""
        summary = {}

        for key, results in self.validation_scores.items():
            player_type, metric_type = key.split('_')

            # Find best individual model
            model_scores = []
            if 'randomforest_mean_r2' in results:
                model_scores.append(('randomforest', results['randomforest_mean_r2']))
            if 'xgboost_mean_r2' in results:
                model_scores.append(('xgboost', results['xgboost_mean_r2']))
            if 'lightgbm_mean_r2' in results:
                model_scores.append(('lightgbm', results['lightgbm_mean_r2']))

            best_model = max(model_scores, key=lambda x: x[1])[0] if model_scores else 'unknown'

            summary[key] = {
                'player_type': player_type,
                'metric_type': metric_type,
                'best_individual_model': best_model,
                'ensemble_performance': results.get('ensemble_mean_r2', 0),
                'improvement_over_best': results.get('ensemble_improvement', 0)
            }

        return summary


def create_ensemble_for_data(hitter_data, pitcher_data, holdout_year=2024):
    """
    Create and train ensemble models for all data types

    Args:
        hitter_data: Dictionary with 'warp' and 'war' data for hitters
        pitcher_data: Dictionary with 'warp' and 'war' data for pitchers
        holdout_year: Year to hold out for validation

    Returns:
        EnsembleTreePredictor: Trained ensemble predictor
    """
    ensemble = EnsembleTreePredictor()

    # Train models for each combination
    for player_type, data_dict in [('hitter', hitter_data), ('pitcher', pitcher_data)]:
        if not data_dict:
            continue

        for metric_type, data in data_dict.items():
            if not data:
                continue

            # Handle years data
            years_data = data.get('years', [])
            if isinstance(years_data, tuple) and len(years_data) == 1:
                years_data = years_data[0]

            # Create boolean mask for filtering
            if holdout_year is not None:
                mask = np.array(years_data) != holdout_year
            else:
                mask = np.ones(len(years_data) if years_data else len(data['y']), dtype=bool)

            if not mask.any():
                continue

            # Extract training data
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

    Args:
        hitter_data: Dictionary with 'warp' and 'war' data for hitters
        pitcher_data: Dictionary with 'warp' and 'war' data for pitchers
        use_cached: If True, try to load from cache
        cache_path: Path to cached model file
        holdout_year: Year to hold out for validation
        force_retrain: If True, retrain even if cache exists

    Returns:
        EnsembleTreePredictor: Trained ensemble model
    """
    # Use config defaults
    if cache_path is None:
        # Use different cache name for trees ensemble
        cache_path = Path(str(MODEL_CACHE_PATH).replace('.pkl', '_trees.pkl'))
    else:
        cache_path = Path(cache_path)

    holdout_year = holdout_year if holdout_year is not None else DEFAULT_HOLDOUT_YEAR

    # Try to load from cache
    if use_cached and not force_retrain and cache_path.exists():
        logger.info(f"Loading cached tree ensemble from {cache_path}")
        try:
            with open(cache_path, 'rb') as f:
                ensemble = pickle.load(f)
            logger.info("Cached tree model loaded successfully")
            return ensemble
        except Exception as e:
            logger.warning(f"Failed to load cache ({e}). Training new model...")

    # Train new model
    logger.info("Training new three-tree ensemble...")
    logger.info(f"Holdout year: {holdout_year}")
    ensemble = create_ensemble_for_data(hitter_data, pitcher_data, holdout_year)

    # Save to cache if requested
    if use_cached or force_retrain:
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, 'wb') as f:
                pickle.dump(ensemble, f)
            logger.info(f"Tree model saved to {cache_path}")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")

    return ensemble
