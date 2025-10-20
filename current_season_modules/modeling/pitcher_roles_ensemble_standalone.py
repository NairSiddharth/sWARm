"""
Three-Path Pitcher Ensemble - Phase 1 Standalone Implementation.

This module implements rate-based WAR prediction for pitchers using separate
RF + Keras ensembles for starters and relievers, with blending for mixed roles.

IMPORTANT: This is a STANDALONE TESTING FILE. Do not integrate into production
until validation confirms >30% improvement over current method.

Model Choice: RF + Keras (Phase 1)
Training: Rate-based (WAR per 162 IP) to avoid train/test mismatch
Routing: Three-path based on GS/G ratio

Testing notebook: sWARm_CS_pitching.ipynb
Validation script: current_season_modules/modeling/validate_phase1.py
Documentation: docs/pitcher_war_prediction_fix.md
"""

# Standard library imports
import warnings
from typing import Dict, List, Optional

# Third-party imports
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    RandomForestRegressor,
    HistGradientBoostingRegressor,
    StackingRegressor
)
from sklearn.preprocessing import StandardScaler

# XGBoost for quantile regression
try:
    from xgboost import XGBRegressor
except ImportError:
    warnings.warn(
        "XGBoost not available. Install with: pip install xgboost",
        ImportWarning,
        stacklevel=2
    )

# SciKeras wrapper for Keras sklearn compatibility
try:
    from scikeras.wrappers import KerasRegressor
except ImportError:
    warnings.warn(
        "SciKeras not available. Install with: pip install scikeras",
        ImportWarning,
        stacklevel=2
    )

# TensorFlow/Keras imports
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
except ImportError:
    warnings.warn(
        "TensorFlow not available. Install with: pip install tensorflow",
        ImportWarning,
        stacklevel=2
    )

# Local imports - Multi-quantile Keras module
try:
    from ...common_modules.multi_quantile_keras import (
        build_multi_quantile_keras,
        MultiQuantileExtractor
    )
except ImportError:
    # Fallback for different import contexts
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    try:
        from common_modules.multi_quantile_keras import (
            build_multi_quantile_keras,
            MultiQuantileExtractor
        )
    except ImportError as e:
        warnings.warn(
            f"Multi-quantile Keras module not available: {e}",
            ImportWarning,
            stacklevel=2
        )

# Constants
QUALIFIED_IP = 162.0  # MLB standard: 1 IP per team game
RANDOM_STATE = 42  # For reproducibility


# =============================================================================
# Reproducibility - Set Random Seeds
# =============================================================================

def set_random_seeds(seed: int = RANDOM_STATE):
    """
    Set random seeds for reproducibility across all libraries.

    This ensures that neural network initialization and training produce
    identical results across multiple runs.

    Args:
        seed: Random seed value (default: RANDOM_STATE)

    Note:
        Must be called before any model creation or training.
        Sets seeds for: Python random, NumPy, TensorFlow/Keras
    """
    import random
    import os

    # Python random
    random.seed(seed)

    # NumPy random
    np.random.seed(seed)

    # TensorFlow random (if available)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)

        # Force deterministic operations
        os.environ['TF_DETERMINISTIC_OPS'] = '1'
        os.environ['PYTHONHASHSEED'] = str(seed)
    except ImportError:
        pass


# Call at module import to ensure reproducibility
set_random_seeds(RANDOM_STATE)

# Pitcher features (rate-based - NO IP)
# Matches actual feature set: 10 features after removing IP (includes WPA/LI)
PITCHER_RATE_FEATURES = [
    # PHASE 1 UPDATE: Removed K-BB%, HBP%, Dominance_Index (redundant/low predictive value)
    # PHASE 1 UPDATE: Added GB%, SwStr%, and 3 SIERA-based interaction features
    'BB%', 'K%', 'ERA',  # Core rate stats (park factors applied correctly)
    'GB%',  # Ground ball percentage (batted ball outcome)
    'SwStr%',  # Swinging strike percentage (skill-based dominance metric)
    'damage_control_ratio',  # LOB% / (HR% + 0.5)
    'Opportunity_Success',  # (K% - BB%) * (LOB% / 100)
    # PHASE 2.5: Removed Contact_Quality_Index (negative importance for starters, redundant with strikeout_contact_quality)
    'Statcast_Launch_Quality_Index',  # Exit velo + launch angle composite
    'WPA/LI',  # Win impact metric
    # SIERA-based interaction features (capture K% value with contact quality/control)
    'strikeout_efficiency',  # K% × (100 - BB%)
    'contact_management',  # GB% × (100 - BB%)
    'strikeout_contact_quality',  # K% × (100 - Hard%)
]

# Role routing thresholds
PURE_RELIEVER_THRESHOLD = 0.1  # GS/G < 0.1
PURE_STARTER_THRESHOLD = 0.7   # GS/G > 0.7


# =============================================================================
# Data Preparation Functions
# =============================================================================

def normalize_pitcher_targets(
    war_values: np.ndarray,
    ip_values: np.ndarray,
    baseline_ip: float = QUALIFIED_IP
) -> np.ndarray:
    """
    Normalize WAR to rate per baseline IP.

    Args:
        war_values: Array of WAR values
        ip_values: Array of innings pitched
        baseline_ip: IP baseline to normalize to (default: 162)

    Returns:
        Array of WAR per baseline IP

    Example:
        >>> war = np.array([6.0, 2.5])
        >>> ip = np.array([200, 60])
        >>> normalize_pitcher_targets(war, ip, 162)
        array([4.86, 6.75])
    """
    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        war_per_162 = np.where(
            ip_values == 0,
            0,
            (war_values / ip_values) * baseline_ip
        )

    return war_per_162


def denormalize_pitcher_war(
    war_per_162: float,
    actual_ip: float,
    baseline_ip: float = QUALIFIED_IP
) -> float:
    """
    Convert rate-based WAR back to absolute WAR for given IP.

    Args:
        war_per_162: WAR rate per baseline IP
        actual_ip: Actual innings pitched
        baseline_ip: IP baseline used for normalization (default: 162)

    Returns:
        Absolute WAR for the actual IP

    Example:
        >>> denormalize_pitcher_war(4.86, 120, 162)
        3.6
    """
    return war_per_162 * (actual_ip / baseline_ip)


def calculate_role_ratio(gs: np.ndarray, g: np.ndarray) -> np.ndarray:
    """
    Calculate GS/G ratio for role classification.

    Args:
        gs: Games started
        g: Games pitched

    Returns:
        Array of GS/G ratios (0 to 1)

    Raises:
        ValueError: If G contains zero values
    """
    if np.any(g == 0):
        raise ValueError("Games pitched (G) cannot be zero")

    return gs / g


def classify_role_by_ratio(role_ratio: float) -> str:
    """
    Classify pitcher role based on GS/G ratio.

    Args:
        role_ratio: GS/G ratio

    Returns:
        Role classification: 'reliever', 'starter', or 'mixed'

    Example:
        >>> classify_role_by_ratio(0.05)
        'reliever'
        >>> classify_role_by_ratio(0.9)
        'starter'
        >>> classify_role_by_ratio(0.4)
        'mixed'
    """
    if role_ratio < PURE_RELIEVER_THRESHOLD:
        return 'reliever'
    elif role_ratio > PURE_STARTER_THRESHOLD:
        return 'starter'
    else:
        return 'mixed'


# =============================================================================
# Three-Path Pitcher Ensemble Class
# =============================================================================

class ThreePathPitcherEnsemble:
    """
    Three-path ensemble predictor for pitcher WAR/WARP.

    Uses separate RF + Keras ensembles for starters and relievers, with
    blending for mixed-role pitchers. All models trained on WAR per 162 IP
    to avoid train/test distribution mismatch.

    Architecture:
        - Pure Reliever (GS/G < 0.1): Reliever-specific RF + Keras
        - Pure Starter (GS/G > 0.7): Starter-specific RF + Keras
        - Mixed (0.1 <= GS/G <= 0.7): Linearly blended predictions

    Attributes:
        random_state: Random seed for reproducibility
        starter_models: Dict of starter models by metric type
        reliever_models: Dict of reliever models by metric type
        scalers: Dict of feature scalers by role and metric
        ensemble_weights: Weights for RF and Keras models

    Example:
        >>> ensemble = ThreePathPitcherEnsemble()
        >>> ensemble.train_starter_ensemble(X_train, y_train, 'war')
        >>> ensemble.train_reliever_ensemble(X_train, y_train, 'war')
        >>> prediction = ensemble.predict(features, GS=19, G=19, IP=120)
    """

    def __init__(self, random_state: int = RANDOM_STATE, use_huber_loss: bool = False, huber_delta: float = 1.0):
        """
        Initialize three-path pitcher ensemble.

        Args:
            random_state: Random seed for reproducibility
            use_huber_loss: If True, use Huber loss instead of MSE for Keras models
            huber_delta: Delta parameter for Huber loss (default: 1.0)
        """
        self.random_state = random_state
        self.use_huber_loss = use_huber_loss
        self.huber_delta = huber_delta

        # Model storage by role and metric
        self.starter_models: Dict[str, Dict[str, any]] = {}
        self.reliever_models: Dict[str, Dict[str, any]] = {}
        self.stacking_models: Dict[str, any] = {}  # Phase 2: Stacking ensembles
        self.scalers: Dict[str, StandardScaler] = {}
        self.use_stacking: bool = False  # Flag to enable stacking predictions

        # Ensemble weights (3-model: RF + Keras + HistGB)
        # HistGB added to target elite pitcher underprediction via residual learning
        self.ensemble_weights = {
            'war': {
                'randomforest': 0.20,    # PHASE 2.5: Increased from 0.15 (more balanced ensemble)
                'keras': 0.45,           # PHASE 2.5: Reduced from 0.50 (reduce WPA/LI dominance)
                'histgradient': 0.35     # Unchanged - targets elite patterns via boosting
            },
            'warp': {
                'randomforest': 0.50,    # Keep high (WARP model works well)
                'keras': 0.15,           # Keep low
                'histgradient': 0.35     # NEW - same reasoning as WAR
            }
        }

    def train_starter_ensemble(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        metric_type: str,
        validation_split: float = 0.2,
        epochs: int = 150,
        batch_size: int = 64
    ) -> None:
        """
        Train RF + Keras + HistGB ensemble on starter data.

        Args:
            X_train: Feature matrix (rate stats, no IP)
            y_train: Target (WAR per 162 IP)
            metric_type: 'war' or 'warp'
            validation_split: Fraction for validation
            epochs: Keras training epochs
            batch_size: Keras batch size

        Raises:
            ValueError: If metric_type invalid or insufficient data
        """
        if metric_type not in ['war', 'warp']:
            raise ValueError(f"Invalid metric_type: {metric_type}")

        if len(X_train) < 50:
            raise ValueError(
                f"Insufficient starter data: {len(X_train)} samples (need >= 50)"
            )

        # Scale features
        scaler_key = f"starter_{metric_type}"
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)
        self.scalers[scaler_key] = scaler

        # Train RandomForest
        rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=self.random_state,
            n_jobs=-1
        )
        rf_model.fit(X_scaled, y_train)

        # Train Keras with early stopping
        keras_model = self._build_keras_model(X_scaled.shape[1], self.use_huber_loss, self.huber_delta)
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True
        )

        keras_model.fit(
            X_scaled, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stop],
            verbose=0
        )

        # Train HistGradientBoosting
        # Configured for elite pitcher pattern learning via sequential error correction
        histgb_model = HistGradientBoostingRegressor(
            max_iter=200,           # More trees → better elite pattern learning
            max_depth=10,           # Deep enough for K% × ERA × WPA/LI interactions
            learning_rate=0.05,     # Slow learning → better generalization
            l2_regularization=0.5,  # Less regularization than RF (target elites)
            min_samples_leaf=5,     # Allow small leaf nodes for rare elite cases
            max_bins=255,           # Fine-grained histogram binning
            random_state=self.random_state
        )
        histgb_model.fit(X_scaled, y_train)

        # Store all three models
        self.starter_models[metric_type] = {
            'randomforest': rf_model,
            'keras': keras_model,
            'histgradient': histgb_model
        }

    def train_reliever_ensemble(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        metric_type: str,
        validation_split: float = 0.2,
        epochs: int = 150,
        batch_size: int = 64
    ) -> None:
        """
        Train RF + Keras + HistGB ensemble on reliever data.

        Args:
            X_train: Feature matrix (rate stats, no IP)
            y_train: Target (WAR per 162 IP)
            metric_type: 'war' or 'warp'
            validation_split: Fraction for validation
            epochs: Keras training epochs
            batch_size: Keras batch size

        Raises:
            ValueError: If metric_type invalid or insufficient data
        """
        if metric_type not in ['war', 'warp']:
            raise ValueError(f"Invalid metric_type: {metric_type}")

        if len(X_train) < 50:
            raise ValueError(
                f"Insufficient reliever data: {len(X_train)} samples (need >= 50)"
            )

        # Scale features
        scaler_key = f"reliever_{metric_type}"
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)
        self.scalers[scaler_key] = scaler

        # Train RandomForest
        rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=self.random_state,
            n_jobs=-1
        )
        rf_model.fit(X_scaled, y_train)

        # Train Keras with early stopping
        keras_model = self._build_keras_model(X_scaled.shape[1], self.use_huber_loss, self.huber_delta)
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True
        )

        keras_model.fit(
            X_scaled, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stop],
            verbose=0
        )

        # Train HistGradientBoosting
        # Same configuration as starters - targets elite reliever patterns
        histgb_model = HistGradientBoostingRegressor(
            max_iter=200,
            max_depth=10,
            learning_rate=0.05,
            l2_regularization=0.5,
            min_samples_leaf=5,
            max_bins=255,
            random_state=self.random_state
        )
        histgb_model.fit(X_scaled, y_train)

        # Store all three models
        self.reliever_models[metric_type] = {
            'randomforest': rf_model,
            'keras': keras_model,
            'histgradient': histgb_model
        }

    def train_starter_stacking(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        metric_type: str
    ) -> None:
        """
        Train stacking ensemble for starters (Phase 2).

        Uses RF, Keras, and HistGB as base estimators with HistGB meta-learner
        to learn optimal combination weights.

        Args:
            X_train: Feature matrix (rate stats, no IP)
            y_train: Target (WAR per 162 IP)
            metric_type: 'war' or 'warp'

        Raises:
            ValueError: If metric_type invalid or insufficient data
        """
        if metric_type not in ['war', 'warp']:
            raise ValueError(f"Invalid metric_type: {metric_type}")

        if len(X_train) < 50:
            raise ValueError(
                f"Insufficient starter data: {len(X_train)} samples (need >= 50)"
            )

        # Scale features
        scaler_key = f"starter_{metric_type}_stack"
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)
        self.scalers[scaler_key] = scaler

        # Base estimators (same as individual training)
        rf_estimator = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=self.random_state,
            n_jobs=-1
        )

        histgb_estimator = HistGradientBoostingRegressor(
            max_iter=200,
            max_depth=10,
            learning_rate=0.05,
            l2_regularization=0.5,
            min_samples_leaf=5,
            max_bins=255,
            random_state=self.random_state
        )

        # For Keras, we need to use the functional approach
        # Build and compile Keras model
        keras_model = self._build_keras_model(
            X_scaled.shape[1],
            self.use_huber_loss,
            self.huber_delta
        )

        # Fit Keras separately first
        keras_model.fit(
            X_scaled, y_train,
            epochs=150,
            batch_size=64,
            validation_split=0.2,
            callbacks=[EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)],
            verbose=0
        )

        # Meta-learner: Simple HistGB (learns when to trust each model)
        meta_learner = HistGradientBoostingRegressor(
            max_iter=50,      # Few iterations (simple meta-learner)
            max_depth=3,      # Shallow to prevent overfitting
            learning_rate=0.1,
            random_state=self.random_state
        )

        # Create stacking ensemble
        # Note: Can't include Keras directly in StackingRegressor, so we'll use manual stacking
        # Get predictions from base models for meta-training
        rf_estimator.fit(X_scaled, y_train)
        histgb_estimator.fit(X_scaled, y_train)

        # Store base models (for stacking)
        self.starter_models[f"{metric_type}_stack_base"] = {
            'randomforest': rf_estimator,
            'keras': keras_model,
            'histgradient': histgb_estimator
        }

        # Also store as regular models (so predict() method works)
        self.starter_models[metric_type] = {
            'randomforest': rf_estimator,
            'keras': keras_model,
            'histgradient': histgb_estimator
        }

        # Train meta-learner on base model predictions (5-fold CV to avoid overfitting)
        from sklearn.model_selection import cross_val_predict

        rf_preds = cross_val_predict(rf_estimator, X_scaled, y_train, cv=5)
        histgb_preds = cross_val_predict(histgb_estimator, X_scaled, y_train, cv=5)

        # Get Keras predictions (manual CV since it's not sklearn)
        from sklearn.model_selection import KFold
        keras_preds = np.zeros(len(y_train))
        kf = KFold(n_splits=5, shuffle=True, random_state=self.random_state)

        for train_idx, val_idx in kf.split(X_scaled):
            X_fold_train, X_fold_val = X_scaled[train_idx], X_scaled[val_idx]
            y_fold_train = y_train[train_idx]

            fold_model = self._build_keras_model(X_scaled.shape[1], self.use_huber_loss, self.huber_delta)
            fold_model.fit(X_fold_train, y_fold_train, epochs=150, batch_size=64, verbose=0)
            keras_preds[val_idx] = fold_model.predict(X_fold_val, verbose=0).flatten()

        # Meta-features: predictions from base models
        meta_features = np.column_stack([rf_preds, keras_preds, histgb_preds])

        # Train meta-learner
        meta_learner.fit(meta_features, y_train)

        # Store stacking model
        self.stacking_models[f"starter_{metric_type}"] = meta_learner

    def train_reliever_stacking(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        metric_type: str
    ) -> None:
        """
        Train stacking ensemble for relievers (Phase 2).

        Same approach as starter stacking.

        Args:
            X_train: Feature matrix (rate stats, no IP)
            y_train: Target (WAR per 162 IP)
            metric_type: 'war' or 'warp'

        Raises:
            ValueError: If metric_type invalid or insufficient data
        """
        if metric_type not in ['war', 'warp']:
            raise ValueError(f"Invalid metric_type: {metric_type}")

        if len(X_train) < 50:
            raise ValueError(
                f"Insufficient reliever data: {len(X_train)} samples (need >= 50)"
            )

        # Scale features
        scaler_key = f"reliever_{metric_type}_stack"
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)
        self.scalers[scaler_key] = scaler

        # Base estimators
        rf_estimator = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=self.random_state,
            n_jobs=-1
        )

        histgb_estimator = HistGradientBoostingRegressor(
            max_iter=200,
            max_depth=10,
            learning_rate=0.05,
            l2_regularization=0.5,
            min_samples_leaf=5,
            max_bins=255,
            random_state=self.random_state
        )

        keras_model = self._build_keras_model(X_scaled.shape[1], self.use_huber_loss, self.huber_delta)
        keras_model.fit(
            X_scaled, y_train,
            epochs=150,
            batch_size=64,
            validation_split=0.2,
            callbacks=[EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)],
            verbose=0
        )

        meta_learner = HistGradientBoostingRegressor(
            max_iter=50,
            max_depth=3,
            learning_rate=0.1,
            random_state=self.random_state
        )

        # Train base models
        rf_estimator.fit(X_scaled, y_train)
        histgb_estimator.fit(X_scaled, y_train)

        # Store base models (for stacking)
        self.reliever_models[f"{metric_type}_stack_base"] = {
            'randomforest': rf_estimator,
            'keras': keras_model,
            'histgradient': histgb_estimator
        }

        # Also store as regular models (so predict() method works)
        self.reliever_models[metric_type] = {
            'randomforest': rf_estimator,
            'keras': keras_model,
            'histgradient': histgb_estimator
        }

        # Get CV predictions for meta-training
        from sklearn.model_selection import cross_val_predict, KFold

        rf_preds = cross_val_predict(rf_estimator, X_scaled, y_train, cv=5)
        histgb_preds = cross_val_predict(histgb_estimator, X_scaled, y_train, cv=5)

        keras_preds = np.zeros(len(y_train))
        kf = KFold(n_splits=5, shuffle=True, random_state=self.random_state)

        for train_idx, val_idx in kf.split(X_scaled):
            X_fold_train, X_fold_val = X_scaled[train_idx], X_scaled[val_idx]
            y_fold_train = y_train[train_idx]

            fold_model = self._build_keras_model(X_scaled.shape[1], self.use_huber_loss, self.huber_delta)
            fold_model.fit(X_fold_train, y_fold_train, epochs=150, batch_size=64, verbose=0)
            keras_preds[val_idx] = fold_model.predict(X_fold_val, verbose=0).flatten()

        # Meta-features
        meta_features = np.column_stack([rf_preds, keras_preds, histgb_preds])

        # Train meta-learner
        meta_learner.fit(meta_features, y_train)

        # Store stacking model
        self.stacking_models[f"reliever_{metric_type}"] = meta_learner

    def _build_keras_model(self, input_dim: int, use_huber_loss: bool = False, huber_delta: float = 1.0):
        """
        Build Keras neural network (same architecture as hitter ensemble).

        Args:
            input_dim: Number of input features
            use_huber_loss: If True, use Huber loss instead of MSE (default: False)
            huber_delta: Delta parameter for Huber loss (default: 1.0)

        Returns:
            Compiled Keras Sequential model

        Architecture:
            - Input layer: input_dim neurons
            - Hidden: 128 -> 64 -> 32 -> 16 neurons with dropout
            - Output: 1 neuron (regression)
            - Activation: ReLU for hidden layers
            - Optimizer: Adam
            - Loss: MSE (default) or Huber loss (if use_huber_loss=True)
        """
        model = Sequential([
            Dense(128, activation='relu', input_dim=input_dim),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1)
        ])

        # Choose loss function
        if use_huber_loss:
            try:
                from tensorflow.keras.losses import Huber
                loss_fn = Huber(delta=huber_delta)
                loss_name = f'huber(δ={huber_delta})'
            except ImportError:
                warnings.warn("Huber loss not available, falling back to MSE", UserWarning)
                loss_fn = 'mse'
                loss_name = 'mse'
        else:
            loss_fn = 'mse'
            loss_name = 'mse'

        model.compile(optimizer='adam', loss=loss_fn, metrics=['mae'])

        # Store loss function name for debugging
        if not hasattr(self, 'loss_function'):
            self.loss_function = loss_name

        return model

    def predict(
        self,
        features: np.ndarray,
        GS: int,
        G: int,
        IP: float,
        metric_type: str = 'war',
        projected_ip: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Predict current and projected WAR for a pitcher.

        Args:
            features: Feature vector (rate stats, no IP)
            GS: Games started
            G: Games pitched
            IP: Current innings pitched
            metric_type: 'war' or 'warp'
            projected_ip: Projected full season IP (optional)

        Returns:
            Dictionary containing:
                - role: Pitcher role classification
                - role_ratio: GS/G ratio
                - war_per_162: Skill rate prediction
                - current_war: Current WAR based on actual IP
                - projected_war: Full season projection (if projected_ip given)
                - current_ip: Actual IP
                - projected_ip: Projected IP (if given)

        Raises:
            ValueError: If models not trained for metric_type
            ValueError: If G is zero

        Example:
            >>> result = ensemble.predict(
            ...     features=np.array([0.28, 0.015, 0.265, 2.2, ...]),
            ...     GS=19, G=19, IP=120, metric_type='war',
            ...     projected_ip=195
            ... )
            >>> print(f"Current: {result['current_war']:.2f} WAR")
            >>> print(f"Projected: {result['projected_war']:.2f} WAR")
        """
        if metric_type not in self.starter_models:
            raise ValueError(
                f"Models not trained for {metric_type}. "
                f"Call train_starter_ensemble() and train_reliever_ensemble() first."
            )

        if G == 0:
            raise ValueError("Games pitched (G) cannot be zero")

        # Calculate role
        role_ratio = GS / G
        role = classify_role_by_ratio(role_ratio)

        # Predict skill rate (WAR per 162 IP)
        war_per_162 = self._predict_war_rate(features, role_ratio, metric_type)

        # Calculate current WAR
        current_war = denormalize_pitcher_war(war_per_162, IP)

        # Build result dictionary
        result = {
            'role': role,
            'role_ratio': role_ratio,
            'war_per_162': war_per_162,
            'current_war': current_war,
            'current_ip': IP
        }

        # Add projection if requested
        if projected_ip is not None:
            result['projected_war'] = denormalize_pitcher_war(
                war_per_162, projected_ip
            )
            result['projected_ip'] = projected_ip

        return result

    def _predict_war_rate(
        self,
        features: np.ndarray,
        role_ratio: float,
        metric_type: str
    ) -> float:
        """
        Predict WAR per 162 IP using role-based routing.

        Args:
            features: Feature vector
            role_ratio: GS/G ratio
            metric_type: 'war' or 'warp'

        Returns:
            WAR per 162 IP prediction

        Routing Logic:
            - GS/G < 0.1: Pure reliever model
            - GS/G > 0.7: Pure starter model
            - 0.1 <= GS/G <= 0.7: Linear blend between models
        """
        # Reshape features if needed
        if features.ndim == 1:
            features = features.reshape(1, -1)

        if role_ratio < PURE_RELIEVER_THRESHOLD:
            # Pure reliever path
            return self._get_ensemble_prediction(
                features, 'reliever', metric_type
            )

        elif role_ratio > PURE_STARTER_THRESHOLD:
            # Pure starter path
            return self._get_ensemble_prediction(
                features, 'starter', metric_type
            )

        else:
            # Mixed role - blend predictions
            blend_weight = (role_ratio - PURE_RELIEVER_THRESHOLD) / (
                PURE_STARTER_THRESHOLD - PURE_RELIEVER_THRESHOLD
            )

            reliever_pred = self._get_ensemble_prediction(
                features, 'reliever', metric_type
            )
            starter_pred = self._get_ensemble_prediction(
                features, 'starter', metric_type
            )

            return (1 - blend_weight) * reliever_pred + blend_weight * starter_pred

    def _get_ensemble_prediction(
        self,
        features: np.ndarray,
        role: str,
        metric_type: str
    ) -> float:
        """
        Get ensemble prediction - either weighted or stacking-based.

        If use_stacking=True, uses meta-learner to combine predictions.
        Otherwise uses fixed weights (Phase 1 approach).

        Args:
            features: Feature vector
            role: 'starter' or 'reliever'
            metric_type: 'war' or 'warp'

        Returns:
            Ensemble prediction
        """
        # Check if using stacking (Phase 2)
        stacking_key = f"{role}_{metric_type}"
        if self.use_stacking and stacking_key in self.stacking_models:
            # Phase 2: Stacking meta-learner
            models = self.starter_models if role == 'starter' else self.reliever_models
            scaler_key = f"{role}_{metric_type}_stack"

            # Scale features
            X_scaled = self.scalers[scaler_key].transform(features)

            # Get predictions from base models
            base_key = f"{metric_type}_stack_base"
            rf_pred = models[base_key]['randomforest'].predict(X_scaled)[0]
            keras_pred = models[base_key]['keras'].predict(X_scaled, verbose=0)[0][0]
            histgb_pred = models[base_key]['histgradient'].predict(X_scaled)[0]

            # Meta-features: predictions from base models
            meta_features = np.array([[rf_pred, keras_pred, histgb_pred]])

            # Meta-learner combines predictions
            final_pred = self.stacking_models[stacking_key].predict(meta_features)[0]

            return float(final_pred)

        else:
            # Phase 1: Fixed weighted ensemble
            models = self.starter_models if role == 'starter' else self.reliever_models
            scaler_key = f"{role}_{metric_type}"

            # Scale features
            X_scaled = self.scalers[scaler_key].transform(features)

            # Get predictions from all three models
            rf_pred = models[metric_type]['randomforest'].predict(X_scaled)[0]
            keras_pred = models[metric_type]['keras'].predict(X_scaled, verbose=0)[0][0]
            histgb_pred = models[metric_type]['histgradient'].predict(X_scaled)[0]

            # Weighted ensemble (3 models)
            weights = self.ensemble_weights[metric_type]
            ensemble_pred = (
                weights['randomforest'] * rf_pred +
                weights['keras'] * keras_pred +
                weights['histgradient'] * histgb_pred
            )

            return float(ensemble_pred)


# =============================================================================
# Training and Prediction Helper Functions
# =============================================================================

def prepare_pitcher_data_for_training(
    pitcher_df: pd.DataFrame,
    features: List[str],
    target_col: str = 'WAR',
    holdout_year: Optional[int] = None
) -> Dict[str, np.ndarray]:
    """
    Prepare pitcher data for rate-based ensemble training.

    Args:
        pitcher_df: DataFrame with pitcher statistics
        features: List of feature column names (should not include IP)
        target_col: Target column name ('WAR' or 'WARP')
        holdout_year: Optional year to hold out for validation

    Returns:
        Dictionary containing:
            - X: Feature matrix
            - y: Normalized target (WAR per 162 IP)
            - GS: Games started
            - G: Games pitched
            - IP: Innings pitched
            - years: Season years
            - role_ratio: GS/G ratios
            - train_mask: Boolean mask for training data (if holdout_year given)

    Raises:
        ValueError: If required columns missing
        ValueError: If IP contains zeros or negative values
    """
    # Validate required columns
    required_cols = features + [target_col, 'GS', 'G', 'IP', 'Season']
    missing_cols = set(required_cols) - set(pitcher_df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Filter out invalid IP values
    valid_ip_mask = pitcher_df['IP'] > 0
    if not valid_ip_mask.all():
        n_invalid = (~valid_ip_mask).sum()
        warnings.warn(
            f"Filtering {n_invalid} rows with IP <= 0",
            UserWarning,
            stacklevel=2
        )
        pitcher_df = pitcher_df[valid_ip_mask].copy()

    # Extract arrays
    X = pitcher_df[features].values
    y = pitcher_df[target_col].values
    GS = pitcher_df['GS'].values
    G = pitcher_df['G'].values
    IP = pitcher_df['IP'].values
    years = pitcher_df['Season'].values

    # Normalize targets
    y_normalized = normalize_pitcher_targets(y, IP)

    # Calculate role ratios
    role_ratios = calculate_role_ratio(GS, G)

    # Create result dictionary
    result = {
        'X': X,
        'y': y_normalized,
        'GS': GS,
        'G': G,
        'IP': IP,
        'years': years,
        'role_ratio': role_ratios
    }

    # Add train mask if holdout specified
    if holdout_year is not None:
        result['train_mask'] = years != holdout_year

    return result


def train_three_path_ensemble(
    pitcher_data_dict: Dict[str, pd.DataFrame],
    holdout_year: Optional[int] = None,
    use_huber_loss: bool = False,
    huber_delta: float = 1.0
) -> ThreePathPitcherEnsemble:
    """
    Train three-path pitcher ensemble on historical data.

    Args:
        pitcher_data_dict: Dictionary with 'war' and 'warp' DataFrames
        holdout_year: Optional year to hold out for validation
        use_huber_loss: If True, use Huber loss instead of MSE for Keras models
        huber_delta: Delta parameter for Huber loss (default: 1.0)

    Returns:
        Trained ThreePathPitcherEnsemble instance

    Raises:
        ValueError: If required data missing or insufficient samples

    Example:
        >>> pitcher_data = {'war': war_df, 'warp': warp_df}
        >>> ensemble = train_three_path_ensemble(pitcher_data, holdout_year=2024)
        >>> ensemble_huber = train_three_path_ensemble(pitcher_data, use_huber_loss=True, huber_delta=1.5)
    """
    ensemble = ThreePathPitcherEnsemble(use_huber_loss=use_huber_loss, huber_delta=huber_delta)

    for metric_type in ['war', 'warp']:
        if metric_type not in pitcher_data_dict:
            warnings.warn(
                f"Skipping {metric_type} - not in pitcher_data_dict",
                UserWarning,
                stacklevel=2
            )
            continue

        # Prepare data
        features_to_use = custom_features if custom_features is not None else PITCHER_RATE_FEATURES
        data = prepare_pitcher_data_for_training(
            pitcher_data_dict[metric_type],
            features_to_use,
            target_col=metric_type.upper(),
            holdout_year=holdout_year
        )

        # Split by role
        role_ratios = data['role_ratio']
        starter_mask = role_ratios > PURE_STARTER_THRESHOLD
        reliever_mask = role_ratios < PURE_RELIEVER_THRESHOLD

        # Apply holdout if specified
        if holdout_year is not None:
            train_mask = data['train_mask']
            starter_train_mask = starter_mask & train_mask
            reliever_train_mask = reliever_mask & train_mask
        else:
            starter_train_mask = starter_mask
            reliever_train_mask = reliever_mask

        # Extract starter training data
        X_starters = data['X'][starter_train_mask]
        y_starters = data['y'][starter_train_mask]

        if len(X_starters) < 50:
            raise ValueError(
                f"Insufficient starter data for {metric_type}: "
                f"{len(X_starters)} samples (need >= 50)"
            )

        # Extract reliever training data
        X_relievers = data['X'][reliever_train_mask]
        y_relievers = data['y'][reliever_train_mask]

        if len(X_relievers) < 50:
            raise ValueError(
                f"Insufficient reliever data for {metric_type}: "
                f"{len(X_relievers)} samples (need >= 50)"
            )

        # Train ensembles
        print(f"Training {metric_type.upper()} starter ensemble "
              f"({len(X_starters)} samples)...")
        ensemble.train_starter_ensemble(X_starters, y_starters, metric_type)

        print(f"Training {metric_type.upper()} reliever ensemble "
              f"({len(X_relievers)} samples)...")
        ensemble.train_reliever_ensemble(X_relievers, y_relievers, metric_type)

    return ensemble


def train_three_path_stacking(
    pitcher_data_dict: Dict[str, pd.DataFrame],
    holdout_year: Optional[int] = None,
    use_huber_loss: bool = False,
    huber_delta: float = 1.0,
    custom_features: Optional[List[str]] = None
) -> ThreePathPitcherEnsemble:
    """
    Train three-path pitcher ensemble with stacking meta-learner (Phase 2).

    Args:
        pitcher_data_dict: Dictionary with 'war' and 'warp' DataFrames
        custom_features: Optional list of feature names to use instead of PITCHER_RATE_FEATURES
        holdout_year: Optional year to hold out for validation
        use_huber_loss: If True, use Huber loss instead of MSE for Keras models
        huber_delta: Delta parameter for Huber loss (default: 1.0)

    Returns:
        Trained ThreePathPitcherEnsemble with stacking enabled

    Example:
        >>> pitcher_data = {'war': war_df, 'warp': warp_df}
        >>> ensemble = train_three_path_stacking(pitcher_data, holdout_year=None)
        >>> ensemble.use_stacking  # True
    """
    ensemble = ThreePathPitcherEnsemble(use_huber_loss=use_huber_loss, huber_delta=huber_delta)
    ensemble.use_stacking = True  # Enable stacking predictions

    for metric_type in ['war', 'warp']:
        if metric_type not in pitcher_data_dict:
            warnings.warn(
                f"Skipping {metric_type} - not in pitcher_data_dict",
                UserWarning,
                stacklevel=2
            )
            continue

        # Prepare data
        features_to_use = custom_features if custom_features is not None else PITCHER_RATE_FEATURES
        data = prepare_pitcher_data_for_training(
            pitcher_data_dict[metric_type],
            features_to_use,
            target_col=metric_type.upper(),
            holdout_year=holdout_year
        )

        # Split by role
        role_ratios = data['role_ratio']
        starter_mask = role_ratios > PURE_STARTER_THRESHOLD
        reliever_mask = role_ratios < PURE_RELIEVER_THRESHOLD

        # Apply holdout if specified
        if holdout_year is not None:
            train_mask = data['train_mask']
            starter_train_mask = starter_mask & train_mask
            reliever_train_mask = reliever_mask & train_mask
        else:
            starter_train_mask = starter_mask
            reliever_train_mask = reliever_mask

        # Extract starter training data
        X_starters = data['X'][starter_train_mask]
        y_starters = data['y'][starter_train_mask]

        if len(X_starters) < 50:
            raise ValueError(
                f"Insufficient starter data for {metric_type}: "
                f"{len(X_starters)} samples (need >= 50)"
            )

        # Extract reliever training data
        X_relievers = data['X'][reliever_train_mask]
        y_relievers = data['y'][reliever_train_mask]

        if len(X_relievers) < 50:
            raise ValueError(
                f"Insufficient reliever data for {metric_type}: "
                f"{len(X_relievers)} samples (need >= 50)"
            )

        # Train stacking ensembles
        print(f"Training {metric_type.upper()} starter stacking ensemble "
              f"({len(X_starters)} samples)...")
        ensemble.train_starter_stacking(X_starters, y_starters, metric_type)

        print(f"Training {metric_type.upper()} reliever stacking ensemble "
              f"({len(X_relievers)} samples)...")
        ensemble.train_reliever_stacking(X_relievers, y_relievers, metric_type)

    return ensemble


# =============================================================================
# Quantile Stacking Ensemble (Phase 2 - Advanced)
# =============================================================================

class QuantileStackingEnsemble:
    """
    Quantile-based stacking ensemble for pitcher WAR prediction.

    Uses sklearn StackingRegressor with quantile loss to address elite
    pitcher underprediction. Provides same API as ThreePathPitcherEnsemble
    for easy A/B testing.

    Architecture:
        Base Learners:
            - RandomForest (MSE) - conservative baseline
            - Multi-Quantile Keras [q50, q75, q90] - flexible nonlinear
            - XGBoost (quantile q=0.9) - aggressive elite targeting

        Meta-Estimator:
            - XGBoost (quantile q=0.75) - learns optimal blend

    Attributes:
        starter_models: Dict of StackingRegressor by metric ('war', 'warp')
        reliever_models: Dict of StackingRegressor by metric
        scalers: Dict of StandardScaler by role and metric
        random_state: Seed for reproducibility

    Example:
        >>> ensemble = QuantileStackingEnsemble()
        >>> ensemble.train_starter_ensemble(X_train, y_train, 'war', input_dim=10)
        >>> result = ensemble.predict(features, GS=20, G=20, IP=130, metric_type='war')
    """

    def __init__(self, random_state: int = RANDOM_STATE):
        """Initialize quantile stacking ensemble."""
        self.random_state = random_state
        self.starter_models = {}
        self.reliever_models = {}
        self.scalers = {}

    def train_starter_ensemble(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        metric_type: str,
        input_dim: Optional[int] = None
    ) -> None:
        """
        Train quantile stacking ensemble for starters.

        Args:
            X_train: Feature matrix (rate stats, no IP)
            y_train: Target (WAR per 162 IP)
            metric_type: 'war' or 'warp'
            input_dim: Number of input features (auto-detected if None)

        Raises:
            ValueError: If metric_type invalid or insufficient data
        """
        if metric_type not in ['war', 'warp']:
            raise ValueError(f"Invalid metric_type: {metric_type}")

        if len(X_train) < 50:
            raise ValueError(
                f"Insufficient starter data: {len(X_train)} samples (need >= 50)"
            )

        if input_dim is None:
            input_dim = X_train.shape[1]

        # Scale features
        scaler_key = f"starter_{metric_type}"
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)
        self.scalers[scaler_key] = scaler

        # Build base estimators
        base_estimators = self._build_base_estimators(input_dim)

        # Build meta-estimator
        meta_estimator = self._build_meta_estimator()

        # Create stacking ensemble
        # NOTE: n_jobs=1 to avoid pickling issues with custom Keras loss functions
        stacking_model = StackingRegressor(
            estimators=base_estimators,
            final_estimator=meta_estimator,
            cv=5,
            n_jobs=1,  # Disable parallel to avoid serialization issues
            passthrough=False  # Only use base predictions
        )

        # Train
        stacking_model.fit(X_scaled, y_train)

        # Store
        self.starter_models[metric_type] = stacking_model

    def train_reliever_ensemble(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        metric_type: str,
        input_dim: Optional[int] = None
    ) -> None:
        """Train quantile stacking ensemble for relievers (same as starters)."""
        if metric_type not in ['war', 'warp']:
            raise ValueError(f"Invalid metric_type: {metric_type}")

        if len(X_train) < 50:
            raise ValueError(
                f"Insufficient reliever data: {len(X_train)} samples (need >= 50)"
            )

        if input_dim is None:
            input_dim = X_train.shape[1]

        scaler_key = f"reliever_{metric_type}"
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)
        self.scalers[scaler_key] = scaler

        base_estimators = self._build_base_estimators(input_dim)
        meta_estimator = self._build_meta_estimator()

        # NOTE: n_jobs=1 to avoid pickling issues with custom Keras loss functions
        stacking_model = StackingRegressor(
            estimators=base_estimators,
            final_estimator=meta_estimator,
            cv=5,
            n_jobs=1,  # Disable parallel to avoid serialization issues
            passthrough=False
        )

        stacking_model.fit(X_scaled, y_train)
        self.reliever_models[metric_type] = stacking_model

    def _build_base_estimators(self, input_dim: int):
        """Build base estimator list for StackingRegressor."""
        # 1. RandomForest (MSE - conservative baseline)
        rf_estimator = (
            'rf',
            RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=self.random_state,
                n_jobs=-1
            )
        )

        # 2. Multi-quantile Keras
        def build_keras():
            return build_multi_quantile_keras(
                input_dim=input_dim,
                quantiles=[0.5, 0.75, 0.9],
                quantile_weights=[0.2, 0.3, 0.5],  # Emphasize upper quantiles
                hidden_layers=(128, 64, 32, 16),
                dropout_rates=(0.3, 0.3, 0.2, 0.0)
            )

        keras_base = KerasRegressor(
            model=build_keras,
            epochs=150,
            batch_size=64,
            validation_split=0.2,
            callbacks=[EarlyStopping(
                monitor='val_loss',
                patience=20,
                restore_best_weights=True
            )],
            verbose=0,
            random_state=self.random_state
        )

        # Wrap Keras to expose 3 separate quantile outputs
        keras_estimators = [
            ('keras_q50', MultiQuantileExtractor(keras_base, quantile_idx=0)),
            ('keras_q75', MultiQuantileExtractor(keras_base, quantile_idx=1)),
            ('keras_q90', MultiQuantileExtractor(keras_base, quantile_idx=2)),
        ]

        # 3. XGBoost (quantile q=0.9 - aggressive for elites)
        xgb_estimator = (
            'xgb_q90',
            XGBRegressor(
                objective='reg:quantileerror',
                quantile_alpha=0.9,
                n_estimators=200,
                max_depth=10,
                learning_rate=0.05,
                tree_method='hist',
                random_state=self.random_state
            )
        )

        # Combine: [RF, Keras_q50, Keras_q75, Keras_q90, XGB_q90]
        return [rf_estimator] + keras_estimators + [xgb_estimator]

    def _build_meta_estimator(self):
        """
        Build meta-estimator (XGBoost with quantile q=0.75).

        q=0.75 provides balance:
        - Elite accuracy (3:1 underprediction penalty)
        - Average pitcher calibration (not too aggressive)
        """
        return XGBRegressor(
            objective='reg:quantileerror',
            quantile_alpha=0.75,  # Upper-middle quantile
            n_estimators=50,
            max_depth=3,
            learning_rate=0.1,
            tree_method='hist',
            random_state=self.random_state
        )

    def predict(
        self,
        features: np.ndarray,
        GS: int,
        G: int,
        IP: float,
        metric_type: str = 'war',
        projected_ip: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Predict WAR using quantile stacking (same API as ThreePathPitcherEnsemble).

        Args:
            features: Feature vector (rate stats, no IP)
            GS: Games started
            G: Games pitched
            IP: Current innings pitched
            metric_type: 'war' or 'warp'
            projected_ip: Projected full season IP (optional)

        Returns:
            Dictionary with role, predictions, IP (same format as baseline)
        """
        if G == 0:
            raise ValueError("Games pitched (G) cannot be zero")

        # Calculate role
        role_ratio = GS / G
        role = classify_role_by_ratio(role_ratio)

        # Determine which model to use based on role
        if role_ratio < PURE_RELIEVER_THRESHOLD:
            model_role = 'reliever'
        elif role_ratio > PURE_STARTER_THRESHOLD:
            model_role = 'starter'
        else:
            # Mixed role: use starter model (more data typically)
            model_role = 'starter'

        # Get prediction
        war_per_162 = self._predict_war_rate(features, model_role, metric_type)

        # Calculate current WAR
        current_war = denormalize_pitcher_war(war_per_162, IP)

        # Build result dictionary
        result = {
            'role': role,
            'role_ratio': role_ratio,
            'war_per_162': war_per_162,
            'current_war': current_war,
            'current_ip': IP
        }

        # Add projection if requested
        if projected_ip is not None:
            result['projected_war'] = denormalize_pitcher_war(
                war_per_162, projected_ip
            )
            result['projected_ip'] = projected_ip

        return result

    def _predict_war_rate(
        self,
        features: np.ndarray,
        role: str,
        metric_type: str
    ) -> float:
        """Predict WAR per 162 IP using stacking ensemble."""
        # Reshape features if needed
        if features.ndim == 1:
            features = features.reshape(1, -1)

        # Scale features
        scaler_key = f"{role}_{metric_type}"
        X_scaled = self.scalers[scaler_key].transform(features)

        # Get model
        models = self.starter_models if role == 'starter' else self.reliever_models
        stacking_model = models[metric_type]

        # [DIAGNOSTIC] Get base learner predictions
        print(f"\n[QuantileStacking] Predicting for {role} {metric_type}")
        print(f"[QuantileStacking] Input features (scaled): {X_scaled[0][:5]}... (showing first 5)")

        # Get predictions from each base learner
        # Note: stacking_model.estimators is list of (name, estimator) tuples (unfitted)
        #       stacking_model.estimators_ is list of fitted estimators (no names)
        base_predictions = {}
        for i, (name, _) in enumerate(stacking_model.estimators):
            try:
                fitted_estimator = stacking_model.estimators_[i]
                pred = fitted_estimator.predict(X_scaled)[0]
                base_predictions[name] = pred
                print(f"[QuantileStacking]   Base learner '{name}': {pred:.4f}")
            except Exception as e:
                print(f"[QuantileStacking]   Base learner '{name}': FAILED - {e}")

        # Get initial prediction from meta-estimator to determine tier
        meta_prediction = stacking_model.predict(X_scaled)[0]
        print(f"[QuantileStacking] Meta-estimator prediction: {meta_prediction:.4f}")

        # TIERED QUANTILE SELECTION based on predicted WAR magnitude
        # - High WAR (>3.5): Use aggressive q90 to avoid under-predicting elites
        # - Mid WAR (2.0-3.5): Use balanced q75
        # - Low WAR (<2.0): Use conservative q50 to avoid over-predicting weak/injured
        if meta_prediction > 3.5:
            selected_quantile = 'keras_q90'
            final_prediction = base_predictions.get('keras_q90', meta_prediction)
            tier = 'ELITE (>3.5)'
        elif meta_prediction >= 2.0:
            selected_quantile = 'keras_q75'
            final_prediction = base_predictions.get('keras_q75', meta_prediction)
            tier = 'AVERAGE (2.0-3.5)'
        else:
            selected_quantile = 'keras_q50'
            final_prediction = base_predictions.get('keras_q50', meta_prediction)
            tier = 'LOW (<2.0)'

        print(f"[QuantileStacking] Tier: {tier} -> Selected: {selected_quantile}")
        print(f"[QuantileStacking] Final prediction: {final_prediction:.4f}\n")

        return float(final_prediction)


def train_quantile_stacking_ensemble(
    pitcher_data_dict: Dict[str, pd.DataFrame],
    holdout_year: Optional[int] = None,
    custom_features: Optional[List[str]] = None
) -> QuantileStackingEnsemble:
    """
    Train quantile-based stacking ensemble for pitcher WAR prediction.

    Uses sklearn StackingRegressor with:
    - Base learners: RF (MSE), Multi-Q Keras [0.5, 0.75, 0.9], XGBoost (q=0.9)
    - Meta-estimator: XGBoost (quantile q=0.75)

    This approach addresses elite pitcher underprediction by using quantile loss
    to penalize low predictions more heavily than overpredictions.

    Args:
        pitcher_data_dict: Dictionary with 'war' and 'warp' DataFrames
        holdout_year: Optional year to hold out for validation
        custom_features: Optional list of feature names (default: PITCHER_RATE_FEATURES)

    Returns:
        Trained QuantileStackingEnsemble instance with .predict() method

    Raises:
        ValueError: If required data missing or insufficient samples

    Example:
        >>> pitcher_data = {'war': war_df, 'warp': warp_df}
        >>> ensemble = train_quantile_stacking_ensemble(pitcher_data)
        >>> result = ensemble.predict(
        ...     features=pitcher_features,
        ...     GS=20, G=20, IP=130, metric_type='war'
        ... )

    Notes:
        - Requires xgboost>=2.0, scikeras>=0.13
        - Training time: ~3-5 min for 3000 samples
        - See test_quantile_stacking_elite.py for validation
    """
    ensemble = QuantileStackingEnsemble(random_state=RANDOM_STATE)

    for metric_type in ['war', 'warp']:
        if metric_type not in pitcher_data_dict:
            warnings.warn(
                f"Skipping {metric_type} - not in pitcher_data_dict",
                UserWarning,
                stacklevel=2
            )
            continue

        # Prepare data
        features_to_use = custom_features if custom_features is not None else PITCHER_RATE_FEATURES
        data = prepare_pitcher_data_for_training(
            pitcher_data_dict[metric_type],
            features_to_use,
            target_col=metric_type.upper(),
            holdout_year=holdout_year
        )

        # Split by role
        role_ratios = data['role_ratio']
        starter_mask = role_ratios > PURE_STARTER_THRESHOLD
        reliever_mask = role_ratios < PURE_RELIEVER_THRESHOLD

        # Apply holdout if specified
        if holdout_year is not None:
            train_mask = data['train_mask']
            starter_train_mask = starter_mask & train_mask
            reliever_train_mask = reliever_mask & train_mask
        else:
            starter_train_mask = starter_mask
            reliever_train_mask = reliever_mask

        # Extract starter training data
        X_starters = data['X'][starter_train_mask]
        y_starters = data['y'][starter_train_mask]

        if len(X_starters) < 50:
            raise ValueError(
                f"Insufficient starter data for {metric_type}: "
                f"{len(X_starters)} samples (need >= 50)"
            )

        # Extract reliever training data
        X_relievers = data['X'][reliever_train_mask]
        y_relievers = data['y'][reliever_train_mask]

        if len(X_relievers) < 50:
            raise ValueError(
                f"Insufficient reliever data for {metric_type}: "
                f"{len(X_relievers)} samples (need >= 50)"
            )

        # Train quantile stacking ensembles
        print(f"Training {metric_type.upper()} starter quantile stacking ensemble "
              f"({len(X_starters)} samples)...")
        ensemble.train_starter_ensemble(
            X_starters, y_starters, metric_type,
            input_dim=X_starters.shape[1]
        )

        print(f"Training {metric_type.upper()} reliever quantile stacking ensemble "
              f"({len(X_relievers)} samples)...")
        ensemble.train_reliever_ensemble(
            X_relievers, y_relievers, metric_type,
            input_dim=X_relievers.shape[1]
        )

    return ensemble


if __name__ == "__main__":
    print("Three-Path Pitcher Ensemble - Phase 1")
    print("=" * 60)
    print("This is a standalone testing file.")
    print("Run validation via: validate_phase1.py")
    print("Testing notebook: sWARm_CS_pitching.ipynb")
    print("\nModel: RF + Keras (rate-based)")
    print(f"Features: {len(PITCHER_RATE_FEATURES)} rate stats (no IP)")
    print(f"Routing: GS/G ratio ({PURE_RELIEVER_THRESHOLD}/{PURE_STARTER_THRESHOLD})")
    print("=" * 60)
