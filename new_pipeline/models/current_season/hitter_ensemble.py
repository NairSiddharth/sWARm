"""
Hitter ensemble with unified model (no role splits).

Single tier-based multi-quantile ensemble for all hitters.
"""

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler

from .base_ensemble import TieredQuantileEnsemble
from .keras_utils import build_multi_quantile_keras_adamw, get_keras_callbacks
from .multi_quantile_histgb import MultiQuantileHistGB

from new_pipeline.common.logging_config import get_logger

logger = get_logger(__name__)


# Monotonic constraints for 9 hitter features
# Order: K%, BB%, AVG, OBP, SLG, GDP, Positional_WAR, Enhanced_BR, Enhanced_Def
HITTER_MONOTONIC_CONSTRAINTS = [
    -1,  # K% – lower is better
    1,   # BB% – higher is better
    1,   # AVG – higher is better
    1,   # OBP – higher is better
    1,   # SLG – higher is better
    -1,  # GDP – lower is better
    1,   # Positional_WAR – higher is better
    1,   # Enhanced_Baserunning – higher is better
    1    # Enhanced_Defense – higher is better
]


class HitterEnsemble(TieredQuantileEnsemble):
    """
    Hitter ensemble with single unified model.

    No role splits (unlike pitchers) - all hitters use same ensemble.

    Components:
    - ExtraTreesRegressor (criterion='friedman_mse', max_features='sqrt')
    - Keras Multi-Quantile with AdamW
    - MultiQuantileHistGB with monotonic constraints

    Predictions use tier-based blending:
    - Average (<2.0 WAR): Conservative
    - Good (2.0-3.5 WAR): Moderate
    - Elite (e3.5 WAR): Aggressive
    """

    def __init__(self):
        """Initialize hitter ensemble."""
        super().__init__()

    def _build_extratrees(self) -> ExtraTreesRegressor:
        """
        Build ExtraTreesRegressor for hitters.

        Uses friedman_mse criterion and sqrt max_features.

        Returns:
            Configured ExtraTreesRegressor
        """
        return ExtraTreesRegressor(
            n_estimators=100,
            max_depth=10,
            criterion='friedman_mse',
            max_features='sqrt',
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        )

    def _build_keras(self, input_dim: int):
        """
        Build Keras multi-quantile model.

        Args:
            input_dim: Number of input features

        Returns:
            Compiled Keras Sequential model
        """
        return build_multi_quantile_keras_adamw(input_dim)

    def _build_histgb(self) -> MultiQuantileHistGB:
        """
        Build MultiQuantileHistGB with improved parameters and monotonic constraints.

        Improvements from legacy implementation:
        - max_leaf_nodes=63, learning_rate=0.03, l2_regularization=0.3
        - max_iter=300, n_iter_no_change=15
        - Hitter-specific monotonic constraints

        Returns:
            MultiQuantileHistGB instance with hitter monotonic constraints
        """
        return MultiQuantileHistGB(
            random_state=42,
            monotonic_cst=HITTER_MONOTONIC_CONSTRAINTS
        )

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Train unified hitter ensemble.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target values (n_samples,)
        """
        logger.info(f"Training hitter ensemble ({len(X)} samples)...")

        input_dim = X.shape[1]

        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Build and train models
        logger.info(f"  Training ExtraTrees...")
        self.extratrees_model = self._build_extratrees()
        self.extratrees_model.fit(X_scaled, y)

        logger.info(f"  Training Keras (AdamW + Swish + BatchNorm)...")
        self.keras_model = self._build_keras(input_dim)
        self.keras_model.fit(
            X_scaled, y,
            epochs=200,
            batch_size=32,
            validation_split=0.2,
            callbacks=get_keras_callbacks(),
            verbose=0
        )

        logger.info(f"  Training MultiQuantileHistGB...")
        self.histgb_model = self._build_histgb()
        self.histgb_model.fit(X_scaled, y)

        self._is_fitted = True
        logger.info(f"  Hitter ensemble training complete")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict with tier-based blending.

        Args:
            X: Feature matrix (n_samples, n_features)

        Returns:
            Predictions (n_samples,)

        Raises:
            RuntimeError: If model not fitted yet
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted yet. Call fit() before predict().")

        # Scale features
        X_scaled = self.scaler.transform(X)

        # Get predictions from all models
        et_pred = self.extratrees_model.predict(X_scaled)
        keras_quantiles = self.keras_model.predict(X_scaled, verbose=0)
        histgb_quantiles = self.histgb_model.get_quantile_predictions(X_scaled)

        # Apply tier-based blending
        predictions = self._blend_predictions(et_pred, keras_quantiles, histgb_quantiles)

        return predictions
