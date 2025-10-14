"""
Multi-Quantile HistGradientBoosting for ROS Prediction

Primary model (50% weight) - combines feature-based regression with
uncertainty quantification via quantile prediction.

IMPROVEMENTS from reference implementation:
- max_iter: 200 -> 300 (more iterations for convergence)
- learning_rate: 0.05 -> 0.03 (slower for better generalization)
- max_depth: 8 -> None (no depth limit, use max_leaf_nodes instead)
- max_leaf_nodes: None -> 63 (direct complexity control)
- min_samples_leaf: 20 -> 10 (ROS has less data than full season)
- l2_regularization: 0.0 -> 0.3 (less regularization to allow elite predictions)
- Added early stopping with validation_fraction=0.2
- Added n_iter_no_change=15 for more patience
"""

import numpy as np
from typing import List, Dict, Optional
from sklearn.ensemble import HistGradientBoostingRegressor
from .base import BaseROSModel


class MultiQuantileHistGB(BaseROSModel):
    """
    Multi-quantile prediction using HistGradientBoostingRegressor.

    Fits separate models for each quantile using pinball loss.
    Provides uncertainty bands and natural elite protection (higher floors).

    Parameters aligned with current season validated implementation,
    adapted for smaller ROS dataset size.
    """

    def __init__(
        self,
        player_type: str = 'hitter',
        quantiles: List[float] = [0.1, 0.25, 0.5, 0.75, 0.9],
        max_iter: int = 300,
        learning_rate: float = 0.03,
        max_depth: Optional[int] = None,
        max_leaf_nodes: int = 63,
        min_samples_leaf: int = 10,
        l2_regularization: float = 0.3,
        random_state: int = 42
    ):
        """
        Initialize multi-quantile model.

        Args:
            player_type: 'hitter' or 'pitcher'
            quantiles: Target quantiles to predict
            max_iter: Boosting iterations (300 for convergence)
            learning_rate: Boosting learning rate (0.03 for generalization)
            max_depth: Tree depth (None = no limit)
            max_leaf_nodes: Direct complexity control (~2^6-1)
            min_samples_leaf: Minimum samples per leaf (10 for ROS dataset size)
            l2_regularization: L2 penalty (0.3 to allow elite predictions)
            random_state: Random seed for reproducibility
        """
        super().__init__(player_type)
        self.quantiles = quantiles
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.max_leaf_nodes = max_leaf_nodes
        self.min_samples_leaf = min_samples_leaf
        self.l2_regularization = l2_regularization
        self.random_state = random_state

        # Store model for each quantile
        self.quantile_models: Dict[float, HistGradientBoostingRegressor] = {}

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'MultiQuantileHistGB':
        """
        Fit quantile models to training data.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target values (n_samples,)

        Returns:
            self (fitted model)

        Example:
            >>> model = MultiQuantileHistGB()
            >>> model.fit(X_train, y_train)
            >>> predictions = model.predict(X_test)  # Uses median (q50)
        """
        for q in self.quantiles:
            # Create model for this quantile
            model = HistGradientBoostingRegressor(
                # Loss function
                loss='quantile',
                quantile=q,

                # Iterations & early stopping
                max_iter=self.max_iter,
                early_stopping=True,
                validation_fraction=0.2,
                n_iter_no_change=15,

                # Tree structure
                max_depth=self.max_depth,
                max_leaf_nodes=self.max_leaf_nodes,

                # Learning
                learning_rate=self.learning_rate,
                l2_regularization=self.l2_regularization,

                # Leaf constraints
                min_samples_leaf=self.min_samples_leaf,

                # Binning
                max_bins=255,

                # Reproducibility
                random_state=self.random_state
            )

            # Fit model
            model.fit(X, y)

            # Store
            self.quantile_models[q] = model

        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate median (q50) predictions.

        Args:
            X: Feature matrix (n_samples, n_features)

        Returns:
            Median predictions (n_samples,)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        # Return median quantile
        median_model = self.quantile_models[0.5]
        return median_model.predict(X)

    def predict_quantiles(
        self,
        X: np.ndarray,
        quantiles: Optional[List[float]] = None
    ) -> Dict[float, np.ndarray]:
        """
        Generate predictions for all quantiles.

        Args:
            X: Feature matrix
            quantiles: Quantiles to predict (default: self.quantiles)

        Returns:
            Dictionary mapping quantile -> predictions

        Example:
            >>> preds = model.predict_quantiles(X_test)
            >>> preds[0.1]  # 10th percentile (floor)
            array([2.5, 3.2, ...])
            >>> preds[0.9]  # 90th percentile (ceiling)
            array([5.8, 7.1, ...])
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        if quantiles is None:
            quantiles = self.quantiles

        predictions = {}
        for q in quantiles:
            if q not in self.quantile_models:
                raise ValueError(f"Quantile {q} not fitted. Available: {list(self.quantile_models.keys())}")

            predictions[q] = self.quantile_models[q].predict(X)

        return predictions

    def get_uncertainty_band(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate uncertainty band (q90 - q10).

        Args:
            X: Feature matrix

        Returns:
            Uncertainty width for each sample

        Example:
            >>> uncertainty = model.get_uncertainty_band(X_test)
            >>> # Elite players have wider bands (higher ceiling potential)
            >>> uncertainty[elite_idx]
            3.2  # Wide band
            >>> uncertainty[average_idx]
            1.8  # Narrow band
        """
        preds = self.predict_quantiles(X, [0.1, 0.9])
        return preds[0.9] - preds[0.1]

    def get_feature_importance(self) -> np.ndarray:
        """
        Get feature importance from median model.

        Returns:
            Feature importance array
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        median_model = self.quantile_models[0.5]
        return median_model.feature_importances_
