"""
Improved Multi-Quantile HistGradientBoosting with recommended parameters.

IMPROVEMENTS from current_season_modules version:
- max_iter: 200 -> 300 (more iterations)
- max_depth: 10 -> None (no depth limit)
- max_leaf_nodes: None -> 63 (better complexity control)
- learning_rate: 0.05 -> 0.03 (better generalization)
- l2_regularization: 0.5 -> 0.3 (allow elite predictions)
- n_iter_no_change: 10 -> 15 (more patience)
- Added monotonic_cst parameter support
"""

from typing import Optional, List
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor


class MultiQuantileHistGB:
    """
    Multi-quantile HistGradientBoosting ensemble with tier-based blending.

    Trains three quantile models (q=0.5, 0.75, 0.9) and blends predictions
    based on predicted performance tier:
    - Average pitchers (< 2.0 WAR): Use median only (conservative)
    - Good pitchers (2.0-3.5 WAR): Blend median + q75 (moderate)
    - Elite pitchers (e 3.5 WAR): Emphasize q90 (aggressive)

    This asymmetric approach improves elite predictions without risking
    overprediction for average players.

    IMPROVED PARAMETERS (vs legacy version):
    - Higher max_iter (300) for better convergence
    - No max_depth limit (use max_leaf_nodes instead)
    - Direct complexity control via max_leaf_nodes=63
    - Lower learning_rate (0.03) for fine-tuning
    - Less regularization (0.3) to allow extreme predictions
    - More patience (n_iter_no_change=15)
    - Support for monotonic constraints
    """

    def __init__(
        self,
        random_state: int = 42,
        monotonic_cst: Optional[List[int]] = None
    ):
        """
        Initialize multi-quantile HistGB ensemble.

        Args:
            random_state: Random seed for reproducibility
            monotonic_cst: Monotonic constraints for features (e.g., [-1, 1, 0, ...])
                -1 = decreasing (higher feature � lower target)
                0 = no constraint
                1 = increasing (higher feature � higher target)
        """
        self.random_state = random_state
        self.monotonic_cst = monotonic_cst
        self.models = {}

    def fit(self, X, y):
        """
        Train three quantile models (q=0.5, 0.75, 0.9).

        Args:
            X: Feature matrix (already scaled if needed)
            y: Target values (WAR per 162 IP or WAR per 600 PA)
        """
        quantiles = [0.5, 0.75, 0.9]

        for q in quantiles:
            model = HistGradientBoostingRegressor(
                # Loss function
                loss='quantile',
                quantile=q,

                # Iterations & early stopping
                max_iter=300,  #  More iterations (was 200)
                early_stopping=True,
                validation_fraction=0.2,
                n_iter_no_change=15,  #  More patience (was 10)

                # Tree structure
                max_depth=None,  # = No depth limit (was 10)
                max_leaf_nodes=63,  #  NEW - Direct complexity control (~2^6-1)

                # Learning
                learning_rate=0.03,  #  Slower for better generalization (was 0.05)
                l2_regularization=0.3,  #  Less regularization to allow elites (was 0.5)

                # Leaf constraints
                min_samples_leaf=5,  #  KEEP

                # Binning
                max_bins=255,  #  KEEP

                # Domain knowledge constraints
                monotonic_cst=self.monotonic_cst,  #  NEW - Baseball logic enforcement

                # Reproducibility
                random_state=self.random_state
            )
            model.fit(X, y)
            self.models[f'q{int(q*100)}'] = model

    def predict(self, X):
        """
        Predict using tier-based quantile blending.

        Blending strategy:
        - If median_pred < 2.0: Use median only (conservative)
        - If 2.0 d median_pred < 3.5: 70% median + 30% q75 (moderate)
        - If median_pred e 3.5: 30% median + 30% q75 + 40% q90 (aggressive)

        Args:
            X: Feature matrix (same shape as training)

        Returns:
            Array of blended predictions
        """
        # Get predictions from all three models
        median_pred = self.models['q50'].predict(X)
        q75_pred = self.models['q75'].predict(X)
        q90_pred = self.models['q90'].predict(X)

        # Tier-based blending
        predictions = np.zeros(len(X))

        for i in range(len(X)):
            med = median_pred[i]
            q75 = q75_pred[i]
            q90 = q90_pred[i]

            if med < 2.0:
                # Average tier: conservative (median only)
                predictions[i] = med
            elif 2.0 <= med < 3.5:
                # Good tier: moderate blend
                predictions[i] = 0.7 * med + 0.3 * q75
            else:
                # Elite tier: aggressive blend (emphasize q90)
                predictions[i] = 0.3 * med + 0.3 * q75 + 0.4 * q90

        return predictions

    def get_quantile_predictions(self, X):
        """
        Get individual quantile predictions (for ensemble blending).

        Args:
            X: Feature matrix

        Returns:
            dict: {'q50': array, 'q75': array, 'q90': array}
        """
        return {
            'q50': self.models['q50'].predict(X),
            'q75': self.models['q75'].predict(X),
            'q90': self.models['q90'].predict(X)
        }
