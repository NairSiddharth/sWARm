"""
Multi-Quantile HistGradientBoosting for Phase 3.

Implements tier-based quantile regression to address elite pitcher underprediction
while maintaining conservative predictions for average pitchers.
"""

from sklearn.ensemble import HistGradientBoostingRegressor
import numpy as np


class MultiQuantileHistGB:
    """
    Multi-quantile HistGradientBoosting ensemble with tier-based blending.

    Trains three quantile models (q=0.5, 0.75, 0.9) and blends predictions
    based on predicted performance tier:
    - Average pitchers (< 2.0 WAR): Use median only (conservative)
    - Good pitchers (2.0-3.5 WAR): Blend median + q75 (moderate)
    - Elite pitchers (≥ 3.5 WAR): Emphasize q90 (aggressive)

    This asymmetric approach improves elite predictions without risking
    overprediction for average pitchers.
    """

    def __init__(self, random_state=42):
        """
        Initialize multi-quantile HistGB ensemble.

        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.models = {}

    def fit(self, X, y):
        """
        Train three quantile models (q=0.5, 0.75, 0.9).

        Args:
            X: Feature matrix (already scaled if needed)
            y: Target values (WAR per 162 IP)
        """
        quantiles = [0.5, 0.75, 0.9]

        for q in quantiles:
            model = HistGradientBoostingRegressor(
                loss='quantile',
                quantile=q,
                max_iter=200,
                max_depth=10,
                learning_rate=0.05,
                l2_regularization=0.5,
                min_samples_leaf=5,
                max_bins=255,
                early_stopping=True,
                validation_fraction=0.2,
                n_iter_no_change=10,
                random_state=self.random_state
            )
            model.fit(X, y)
            self.models[f'q{int(q*100)}'] = model

    def predict(self, X):
        """
        Predict using tier-based quantile blending.

        Blending strategy:
        - If median_pred < 2.0: Use median only (conservative)
        - If 2.0 ≤ median_pred < 3.5: 70% median + 30% q75 (moderate)
        - If median_pred ≥ 3.5: 30% median + 30% q75 + 40% q90 (aggressive)

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
        Get raw predictions from all three quantile models.

        Useful for debugging and analysis.

        Args:
            X: Feature matrix

        Returns:
            Dictionary with keys 'q50', 'q75', 'q90' containing predictions
        """
        return {
            'q50': self.models['q50'].predict(X),
            'q75': self.models['q75'].predict(X),
            'q90': self.models['q90'].predict(X)
        }
