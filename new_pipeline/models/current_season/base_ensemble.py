"""
Base ensemble class with tier-based quantile blending.

Provides common functionality for pitcher and hitter ensembles.
"""

from typing import Optional, Dict
import numpy as np
import pickle
from pathlib import Path
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler


class TieredQuantileEnsemble(BaseEstimator, RegressorMixin):
    """
    Base class for tier-based multi-quantile ensembles.

    Three-model ensemble:
    1. ExtraTreesRegressor (MSE) - Conservative anchor
    2. Keras Multi-Quantile [q50, q75, q90] with AdamW
    3. HistGB Multi-Quantile [q50, q75, q90]

    Tier-based blending (based on initial conservative estimate):
    - Average tier (<2.0 WAR): Conservative - heavy MSE, median quantiles only
    - Good tier (2.0-3.5 WAR): Moderate - balanced mix of q50 and q75
    - Elite tier (e3.5 WAR): Aggressive - emphasize q90 to avoid underprediction

    This asymmetric approach improves elite predictions without risking
    overprediction for average players.
    """

    def __init__(self):
        """Initialize base ensemble (models created by subclasses)."""
        self.scaler: Optional[StandardScaler] = None
        self.extratrees_model = None
        self.keras_model = None
        self.histgb_model = None
        self._is_fitted = False

    def _get_initial_estimate(
        self,
        et_pred: np.ndarray,
        keras_quantiles: np.ndarray,
        histgb_q50: np.ndarray
    ) -> np.ndarray:
        """
        Calculate initial conservative estimate for tier classification.

        Uses conservative blend: MSE anchor + median quantiles.

        Args:
            et_pred: ExtraTrees predictions (MSE)
            keras_quantiles: Keras quantiles (batch, 3) - [q50, q75, q90]
            histgb_q50: HistGB median predictions

        Returns:
            Initial estimates for tier classification
        """
        keras_q50 = keras_quantiles[:, 0]
        return 0.4 * et_pred + 0.3 * keras_q50 + 0.3 * histgb_q50

    def _blend_predictions(
        self,
        et_pred: np.ndarray,
        keras_quantiles: np.ndarray,
        histgb_quantiles: Dict[str, np.ndarray],
        tier_thresholds: tuple = (2.0, 3.5),
        role: str = 'starter'
    ) -> np.ndarray:
        """
        Apply tier-based blending to ensemble predictions with role-specific weights.

        Starter/Swing tier definitions:
        - Average (<avg_threshold): 50% ET + 30% keras_q50 + 20% histgb_q50
        - Good: 30% ET + 15% keras_q50 + 25% keras_q75 + 10% histgb_q50 + 20% histgb_q75 (45% q75)
        - Elite: 15% ET + 10% keras_q50 + 20% keras_q75 + 30% keras_q90 + 10% histgb_q75 + 15% histgb_q90 (45% q90)

        Reliever tier definitions (more aggressive):
        - Average (<avg_threshold): Same as starters
        - Good: 20% ET + 10% keras_q50 + 25% keras_q75 + 15% keras_q90 + 10% histgb_q75 + 20% histgb_q90 (50% q75 + 35% q90)
        - Elite: 10% ET + 5% keras_q50 + 10% keras_q75 + 40% keras_q90 + 10% histgb_q75 + 25% histgb_q90 (65% q90)

        Args:
            et_pred: ExtraTrees predictions (n_samples,)
            keras_quantiles: Keras quantiles (n_samples, 3) - [q50, q75, q90]
            histgb_quantiles: Dict with 'q50', 'q75', 'q90' keys
            tier_thresholds: Tuple of (avg_threshold, elite_threshold) in WAR
                            Default: (2.0, 3.5) for full-season predictions
            role: Player role ('starter', 'reliever', 'swing') - affects blend weights

        Returns:
            Blended predictions (n_samples,)
        """
        # Extract quantiles
        keras_q50 = keras_quantiles[:, 0]
        keras_q75 = keras_quantiles[:, 1]
        keras_q90 = keras_quantiles[:, 2]
        histgb_q50 = histgb_quantiles['q50']
        histgb_q75 = histgb_quantiles['q75']
        histgb_q90 = histgb_quantiles['q90']

        # Calculate initial conservative estimate for tier classification
        initial = self._get_initial_estimate(et_pred, keras_quantiles, histgb_q50)

        # Unpack tier thresholds
        avg_threshold, elite_threshold = tier_thresholds

        # Apply tier-based blending with role-specific weights
        n_samples = len(et_pred)
        final_predictions = np.zeros(n_samples)

        # Determine if we should use reliever weights
        use_reliever_weights = (role == 'reliever')

        for i in range(n_samples):
            init_war = initial[i]

            if init_war < avg_threshold:
                # Average tier: Conservative (don't overpredict)
                # Same for all roles - heavy weight on MSE anchor, use median quantiles only
                final_predictions[i] = (
                    0.50 * et_pred[i] +
                    0.30 * keras_q50[i] +
                    0.20 * histgb_q50[i]
                )

            elif init_war < elite_threshold:
                # Good tier: Moderate (balanced approach)
                if use_reliever_weights:
                    # Relievers: More aggressive (50% q75 + 35% q90)
                    final_predictions[i] = (
                        0.20 * et_pred[i] +
                        0.10 * keras_q50[i] +
                        0.25 * keras_q75[i] +
                        0.15 * keras_q90[i] +
                        0.10 * histgb_q75[i] +
                        0.20 * histgb_q90[i]
                    )
                else:
                    # Starters/Swing: Standard (45% q75)
                    final_predictions[i] = (
                        0.30 * et_pred[i] +
                        0.15 * keras_q50[i] +
                        0.25 * keras_q75[i] +
                        0.10 * histgb_q50[i] +
                        0.20 * histgb_q75[i]
                    )

            else:  # init_war >= elite_threshold
                # Elite tier: Aggressive (avoid underprediction)
                if use_reliever_weights:
                    # Relievers: Very aggressive (65% q90)
                    final_predictions[i] = (
                        0.10 * et_pred[i] +
                        0.05 * keras_q50[i] +
                        0.10 * keras_q75[i] +
                        0.40 * keras_q90[i] +
                        0.10 * histgb_q75[i] +
                        0.25 * histgb_q90[i]
                    )
                else:
                    # Starters/Swing: Standard (45% q90)
                    final_predictions[i] = (
                        0.15 * et_pred[i] +
                        0.10 * keras_q50[i] +
                        0.20 * keras_q75[i] +
                        0.30 * keras_q90[i] +
                        0.10 * histgb_q75[i] +
                        0.15 * histgb_q90[i]
                    )

        return final_predictions

    def _get_dynamic_thresholds(self, season_pct: float, role: str) -> tuple:
        """
        Calculate tier thresholds based on season progress.

        Returns thresholds on same scale as model predictions (WAR_per_[denominator]).
        Season scaling is applied after normalization to account for confidence level.

        Uses role-specific scaling formulas:
        - Starters: Quadratic (1.22*x² - 0.83*x + 0.61) fitted to starter WAR accumulation
        - Relievers: Square root (x^0.7) for more conservative early-season thresholds
        - Swing: Square root (x^0.7) similar to relievers
        - Hitters: Linear (x) for more consistent WAR accumulation

        Threshold scales by role:
        - Starters: WAR_per_162
        - Relievers: WAR_per_48.2
        - Swing: WAR_per_110
        - Hitters: WAR_per_600

        Args:
            season_pct: Fraction of season completed (0.0 to 1.0)
            role: Player role ('starter', 'reliever', 'swing', 'hitter')

        Returns:
            tuple: (good_threshold, elite_threshold) in WAR_per_[denominator]

        Example:
            >>> # Starter thresholds at 70% season (WAR_per_162):
            >>> thresholds = self._get_dynamic_thresholds(0.7, 'starter')
            >>> # Returns: (2.24, 3.52) - scaled from base (3.5, 5.5)
            >>>
            >>> # Reliever thresholds at 58.6% season (WAR_per_48.2):
            >>> thresholds = self._get_dynamic_thresholds(0.586, 'reliever')
            >>> # Returns: (0.68, 1.02) - scaled from base (1.03, 1.55)
            >>>
            >>> # Hitter thresholds at 70% season (WAR_per_600):
            >>> thresholds = self._get_dynamic_thresholds(0.7, 'hitter')
            >>> # Returns: (1.40, 2.45) - scaled from base (2.0, 3.5)
        """
        if role == 'starter':
            # Quadratic scaling for starters: 1.22*x² - 0.83*x + 0.61
            # Fitted to starter WAR accumulation patterns
            scaling = min(
                1.22 * (season_pct ** 2) - 0.83 * season_pct + 0.61,
                1.0
            )
            # Base values in WAR_per_162 scale
            base_elite = 5.0   # Ace starters (top 13-14 per year)
            base_good = 3.3    # Top #1/#2 starters (All-Star caliber)

        elif role == 'reliever':
            # Square root scaling for relievers: x^0.7
            # More conservative early season to handle high variance in small IP samples
            scaling = min(season_pct ** 0.7, 1.0)

            # Convert base values from full-season actual WAR to WAR_per_48.2
            # Elite relievers: 2.25 WAR in ~70 IP typical
            # Good relievers: 1.5 WAR in ~70 IP typical
            base_elite_full = 2.25
            base_good_full = 1.5
            typical_ip = 70  # Typical full-season IP for quality reliever

            # Normalize to WAR_per_48.2: (WAR / typical_IP) * 48.2
            base_elite = base_elite_full / typical_ip * 48.2  # = 1.55
            base_good = base_good_full / typical_ip * 48.2    # = 1.03

        elif role == 'swing':
            # Square root scaling for swing (similar to relievers)
            scaling = min(season_pct ** 0.7, 1.0)

            # Base values in WAR_per_110 scale
            base_elite = 3.5  # Full season target
            base_good = 2.5   # Full season target

        elif role == 'hitter':
            # Linear scaling for hitters: x
            # Hitters accumulate WAR more linearly than pitchers
            scaling = min(season_pct, 1.0)

            # Base values in WAR_per_600 scale (full season thresholds)
            base_elite = 3.5  # Elite hitters: >3.5 WAR per 600 PA
            base_good = 2.0   # Good hitters: 2.0-3.5 WAR per 600 PA

        else:
            raise ValueError(f"Invalid role: {role}. Must be 'starter', 'reliever', 'swing', or 'hitter'")

        # Apply season scaling to normalized base values
        return base_good * scaling, base_elite * scaling

    def predict_with_diagnostics(self, X: np.ndarray) -> Dict:
        """
        Predict with detailed diagnostics for debugging.

        Args:
            X: Feature matrix (same shape as training)

        Returns:
            dict with:
                - predictions: Final blended predictions
                - tier: Tier classification for each sample
                - initial_estimate: Conservative initial estimate
                - extratrees: ExtraTrees predictions
                - keras_q50/q75/q90: Keras quantile predictions
                - histgb_q50/q75/q90: HistGB quantile predictions

        Raises:
            RuntimeError: If model not fitted yet
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted yet. Call fit() before predict().")

        # Scale features
        X_scaled = self.scaler.transform(X)

        # Get all predictions
        et_pred = self.extratrees_model.predict(X_scaled)
        keras_quantiles = self.keras_model.predict(X_scaled, verbose=0)
        histgb_quantiles = self.histgb_model.get_quantile_predictions(X_scaled)

        # Calculate initial estimate for tier classification
        initial = self._get_initial_estimate(et_pred, keras_quantiles, histgb_quantiles['q50'])

        # Classify tiers
        tier_labels = np.array([
            'average' if w < 2.0 else 'good' if w < 3.5 else 'elite'
            for w in initial
        ])

        # Get final predictions
        final = self._blend_predictions(et_pred, keras_quantiles, histgb_quantiles)

        return {
            'predictions': final,
            'tier': tier_labels,
            'initial_estimate': initial,
            'extratrees': et_pred,
            'keras_q50': keras_quantiles[:, 0],
            'keras_q75': keras_quantiles[:, 1],
            'keras_q90': keras_quantiles[:, 2],
            'histgb_q50': histgb_quantiles['q50'],
            'histgb_q75': histgb_quantiles['q75'],
            'histgb_q90': histgb_quantiles['q90']
        }

    def save(self, filepath: str):
        """
        Save ensemble model to disk.

        Args:
            filepath: Path to save model (will create .pkl and .keras files)

        Example:
            >>> ensemble.save('models/pitcher_starter_ensemble')
            >>> # Creates: models/pitcher_starter_ensemble.pkl (sklearn models)
            >>> #          models/pitcher_starter_ensemble_keras.keras (Keras model)
        """
        filepath = Path(filepath).with_suffix('.pkl')
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Save sklearn models with pickle
        model_data = {
            'scaler': self.scaler,
            'extratrees_model': self.extratrees_model,
            'histgb_model': self.histgb_model,
            '_is_fitted': self._is_fitted
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

        # Save Keras model separately using native format
        if self.keras_model is not None:
            keras_path = filepath.with_stem(f"{filepath.stem}_keras").with_suffix('.keras')
            self.keras_model.save(str(keras_path))

    def load(self, filepath: str):
        """
        Load ensemble model from disk.

        Args:
            filepath: Path to saved model (.pkl file)

        Example:
            >>> ensemble = TieredQuantileEnsemble()
            >>> ensemble.load('models/pitcher_starter_ensemble.pkl')
        """
        filepath = Path(filepath).with_suffix('.pkl')

        # Load sklearn models from pickle
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        self.scaler = model_data['scaler']
        self.extratrees_model = model_data['extratrees_model']
        self.histgb_model = model_data['histgb_model']
        self._is_fitted = model_data['_is_fitted']

        # Load Keras model from separate file
        keras_path = filepath.with_stem(f"{filepath.stem}_keras").with_suffix('.keras')
        if keras_path.exists():
            import keras
            from .keras_utils import multi_quantile_loss
            # Provide custom loss function for loading
            custom_objects = {
                'multi_quantile_loss_[0.5, 0.75, 0.9]': multi_quantile_loss([0.5, 0.75, 0.9], [0.2, 0.3, 0.5])
            }
            self.keras_model = keras.saving.load_model(str(keras_path), custom_objects=custom_objects)
        else:
            self.keras_model = None
