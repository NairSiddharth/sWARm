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
        histgb_quantiles: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """
        Apply tier-based blending to ensemble predictions.

        Tier definitions:
        - Average (<2.0 WAR): 50% ET + 30% keras_q50 + 20% histgb_q50
        - Good (2.0-3.5 WAR): 30% ET + 15% keras_q50 + 25% keras_q75 + 10% histgb_q50 + 20% histgb_q75
        - Elite (e3.5 WAR): 15% ET + 10% keras_q50 + 20% keras_q75 + 30% keras_q90 + 10% histgb_q75 + 15% histgb_q90

        Args:
            et_pred: ExtraTrees predictions (n_samples,)
            keras_quantiles: Keras quantiles (n_samples, 3) - [q50, q75, q90]
            histgb_quantiles: Dict with 'q50', 'q75', 'q90' keys

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

        # Apply tier-based blending
        n_samples = len(et_pred)
        final_predictions = np.zeros(n_samples)

        for i in range(n_samples):
            init_war = initial[i]

            if init_war < 2.0:
                # Average tier: Conservative (don't overpredict)
                # Heavy weight on MSE anchor, use median quantiles only
                final_predictions[i] = (
                    0.50 * et_pred[i] +
                    0.30 * keras_q50[i] +
                    0.20 * histgb_q50[i]
                )

            elif 2.0 <= init_war < 3.5:
                # Good tier: Moderate (balanced approach)
                # Mix of median and upper-mid quantiles
                final_predictions[i] = (
                    0.30 * et_pred[i] +
                    0.15 * keras_q50[i] +
                    0.25 * keras_q75[i] +
                    0.10 * histgb_q50[i] +
                    0.20 * histgb_q75[i]
                )

            else:  # init_war >= 3.5
                # Elite tier: Aggressive (avoid underprediction)
                # Heavy emphasis on q90 from both quantile models
                final_predictions[i] = (
                    0.15 * et_pred[i] +
                    0.10 * keras_q50[i] +
                    0.20 * keras_q75[i] +
                    0.30 * keras_q90[i] +
                    0.10 * histgb_q75[i] +
                    0.15 * histgb_q90[i]
                )

        return final_predictions

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
