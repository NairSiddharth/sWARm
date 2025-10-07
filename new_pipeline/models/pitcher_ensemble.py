"""
Pitcher ensemble with 3 role-specific models (starter/reliever/swing).

Each role has its own tier-based multi-quantile ensemble.
"""

from typing import Dict, Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler

from .base_ensemble import TieredQuantileEnsemble
from .keras_utils import build_multi_quantile_keras_adamw, get_keras_callbacks
from .multi_quantile_histgb import MultiQuantileHistGB

from new_pipeline.common.logging_config import get_logger

logger = get_logger(__name__)


# Monotonic constraints for 13 pitcher features
# Order: BB%, K%, ERA, GB%, SwStr%, WPA/LI, damage_control, Opportunity, strikeout_eff, contact_mgmt, strikeout_contact, Launch_Quality, Running_Control
PITCHER_MONOTONIC_CONSTRAINTS = [
    -1,  # BB% - lower is better
     1,  # K% - higher is better
    -1,  # ERA - lower is better
     1,  # GB% - higher is better
     1,  # SwStr% - higher is better
     1,  # WPA/LI - higher is better
     1,  # damage_control_ratio - higher is better
     1,  # Opportunity_Success - higher is better
     1,  # strikeout_efficiency - higher is better
     1,  # contact_management - higher is better
     1,  # strikeout_contact_quality - higher is better
     0,  # Statcast_Launch_Quality_Index - uncertain direction
     1   # Running_Control - higher is better
]


class PitcherRoleEnsemble(TieredQuantileEnsemble):
    """
    Pitcher ensemble with 3 role-specific models.

    Maintains separate ensembles for:
    - Starters (GS/G >= 0.8)
    - Relievers (GS/G < 0.3)
    - Swing pitchers (0.3 <= GS/G < 0.8)

    Each role has:
    - ExtraTreesRegressor (criterion='friedman_mse', max_features='sqrt')
    - Keras Multi-Quantile with AdamW
    - MultiQuantileHistGB with monotonic constraints

    Predictions use tier-based blending specific to each role.
    """

    def __init__(self):
        """Initialize pitcher ensemble (3 role-specific models)."""
        super().__init__()
        self.models: Dict[str, Dict] = {
            'starter': {},
            'reliever': {},
            'swing': {}
        }
        self.scalers: Dict[str, StandardScaler] = {}

    def _build_extratrees(self) -> ExtraTreesRegressor:
        """
        Build ExtraTreesRegressor for pitchers.

        Uses friedman_mse criterion (better for correlated features like K% variants)
        and sqrt max_features for feature diversity.

        Returns:
            Configured ExtraTreesRegressor
        """
        return ExtraTreesRegressor(
            n_estimators=100,
            max_depth=10,
            criterion='friedman_mse',  # Better splits with correlated features
            max_features='sqrt',  # ~3-4 features per split from 12 total
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
        - max_leaf_nodes=63 (better complexity control than max_depth alone)
        - learning_rate=0.03 (down from 0.05, better generalization)
        - l2_regularization=0.3 (down from 0.5, allow elite predictions)
        - max_iter=300 (up from 200, more iterations)
        - n_iter_no_change=15 (up from 10, more patience)
        - monotonic_cst for baseball logic enforcement

        Returns:
            MultiQuantileHistGB instance with pitcher monotonic constraints
        """
        return MultiQuantileHistGB(
            random_state=42,
            monotonic_cst=PITCHER_MONOTONIC_CONSTRAINTS
        )

    def fit(self, X: np.ndarray, y: np.ndarray, role_column: np.ndarray):
        """
        Train 3 role-specific ensembles.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target values (n_samples,)
            role_column: Role classification for each sample ('starter', 'reliever', 'swing')

        Raises:
            ValueError: If role_column contains invalid values
        """
        valid_roles = ['starter', 'reliever', 'swing']
        unique_roles = np.unique(role_column)

        if not all(role in valid_roles for role in unique_roles):
            invalid = [r for r in unique_roles if r not in valid_roles]
            raise ValueError(f"Invalid roles found: {invalid}. Must be one of {valid_roles}")

        input_dim = X.shape[1]

        # Train ensemble for each role
        for role in valid_roles:
            role_mask = role_column == role

            if not role_mask.any():
                continue  # Skip roles with no samples

            X_role = X[role_mask]
            y_role = y[role_mask]

            logger.info(f"Training {role} ensemble ({len(X_role)} samples)...")

            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_role)
            self.scalers[role] = scaler

            # Build and train models
            logger.info(f"  Training ExtraTrees...")
            et_model = self._build_extratrees()
            et_model.fit(X_scaled, y_role)

            logger.info(f"  Training Keras (AdamW + Swish + BatchNorm)...")
            keras_model = self._build_keras(input_dim)
            keras_model.fit(
                X_scaled, y_role,
                epochs=200,
                batch_size=32,
                validation_split=0.2,
                callbacks=get_keras_callbacks(),
                verbose=0
            )

            logger.info(f"  Training MultiQuantileHistGB...")
            histgb_model = self._build_histgb()
            histgb_model.fit(X_scaled, y_role)

            # Store models for this role
            self.models[role] = {
                'extratrees': et_model,
                'keras': keras_model,
                'histgb': histgb_model
            }

            logger.info(f"  {role.capitalize()} ensemble training complete")

        self._is_fitted = True

    def predict(self, X: np.ndarray, role_column: np.ndarray) -> np.ndarray:
        """
        Predict with tier-based blending per role.

        Args:
            X: Feature matrix (n_samples, n_features)
            role_column: Role classification for each sample

        Returns:
            Predictions (n_samples,)

        Raises:
            RuntimeError: If model not fitted yet
            ValueError: If role_column contains invalid/unknown roles
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted yet. Call fit() before predict().")

        predictions = np.zeros(len(X))

        # Predict for each role separately
        for role in ['starter', 'reliever', 'swing']:
            role_mask = role_column == role

            if not role_mask.any():
                continue

            if role not in self.models or not self.models[role]:
                raise ValueError(f"No model trained for role '{role}'")

            X_role = X[role_mask]

            # Scale features
            X_scaled = self.scalers[role].transform(X_role)

            # Get predictions from all models
            et_pred = self.models[role]['extratrees'].predict(X_scaled)
            keras_quantiles = self.models[role]['keras'].predict(X_scaled, verbose=0)
            histgb_quantiles = self.models[role]['histgb'].get_quantile_predictions(X_scaled)

            # Apply tier-based blending
            role_predictions = self._blend_predictions(et_pred, keras_quantiles, histgb_quantiles)

            # Store predictions for this role
            predictions[role_mask] = role_predictions

        return predictions

    def save(self, base_filepath: str):
        """
        Save all 3 role-specific ensembles.

        Args:
            base_filepath: Base path (will create 3 files: *_starter.pkl, *_reliever.pkl, *_swing.pkl)

        Example:
            >>> ensemble.save('models/pitcher_ensemble')
            >>> # Creates:
            >>> #   models/pitcher_ensemble_starter.pkl
            >>> #   models/pitcher_ensemble_reliever.pkl
            >>> #   models/pitcher_ensemble_swing.pkl
        """
        from pathlib import Path
        import pickle

        base_path = Path(base_filepath)
        base_path.parent.mkdir(parents=True, exist_ok=True)

        for role in ['starter', 'reliever', 'swing']:
            if role in self.models and self.models[role]:
                # Save sklearn models (scaler, extratrees, histgb) with pickle
                role_path = base_path.parent / f"{base_path.stem}_{role}.pkl"

                model_data = {
                    'scaler': self.scalers.get(role),
                    'extratrees': self.models[role].get('extratrees'),
                    'histgb': self.models[role].get('histgb')
                }

                with open(role_path, 'wb') as f:
                    pickle.dump(model_data, f)

                # Save Keras model separately using native format
                keras_model = self.models[role].get('keras')
                if keras_model is not None:
                    keras_path = base_path.parent / f"{base_path.stem}_{role}_keras.keras"
                    keras_model.save(str(keras_path))

        # Save fitted flag
        meta_path = base_path.parent / f"{base_path.stem}_meta.pkl"
        with open(meta_path, 'wb') as f:
            pickle.dump({'_is_fitted': self._is_fitted}, f)

    def load(self, base_filepath: str):
        """
        Load all 3 role-specific ensembles.

        Args:
            base_filepath: Base path (looks for *_starter.pkl, *_reliever.pkl, *_swing.pkl)

        Example:
            >>> ensemble = PitcherRoleEnsemble()
            >>> ensemble.load('models/pitcher_ensemble')
        """
        from pathlib import Path
        import pickle

        base_path = Path(base_filepath)

        for role in ['starter', 'reliever', 'swing']:
            role_path = base_path.parent / f"{base_path.stem}_{role}.pkl"

            if role_path.exists():
                # Load sklearn models from pickle
                with open(role_path, 'rb') as f:
                    model_data = pickle.load(f)

                self.scalers[role] = model_data['scaler']

                # Load Keras model from separate file
                keras_path = base_path.parent / f"{base_path.stem}_{role}_keras.keras"
                keras_model = None
                if keras_path.exists():
                    import keras
                    from .keras_utils import multi_quantile_loss
                    # Provide custom loss function for loading
                    custom_objects = {
                        'multi_quantile_loss_[0.5, 0.75, 0.9]': multi_quantile_loss([0.5, 0.75, 0.9], [0.2, 0.3, 0.5])
                    }
                    keras_model = keras.saving.load_model(str(keras_path), custom_objects=custom_objects)

                self.models[role] = {
                    'extratrees': model_data['extratrees'],
                    'keras': keras_model,
                    'histgb': model_data['histgb']
                }

        # Load fitted flag
        meta_path = base_path.parent / f"{base_path.stem}_meta.pkl"
        if meta_path.exists():
            with open(meta_path, 'rb') as f:
                meta = pickle.load(f)
                self._is_fitted = meta['_is_fitted']
