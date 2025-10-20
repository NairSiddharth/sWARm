"""
Pitcher ensemble with 3 role-specific models (starter/reliever/swing).

Each role has its own tier-based multi-quantile ensemble.
"""

from typing import Dict, Optional
import random
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler

from .base_ensemble import TieredQuantileEnsemble
from .keras_utils import build_multi_quantile_keras_adamw, get_keras_callbacks, set_seed, OPTIMAL_SEED
from .multi_quantile_histgb import MultiQuantileHistGB

from new_pipeline.common.logging_config import get_logger

logger = get_logger(__name__)


# Monotonic constraints for 14 pitcher features
# Order: BB%, K%, ERA, GB%, SwStr%, WPA/LI, damage_control, Opportunity, strikeout_eff, contact_mgmt, strikeout_contact, Launch_Quality, Running_Control, SD_MD_Net
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
     1,  # Running_Control - higher is better
     1   # SD_MD_Net - higher is better (reliever-specific signal)
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
        self._current_season_pct = 0.5  # Default to halfway point

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

    def _build_keras(self, input_dim: int, role: str):
        """
        Build Keras multi-quantile model with role-specific hyperparameters.

        Note: After implementing WAR_per_48.2 normalization for relievers, all roles
        now use the same hyperparameters since they're on similar scales.

        Unified settings for all roles:
        - weight_decay=0.03
        - quantile_weights=[0.25, 0.3, 0.45]

        Args:
            input_dim: Number of input features
            role: Pitcher role ('starter', 'reliever', 'swing')

        Returns:
            Compiled Keras Sequential model
        """
        # Role-specific hyperparameters
        # Note: With WAR_per_48.2 normalization, relievers are on similar scale to starters
        # so we use the same hyperparameters for all roles
        hyperparams = {
            'starter': {
                'weight_decay': 0.03,
                'quantile_weights': [0.25, 0.3, 0.45]
            },
            'reliever': {
                'weight_decay': 0.03,  # Same as starters (normalization fixed the issue)
                'quantile_weights': [0.25, 0.3, 0.45]  # Same as starters
            },
            'swing': {
                'weight_decay': 0.03,  # Same as starters
                'quantile_weights': [0.25, 0.3, 0.45]  # Same as starters
            }
        }

        params = hyperparams[role]
        return build_multi_quantile_keras_adamw(
            input_dim,
            weight_decay=params['weight_decay'],
            quantile_weights=params['quantile_weights']
        )

    def _build_histgb(self, role: str) -> MultiQuantileHistGB:
        """
        Build MultiQuantileHistGB with role-specific hyperparameters and monotonic constraints.

        Note: After implementing WAR_per_48.2 normalization for relievers, all roles
        now use the same hyperparameters since they're on similar scales.

        Unified settings for all roles:
        - learning_rate=0.03
        - l2_regularization=0.4

        Improvements from legacy implementation:
        - max_leaf_nodes=63 (better complexity control than max_depth alone)
        - max_iter=300 (up from 200, more iterations)
        - n_iter_no_change=15 (up from 10, more patience)
        - monotonic_cst for baseball logic enforcement

        Args:
            role: Pitcher role ('starter', 'reliever', 'swing')

        Returns:
            MultiQuantileHistGB instance with pitcher monotonic constraints
        """
        # Role-specific hyperparameters
        # Note: With WAR_per_48.2 normalization, relievers are on similar scale to starters
        # so we use the same hyperparameters for all roles
        hyperparams = {
            'starter': {
                'learning_rate': 0.03,
                'l2_regularization': 0.4
            },
            'reliever': {
                'learning_rate': 0.03,  # Same as starters (normalization fixed the issue)
                'l2_regularization': 0.4  # Same as starters
            },
            'swing': {
                'learning_rate': 0.03,  # Same as starters
                'l2_regularization': 0.4  # Same as starters
            }
        }

        params = hyperparams[role]
        return MultiQuantileHistGB(
            random_state=42,
            monotonic_cst=PITCHER_MONOTONIC_CONSTRAINTS,
            learning_rate=params['learning_rate'],
            l2_regularization=params['l2_regularization']
        )

    def set_season_progress(self, season_pct: float):
        """
        Set current season progress for dynamic threshold calculation.

        Args:
            season_pct: Season completion percentage (0.0 to 1.0)
                       0.0 = season start, 0.5 = All-Star break, 1.0 = season end

        Example:
            >>> # For All-Star break predictions
            >>> model.set_season_progress(0.5)
            >>> predictions = model.predict(X, roles)
        """
        self._current_season_pct = min(max(season_pct, 0.0), 1.0)  # Clamp to [0, 1]

    def fit(self, X: np.ndarray, y: np.ndarray, role_column: np.ndarray, seed: int = OPTIMAL_SEED, enable_determinism: bool = True):
        """
        Train 3 role-specific ensembles.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target values (n_samples,)
            role_column: Role classification for each sample ('starter', 'reliever', 'swing')
            seed: Random seed for reproducibility (default: OPTIMAL_SEED=3141)
            enable_determinism: Whether to enable TensorFlow op determinism (default: True)

        Raises:
            ValueError: If role_column contains invalid values
        """
        # Set all random seeds for reproducibility
        set_seed(seed, enable_determinism=enable_determinism)

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

            logger.info(f"  Training Keras (AdamW + Swish + BatchNorm) with role-specific hyperparameters...")
            keras_model = self._build_keras(input_dim, role)
            keras_model.fit(
                X_scaled, y_role,
                epochs=200,
                batch_size=32,
                validation_split=0.2,
                callbacks=get_keras_callbacks(),
                verbose=0
            )

            logger.info(f"  Training MultiQuantileHistGB with role-specific hyperparameters...")
            histgb_model = self._build_histgb(role)
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

            # Get dynamic thresholds based on season progress and role
            from . import tier_thresholds as tier_thresh
            tier_thresholds = tier_thresh.get_thresholds(self._current_season_pct, role)

            # Apply tier-based blending with dynamic thresholds and role-specific weights
            role_predictions = self._blend_predictions(
                et_pred, keras_quantiles, histgb_quantiles,
                tier_thresholds=tier_thresholds,
                role=role
            )

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

                    # Role-specific quantile weights (must match training)
                    # Note: All roles now use same weights after normalization fix
                    role_weights = {
                        'starter': [0.25, 0.3, 0.45],
                        'reliever': [0.25, 0.3, 0.45],  # Unified with starters
                        'swing': [0.25, 0.3, 0.45]  # Unified with starters
                    }

                    # Provide custom loss function for loading
                    custom_objects = {
                        'multi_quantile_loss_[0.5, 0.75, 0.9]': multi_quantile_loss(
                            [0.5, 0.75, 0.9],
                            role_weights[role]
                        )
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
