"""
Longitudinal Model for Year-to-Year WAR Prediction

Predicts Year N+1 WAR from Year N performance using RandomForest regression.
Uses proven hyperparameters and new pipeline features.

See FUTURE_PROJECTIONS_MODEL_ARCHITECTURE.md Module 1 for design specs.
See FUTURE_PROJECTIONS_MIGRATION_GUIDE.md Section 3B for migration notes.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from new_pipeline.common.constants import HITTER_MODEL_FEATURES, PITCHER_MODEL_FEATURES


class LongitudinalModel:
    """
    Year-to-year WAR prediction using RandomForest regression.

    Predicts Year N+1 WAR from Year N features using sequences from data_preparation.py.

    Key features:
    - Uses new pipeline features (9 hitter / 14 pitcher)
    - Proven RandomForest hyperparameters from old system
    - Standard scaling for feature normalization
    - Temporal validation (no future data leakage)
    """

    def __init__(self, player_type: str):
        """
        Initialize longitudinal model.

        Args:
            player_type: 'hitter' or 'pitcher'
        """
        if player_type not in ['hitter', 'pitcher']:
            raise ValueError(f"player_type must be 'hitter' or 'pitcher', got {player_type}")

        self.player_type = player_type
        self.model_features = HITTER_MODEL_FEATURES if player_type == 'hitter' else PITCHER_MODEL_FEATURES

        # Proven hyperparameters from MODEL_ARCHITECTURE.md
        # Based on old system + research tuning
        self.model = RandomForestRegressor(
            n_estimators=200,
            max_depth=12,
            min_samples_split=20,
            min_samples_leaf=10,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )

        self.scaler = StandardScaler()
        self.is_fitted = False

    def prepare_features(self, sequences_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract features and target from sequences dataframe.

        Args:
            sequences_df: Output from build_longitudinal_sequences() + add_age_context_features()

        Returns:
            (X, y) where:
                X: Features (n_samples, n_features)
                y: Target WAR values (n_samples,)
        """
        # Model features from Year N (e.g., 'K%_n', 'BB%_n', ...)
        feature_cols_n = [f'{feat}_n' for feat in self.model_features]

        # Age/career context features (added by add_age_context_features())
        context_features = [
            'age_n',
            'age_squared',
            'years_from_peak',
            'age_group_young',
            'age_group_prime',
            'age_group_veteran',
            'war_n'  # Include Year N WAR for context
        ]

        # Combine all features
        all_features = feature_cols_n + context_features

        # Check if all features exist
        missing = [f for f in all_features if f not in sequences_df.columns]
        if missing:
            raise ValueError(f"Missing features in sequences_df: {missing}")

        X = sequences_df[all_features].values

        # Target column may not exist during prediction
        if 'war_n_plus_1' in sequences_df.columns:
            y = sequences_df['war_n_plus_1'].values
        else:
            y = None

        return X, y

    def train(self, train_df: pd.DataFrame) -> Dict[str, float]:
        """
        Train longitudinal model on training sequences.

        Args:
            train_df: Training sequences from create_temporal_splits()

        Returns:
            Training metrics: {'r2': float, 'rmse': float, 'mae': float}
        """
        print(f"Training {self.player_type} longitudinal model...")
        print(f"  Sequences: {len(train_df)}")

        # Prepare features
        X_train, y_train = self.prepare_features(train_df)

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)

        print(f"  Features: {X_train_scaled.shape[1]} (9 hitter/{14} pitcher + 7 context)")
        print(f"  Target range: {y_train.min():.2f} to {y_train.max():.2f} WAR")

        # Train model
        self.model.fit(X_train_scaled, y_train)
        self.is_fitted = True

        # Calculate training metrics
        y_pred_train = self.model.predict(X_train_scaled)
        metrics = {
            'r2': r2_score(y_train, y_pred_train),
            'rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'mae': mean_absolute_error(y_train, y_pred_train),
            'n_samples': len(y_train)
        }

        print(f"  Training R²: {metrics['r2']:.3f}")
        print(f"  Training RMSE: {metrics['rmse']:.3f}")
        print(f"  Training MAE: {metrics['mae']:.3f}")

        return metrics

    def predict(self, test_df: pd.DataFrame) -> np.ndarray:
        """
        Predict Year N+1 WAR for test sequences.

        Args:
            test_df: Test sequences (does not need war_n_plus_1 column)

        Returns:
            Predicted WAR values (n_samples,)
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call train() first.")

        X_test, _ = self.prepare_features(test_df)  # y will be None for prediction
        X_test_scaled = self.scaler.transform(X_test)

        return self.model.predict(X_test_scaled)

    def validate(self, val_df: pd.DataFrame) -> Dict[str, float]:
        """
        Validate model on held-out data.

        Args:
            val_df: Validation sequences

        Returns:
            Validation metrics: {'r2': float, 'rmse': float, 'mae': float}
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call train() first.")

        X_val, y_val = self.prepare_features(val_df)
        X_val_scaled = self.scaler.transform(X_val)
        y_pred = self.model.predict(X_val_scaled)

        metrics = {
            'r2': r2_score(y_val, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_val, y_pred)),
            'mae': mean_absolute_error(y_val, y_pred),
            'n_samples': len(y_val)
        }

        print(f"\n{self.player_type.capitalize()} validation metrics:")
        print(f"  R²: {metrics['r2']:.3f}")
        print(f"  RMSE: {metrics['rmse']:.3f} WAR")
        print(f"  MAE: {metrics['mae']:.3f} WAR")
        print(f"  Samples: {metrics['n_samples']}")

        return metrics

    def get_feature_importance(self, top_n: int = 10) -> pd.DataFrame:
        """
        Get feature importance from trained model.

        Args:
            top_n: Number of top features to return

        Returns:
            DataFrame with feature names and importance scores
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call train() first.")

        # Get feature names
        feature_cols_n = [f'{feat}_n' for feat in self.model_features]
        context_features = ['age_n', 'age_squared', 'years_from_peak',
                           'age_group_young', 'age_group_prime', 'age_group_veteran', 'war_n']
        all_features = feature_cols_n + context_features

        # Get importance scores
        importance = self.model.feature_importances_

        # Create dataframe
        importance_df = pd.DataFrame({
            'feature': all_features,
            'importance': importance
        }).sort_values('importance', ascending=False).head(top_n)

        return importance_df
