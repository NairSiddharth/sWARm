"""
Ensemble Longitudinal Model - 3-Model Darts Ensemble for Future Projections

This module implements a time-series ensemble combining:
- Darts XGBModel (gradient boosting with lags)
- Darts RNNModel with GRU (sequential trajectory learning)
- Darts SKLearnModel wrapping ExtraTrees (proven baseline)

Weighting: 2D adaptive weighting based on:
- Player tier (elite/average/below_average based on recent WAR)
- Consistency (high/medium/low based on coefficient of variation)
- History length (short history uses baseline weights)

Author: Claude Code (Phase 2 Implementation)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import sys

# Darts imports
from darts import TimeSeries
from darts.models import XGBModel, RNNModel
from darts.models.forecasting.sklearn_model import SKLearnModel

# sklearn for ExtraTrees and preprocessing
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from new_pipeline.models.future_season.constants import (
    FUTURE_HITTER_MODEL_FEATURES,
    FUTURE_PITCHER_MODEL_FEATURES
)


def get_baseline_ensemble_weights(player_history_length: int) -> List[float]:
    """
    Get ensemble weights based on player career length (baseline - no HMM).

    Baseline 1D weighting strategy:
    - Veterans (5+ years): Trust RNN trajectory learning equally with XGBoost
    - Mid-career (3-4 years): Favor XGBoost over incomplete RNN patterns
    - Rookies (<3 years): Skip RNN entirely, use cross-sectional models only

    Args:
        player_history_length: Number of consecutive MLB seasons

    Returns:
        [xgboost_weight, rnn_weight, extratrees_weight]
    """
    if player_history_length >= 5:
        # Veterans: Trust RNN trajectory learning
        return [0.35, 0.35, 0.30]
    elif player_history_length >= 3:
        # Mid-career: Blend all three, favor XGBoost
        return [0.45, 0.25, 0.30]
    else:
        # Rookies: Skip RNN, use cross-sectional only
        return [0.55, 0.00, 0.45]


def get_weighted_recent_war(war_history: List[float], decay_lambda: float = 0.5) -> float:
    """
    Calculate weighted average WAR with exponential decay.

    More recent years get higher weight, which smooths out single-year noise
    while still giving most weight to recent performance.

    With decay_lambda=0.5:
    - Last year: weight 1.0
    - 2 years ago: weight 0.61
    - 3 years ago: weight 0.37
    - 4 years ago: weight 0.22

    Args:
        war_history: List of historical WAR values (oldest to newest)
        decay_lambda: Decay rate (higher = faster decay, more weight on recent)

    Returns:
        Exponentially weighted average WAR
    """
    if len(war_history) == 0:
        return 0.0

    n = len(war_history)
    # Calculate weights: most recent year gets highest weight
    # years_ago = 0 for most recent, 1 for one year ago, etc.
    weights = [np.exp(-decay_lambda * (n - 1 - i)) for i in range(n)]
    weights = np.array(weights) / sum(weights)  # Normalize to sum to 1

    return float(np.dot(war_history, weights))


def get_player_tier(weighted_war: float) -> str:
    """
    Classify player into tier based on weighted recent WAR.

    Tiers:
    - elite: 4+ WAR (star-level performance)
    - average: 2-4 WAR (solid regular)
    - below_average: <2 WAR (replacement to below-average)

    Args:
        weighted_war: Exponentially weighted average WAR from get_weighted_recent_war()

    Returns:
        Tier string: 'elite', 'average', or 'below_average'
    """
    if weighted_war >= 4.0:
        return 'elite'
    elif weighted_war >= 2.0:
        return 'average'
    else:
        return 'below_average'


def get_consistency_bucket(war_history: List[float]) -> str:
    """
    Classify player consistency based on coefficient of variation.

    Thresholds based on empirical analysis of MLB WAR data (terciles):
    - high: CV < 0.9 (top third - most stable performers)
    - medium: CV 0.9-1.7 (middle third - typical variation)
    - low: CV > 1.7 (bottom third - volatile performers)

    Args:
        war_history: List of historical WAR values

    Returns:
        Consistency string: 'high', 'medium', or 'low'
    """
    if len(war_history) < 2:
        return 'medium'  # Default for insufficient data

    war_array = np.array(war_history)
    mean_war = np.mean(war_array)

    if mean_war <= 0:
        return 'low'  # Can't calculate CV meaningfully for non-positive mean

    cv = np.std(war_array) / mean_war

    if cv < 0.9:
        return 'high'
    elif cv < 1.7:
        return 'medium'
    else:
        return 'low'


def get_adaptive_ensemble_weights(
    player_history_length: int,
    player_tier: str,
    consistency: str
) -> List[float]:
    """
    Get ensemble weights based on history length, tier, and consistency.

    2D weight matrix rationale:
    - Elite + consistent players: RNN can learn reliable trajectory patterns (0.50)
    - Volatile + below-average: Historical patterns less predictive, skip RNN (0.00)

    Weight range widened from [0.15-0.40] to [0.00-0.50] RNN to allow the
    tier/consistency signal to have meaningful impact on ensemble behavior.

    Args:
        player_history_length: Number of consecutive MLB seasons
        player_tier: 'elite', 'average', or 'below_average'
        consistency: 'high', 'medium', or 'low'

    Returns:
        [xgboost_weight, rnn_weight, extratrees_weight]
    """
    # Short history: use baseline (RNN needs sequences)
    if player_history_length < 3:
        return [0.55, 0.00, 0.45]

    # 2D weight matrix: [XGBoost, RNN, ExtraTrees]
    # Widened RNN weight range: 0.00 (volatile/below-avg) to 0.50 (elite/consistent)
    # This allows tier/consistency to meaningfully differentiate weighting
    weight_matrix = {
        # Elite tier: Trust RNN most for consistent elite players
        ('elite', 'high'): [0.25, 0.50, 0.25],      # RNN learns stable elite trajectories
        ('elite', 'medium'): [0.30, 0.40, 0.30],    # Still trust RNN with some caution
        ('elite', 'low'): [0.40, 0.25, 0.35],       # Volatile elite - less RNN trust

        # Average tier: Moderate RNN trust for consistent players
        ('average', 'high'): [0.30, 0.40, 0.30],    # Consistent average - RNN useful
        ('average', 'medium'): [0.40, 0.30, 0.30],  # Balanced blend
        ('average', 'low'): [0.50, 0.15, 0.35],     # Volatile average - favor XGBoost

        # Below-average tier: Minimal to no RNN trust
        ('below_average', 'high'): [0.40, 0.30, 0.30],   # Consistent below-avg - some RNN
        ('below_average', 'medium'): [0.55, 0.10, 0.35], # Limited RNN value
        ('below_average', 'low'): [0.60, 0.00, 0.40],    # Skip RNN entirely
    }

    return weight_matrix.get((player_tier, consistency), [0.40, 0.30, 0.30])


class XGBoostTimeSeriesModel:
    """
    Darts XGBModel wrapper for longitudinal WAR prediction.

    Uses lagged features from last 3 years to predict next year.
    Hyperparameters inherited from HistGradientBoosting config.
    """

    def __init__(self, player_type: str, lags: Optional[int] = 1):
        """
        Initialize XGBoost time series model.

        Args:
            player_type: 'hitter' or 'pitcher'
            lags: Number of past WAR values to use as features (default: 1)
                  Using lags=1 provides regression-to-mean signal while minimizing
                  redundancy with component stats. Set to None for stats-only forecasting.
        """
        if player_type not in ['hitter', 'pitcher']:
            raise ValueError(f"player_type must be 'hitter' or 'pitcher'")

        self.player_type = player_type
        self.lags = lags

        # Build model kwargs
        model_kwargs = {
            # Time series parameters
            'lags_past_covariates': 3,      # Always use last 3 years of covariates
            'output_chunk_length': 1,       # Predict 1 year ahead

            # XGBoost parameters (inherited from HistGB config)
            'n_estimators': 300,            # From max_iter
            'max_depth': None,              # No depth limit (use max_leaves)
            'max_leaves': 63,               # From max_leaf_nodes
            'learning_rate': 0.03,          # From HistGB

            # Regularization
            'reg_alpha': 0.1,               # L1 regularization
            'reg_lambda': 0.4,              # From l2_regularization

            # Sampling for regularization
            'subsample': 0.8,               # Row sampling per tree
            'colsample_bytree': 0.8,        # Column sampling per tree

            # Reproducibility
            'random_state': 42
        }

        # Only add lags if specified (allows lags=None for component-stats-only)
        if lags is not None and lags > 0:
            model_kwargs['lags'] = lags

        self.model = XGBModel(**model_kwargs)
        self.is_fitted = False

    def train(
        self,
        target_series: List[TimeSeries],
        covariate_series: List[TimeSeries]
    ):
        """
        Train XGBoost model on multiple player time series.

        Args:
            target_series: List of WAR TimeSeries (one per player)
            covariate_series: List of feature TimeSeries (one per player)
        """
        # Darts handles multi-series training automatically
        self.model.fit(
            series=target_series,
            past_covariates=covariate_series
        )
        self.is_fitted = True

    def predict(
        self,
        target_series: TimeSeries,
        covariate_series: TimeSeries,
        n: int = 1
    ) -> float:
        """
        Predict next n years for a single player.

        Args:
            target_series: Player's historical WAR TimeSeries
            covariate_series: Player's historical features TimeSeries
            n: Number of years to predict (default: 1)

        Returns:
            Predicted WAR for next year
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call train() first.")

        # Darts prediction
        prediction = self.model.predict(
            n=n,
            series=target_series,
            past_covariates=covariate_series
        )

        # Extract scalar value
        return prediction.values()[0, 0]


class RNNTimeSeriesModel:
    """
    Darts RNNModel (GRU) for player-specific trajectory learning.

    Requires minimum 3 consecutive seasons. Players with shorter
    histories cannot be predicted by this model.

    Hyperparameters inherited from Keras config (but smaller network
    due to limited data - only ~150 players with 3+ consecutive seasons).
    """

    def __init__(self, player_type: str, min_sequence_length: int = 3):
        """
        Initialize RNN time series model.

        Args:
            player_type: 'hitter' or 'pitcher'
            min_sequence_length: Minimum consecutive seasons required
        """
        if player_type not in ['hitter', 'pitcher']:
            raise ValueError(f"player_type must be 'hitter' or 'pitcher'")

        self.player_type = player_type
        self.min_sequence_length = min_sequence_length
        self.model = RNNModel(
            # Architecture (increased capacity for trajectory learning)
            model='GRU',                 # Faster and more robust than LSTM
            hidden_dim=48,               # Increased from 16 for better trajectory capture
            n_rnn_layers=2,              # Added second layer for deeper patterns
            dropout=0.35,                # Increased regularization for larger network

            # Time series parameters
            input_chunk_length=2,        # Use last 2 years (reduced from 3 due to RNN minimum requirements)
            training_length=3,           # Minimum training sequence length

            # Training (adjusted for larger network)
            n_epochs=150,                # Increased from 100 for convergence
            batch_size=16,
            optimizer_kwargs={'lr': 0.001},

            # Reproducibility
            random_state=42,

            # Logging
            pl_trainer_kwargs={
                'accelerator': 'cpu',
                'enable_progress_bar': False
            }
        )
        self.is_fitted = False
        self.trainable_players = []

    def train(
        self,
        target_series: List[TimeSeries],
        covariate_series: List[TimeSeries]
    ):
        """
        Train RNN on players with sufficient history.

        RNN learns trajectory patterns from WAR history only (no covariates).
        RNNModel supports future_covariates but not past_covariates.

        Args:
            target_series: List of WAR TimeSeries (one per player)
            covariate_series: List of feature TimeSeries (not used for RNN)
        """
        # Filter to series with min_sequence_length
        valid_series = []

        for i, ts in enumerate(target_series):
            if len(ts) >= self.min_sequence_length:
                valid_series.append(ts)
                self.trainable_players.append(i)

        if len(valid_series) == 0:
            print("WARNING: No players with sufficient sequence length for RNN")
            self.is_fitted = False
            return

        print(f"Training RNN on {len(valid_series)} players with {self.min_sequence_length}+ seasons")
        print(f"  (RNN learns WAR trajectories only, no covariates)")

        # Train on valid series (no covariates - RNN learns from WAR history only)
        self.model.fit(series=valid_series)
        self.is_fitted = True

    def predict(
        self,
        target_series: TimeSeries,
        covariate_series: TimeSeries,
        n: int = 1
    ) -> Optional[float]:
        """
        Predict next n years for a single player.

        Args:
            target_series: Player's historical WAR TimeSeries
            covariate_series: Player's historical features TimeSeries (not used)
            n: Number of years to predict

        Returns:
            Predicted WAR or None if player has insufficient history
        """
        if not self.is_fitted:
            return None

        # Check sequence length
        if len(target_series) < self.min_sequence_length:
            return None

        # Predict (no covariates - RNN uses WAR history only)
        try:
            prediction = self.model.predict(n=n, series=target_series)
            return prediction.values()[0, 0]
        except Exception as e:
            print(f"RNN prediction failed: {str(e)}")
            return None


class ExtraTreesTimeSeriesModel:
    """
    Darts SKLearnModel wrapping ExtraTreesRegressor.

    Uses proven current season configuration, adapted for time series.
    Exact hyperparameters from current_season/hitter_ensemble.py.
    """

    def __init__(self, player_type: str):
        """
        Initialize ExtraTrees time series model.

        Args:
            player_type: 'hitter' or 'pitcher'
        """
        if player_type not in ['hitter', 'pitcher']:
            raise ValueError(f"player_type must be 'hitter' or 'pitcher'")

        self.player_type = player_type
        self.model = SKLearnModel(
            # Time series parameters
            lags=3,                      # Use last 3 years as features
            lags_past_covariates=3,      # Use last 3 years of covariates
            output_chunk_length=1,       # Predict 1 year ahead

            # Proven ExtraTrees config (from current_season/hitter_ensemble.py)
            model=ExtraTreesRegressor(
                n_estimators=100,
                max_depth=10,
                criterion='friedman_mse',
                max_features='sqrt',
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1
            )
        )
        self.is_fitted = False

    def train(
        self,
        target_series: List[TimeSeries],
        covariate_series: List[TimeSeries]
    ):
        """
        Train ExtraTrees model on multiple player time series.

        Args:
            target_series: List of WAR TimeSeries (one per player)
            covariate_series: List of feature TimeSeries (one per player)
        """
        self.model.fit(
            series=target_series,
            past_covariates=covariate_series
        )
        self.is_fitted = True

    def predict(
        self,
        target_series: TimeSeries,
        covariate_series: TimeSeries,
        n: int = 1
    ) -> float:
        """
        Predict next n years for a single player.

        Args:
            target_series: Player's historical WAR TimeSeries
            covariate_series: Player's historical features TimeSeries
            n: Number of years to predict

        Returns:
            Predicted WAR for next year
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call train() first.")

        prediction = self.model.predict(
            n=n,
            series=target_series,
            past_covariates=covariate_series
        )

        return prediction.values()[0, 0]


class ExtraTreesFallbackModel:
    """
    ExtraTrees fallback model for short-history players (<4 seasons).

    Non-Darts sklearn-based ExtraTrees for players with insufficient
    history for Darts models. Uses same hyperparameters as current season
    ExtraTrees for consistency.

    This is a global model trained on all players, learning cross-player
    patterns to make informed predictions for short-history players.
    """

    def __init__(self, player_type: str):
        """
        Initialize ExtraTrees fallback model.

        Args:
            player_type: 'hitter' or 'pitcher'
        """
        if player_type not in ['hitter', 'pitcher']:
            raise ValueError(f"player_type must be 'hitter' or 'pitcher'")

        self.player_type = player_type
        self.model_features = FUTURE_HITTER_MODEL_FEATURES if player_type == 'hitter' else FUTURE_PITCHER_MODEL_FEATURES

        # Same hyperparameters as current season ExtraTrees (from hitter_ensemble.py)
        self.model = ExtraTreesRegressor(
            n_estimators=100,
            max_depth=10,
            criterion='friedman_mse',
            max_features='sqrt',
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        )

        self.scaler = StandardScaler()
        self.is_fitted = False

    def prepare_features(self, sequences_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract features and target from sequences dataframe.

        Args:
            sequences_df: DataFrame with year-to-year sequences

        Returns:
            (X, y) where:
                X: Features (n_samples, n_features)
                y: Target WAR values (n_samples,) or None if not available
        """

        # Model features from Year N
        feature_cols_n = [f'{feat}_n' for feat in self.model_features]

        # Age/career context features
        context_features = [
            'age_n',
            'age_squared',
            'years_from_peak',
            'age_group_young',
            'age_group_prime',
            'age_group_veteran',
            'war_n',
            'career_war',
            'seasons_played',
            'peak_war',
            'peak_percentage'
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
        Train ExtraTrees fallback model on training sequences.

        Args:
            train_df: Training sequences from TimeSeries conversion

        Returns:
            Training metrics: {'r2': float, 'rmse': float, 'mae': float}
        """
        print(f"Training {self.player_type} ExtraTrees fallback model...")
        print(f"  Sequences: {len(train_df)}")

        # Prepare features
        X_train, y_train = self.prepare_features(train_df)

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)

        print(f"  Features: {X_train_scaled.shape[1]}")
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

        X_test, _ = self.prepare_features(test_df)
        X_test_scaled = self.scaler.transform(X_test)

        return self.model.predict(X_test_scaled)


class EnsembleLongitudinalModel:
    """
    Ensemble of 3 Darts models with adaptive weighting + ExtraTrees fallback for short histories.

    2D Adaptive Weighting: Weights based on (player_tier, consistency) matrix.

    Models:
        1. XGBoost - Gradient boosting with lagged features (requires 4+ seasons)
        2. RNN (GRU) - Sequential trajectory learning (requires 3+ seasons)
        3. ExtraTrees (Darts) - Conservative baseline (requires 4+ seasons)
        4. ExtraTrees (Fallback) - For players with <4 seasons

    Weighting Strategy (2D matrix by tier and consistency):
        - Elite + consistent: [0.30, 0.40, 0.30] - Trust RNN trajectory learning
        - Volatile + below-average: [0.55, 0.15, 0.30] - Favor XGBoost cross-sectional
        - Short history (<3 seasons): [0.55, 0.00, 0.45] - Skip RNN
        - Very short history (<4 seasons): ExtraTrees fallback only
    """

    def __init__(self, player_type: str, xgboost_lags: Optional[int] = 1):
        """
        Initialize ensemble model.

        Args:
            player_type: 'hitter' or 'pitcher'
            xgboost_lags: Number of past WAR values for XGBoost to use (default: 1)
                         Default of 1 balances regression-to-mean with component stats forecasting.
                         Set to None for pure component-stats-only predictions.
        """
        if player_type not in ['hitter', 'pitcher']:
            raise ValueError(f"player_type must be 'hitter' or 'pitcher'")

        self.player_type = player_type
        self.xgboost_lags = xgboost_lags

        # Initialize all 3 Darts models
        self.xgboost_model = XGBoostTimeSeriesModel(player_type, lags=xgboost_lags)
        self.rnn_model = RNNTimeSeriesModel(player_type, min_sequence_length=3)
        self.extratrees_model = ExtraTreesTimeSeriesModel(player_type)

        # Initialize ExtraTrees fallback for players with insufficient Darts history
        self.fallback_model = ExtraTreesFallbackModel(player_type)

        # Darts models require lags + output_chunk_length minimum timesteps
        self.min_darts_history = 4  # lags=3 + output=1

        self.is_fitted = False

    def _convert_timeseries_to_sequences(
        self,
        target_series: List[TimeSeries],
        covariate_series: List[TimeSeries]
    ) -> pd.DataFrame:
        """
        Convert TimeSeries format to sequences_df format for fallback model.

        Creates year-to-year sequences (Year N → Year N+1) from TimeSeries.

        Args:
            target_series: List of WAR TimeSeries
            covariate_series: List of feature TimeSeries

        Returns:
            DataFrame with sequences (same format as build_longitudinal_sequences())
        """
        sequences = []
        model_features = FUTURE_HITTER_MODEL_FEATURES if self.player_type == 'hitter' else FUTURE_PITCHER_MODEL_FEATURES

        for i, (target_ts, cov_ts) in enumerate(zip(target_series, covariate_series)):
            # Get values as arrays
            war_values = target_ts.values().flatten()

            # Handle both datetime and integer indices
            time_index = target_ts.time_index
            if hasattr(time_index, 'year'):
                # DatetimeIndex
                years = time_index.year.values
            else:
                # Integer index (just year values)
                years = time_index.values

            # Need at least 2 consecutive years for a sequence
            if len(war_values) < 2:
                continue

            # Extract covariate values (features + Age)
            cov_df = cov_ts.pd_dataframe() if hasattr(cov_ts, 'pd_dataframe') else cov_ts.to_dataframe()

            # Create sequences for each consecutive year pair
            for j in range(len(war_values) - 1):
                year_n = int(years[j])
                year_n_plus_1 = int(years[j + 1])

                # Check consecutive
                if year_n_plus_1 != year_n + 1:
                    continue

                # Build sequence
                sequence = {
                    'playerid': i,  # Use index as playerid
                    'year_n': int(year_n),
                    'year_n_plus_1': int(year_n_plus_1),
                    'war_n': float(war_values[j]),
                    'war_n_plus_1': float(war_values[j + 1])
                }

                # Get age from covariates
                if 'Age' in cov_df.columns:
                    sequence['age_n'] = float(cov_df.iloc[j]['Age'])
                else:
                    sequence['age_n'] = 25.0  # Default fallback

                # Add features from Year N
                for feat in model_features:
                    if feat in cov_df.columns:
                        sequence[f'{feat}_n'] = float(cov_df.iloc[j][feat])

                sequences.append(sequence)

        sequences_df = pd.DataFrame(sequences)

        # Add age context features (from data_preparation.py)
        if len(sequences_df) > 0:
            sequences_df['age_squared'] = sequences_df['age_n'] ** 2
            peak_age = 27 if self.player_type == 'pitcher' else 28
            sequences_df['years_from_peak'] = sequences_df['age_n'] - peak_age
            sequences_df['age_group_young'] = (sequences_df['age_n'] < 26).astype(int)
            sequences_df['age_group_prime'] = ((sequences_df['age_n'] >= 26) & (sequences_df['age_n'] <= 30)).astype(int)
            sequences_df['age_group_veteran'] = (sequences_df['age_n'] > 30).astype(int)

            # Career context features - compute from sequences
            sequences_df['career_war'] = sequences_df.groupby('playerid')['war_n'].cumsum()
            sequences_df['seasons_played'] = sequences_df.groupby('playerid').cumcount() + 1
            sequences_df['peak_war'] = sequences_df.groupby('playerid')['war_n'].cummax()
            sequences_df['peak_percentage'] = sequences_df['war_n'] / sequences_df['peak_war'].clip(lower=0.1)
            sequences_df['peak_percentage'] = sequences_df['peak_percentage'].clip(upper=2.0)

        return sequences_df

    def _convert_single_timeseries_to_row(
        self,
        target_series: TimeSeries,
        covariate_series: TimeSeries
    ) -> pd.DataFrame:
        """
        Convert single player TimeSeries to single-row DataFrame for fallback prediction.

        Uses the most recent year as Year N to predict Year N+1.

        Args:
            target_series: Player's WAR TimeSeries
            covariate_series: Player's feature TimeSeries

        Returns:
            Single-row DataFrame with columns matching sequences_df format
        """
        model_features = FUTURE_HITTER_MODEL_FEATURES if self.player_type == 'hitter' else FUTURE_PITCHER_MODEL_FEATURES

        # Get most recent year data
        war_values = target_series.values().flatten()

        # Handle both datetime and integer indices
        time_index = target_series.time_index
        if hasattr(time_index, 'year'):
            years = time_index.year.values
        else:
            years = time_index.values

        cov_df = covariate_series.pd_dataframe() if hasattr(covariate_series, 'pd_dataframe') else covariate_series.to_dataframe()

        # Use last available year as Year N
        idx = len(war_values) - 1
        year_n = int(years[idx])

        # Build row
        row = {
            'playerid': 0,  # Dummy ID for single prediction
            'year_n': year_n,
            'year_n_plus_1': year_n + 1,
            'war_n': float(war_values[idx])
        }

        # Get age
        if 'Age' in cov_df.columns:
            age_n = float(cov_df.iloc[idx]['Age'])
        else:
            age_n = 25.0  # Default fallback

        row['age_n'] = age_n

        # Add features from Year N
        for feat in model_features:
            if feat in cov_df.columns:
                row[f'{feat}_n'] = float(cov_df.iloc[idx][feat])
            else:
                row[f'{feat}_n'] = 0.0  # Fallback

        # Create DataFrame
        row_df = pd.DataFrame([row])

        # Add age context features
        row_df['age_squared'] = row_df['age_n'] ** 2
        peak_age = 27 if self.player_type == 'pitcher' else 28
        row_df['years_from_peak'] = row_df['age_n'] - peak_age
        row_df['age_group_young'] = (row_df['age_n'] < 26).astype(int)
        row_df['age_group_prime'] = ((row_df['age_n'] >= 26) & (row_df['age_n'] <= 30)).astype(int)
        row_df['age_group_veteran'] = (row_df['age_n'] > 30).astype(int)

        # Career context features - fallback values for single-row prediction
        # (Single row doesn't have full history, use current year as proxy)
        row_df['career_war'] = row_df['war_n']
        row_df['seasons_played'] = 1
        row_df['peak_war'] = row_df['war_n'].clip(lower=0.1)
        row_df['peak_percentage'] = 1.0

        return row_df

    def train(
        self,
        target_series: List[TimeSeries],
        covariate_series: List[TimeSeries]
    ) -> Dict[str, Dict[str, any]]:
        """
        Train all models (Darts + fallback) on player time series.

        Darts models (XGBoost, RNN, ExtraTrees) are trained only on players
        with 4+ seasons. Fallback model (RandomForest) is trained on all players.

        Args:
            target_series: List of WAR TimeSeries (one per player)
            covariate_series: List of feature TimeSeries (one per player)

        Returns:
            Training metrics for each model
        """
        print("Training ensemble models with Darts...")

        # Filter to players with sufficient history for Darts models
        darts_target_series = []
        darts_covariate_series = []
        short_history_count = 0

        for i, (ts, cov) in enumerate(zip(target_series, covariate_series)):
            if len(ts) >= self.min_darts_history:
                darts_target_series.append(ts)
                darts_covariate_series.append(cov)
            else:
                short_history_count += 1

        print(f"  Players with 4+ seasons: {len(darts_target_series)} (Darts-eligible)")
        print(f"  Players with <4 seasons: {short_history_count} (fallback only)")

        # Train Darts models on filtered data
        print("  Training XGBoost (Darts XGBModel)...")
        self.xgboost_model.train(darts_target_series, darts_covariate_series)

        print("  Training RNN (Darts RNNModel with GRU)...")
        self.rnn_model.train(darts_target_series, darts_covariate_series)

        print("  Training ExtraTrees (Darts SKLearnModel)...")
        self.extratrees_model.train(darts_target_series, darts_covariate_series)

        # Train fallback model on ALL players (convert TimeSeries to sequences_df)
        print("  Training Fallback (RandomForest) on all players...")
        sequences_df = self._convert_timeseries_to_sequences(target_series, covariate_series)
        print(f"    Converted {len(sequences_df)} sequences from {len(target_series)} players")

        fallback_metrics = self.fallback_model.train(sequences_df)

        self.is_fitted = True
        print("  Ensemble training complete")

        return {
            'xgboost': {'trained': self.xgboost_model.is_fitted},
            'rnn': {
                'trained': self.rnn_model.is_fitted,
                'trainable_players': len(self.rnn_model.trainable_players)
            },
            'extratrees': {'trained': self.extratrees_model.is_fitted},
            'fallback': {
                'trained': self.fallback_model.is_fitted,
                'r2': fallback_metrics['r2'],
                'rmse': fallback_metrics['rmse'],
                'n_samples': fallback_metrics['n_samples']
            },
            'short_history_players': short_history_count
        }

    def predict(
        self,
        target_series: TimeSeries,
        covariate_series: TimeSeries,
        player_history_length: int
    ) -> float:
        """
        Generate ensemble prediction with adaptive weighting.

        Routes to fallback model if player has <4 seasons (insufficient for Darts).
        Otherwise uses Darts ensemble with adaptive weights.

        Args:
            target_series: Player's historical WAR TimeSeries
            covariate_series: Player's historical features TimeSeries
            player_history_length: Number of consecutive seasons in history

        Returns:
            Ensemble prediction (weighted average of Darts models or fallback)
        """
        if not self.is_fitted:
            raise RuntimeError("Ensemble not fitted. Call train() first.")

        # Check if player has sufficient history for Darts models
        if len(target_series) < self.min_darts_history:
            # Route to fallback model
            row_df = self._convert_single_timeseries_to_row(target_series, covariate_series)
            fallback_pred = self.fallback_model.predict(row_df)[0]
            return fallback_pred

        # Use Darts ensemble for players with 4+ seasons
        # Get predictions from each model
        xgb_pred = self.xgboost_model.predict(target_series, covariate_series)
        rnn_pred = self.rnn_model.predict(target_series, covariate_series)  # May be None
        et_pred = self.extratrees_model.predict(target_series, covariate_series)

        # Extract WAR history for tier/consistency calculation
        war_values = target_series.values().flatten()
        war_history = list(war_values)

        # Use exponentially weighted WAR for tier (smooths single-year noise)
        weighted_war = get_weighted_recent_war(war_history)

        # Get adaptive weights (2D - tier and consistency)
        player_tier = get_player_tier(weighted_war)
        consistency = get_consistency_bucket(war_history)
        weights = get_adaptive_ensemble_weights(player_history_length, player_tier, consistency)

        # Compute weighted ensemble
        if rnn_pred is not None:
            # All models available
            ensemble_pred = (weights[0] * xgb_pred +
                           weights[1] * rnn_pred +
                           weights[2] * et_pred)
        else:
            # RNN not available (insufficient history), renormalize weights
            total_weight = weights[0] + weights[2]
            ensemble_pred = ((weights[0] / total_weight) * xgb_pred +
                           (weights[2] / total_weight) * et_pred)

        return ensemble_pred

    def get_model_contributions(
        self,
        target_series: TimeSeries,
        covariate_series: TimeSeries,
        player_history_length: int
    ) -> Dict[str, float]:
        """
        Get individual model predictions for analysis.

        Args:
            target_series: Player's historical WAR TimeSeries
            covariate_series: Player's historical features TimeSeries
            player_history_length: Number of consecutive seasons in history

        Returns:
            Dict with individual predictions and ensemble result
        """
        if not self.is_fitted:
            raise RuntimeError("Ensemble not fitted. Call train() first.")

        # Check if using fallback
        if len(target_series) < self.min_darts_history:
            row_df = self._convert_single_timeseries_to_row(target_series, covariate_series)
            fallback_pred = self.fallback_model.predict(row_df)[0]

            return {
                'xgboost_pred': None,
                'rnn_pred': None,
                'extratrees_pred': None,
                'fallback_pred': fallback_pred,
                'ensemble_pred': fallback_pred,
                'weights': [0.0, 0.0, 0.0],  # Fallback used
                'used_fallback': True
            }

        # Use Darts models
        xgb_pred = self.xgboost_model.predict(target_series, covariate_series)
        rnn_pred = self.rnn_model.predict(target_series, covariate_series)
        et_pred = self.extratrees_model.predict(target_series, covariate_series)

        ensemble_pred = self.predict(target_series, covariate_series, player_history_length)

        # Extract WAR history for tier/consistency calculation
        war_values = target_series.values().flatten()
        war_history = list(war_values)

        # Use exponentially weighted WAR for tier (smooths single-year noise)
        weighted_war = get_weighted_recent_war(war_history)

        # Get adaptive weights (2D - tier and consistency)
        player_tier = get_player_tier(weighted_war)
        consistency = get_consistency_bucket(war_history)
        weights = get_adaptive_ensemble_weights(player_history_length, player_tier, consistency)

        return {
            'xgboost_pred': xgb_pred,
            'rnn_pred': rnn_pred,  # May be None
            'extratrees_pred': et_pred,
            'fallback_pred': None,
            'ensemble_pred': ensemble_pred,
            'weights': weights,
            'player_tier': player_tier,
            'consistency': consistency,
            'weighted_war': weighted_war,
            'used_fallback': False
        }

    def _build_features_from_dataframe_row(
        self,
        player_row: pd.Series
    ) -> pd.DataFrame:
        """
        Build fallback model features from a single DataFrame row.

        Converts most recent season's data to the format expected by fallback model.

        Args:
            player_row: Single row from historical DataFrame with current year stats

        Returns:
            Single-row DataFrame with fallback model features
        """
        model_features = FUTURE_HITTER_MODEL_FEATURES if self.player_type == 'hitter' else FUTURE_PITCHER_MODEL_FEATURES

        # Build row
        row = {
            'playerid': int(player_row.get('playerid', 0)),
            'year_n': int(player_row.get('Year', 2023)),
            'year_n_plus_1': int(player_row.get('Year', 2023)) + 1,
            'war_n': float(player_row.get('WAR', 0.0))
        }

        # Get age
        age_n = float(player_row.get('Age', 25.0))
        row['age_n'] = age_n

        # Add features from current year
        for feat in model_features:
            if feat in player_row.index:
                row[f'{feat}_n'] = float(player_row[feat])
            else:
                row[f'{feat}_n'] = 0.0  # Fallback

        # Create DataFrame
        row_df = pd.DataFrame([row])

        # Add age context features
        row_df['age_squared'] = row_df['age_n'] ** 2
        peak_age = 27 if self.player_type == 'pitcher' else 28
        row_df['years_from_peak'] = row_df['age_n'] - peak_age
        row_df['age_group_young'] = (row_df['age_n'] < 26).astype(int)
        row_df['age_group_prime'] = ((row_df['age_n'] >= 26) & (row_df['age_n'] <= 30)).astype(int)
        row_df['age_group_veteran'] = (row_df['age_n'] > 30).astype(int)

        # Career context features - fallback values for single DataFrame row
        # (Single row doesn't have full history, use current year as proxy)
        row_df['career_war'] = row_df['war_n']
        row_df['seasons_played'] = 1
        row_df['peak_war'] = row_df['war_n'].clip(lower=0.1)
        row_df['peak_percentage'] = 1.0

        return row_df

    def predict_from_dataframe(
        self,
        player_history_df: pd.DataFrame
    ) -> float:
        """
        Predict using raw player history DataFrame (production-ready method).

        Handles both consecutive and non-consecutive years:
        - Tries TimeSeries creation for consecutive history with 4+ seasons
        - Falls back to most recent year features for:
          - Non-consecutive years (gaps in career)
          - <4 seasons of history
          - TimeSeries creation failures

        Args:
            player_history_df: Player's historical seasons sorted by Year
                              Must have columns: Year, WAR, Age, and model features

        Returns:
            Predicted WAR for next season

        Raises:
            ValueError: If player_history_df is empty
        """
        if len(player_history_df) == 0:
            raise ValueError("Cannot predict from empty history")

        if not self.is_fitted:
            raise RuntimeError("Ensemble not fitted. Call train() first.")

        model_features = FUTURE_HITTER_MODEL_FEATURES if self.player_type == 'hitter' else FUTURE_PITCHER_MODEL_FEATURES

        # Sort by year to ensure correct ordering
        player_history_df = player_history_df.sort_values('Year').copy()

        # Try creating TimeSeries for Darts models (requires consecutive years + 4+ seasons)
        try:
            # Check for consecutive years
            years = player_history_df['Year'].values
            if len(years) >= 4 and np.all(np.diff(years) == 1):
                # Consecutive years with sufficient history - use Darts ensemble
                target_series = TimeSeries.from_dataframe(
                    df=player_history_df,
                    time_col='Year',
                    value_cols=['WAR'],
                    fill_missing_dates=False
                )

                covariate_cols = [c for c in model_features + ['Age'] if c in player_history_df.columns]
                covariate_series = TimeSeries.from_dataframe(
                    df=player_history_df,
                    time_col='Year',
                    value_cols=covariate_cols,
                    fill_missing_dates=False
                )

                return self.predict(target_series, covariate_series, len(target_series))

        except (ValueError, Exception):
            # TimeSeries creation failed - fall through to fallback
            pass

        # Route to fallback using most recent year's features
        most_recent = player_history_df.iloc[-1]
        row_df = self._build_features_from_dataframe_row(most_recent)
        return self.fallback_model.predict(row_df)[0]
