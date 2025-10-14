"""
Darts Temporal Models for ROS Prediction

Implements TCN, TSMixer, and AutoARIMA using Darts library.
Combines into temporal ensemble (40% weight).

IMPROVEMENTS from reference implementation:

TCN:
- num_filters: 32 -> 8 (reference was unjustified 10x increase from default)
- num_layers: 3 -> 2 (shallower for limited ROS data)
- weight_norm: False -> True (stabilize training)
- n_epochs: 100 -> 200 (allow more training)
- batch_size: 32 -> 16 (smaller for limited data)
- Added AdamW optimizer with weight_decay=0.01
- Added ReduceLROnPlateau scheduler
- Added bias tracking metrics
- Silent training (matches current season)

TSMixer:
- activation: 'ReLU' -> 'GELU' (transformer standard, like Swish in Keras)
- n_epochs: 100 -> 200
- batch_size: 32 -> 16
- Added AdamW optimizer with weight_decay=0.01
- Added ReduceLROnPlateau scheduler
- Added bias tracking metrics
- Silent training

AutoARIMA:
- seasonal: True -> False (CRITICAL FIX - ROS is not seasonal)
- max_p: 5 -> 3 (tighter search for baseball)
- max_q: 5 -> 3
- max_P: 2 -> 0 (no seasonal component)
- max_Q: 2 -> 0
- start_p: 2 -> 1 (start simpler)
- start_q: 2 -> 1
- max_order: 5 -> 4
- Added random_state=42 for reproducibility
"""

import numpy as np
import pandas as pd
import torch
from typing import List, Optional, Dict
from darts import TimeSeries
from darts.models import TCNModel, TSMixerModel, AutoARIMA
from pytorch_lightning.callbacks import EarlyStopping
from .base import BaseEnsemble
from .metrics import get_ros_metrics

# Optimize RTX 4080 Tensor Cores performance
torch.set_float32_matmul_precision('high')


class DartsTemporalEnsemble(BaseEnsemble):
    """
    Ensemble of temporal forecasting models using Darts.

    Components:
    - TCN (Temporal Convolutional Network) - 50% of temporal weight
    - TSMixer (Transformer) - 30% of temporal weight
    - AutoARIMA (Statistical) - 20% of temporal weight

    All models configured with:
    - AdamW optimizer (weight_decay=0.01, like Keras current season)
    - ReduceLROnPlateau scheduler
    - Bias tracking metrics
    - Silent training (no epoch-by-epoch output)
    - Random state for reproducibility
    """

    def __init__(
        self,
        player_type: str = 'hitter',
        component_weights: Optional[List[float]] = None,
        tcn_params: Optional[Dict] = None,
        tsmixer_params: Optional[Dict] = None,
        arima_params: Optional[Dict] = None,
        elite_threshold: float = 4.0,
        random_state: int = 42
    ):
        """
        Initialize temporal ensemble.

        Args:
            player_type: 'hitter' or 'pitcher'
            component_weights: Weights for [TCN, TSMixer, AutoARIMA]
                Default: [0.5, 0.3, 0.2]
            tcn_params: Optional TCN hyperparameters (overrides defaults)
            tsmixer_params: Optional TSMixer hyperparameters (overrides defaults)
            arima_params: Optional AutoARIMA hyperparameters (overrides defaults)
            elite_threshold: WAR threshold for elite bias tracking
            random_state: Random seed for reproducibility
        """
        if component_weights is None:
            component_weights = [0.5, 0.3, 0.2]

        super().__init__(player_type, weights=component_weights)

        self.random_state = random_state
        self.elite_threshold = elite_threshold

        # Setup metrics and callbacks
        metrics = get_ros_metrics(elite_threshold=elite_threshold)

        early_stopping = EarlyStopping(
            monitor='train_loss',
            patience=25,
            min_delta=1e-4,
            mode='min',
            verbose=True  # Print when stopping
        )

        # TCN parameters
        if tcn_params is None:
            tcn_params = {
                # Architecture
                'input_chunk_length': 3,
                'output_chunk_length': 1,
                'kernel_size': 2,  # Must be < input_chunk_length (Darts validation requirement)
                'num_filters': 8,
                'num_layers': 2,
                'dilation_base': 2,
                'weight_norm': True,
                'dropout': 0.2,

                # Training
                'n_epochs': 200,
                'batch_size': 16,

                # Loss & Optimization
                'loss_fn': torch.nn.MSELoss(),
                'torch_metrics': metrics,
                'optimizer_cls': torch.optim.AdamW,
                'optimizer_kwargs': {
                    'lr': 0.001,
                    'weight_decay': 0.01,
                    'eps': 1e-8
                },
                'lr_scheduler_cls': torch.optim.lr_scheduler.ReduceLROnPlateau,
                'lr_scheduler_kwargs': {
                    'monitor': 'train_loss',
                    'mode': 'min',
                    'factor': 0.5,
                    'patience': 10,
                    'min_lr': 1e-6
                },

                # PyTorch Lightning
                'pl_trainer_kwargs': {
                    'gradient_clip_val': 1.0,
                    'accelerator': 'auto',
                    'enable_progress_bar': False,
                    'enable_model_summary': False,
                    'callbacks': [early_stopping],
                    'logger': False
                },

                # Reproducibility & Saving
                'random_state': random_state,
                'save_checkpoints': False,  # Don't save checkpoints automatically
                'force_reset': True,
                'log_tensorboard': False
            }

        # TSMixer parameters
        if tsmixer_params is None:
            tsmixer_params = {
                # Architecture
                'input_chunk_length': 3,
                'output_chunk_length': 1,
                'hidden_size': 64,
                'num_blocks': 2,
                'ff_size': 64,
                'dropout': 0.1,
                'activation': 'GELU',
                'norm_type': 'LayerNorm',

                # Training
                'n_epochs': 200,
                'batch_size': 16,

                # Loss & Optimization
                'loss_fn': torch.nn.MSELoss(),
                'torch_metrics': metrics,
                'optimizer_cls': torch.optim.AdamW,
                'optimizer_kwargs': {
                    'lr': 0.001,
                    'weight_decay': 0.01,
                    'eps': 1e-8,
                    'betas': (0.9, 0.999)
                },
                'lr_scheduler_cls': torch.optim.lr_scheduler.ReduceLROnPlateau,
                'lr_scheduler_kwargs': {
                    'monitor': 'train_loss',
                    'mode': 'min',
                    'factor': 0.5,
                    'patience': 10,
                    'min_lr': 1e-6
                },

                # PyTorch Lightning
                'pl_trainer_kwargs': {
                    'accelerator': 'auto',
                    'gradient_clip_val': 1.0,
                    'enable_progress_bar': False,
                    'enable_model_summary': False,
                    'callbacks': [early_stopping],
                    'logger': False
                },

                # Reproducibility & Saving
                'random_state': random_state,
                'save_checkpoints': False,  # Don't save checkpoints automatically
                'force_reset': True,
                'log_tensorboard': False
            }

        # AutoARIMA parameters
        if arima_params is None:
            arima_params = {
                # Seasonal (CRITICAL: False for ROS)
                'seasonal': False,
                'season_length': 1,

                # Search space (tightened for baseball)
                'max_p': 3,
                'max_q': 3,
                'max_P': 0,
                'max_Q': 0,
                'start_p': 1,
                'start_q': 1,
                'max_order': 4,
                'max_d': 2,
                'max_D': 1,

                # Optimization
                'ic': 'aicc',
                'stepwise': True,
                'approximation': False,

                # Stationarity
                'stationary': False,
                'test': 'kpss',
                'allowdrift': True,
                'allowmean': True,

                # Reproducibility
                'random_state': random_state
            }

        # Initialize models
        self.tcn = TCNModel(**tcn_params)
        self.tsmixer = TSMixerModel(**tsmixer_params)
        self.arima_params = arima_params  # Store for per-series fitting

        self.component_models = [self.tcn, self.tsmixer]

    def fit(
        self,
        series_list: List[TimeSeries],
        past_covariates_list: Optional[List[TimeSeries]] = None,
        static_covariates: Optional[pd.DataFrame] = None
    ) -> 'DartsTemporalEnsemble':
        """
        Fit all temporal models.

        Args:
            series_list: List of target time series (one per player)
            past_covariates_list: List of covariate time series (optional)
            static_covariates: Static covariates DataFrame (optional)

        Returns:
            self (fitted ensemble)

        Example:
            >>> # Create time series for each player
            >>> series_list = []
            >>> for player_id in player_ids:
            ...     player_data = historical[historical['playerid'] == player_id]
            ...     ts = TimeSeries.from_dataframe(
            ...         player_data,
            ...         time_col='Year',
            ...         value_cols='WAR_per_600'
            ...     )
            ...     series_list.append(ts)
            >>>
            >>> ensemble.fit(series_list)
        """
        # Fit TCN
        self.tcn.fit(
            series=series_list,
            past_covariates=past_covariates_list
        )

        # Fit TSMixer
        self.tsmixer.fit(
            series=series_list,
            past_covariates=past_covariates_list
        )

        # Fit AutoARIMA (on each series individually)
        # Note: AutoARIMA requires 10+ years, will skip players with less
        self.arima_models = []
        skipped_count = 0
        for series in series_list:
            try:
                arima = AutoARIMA(**self.arima_params)
                arima.fit(series)
                self.arima_models.append(arima)
            except ValueError as e:
                # Skip if insufficient data (< 10 years for AutoARIMA)
                if "requires at least" in str(e):
                    self.arima_models.append(None)
                    skipped_count += 1
                else:
                    raise  # Re-raise if it's a different error

        if skipped_count > 0:
            print(f"    Note: AutoARIMA skipped for {skipped_count}/{len(series_list)} players (<10 years data)")

        self.is_fitted = True
        return self

    def predict(
        self,
        n: int = 1,
        series: Optional[TimeSeries] = None,
        past_covariates: Optional[TimeSeries] = None
    ) -> np.ndarray:
        """
        Generate ensemble prediction.

        Args:
            n: Forecast horizon (default: 1 for ROS)
            series: Historical series for prediction
            past_covariates: Covariates for prediction period

        Returns:
            Ensemble predictions (n_samples,)
        """
        if not self.is_fitted:
            raise ValueError("Ensemble not fitted. Call fit() first.")

        # Get predictions from each model
        tcn_pred = self.tcn.predict(n=n, series=series, past_covariates=past_covariates)
        tsmixer_pred = self.tsmixer.predict(n=n, series=series, past_covariates=past_covariates)

        # ARIMA prediction (only if model exists for this series)
        # For batch prediction, match series to fitted models
        # Simplified: use first ARIMA model if available
        arima_pred = None
        if self.arima_models and self.arima_models[0] is not None:
            arima_pred = self.arima_models[0].predict(n=n)

        # Extract values and combine
        if arima_pred is not None:
            predictions = [
                tcn_pred.values().flatten(),
                tsmixer_pred.values().flatten(),
                arima_pred.values().flatten()
            ]
            # Use full ensemble weights [0.5, 0.3, 0.2]
            return self.weighted_average(predictions)
        else:
            # No ARIMA - use TCN and TSMixer only
            predictions = [
                tcn_pred.values().flatten(),
                tsmixer_pred.values().flatten()
            ]
            # Renormalize weights: 0.5/(0.5+0.3) = 0.625, 0.3/(0.5+0.3) = 0.375
            tcn_weight = self.weights[0] / (self.weights[0] + self.weights[1])
            tsmixer_weight = self.weights[1] / (self.weights[0] + self.weights[1])
            return self.weighted_average(predictions, weights=[tcn_weight, tsmixer_weight])
