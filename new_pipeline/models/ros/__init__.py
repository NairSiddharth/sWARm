"""
ROS (Rest of Season) Prediction Models
"""

# Base classes
from .base import BaseROSModel, BaseEnsemble

# Component models
from .quantile_model import MultiQuantileHistGB
from .direct_forecaster import DirectROSForecaster
from .temporal_models import DartsTemporalEnsemble

# Complete ensembles
from .hitter_ros import HitterROSEnsemble, ROS_HITTER_FEATURES
from .pitcher_ros import PitcherROSEnsemble, ROS_PITCHER_FEATURES

# Metrics
from .metrics import MeanError, EliteBias, EliteMAE, get_ros_metrics

# Data utilities
from .data_utils import (
    convert_to_sktime_format,
    convert_to_darts_format,
    prepare_player_data,
    validate_time_series_data,
    split_features_target,
    create_prediction_data
)

# Training utilities
from .training_utils import (
    prepare_ros_training_data,
    temporal_cv_split,
    calculate_ros_metrics,
    calculate_component_metrics,
    optimize_ensemble_weights
)

__all__ = [
    # Base classes
    'BaseROSModel',
    'BaseEnsemble',

    # Component models
    'MultiQuantileHistGB',
    'DirectROSForecaster',
    'DartsTemporalEnsemble',

    # Complete ensembles
    'HitterROSEnsemble',
    'PitcherROSEnsemble',
    'ROS_HITTER_FEATURES',
    'ROS_PITCHER_FEATURES',

    # Metrics
    'MeanError',
    'EliteBias',
    'EliteMAE',
    'get_ros_metrics',

    # Data utilities
    'convert_to_sktime_format',
    'convert_to_darts_format',
    'prepare_player_data',
    'validate_time_series_data',
    'split_features_target',
    'create_prediction_data',

    # Training utilities
    'prepare_ros_training_data',
    'temporal_cv_split',
    'calculate_ros_metrics',
    'calculate_component_metrics',
    'optimize_ensemble_weights'
]
