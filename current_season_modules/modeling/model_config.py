"""Configuration settings for predictive modeling.

This module contains all configuration parameters for model training,
feature engineering, and cross-validation.
"""

from typing import List, Dict, Any

# Model hyperparameters
MODEL_CONFIGS = {
    'ridge': {
        'alpha': 1.0,
        'fit_intercept': True,
        'normalize': False,
        'max_iter': None,
        'random_state': 42
    },
    'randomforest': {
        'n_estimators': 100,
        'max_depth': None,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'random_state': 42,
        'n_jobs': -1
    },
    'svr': {
        'kernel': 'rbf',
        'degree': 3,
        'gamma': 'scale',
        'C': 1.0,
        'epsilon': 0.1
    },
    'xgboost': {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.3,
        'objective': 'reg:squarederror',
        'random_state': 42,
        'n_jobs': -1,
    },
}

# Keras model configuration
KERAS_CONFIG = {
    'layers': [
        {'units': 64, 'activation': 'relu', 'dropout': 0.2},
        {'units': 32, 'activation': 'relu', 'dropout': 0.1},
        {'units': 1, 'activation': 'linear', 'dropout': None}
    ],
    'optimizer': 'adam',
    'loss': 'mse',
    'metrics': ['mae'],
    'epochs': 50,
    'batch_size': 32,
    'validation_split': 0.2,
    'verbose': 0,
}

# Cross - validation settings
CV_CONFIG = {
    'n_splits': 5,
    'shuffle': False,  # GroupKFold doesn't support shuffling
    'random_state': 42,
}

# Feature configurations for hitters
HITTER_FEATURES = {
    'core': ['K%', 'BB%', 'AVG', 'OBP', 'SLG'],
    'volume': ['PA'],
    'situational': ['GDP_rate'],
    'enhanced': ['Enhanced_Baserunning', 'Enhanced_Defense'],
    'positional': ['Positional_WAR'],
}

# All hitter features combined
HITTER_FEATURE_LIST: List[str] = (
    HITTER_FEATURES['core'] +
    HITTER_FEATURES['volume'] +
    HITTER_FEATURES['situational'] +
    HITTER_FEATURES['enhanced'] +
    HITTER_FEATURES['positional'],
)

# Feature configurations for pitchers
PITCHER_FEATURES = {
    'core': ['IP', 'BB%', 'K%', 'K-BB%', 'ERA'],
    'advanced': [
        'damage_control_ratio',
        'Opportunity_Success',
        'Contact_Quality_Index'
    ],
    'command': ['HBP%'],
    'statcast': ['Statcast_Launch_Quality_Index'],
}

# All pitcher features combined
PITCHER_FEATURE_LIST: List[str] = (
    PITCHER_FEATURES['core'] +
    PITCHER_FEATURES['advanced'] +
    PITCHER_FEATURES['command'] +
    PITCHER_FEATURES['statcast'],
)

# Data source configurations
DATA_SOURCES = {
    'warp': {
        'year_column': 'Season',
        'value_column': 'WARP',
        'id_column': 'mlbid',
        'name_column': 'Name'
    },
    'war': {
        'year_column': 'Year',
        'value_column': 'WAR',
        'id_column': 'MLBAMID',
        'name_column': 'Name',
    },
}

# Filtering thresholds
FILTERING_THRESHOLDS = {
    'min_pa_for_qualified': 502,  # MLB qualification threshold
    'min_ip_for_qualified': 162.0,  # 1 IP per team game
    'min_games_for_pitcher': 10,  # Minimum games to be considered
    'max_era_for_valid': 10.0,  # Maximum ERA to consider valid
    'min_year': 2016,  # Earliest year to include in training
    'max_year': 2024  # Latest year to include in training,
}

# Performance metrics to track
METRICS_TO_TRACK = [
    'r2',
    'rmse',
    'mae',
    'mape',  # Mean Absolute Percentage Error
    'max_error',
]

# Output configurations
OUTPUT_CONFIG = {
    'decimal_places': 4,
    'show_feature_importance': True,
    'save_predictions': True,
    'save_models': True,
    'output_dir': 'model_outputs',
}

# Training configurations
TRAINING_CONFIG = {
    'use_scaler': True,  # Whether to standardize features
    'handle_missing': 'fill_zero',  # How to handle missing values
    'remove_outliers': False,  # Whether to remove statistical outliers
    'outlier_std_threshold': 3,  # Standard deviations for outlier detection
    'min_samples_per_year': 50,  # Minimum samples per year to include
    'holdout_years': [2024],  # Years to use for final validation
    'test_size': 0.2,  # Proportion for train / test split
    'stratify_by_year': True  # Whether to stratify splits by year,
}

# Enhanced feature configurations
ENHANCED_FEATURE_CONFIG = {
    'baserunning': {
        'weight': 1.0,
        'normalize': True,
        'fill_value': 0.0
    },
    'defense': {
        'weight': 1.0,
        'normalize': True,
        'fill_value': 0.0
    },
    'positional_adjustment': {
        'scale_by_pa': True,
        'base_pa': 600,
        'fill_value': 0.0,
    },
}

# Model ensemble configurations
ENSEMBLE_CONFIG = {
    'use_ensemble': True,
    'ensemble_method': 'weighted_average',  # or 'voting', 'stacking'
    'model_weights': {
        'ridge': 0.2,
        'randomforest': 0.3,
        'svr': 0.2,
        'xgboost': 0.2,
        'keras': 0.1,
    },
}

# Logging configurations
LOGGING_CONFIG = {
    'log_level': 'INFO',
    'log_training_progress': True,
    'log_feature_importance': True,
    'log_predictions': False,  # Can be verbose
    'log_to_file': True,
    'log_file': 'modeling.log',
}


def get_model_config(model_name: str) -> Dict[str, Any]:
    """Get configuration for a specific model.

    Args:
        model_name: Name of the model

    Returns:
        Dictionary of model configuration parameters

    Raises:
        KeyError: If model name is not found in configurations
    """
    if model_name not in MODEL_CONFIGS:
        raise KeyError(f"Model '{model_name}' not found in configurations")
    return MODEL_CONFIGS[model_name].copy()


def get_features_for_player_type(player_type: str) -> List[str]:
    """Get feature list for a specific player type.

    Args:
        player_type: 'hitter' or 'pitcher'

    Returns:
        List of feature names

    Raises:
        ValueError: If player_type is not 'hitter' or 'pitcher'
    """
    if player_type == 'hitter':
        return HITTER_FEATURE_LIST.copy()
    elif player_type == 'pitcher':
        return PITCHER_FEATURE_LIST.copy()
    else:
        raise ValueError(f"Invalid player_type: {player_type}. Must be 'hitter' or 'pitcher'")


def validate_config() -> bool:
    """Validate that all configuration settings are properly set.

    Returns:
        True if all configurations are valid

    Raises:
        ValueError: If any configuration is invalid
    """
    # Check model weights sum to 1
    if ENSEMBLE_CONFIG['use_ensemble']:
        weight_sum = sum(ENSEMBLE_CONFIG['model_weights'].values())
        if abs(weight_sum - 1.0) > 0.001:
            raise ValueError(f"Model weights sum to {weight_sum}, should sum to 1.0")

    # Check that holdout years are within range
    for year in TRAINING_CONFIG['holdout_years']:
        if year < FILTERING_THRESHOLDS['min_year'] or year > FILTERING_THRESHOLDS['max_year']:
            raise ValueError(f"Holdout year {year} outside valid range")

    # Check CV splits is reasonable
    if CV_CONFIG['n_splits'] < 2 or CV_CONFIG['n_splits'] > 10:
        raise ValueError(f"n_splits={CV_CONFIG['n_splits']} is unreasonable (should be 2 - 10)")

    return True
