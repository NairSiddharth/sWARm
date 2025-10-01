"""Modeling package for current season predictive analytics.

This package contains modules for:
- Cross - validation and model evaluation
- Data loading and preparation
- Model training and optimization
- Configuration management
"""

from .cross_validation import CrossValidationResults
from .data_loading import (
    load_expanded_fangraphs_data,
    load_expanded_fangraphs_pitcher_data,
    load_comprehensive_warp_hitter_data,
    load_comprehensive_fangraphs_hitter_data,
    load_comprehensive_warp_pitcher_data,
    load_comprehensive_fangraphs_pitcher_data,
    load_and_prepare_hitter_data,
    load_and_prepare_hitter_war_data,
    load_and_prepare_pitcher_data,
    load_and_prepare_pitcher_war_data,
)
from .model_training import (
    run_kfold_cross_validation,
    create_keras_model_temp,
    CurrentSeasonPredictor,
)
from .data_preparation import (
    filter_position_players_pitching,
    filter_pitchers_from_hitting_data,
    create_mlbid_mapping,
    prepare_data_for_kfold,
)

__all__ = [
    'CrossValidationResults',
    'load_expanded_fangraphs_data',
    'load_expanded_fangraphs_pitcher_data',
    'load_comprehensive_warp_hitter_data',
    'load_comprehensive_fangraphs_hitter_data',
    'load_comprehensive_warp_pitcher_data',
    'load_comprehensive_fangraphs_pitcher_data',
    'load_and_prepare_hitter_data',
    'load_and_prepare_hitter_war_data',
    'load_and_prepare_pitcher_data',
    'load_and_prepare_pitcher_war_data',
    'run_kfold_cross_validation',
    'create_keras_model_temp',
    'CurrentSeasonPredictor',
    'filter_position_players_pitching',
    'filter_pitchers_from_hitting_data',
    'create_mlbid_mapping',
    'prepare_data_for_kfold',
]
