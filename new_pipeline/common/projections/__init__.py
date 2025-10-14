"""
Projection utilities for oWAR system.

Provides functions for calculating remaining usage (IP/games) based on
current season patterns and team context, plus complete projection
generation that combines current season + ROS models.
"""

from .usage_projections import (
    get_team_games_from_data,
    calculate_pitcher_remaining_ip,
    calculate_hitter_remaining_pa,
    calculate_remaining_usage
)
from .complete_projection import CompleteProjectionGenerator

__all__ = [
    'get_team_games_from_data',
    'calculate_pitcher_remaining_ip',
    'calculate_hitter_remaining_pa',
    'calculate_remaining_usage',
    'CompleteProjectionGenerator'
]
