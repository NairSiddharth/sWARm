"""
Projection utilities for oWAR system.

Provides functions for calculating remaining usage (IP/games) based on
current season patterns and team context.
"""

from .usage_projections import (
    get_team_games_from_data,
    calculate_pitcher_remaining_ip,
    calculate_hitter_remaining_games,
    calculate_remaining_usage
)

__all__ = [
    'get_team_games_from_data',
    'calculate_pitcher_remaining_ip',
    'calculate_hitter_remaining_games',
    'calculate_remaining_usage'
]
