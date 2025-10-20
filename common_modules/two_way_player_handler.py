"""
Two-Way Player Detection and Handling Module

Systematically identifies and handles two-way players (players who both hit and pitch)
for accurate workload projections and WAR calculations.
"""

from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from common_modules.logging import get_logger

logger = get_logger(__name__)


def identify_two_way_players(
    hitter_data: pd.DataFrame,
    pitcher_data: pd.DataFrame,
    min_pitcher_games: int = 3,
    min_hitter_games: int = 10
) -> Dict[str, Dict]:
    """
    Identify players who appear in both hitter and pitcher datasets.

    Args:
        hitter_data: DataFrame with hitter statistics
        pitcher_data: DataFrame with pitcher statistics
        min_pitcher_games: Minimum games pitched to qualify as two-way
        min_hitter_games: Minimum games as hitter to qualify as two-way

    Returns:
        Dictionary mapping player name to their two-way stats
    """
    two_way_players = {}

    # Get player names from both datasets
    hitter_names = set(hitter_data['Name'].unique()) if 'Name' in hitter_data.columns else set()
    pitcher_names = set(pitcher_data['Name'].unique()) if 'Name' in pitcher_data.columns else set()

    # Find intersection
    potential_two_way = hitter_names & pitcher_names

    for player_name in potential_two_way:
        # Get their stats from both sides
        hitter_stats = hitter_data[hitter_data['Name'] == player_name].iloc[0]
        pitcher_stats = pitcher_data[pitcher_data['Name'] == player_name].iloc[0]

        hitter_games = hitter_stats.get('G', 0)
        pitcher_games = pitcher_stats.get('G', 0)

        # Check if they meet minimum thresholds
        if pitcher_games >= min_pitcher_games and hitter_games >= min_hitter_games:
            two_way_players[player_name] = {
                'hitter_games': hitter_games,
                'pitcher_games': pitcher_games,
                'hitter_data': hitter_stats,
                'pitcher_data': pitcher_stats,
                'primary_role': 'hitter' if hitter_games > pitcher_games * 3 else 'pitcher'
            }

            logger.info(f"Identified two-way player: {player_name} "
                       f"(H: {hitter_games}G, P: {pitcher_games}G)")

    return two_way_players


def calculate_two_way_constraints(
    player_name: str,
    two_way_players: Dict[str, Dict],
    hitter_remaining_games: Optional[int] = None,
    pitcher_remaining_games: Optional[int] = None
) -> Dict[str, int]:
    """
    Calculate remaining game constraints for two-way players.

    Two-way players can't exceed the total remaining games available to them
    as a position player, since they play that role more frequently.

    Args:
        player_name: Name of the player
        two_way_players: Dictionary of identified two-way players
        hitter_remaining_games: Already calculated hitter remaining games
        pitcher_remaining_games: Already calculated pitcher remaining games

    Returns:
        Dictionary with constrained remaining games for each role
    """
    if player_name not in two_way_players:
        # Not a two-way player, return unconstrained
        return {
            'is_two_way': False,
            'hitter_remaining': hitter_remaining_games,
            'pitcher_remaining': pitcher_remaining_games,
            'pitcher_constraint': None
        }

    player_info = two_way_players[player_name]

    # For two-way players, pitcher games are constrained by total games available
    # They can't pitch more games than they have available as a player
    if hitter_remaining_games is not None:
        # Use hitter remaining games as the constraint
        pitcher_constraint = hitter_remaining_games

        # But also consider their historical pitch/hit ratio
        pitch_ratio = player_info['pitcher_games'] / player_info['hitter_games']
        expected_pitcher_games = int(hitter_remaining_games * pitch_ratio)

        # Take the minimum of expected and total available
        pitcher_constraint = min(pitcher_constraint, expected_pitcher_games * 2)  # Allow some flexibility

        logger.info(f"Two-way constraint for {player_name}: "
                   f"max {pitcher_constraint} pitching games "
                   f"(based on {hitter_remaining_games} total remaining)")
    else:
        pitcher_constraint = pitcher_remaining_games

    return {
        'is_two_way': True,
        'hitter_remaining': hitter_remaining_games,
        'pitcher_remaining': min(pitcher_remaining_games or float('inf'), pitcher_constraint),
        'pitcher_constraint': pitcher_constraint,
        'primary_role': player_info['primary_role']
    }


def get_two_way_player_names(year: int = 2025) -> List[str]:
    """
    Get list of known two-way players for a given year.

    This can be extended with a database or configuration file
    for historical two-way players.

    Args:
        year: Season year

    Returns:
        List of two-way player names
    """
    # Known two-way players by year
    two_way_by_year = {
        2025: ['Shohei Ohtani'],
        2024: ['Shohei Ohtani'],
        2023: ['Shohei Ohtani', 'Michael Lorenzen'],
        2022: ['Shohei Ohtani'],
        2021: ['Shohei Ohtani', 'Michael Lorenzen'],
        # Add more as needed
    }

    return two_way_by_year.get(year, [])


def apply_two_way_constraints_to_projections(
    player_data: pd.Series,
    player_type: str,
    calculated_remaining_games: int,
    hitter_df: pd.DataFrame,
    pitcher_df: pd.DataFrame
) -> int:
    """
    Apply two-way player constraints to projection calculations.

    This is the main function to use in projection pipelines.

    Args:
        player_data: Current player data row
        player_type: 'hitter' or 'pitcher'
        calculated_remaining_games: Initially calculated remaining games
        hitter_df: Full hitter dataset
        pitcher_df: Full pitcher dataset

    Returns:
        Constrained remaining games accounting for two-way status
    """
    player_name = player_data.get('Name', player_data.get('name', ''))

    # Identify all two-way players
    two_way_players = identify_two_way_players(hitter_df, pitcher_df)

    if player_name not in two_way_players:
        # Not a two-way player, return original
        return calculated_remaining_games

    player_info = two_way_players[player_name]

    if player_type == 'pitcher':
        # For pitchers, constrain by their hitting games
        hitter_remaining = None
        if 'hitter_data' in player_info:
            # Calculate hitter remaining games
            from current_season_modules.participation_rate_calculator import calculate_participation_adjusted_games

            hitter_result = calculate_participation_adjusted_games(
                player_info['hitter_data'],
                current_war=player_info['hitter_data'].get('WAR', 0),
                injury_adjustment=1.0
            )
            hitter_remaining = hitter_result.get('games_remaining', calculated_remaining_games)

        # Apply constraint
        constraints = calculate_two_way_constraints(
            player_name,
            two_way_players,
            hitter_remaining_games=hitter_remaining,
            pitcher_remaining_games=calculated_remaining_games
        )

        constrained_games = constraints['pitcher_remaining']

        if constrained_games != calculated_remaining_games:
            logger.info(f"Two-way constraint applied to {player_name}: "
                       f"{calculated_remaining_games} -> {constrained_games} games")

        return constrained_games

    # For hitters, typically no constraint needed
    return calculated_remaining_games