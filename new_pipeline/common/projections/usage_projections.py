"""
Usage Projection Utilities

Calculates realistic remaining usage (IP for pitchers, games for hitters)
based on current usage patterns and team context.

Key Concept:
- Pitchers: IP/G rate × remaining team games = remaining IP
- Hitters: Games/team_games rate × remaining team games = remaining games

This naturally encodes roles without explicit classification:
- Starters: 6-7 IP/G
- Swing: 3-4 IP/G
- Relievers: 0.4-1.0 IP/G
"""

from typing import Dict, Union
import pandas as pd
import numpy as np


def get_team_games_from_data(df: pd.DataFrame, team_col: str = 'Team', g_col: str = 'G') -> Dict[str, int]:
    """
    Extract team games played from player data.

    Takes the maximum games played by any player on each team as the team's
    games played (since at least one player appears in every game).

    Args:
        df: Player data with Team and G columns
        team_col: Name of team column (default: 'Team')
        g_col: Name of games column (default: 'G')

    Returns:
        dict: {team: games_played}

    Example:
        >>> df = pd.DataFrame({'Team': ['NYY', 'NYY', 'BOS'], 'G': [95, 92, 98]})
        >>> get_team_games_from_data(df)
        {'NYY': 95, 'BOS': 98}
    """
    if team_col not in df.columns or g_col not in df.columns:
        return {}

    return df.groupby(team_col)[g_col].max().to_dict()


def calculate_pitcher_remaining_ip(
    current_ip: float,
    pitcher_games: int,
    team_games_played: int,
    season_length: int = 162,
    performance_boost: float = 1.0,
    min_appearances: int = 5
) -> Dict[str, Union[float, str]]:
    """
    Calculate remaining IP for a pitcher based on appearance rate and IP per appearance.

    Uses two-step projection:
    1. Appearance rate (G / team_games) to project total season appearances
    2. IP per appearance (IP / G) to convert to innings

    Realistic caps by role:
    - Starter (≥5.0 IP/appearance): max 33 starts/season
    - Swing (2.5-5.0 IP/appearance): max 50 appearances/season
    - Reliever (<2.5 IP/appearance): max 70 appearances/season

    Args:
        current_ip: Innings pitched so far
        pitcher_games: Games pitcher has appeared in
        team_games_played: Games team has played
        season_length: Total season games (default: 162)
        performance_boost: Multiplier for elite performers (default: 1.0)
        min_appearances: Minimum appearances to avoid noise (default: 5)

    Returns:
        dict: {
            'ip_per_appearance': IP per game pitched,
            'appearance_rate': Games per team game,
            'remaining_ip': Projected remaining IP,
            'total_projected_ip': Current + remaining,
            'role': Auto-detected role classification
        }

    Example:
        >>> # Starter: 121 IP in 19 G, team has 40 games
        >>> calculate_pitcher_remaining_ip(121, 19, 40)
        {'ip_per_appearance': 6.37, 'appearance_rate': 0.475,
         'remaining_ip': 89.1, 'total_projected_ip': 210.1, 'role': 'starter'}

        >>> # Reliever: 38 IP in 40 G, team has 40 games
        >>> calculate_pitcher_remaining_ip(38, 40, 40)
        {'ip_per_appearance': 0.95, 'appearance_rate': 1.0,
         'remaining_ip': 28.5, 'total_projected_ip': 66.5, 'role': 'reliever'}
    """
    if team_games_played <= 0 or pitcher_games <= 0:
        return {
            'ip_per_appearance': 0.0,
            'appearance_rate': 0.0,
            'remaining_ip': 0.0,
            'total_projected_ip': current_ip,
            'role': 'unknown'
        }

    # Calculate rates
    ip_per_appearance = current_ip / pitcher_games
    appearance_rate = pitcher_games / team_games_played

    # Auto-detect role from IP/appearance
    if ip_per_appearance >= 5.0:
        role = 'starter'
        max_season_appearances = 33
    elif ip_per_appearance >= 2.5:
        role = 'swing'
        max_season_appearances = 50
    else:
        role = 'reliever'
        max_season_appearances = 70

    # Project total season appearances based on current rate
    projected_total_appearances = appearance_rate * season_length

    # Apply performance boost
    projected_total_appearances *= performance_boost

    # Cap at realistic maximum for role
    projected_total_appearances = min(projected_total_appearances, max_season_appearances)

    # Calculate remaining appearances
    remaining_appearances = max(0, projected_total_appearances - pitcher_games)

    # Project remaining IP
    remaining_ip = remaining_appearances * ip_per_appearance

    return {
        'ip_per_appearance': ip_per_appearance,
        'appearance_rate': appearance_rate,
        'remaining_ip': remaining_ip,
        'total_projected_ip': current_ip + remaining_ip,
        'role': role
    }


def calculate_hitter_remaining_games(
    current_games: int,
    team_games_played: int,
    season_length: int = 162,
    performance_boost: float = 1.0,
    role_cap: float = 1.0
) -> Dict[str, Union[float, int, str]]:
    """
    Calculate remaining games for a hitter based on games/team_games rate.

    Args:
        current_games: Games played so far
        team_games_played: Games team has played
        season_length: Total season games (default: 162)
        performance_boost: Multiplier for elite performers (default: 1.0)
        role_cap: Maximum participation rate (default: 1.0 = 100%)

    Returns:
        dict: {
            'games_per_team_game': Current participation rate,
            'remaining_games': Projected remaining games,
            'total_projected_games': Current + remaining,
            'participation_rate': Final rate after adjustments
        }

    Example:
        >>> # Everyday player: 95 of 95 team games
        >>> calculate_hitter_remaining_games(95, 95)
        {'games_per_team_game': 1.0, 'remaining_games': 67, ...}

        >>> # Platoon: 50 of 95 team games
        >>> calculate_hitter_remaining_games(50, 95)
        {'games_per_team_game': 0.53, 'remaining_games': 35, ...}
    """
    if team_games_played <= 0:
        return {
            'games_per_team_game': 0.0,
            'remaining_games': 0,
            'total_projected_games': current_games,
            'participation_rate': 0.0
        }

    # Calculate participation rate
    participation_rate = current_games / team_games_played

    # Apply performance boost
    adjusted_rate = participation_rate * performance_boost

    # Apply role cap (can't play >100% of games)
    final_rate = min(adjusted_rate, role_cap)

    # Calculate remaining games
    remaining_team_games = max(0, season_length - team_games_played)
    remaining_games = int(final_rate * remaining_team_games)

    return {
        'games_per_team_game': participation_rate,
        'remaining_games': remaining_games,
        'total_projected_games': current_games + remaining_games,
        'participation_rate': final_rate
    }


def calculate_remaining_usage(
    player_row: pd.Series,
    player_type: str,
    team_games_dict: Dict[str, int],
    season_length: int = 162
) -> float:
    """
    Calculate remaining usage (IP for pitchers, games for hitters).

    Wrapper function that dispatches to pitcher or hitter-specific logic.

    Args:
        player_row: Player data row with Team, IP/G, etc.
        player_type: 'pitcher' or 'hitter'
        team_games_dict: {team: games_played}
        season_length: Total season games (default: 162)

    Returns:
        float: Remaining usage (IP or games)

    Example:
        >>> team_games = {'NYY': 95}
        >>> pitcher = pd.Series({'Team': 'NYY', 'IP': 100})
        >>> calculate_remaining_usage(pitcher, 'pitcher', team_games)
        70.6  # Remaining IP
    """
    team = player_row.get('Team', None)

    # Get team games, fallback to estimation
    if team and team in team_games_dict:
        team_games_played = team_games_dict[team]
    else:
        # Fallback: estimate from player's games
        player_games = player_row.get('G', 0)
        team_games_played = max(player_games, 95)  # Mid-season estimate

    if player_type == 'pitcher':
        current_ip = player_row.get('IP', 0.0)
        pitcher_games = player_row.get('G', 0)
        result = calculate_pitcher_remaining_ip(current_ip, pitcher_games, team_games_played, season_length)
        return result['remaining_ip']

    elif player_type == 'hitter':
        current_games = player_row.get('G', 0)
        result = calculate_hitter_remaining_games(current_games, team_games_played, season_length)
        return result['remaining_games']

    else:
        return 0.0
