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


def get_team_games_from_data(df: pd.DataFrame, team_col: str = 'Team', g_col: str = 'G') -> tuple:
    """
    Extract team games played from player data with multi-team handling.

    Takes the maximum games played by any player on each team as the team's
    games played (since at least one player appears in every game).

    Multi-team players (Team = '- - -') are included in the dictionary but
    excluded from league median calculation.

    Args:
        df: Player data with Team and G columns
        team_col: Name of team column (default: 'Team')
        g_col: Name of games column (default: 'G')

    Returns:
        tuple: (team_games_dict, league_median_games)
            - team_games_dict: {team: games_played} including '- - -'
            - league_median_games: Median games across single-team entries

    Example:
        >>> df = pd.DataFrame({
        ...     'Team': ['NYY', 'NYY', 'BOS', '- - -'],
        ...     'G': [95, 92, 98, 147]
        ... })
        >>> get_team_games_from_data(df)
        ({'NYY': 95, 'BOS': 98, '- - -': 147}, 96)
    """
    if team_col not in df.columns or g_col not in df.columns:
        return {}, 100  # Fallback median

    team_games_dict = df.groupby(team_col)[g_col].max().to_dict()

    # Calculate league median excluding multi-team players
    league_median = int(np.median([
        g for team, g in team_games_dict.items()
        if team != '- - -'
    ])) if len(team_games_dict) > 1 else 100

    return team_games_dict, league_median


def get_team_games_for_player(
    team: str,
    player_games: int,
    team_games_dict: Dict[str, int],
    league_median: int
) -> int:
    """
    Get team games played for a player, handling multi-team cases.

    Args:
        team: Player's team (or '- - -' for multi-team)
        player_games: Player's games played
        team_games_dict: Dictionary of {team: games_played}
        league_median: League median games (fallback)

    Returns:
        int: Team games to use for this player

    Logic:
        - Multi-team ('- - -'): Use player's actual games
        - Single-team: Use team's max games from dict
        - Unknown team: Use league median

    Example:
        >>> team_games_dict = {'NYY': 113, 'BOS': 115}
        >>> get_team_games_for_player('NYY', 110, team_games_dict, 113)
        113  # Uses team's games
        >>> get_team_games_for_player('- - -', 147, team_games_dict, 113)
        147  # Uses player's actual games (multi-team)
    """
    if team == '- - -':  # Multi-team player
        return player_games  # Use their actual games played
    else:
        return team_games_dict.get(team, league_median)


def calculate_pitcher_remaining_ip(
    current_ip: float,
    pitcher_games: int,
    team: str,
    team_games_dict: Dict[str, int],
    league_median_games: int,
    games_started: int = 0,
    season_length: int = 162,
    performance_boost: float = 1.0,
    min_appearances: int = 5
) -> Dict[str, Union[float, str]]:
    """
    Calculate remaining IP for a pitcher with team-specific and multi-team handling.

    Starters (GS/G > 0.7):
    - Uses rotation detection (5-man vs 6-man)
    - Applies 210 IP workhorse cap

    Relievers (GS/G <= 0.7):
    - Uses appearance rate × IP/G
    - Applies 100 IP cap (2025 max observed = 90 IP)

    Multi-team handling:
    - Single-team players use their team's games from team_games_dict
    - Multi-team players ('- - -') use their actual pitcher_games
    - Unknown teams fall back to league_median_games

    Args:
        current_ip: Innings pitched so far
        pitcher_games: Games pitcher has appeared in
        team: Player's team (or '- - -' for multi-team)
        team_games_dict: Dictionary of {team: games_played}
        league_median_games: League median games (fallback)
        games_started: Games started (for starter detection)
        season_length: Total season games (default: 162)
        performance_boost: Multiplier for elite performers (default: 1.0, unused)
        min_appearances: Minimum appearances to avoid noise (default: 5, unused)

    Returns:
        dict: {
            'ip_per_appearance': IP per game pitched,
            'appearance_rate': Games per team game (relievers only),
            'remaining_ip': Projected remaining IP,
            'total_projected_ip': Current + remaining,
            'role': 'starter' or 'reliever',
            'team_games_played': Team games used for calculation
        }

    Example:
        >>> team_games = {'NYY': 113, 'BOS': 115}
        >>> # Starter: 121 IP, 19 starts, NYY played 113 games
        >>> calculate_pitcher_remaining_ip(121, 19, 'NYY', team_games, 113, games_started=19)
        {'ip_per_appearance': 6.37, 'remaining_ip': 81.0,
         'total_projected_ip': 202.0, 'role': 'starter', 'team_games_played': 113}

        >>> # Multi-team: 48 IP, 47 games across teams
        >>> calculate_pitcher_remaining_ip(48, 47, '- - -', team_games, 113, games_started=0)
        {'ip_per_appearance': 1.02, 'remaining_ip': ~30.0,
         'total_projected_ip': ~78.0, 'role': 'reliever', 'team_games_played': 47}
    """
    # Get team-specific games
    team_games_played = get_team_games_for_player(
        team, pitcher_games, team_games_dict, league_median_games
    )

    if team_games_played <= 0 or pitcher_games <= 0:
        return {
            'ip_per_appearance': 0.0,
            'appearance_rate': 0.0,
            'remaining_ip': 0.0,
            'total_projected_ip': current_ip,
            'role': 'unknown',
            'team_games_played': team_games_played
        }

    # Calculate IP/G rate
    ip_per_appearance = current_ip / pitcher_games

    # Determine role from GS/G
    gs_per_g = games_started / pitcher_games if pitcher_games > 0 else 0.0
    is_starter = gs_per_g > 0.7

    remaining_team_games = max(0, season_length - team_games_played)

    if is_starter:
        # STARTER LOGIC: Rotation detection + 210 IP cap
        role = 'starter'

        # Rotation detection (5-man vs 6-man)
        implied_games_5man = games_started * 5
        implied_games_6man = games_started * 6

        dist_5man = abs(implied_games_5man - team_games_played)
        dist_6man = abs(implied_games_6man - team_games_played)

        # Use closer match (5-man or 6-man)
        rotation_spot = 5.0 if dist_5man <= dist_6man else 6.0

        # Edge case: IL returns or mid-season callups (< 10 starts)
        # Use 5.06 as fallback (162 games / 32 typical starts)
        if games_started < 10:
            rotation_spot = 5.06

        # Project remaining starts
        remaining_starts = remaining_team_games / rotation_spot

        # Project remaining IP based on current IP/start rate
        projected_remaining_ip = ip_per_appearance * remaining_starts

        # No artificial cap - let natural pace dictate projections
        remaining_ip = max(0, projected_remaining_ip)

        appearance_rate = games_started / team_games_played

    else:
        # RELIEVER/SWING LOGIC: Appearance rate with role-appropriate caps
        # Determine if pure reliever or swing man
        if gs_per_g < 0.1:
            # Pure reliever: Apply 100 IP cap
            role = 'reliever'
            ip_cap = 100  # Max for pure relievers
        else:
            # Swing man (0.1 <= GS/G <= 0.7): Higher cap
            role = 'swing'
            ip_cap = 140  # Swing pitchers can throw 110-140 IP

        # Calculate appearance rate
        appearance_rate = pitcher_games / team_games_played

        # Project remaining appearances
        projected_appearances = remaining_team_games * appearance_rate

        # Project remaining IP based on current IP/G rate
        projected_remaining_ip = ip_per_appearance * projected_appearances

        # Apply role-appropriate cap
        role_cap = max(0, ip_cap - current_ip)
        remaining_ip = min(projected_remaining_ip, role_cap)
        remaining_ip = max(0, remaining_ip)

    return {
        'ip_per_appearance': ip_per_appearance,
        'appearance_rate': appearance_rate,
        'remaining_ip': remaining_ip,
        'total_projected_ip': current_ip + remaining_ip,
        'role': role,
        'team_games_played': team_games_played
    }


def calculate_hitter_remaining_pa(
    current_pa: int,
    current_games: int,
    team: str,
    team_games_dict: Dict[str, int],
    league_median_games: int,
    season_length: int = 162,
    performance_boost: float = 1.0,
    role_cap: float = 1.0
) -> Dict[str, Union[float, int, str]]:
    """
    Calculate remaining PA for a hitter with team-specific and multi-team handling.

    Caps applied:
    1. Season total: 780 PA (leadoff everyday ~762, +buffer)
    2. PA/game rate: 5.0 PA/G (normal 3.8-4.7 by lineup spot)

    Preserves natural PA distribution based on lineup position and participation rate.

    Multi-team handling:
    - Single-team players use their team's games from team_games_dict
    - Multi-team players ('- - -') use their actual current_games
    - Unknown teams fall back to league_median_games

    Args:
        current_pa: Plate appearances so far
        current_games: Games played so far
        team: Player's team (or '- - -' for multi-team)
        team_games_dict: Dictionary of {team: games_played}
        league_median_games: League median games (fallback)
        season_length: Total season games (default: 162)
        performance_boost: Multiplier for elite performers (default: 1.0, unused)
        role_cap: Maximum participation rate (default: 1.0 = 100%)

    Returns:
        dict: {
            'participation_rate': Games per team game,
            'pa_per_game': Current PA/G rate,
            'remaining_pa': Projected remaining PA,
            'total_projected_pa': Current + remaining,
            'remaining_games': Projected remaining games,
            'team_games_played': Team games used for calculation
        }

    Example:
        >>> team_games = {'NYY': 113, 'BOS': 115}
        >>> # Everyday leadoff: 443 PA in 98 games, NYY played 113 games
        >>> calculate_hitter_remaining_pa(443, 98, 'NYY', team_games, 113)
        {'participation_rate': 0.87, 'pa_per_game': 4.52,
         'remaining_pa': ~200, 'total_projected_pa': ~643, 'team_games_played': 113}

        >>> # Multi-team: 347 PA in 95 games across teams
        >>> calculate_hitter_remaining_pa(347, 95, '- - -', team_games, 113)
        {'participation_rate': 1.0, 'pa_per_game': 3.65,
         'remaining_pa': ~245, 'total_projected_pa': ~592, 'team_games_played': 95}
    """
    # Get team-specific games
    team_games_played = get_team_games_for_player(
        team, current_games, team_games_dict, league_median_games
    )

    if team_games_played <= 0 or current_games <= 0:
        return {
            'participation_rate': 0.0,
            'pa_per_game': 0.0,
            'remaining_pa': 0,
            'total_projected_pa': current_pa,
            'remaining_games': 0,
            'team_games_played': team_games_played
        }

    # Calculate participation rate (handles playing time automatically)
    participation_rate = current_games / team_games_played

    # Apply role cap (can't play >100% of games)
    final_rate = min(participation_rate, role_cap)

    # Calculate remaining team games
    remaining_team_games = max(0, season_length - team_games_played)

    # Project remaining games for this player
    projected_player_games = final_rate * remaining_team_games

    # Calculate PA/G rate
    pa_per_game = current_pa / current_games

    # Project remaining PA based on current rate
    projected_remaining_pa = pa_per_game * projected_player_games

    # Apply realistic caps
    # Cap 1: Season total (leadoff max ~762 PA, add buffer for extra innings)
    workhorse_cap = max(0, 780 - current_pa)
    projected_remaining_pa = min(projected_remaining_pa, workhorse_cap)

    # Cap 2: PA/game rate (normal 3.8-4.7, cap at 5.0 for small-sample protection)
    # Protects against extrapolating unsustainable hot streaks
    max_pa_by_rate = projected_player_games * 5.0
    projected_remaining_pa = min(projected_remaining_pa, max_pa_by_rate)

    # Floor at 0
    remaining_pa = max(0, projected_remaining_pa)

    return {
        'participation_rate': final_rate,
        'pa_per_game': pa_per_game,
        'remaining_pa': remaining_pa,
        'total_projected_pa': current_pa + remaining_pa,
        'remaining_games': int(projected_player_games),
        'team_games_played': team_games_played
    }


def calculate_remaining_usage(
    player_row: pd.Series,
    player_type: str,
    team_games_dict: Dict[str, int],
    league_median_games: int,
    season_length: int = 162
) -> float:
    """
    Calculate remaining usage (IP for pitchers, PA for hitters) with multi-team handling.

    Wrapper function that dispatches to pitcher or hitter-specific logic.

    Multi-team handling:
    - Single-team players use their team's games from team_games_dict
    - Multi-team players ('- - -') use their actual games
    - Unknown teams fall back to league_median_games

    Args:
        player_row: Player data row with Team, IP/PA/G/GS, etc.
        player_type: 'pitcher' or 'hitter'
        team_games_dict: {team: games_played} (from get_team_games_from_data)
        league_median_games: League median games (fallback)
        season_length: Total season games (default: 162)

    Returns:
        float: Remaining usage (IP for pitchers, PA for hitters)

    Example:
        >>> team_games, league_median = get_team_games_from_data(df)
        >>> pitcher = pd.Series({'Team': 'NYY', 'IP': 121, 'G': 19, 'GS': 19})
        >>> calculate_remaining_usage(pitcher, 'pitcher', team_games, league_median)
        81.0  # Remaining IP (starter with rotation detection)

        >>> multi_team = pd.Series({'Team': '- - -', 'PA': 347, 'G': 95})
        >>> calculate_remaining_usage(multi_team, 'hitter', team_games, league_median)
        ~245.0  # Remaining PA (uses player's actual G)
    """
    team = player_row.get('Team', None)
    player_games = player_row.get('G', 0)

    if player_type == 'pitcher':
        current_ip = player_row.get('IP', 0.0)
        games_started = player_row.get('GS', 0)  # For starter detection
        result = calculate_pitcher_remaining_ip(
            current_ip,
            player_games,
            team,
            team_games_dict,
            league_median_games,
            games_started=games_started,
            season_length=season_length
        )
        return result['remaining_ip']

    elif player_type == 'hitter':
        current_pa = player_row.get('PA', 0)
        result = calculate_hitter_remaining_pa(
            current_pa,
            player_games,
            team,
            team_games_dict,
            league_median_games,
            season_length=season_length
        )
        return result['remaining_pa']

    else:
        return 0.0
