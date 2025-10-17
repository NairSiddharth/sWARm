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

from typing import Dict, Union, List, Optional
import pandas as pd
import numpy as np
from ..data_preparation.team_stint_tracker import calculate_weighted_usage


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
    league_median: int,
    current_team: str = None
) -> int:
    """
    Get team games played for a player, handling multi-team cases.

    Args:
        team: Player's team (e.g., 'NYY' or 'MIA, NYY' for multi-team)
        player_games: Player's games played
        team_games_dict: Dictionary of {team: games_played}
        league_median: League median games (fallback)
        current_team: Player's current team (for multi-team players, optional)

    Returns:
        int: Team games to use for this player

    Logic:
        - If current_team provided: Use current_team's games from dict
        - Multi-team (comma in team name): Use current_team if provided, else player's games
        - Legacy '- - -': Use player's actual games
        - Single-team: Use team's max games from dict
        - Unknown team: Use league median

    Example:
        >>> team_games_dict = {'NYY': 113, 'BOS': 115, 'MIA': 110}
        >>> get_team_games_for_player('NYY', 110, team_games_dict, 113)
        113  # Uses team's games
        >>> get_team_games_for_player('MIA, NYY', 147, team_games_dict, 113, current_team='NYY')
        113  # Uses current team's games (NYY)
        >>> get_team_games_for_player('- - -', 147, team_games_dict, 113)
        147  # Legacy multi-team: uses player's actual games
    """
    # If current_team is explicitly provided, use it
    if current_team:
        return team_games_dict.get(current_team, league_median)

    # Detect multi-team players (comma in team or legacy '- - -')
    if ',' in team or team == '- - -':
        # Multi-team: use player's actual games played
        return player_games

    # Single-team: use team's games from dict
    return team_games_dict.get(team, league_median)


def classify_pitcher_role(
    games_started: int,
    games_pitched: int,
    innings_pitched: float
) -> str:
    """
    Classify pitcher role using robust multi-factor logic.

    Handles edge cases: late-season callups, injured starters, openers, swing pitchers.

    Args:
        games_started: Number of games started (GS)
        games_pitched: Total games pitched (G)
        innings_pitched: Total innings pitched (IP)

    Returns:
        str: 'starter', 'swing', or 'reliever'

    Classification Logic:
        STARTER: Either high GS ratio (>70%) OR significant total starts (>=15)
        - Handles late-season callups with many starts but low ratio
        - Example: 20 G, 15 GS → starter (by total starts)

        SWING: Moderate GS ratio (10-70%) OR high IP/G (2.0-4.5)
        - Captures spot starters, openers, long relievers
        - Example: 20 G, 8 GS, 78 IP → swing (by GS ratio and IP/G)

        RELIEVER: Everything else (low GS ratio AND low IP/G)
        - Pure bullpen arms
        - Example: 43 G, 0 GS, 41 IP → reliever

    Example:
        >>> # Late-season callup (qualified starter)
        >>> classify_pitcher_role(games_started=15, games_pitched=20, innings_pitched=95)
        'starter'

        >>> # Opener/spot starter
        >>> classify_pitcher_role(games_started=8, games_pitched=20, innings_pitched=65)
        'swing'

        >>> # Elite closer
        >>> classify_pitcher_role(games_started=0, games_pitched=43, innings_pitched=41)
        'reliever'
    """
    if games_pitched == 0:
        return 'reliever'  # Default for edge case

    gs_ratio = games_started / games_pitched
    ip_per_g = innings_pitched / games_pitched

    # STARTER: High GS ratio OR significant total starts
    if gs_ratio > 0.7 or games_started >= 15:
        return 'starter'

    # SWING: Moderate GS ratio OR high IP/G (openers/spot starters)
    elif (0.1 < gs_ratio <= 0.7) or (2.0 < ip_per_g <= 4.5):
        return 'swing'

    # RELIEVER: Everything else
    else:
        return 'reliever'


def calculate_pitcher_remaining_ip(
    current_ip: float,
    pitcher_games: int,
    team: str,
    team_games_dict: Dict[str, int],
    league_median_games: int,
    games_started: int = 0,
    season_length: int = 162,
    performance_boost: float = 1.0,
    min_appearances: int = 5,
    current_team: Optional[str] = None,
    stint_data: Optional[List[Dict]] = None,
    weighting_params: Optional[Dict] = None
) -> Dict[str, Union[float, str]]:
    """
    Calculate remaining IP for a pitcher with team-specific and multi-team handling.

    Starters (GS/G > 0.7):
    - Uses rotation detection (5-man vs 6-man)
    - No cap - natural pace dictates projections

    Swing (0.1 <= GS/G <= 0.7):
    - Uses appearance rate × IP/G
    - No cap - natural pace dictates projections

    Relievers (GS/G < 0.1):
    - Uses appearance rate × IP/G
    - No cap - natural pace dictates projections

    Multi-team handling:
    - Single-team players use their team's games from team_games_dict
    - Multi-team players with stint_data use weighted usage calculations
    - Multi-team players without stint_data use their actual pitcher_games
    - Unknown teams fall back to league_median_games

    Args:
        current_ip: Innings pitched so far
        pitcher_games: Games pitcher has appeared in
        team: Player's team (e.g., 'NYY' or 'MIA, NYY' for multi-team)
        team_games_dict: Dictionary of {team: games_played}
        league_median_games: League median games (fallback)
        games_started: Games started (for starter detection)
        season_length: Total season games (default: 162)
        performance_boost: Multiplier for elite performers (default: 1.0, unused)
        min_appearances: Minimum appearances to avoid noise (default: 5, unused)
        current_team: Player's current team (for multi-team players, optional)
        stint_data: List of team stints with usage data (for weighted calculations, optional)
        weighting_params: Parameters for weighted usage calculation (optional)

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

        >>> # Multi-team with weighted usage: 48 IP across 2 teams
        >>> stints = [{'team': 'MIA', 'G': 27, 'IP': 30.0}, {'team': 'NYY', 'G': 20, 'IP': 18.0}]
        >>> calculate_pitcher_remaining_ip(48, 47, 'MIA, NYY', team_games, 113,
        ...     current_team='NYY', stint_data=stints)
        # Uses weighted IP/G calculation across both stints
    """
    # Get team-specific games
    team_games_played = get_team_games_for_player(
        team, pitcher_games, team_games_dict, league_median_games, current_team=current_team
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

    # Calculate IP/G rate (with weighted usage for multi-team players)
    if stint_data and len(stint_data) > 0:
        # Use weighted usage calculation across all team stints
        weighted_result = calculate_weighted_usage(
            stints=stint_data,
            usage_key='IP',
            games_key='G',
            weighting_params=weighting_params
        )
        ip_per_appearance = weighted_result['weighted_usage_per_game']
    else:
        # Use simple average for single-team players
        ip_per_appearance = current_ip / pitcher_games

    # For multi-team players, extract current stint games
    if stint_data and len(stint_data) > 0:
        current_stint_games = stint_data[-1].get('G', pitcher_games)
    else:
        current_stint_games = pitcher_games

    # Classify pitcher role using centralized function
    role = classify_pitcher_role(games_started, pitcher_games, current_ip)

    remaining_team_games = max(0, season_length - team_games_played)

    if role == 'starter':
        # STARTER LOGIC: Rotation detection

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
        # RELIEVER/SWING LOGIC: Appearance rate with natural projections
        # Role already determined by classify_pitcher_role() above

        # Calculate appearance rate using current stint games for multi-team players
        appearance_rate = current_stint_games / team_games_played

        # Project remaining appearances
        projected_appearances = remaining_team_games * appearance_rate

        # Project remaining IP based on current IP/G rate
        projected_remaining_ip = ip_per_appearance * projected_appearances

        # No caps - use natural projection
        remaining_ip = max(0, projected_remaining_ip)

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
    role_cap: float = 1.0,
    current_team: Optional[str] = None,
    stint_data: Optional[List[Dict]] = None,
    weighting_params: Optional[Dict] = None
) -> Dict[str, Union[float, int, str]]:
    """
    Calculate remaining PA for a hitter with team-specific and multi-team handling.

    Caps applied:
    1. Season total: 780 PA (leadoff everyday ~762, +buffer)
    2. PA/game rate: 5.0 PA/G (normal 3.8-4.7 by lineup spot)

    Preserves natural PA distribution based on lineup position and participation rate.

    Multi-team handling:
    - Single-team players use their team's games from team_games_dict
    - Multi-team players with stint_data use weighted usage calculations
    - Multi-team players without stint_data use their actual current_games
    - Unknown teams fall back to league_median_games

    Args:
        current_pa: Plate appearances so far
        current_games: Games played so far
        team: Player's team (e.g., 'NYY' or 'MIA, NYY' for multi-team)
        team_games_dict: Dictionary of {team: games_played}
        league_median_games: League median games (fallback)
        season_length: Total season games (default: 162)
        performance_boost: Multiplier for elite performers (default: 1.0, unused)
        role_cap: Maximum participation rate (default: 1.0 = 100%)
        current_team: Player's current team (for multi-team players, optional)
        stint_data: List of team stints with usage data (for weighted calculations, optional)
        weighting_params: Parameters for weighted usage calculation (optional)

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

        >>> # Multi-team with weighted usage: 347 PA across 2 teams
        >>> stints = [{'team': 'MIA', 'G': 95, 'PA': 280.0}, {'team': 'NYY', 'G': 30, 'PA': 67.0}]
        >>> calculate_hitter_remaining_pa(347, 125, 'MIA, NYY', team_games, 113,
        ...     current_team='NYY', stint_data=stints)
        # Uses weighted PA/G calculation across both stints
    """
    # Get team-specific games
    team_games_played = get_team_games_for_player(
        team, current_games, team_games_dict, league_median_games, current_team=current_team
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

    # For multi-team players, use games with CURRENT team only (not total)
    if stint_data and len(stint_data) > 0:
        # Current stint is last in list
        current_stint_games = stint_data[-1].get('G', current_games)
    else:
        # Single-team player: use total games
        current_stint_games = current_games

    # Calculate participation rate (handles playing time automatically)
    participation_rate = current_stint_games / team_games_played

    # Apply role cap (can't play >100% of games)
    final_rate = min(participation_rate, role_cap)

    # Calculate remaining team games
    remaining_team_games = max(0, season_length - team_games_played)

    # Project remaining games for this player
    projected_player_games = final_rate * remaining_team_games

    # Calculate PA/G rate (with weighted usage for multi-team players)
    if stint_data and len(stint_data) > 0:
        # Use weighted usage calculation across all team stints
        weighted_result = calculate_weighted_usage(
            stints=stint_data,
            usage_key='PA',
            games_key='G',
            weighting_params=weighting_params
        )
        pa_per_game = weighted_result['weighted_usage_per_game']
    else:
        # Use simple average for single-team players
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
    - Multi-team players with stint data use weighted usage calculations
    - Multi-team players without stint data use their actual games
    - Unknown teams fall back to league_median_games

    Args:
        player_row: Player data row with Team, IP/PA/G/GS, etc.
            May include _multi_team_current and _multi_team_stints for multi-team players
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

        >>> # Multi-team with stint data
        >>> multi_team = pd.Series({
        ...     'Team': 'MIA, NYY', 'PA': 621, 'G': 147,
        ...     '_multi_team_current': 'NYY',
        ...     '_multi_team_stints': '[{"team":"MIA","G":101,"PA":430.0},{"team":"NYY","G":46,"PA":191.0}]'
        ... })
        >>> calculate_remaining_usage(multi_team, 'hitter', team_games, league_median)
        ~159.0  # Remaining PA (uses weighted PA/G calculation)
    """
    import json

    # Check for season-ending injury - return 0 immediately
    if player_row.get('season_ending_injury', 0) == 1:
        return 0.0

    team = player_row.get('Team', None)
    player_games = player_row.get('G', 0)

    # Extract multi-team metadata if present
    current_team = player_row.get('_multi_team_current', None)
    stint_data_json = player_row.get('_multi_team_stints', None)

    # Parse stint data if present
    stint_data = None
    if stint_data_json and isinstance(stint_data_json, str):
        try:
            stint_data = json.loads(stint_data_json)
        except (json.JSONDecodeError, TypeError):
            stint_data = None

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
            season_length=season_length,
            current_team=current_team,
            stint_data=stint_data
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
            season_length=season_length,
            current_team=current_team,
            stint_data=stint_data
        )
        return result['remaining_pa']

    else:
        return 0.0
