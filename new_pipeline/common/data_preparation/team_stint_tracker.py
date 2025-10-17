"""
Team Stint Tracking for Multi-Team Players

Provides functionality to track players who were traded during the season
and identify their current team for accurate ROS projections.

Uses efficient per-player Statcast queries with permanent caching to avoid
repeated API calls.
"""

from typing import Dict, List, Optional
from datetime import datetime
import pandas as pd
import numpy as np
import pybaseball
import json
import os

from ..constants import CACHE_DIR


# Default weighting parameters for multi-team usage calculations
DEFAULT_WEIGHTING_PARAMS = {
    'hitter': {
        'transition_midpoint': 5,      # Game 5 = 50/50 split
        'transition_complete': 10,     # Game 10 = 100% new team
        'historical_decay': 0.7,       # Each older stint = 70% of previous
        'curve_type': 'sigmoid'        # Smooth S-curve transition
    },
    'pitcher': {
        'transition_midpoint': 2,      # Appearance 2 = 50/50 split
        'transition_complete': 3,      # Appearance 3 = 100% new team
        'historical_decay': 0.6,       # Faster decay (pitchers adapt quicker?)
        'curve_type': 'sigmoid'
    }
}


def _load_cache(cache_path: str) -> Dict:
    """
    Load stint cache from JSON file.

    Args:
        cache_path: Path to cache file

    Returns:
        Dictionary of cached stint data, or empty dict if file doesn't exist
    """
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            print(f"Warning: Could not load cache from {cache_path}, creating new cache")
            return {}
    return {}


def _save_cache(cache: Dict, cache_path: str) -> None:
    """
    Save stint cache to JSON file.

    Args:
        cache: Dictionary of stint data to save
        cache_path: Path to cache file
    """
    print(f"[STINT CACHE DEBUG] Attempting to save {len(cache)} entries to {cache_path}")

    # Ensure cache directory exists
    cache_dir = os.path.dirname(cache_path)
    if cache_dir and not os.path.exists(cache_dir):
        print(f"[STINT CACHE DEBUG] Creating cache directory: {cache_dir}")
        os.makedirs(cache_dir)

    try:
        with open(cache_path, 'w') as f:
            json.dump(cache, f, indent=2)
        print(f"[STINT CACHE DEBUG] Successfully saved cache to {cache_path}")
    except IOError as e:
        print(f"[STINT CACHE DEBUG] ERROR: Could not save cache to {cache_path}: {e}")


def _sigmoid_curve(x: float, midpoint: float, steepness: float = 4.0) -> float:
    """
    Calculate sigmoid curve value.

    Args:
        x: Input value (current stint games/appearances)
        midpoint: X value where curve = 0.5
        steepness: How steep the curve is (higher = more abrupt transition)

    Returns:
        Sigmoid value between 0 and 1
    """
    # Sigmoid formula: 1 / (1 + e^(-steepness * (x - midpoint)))
    # Normalized so midpoint maps to 0.5
    try:
        return 1.0 / (1.0 + np.exp(-steepness * (x - midpoint) / midpoint))
    except (OverflowError, FloatingPointError):
        # Handle edge cases
        if x >= midpoint:
            return 1.0
        else:
            return 0.0


def get_player_chronological_stints(
    playerid_fg: int,
    year: int,
    player_type: str,
    cache_path: Optional[str] = None
) -> List[Dict]:
    """
    Get chronological team stints for a player, using cache when available.

    Args:
        playerid_fg: FanGraphs player ID
        year: Season year
        player_type: 'hitter' or 'pitcher'
        cache_path: Path to cache file (defaults to ./cache/team_stints_{year}.json)

    Returns:
        List of stints in chronological order:
        [
            {'team': 'MIA', 'G': 101, 'PA': 430, 'order': 0},
            {'team': 'NYY', 'G': 46, 'PA': 191, 'order': 1}
        ]

    Process:
        1. Load stint cache (JSON file)
        2. Check if player_year key exists (e.g., "20454_2024")
        3. If cached: return cached stint order
        4. If not cached:
           a. Load FanGraphs team="0,to" data for this player
           b. Call build_stint_cache() to determine chronological order
           c. Save to cache
           d. Return stint list
    """
    # Default cache path (use centralized CACHE_DIR like park factors)
    if cache_path is None:
        cache_path = str(CACHE_DIR / f'team_stints_{year}.json')

    print(f"[STINT CACHE DEBUG] Year={year}, Cache path={cache_path}")

    # Load cache
    cache = _load_cache(cache_path)
    print(f"[STINT CACHE DEBUG] Loaded cache with {len(cache)} existing entries")

    # Check if player is cached
    cache_key = f"{playerid_fg}_{year}"
    if cache_key in cache:
        print(f"Using cached stint data for player {playerid_fg} ({year})")
        return cache[cache_key]['stints']

    # Not cached - need to query
    print(f"Player {playerid_fg} ({year}) not in cache, querying FanGraphs and Statcast...")

    # Load FanGraphs data with team splits
    if player_type == 'hitter':
        fangraphs_data = pybaseball.batting_stats(year, team='0,to', qual=0)
    elif player_type == 'pitcher':
        fangraphs_data = pybaseball.pitching_stats(year, team='0,to', qual=0)
    else:
        raise ValueError(f"Invalid player_type: {player_type}. Must be 'hitter' or 'pitcher'")

    # Filter to this player
    player_stints = fangraphs_data[fangraphs_data['IDfg'] == playerid_fg]

    if len(player_stints) == 0:
        print(f"Warning: Player {playerid_fg} not found in {year} FanGraphs data")
        return []

    if len(player_stints) == 1:
        # Single team player - shouldn't be calling this function, but handle gracefully
        print(f"Note: Player {playerid_fg} only has one team stint in {year}")
        team = player_stints.iloc[0]['Team']
        games = player_stints.iloc[0]['G']
        usage_key = 'PA' if player_type == 'hitter' else 'IP'
        usage = player_stints.iloc[0][usage_key]

        stint_list = [{
            'team': team,
            'G': int(games),
            usage_key: float(usage),
            'order': 0
        }]

        # Cache it anyway
        cache[cache_key] = {
            'player_name': player_stints.iloc[0].get('Name', 'Unknown'),
            'player_type': player_type,
            'current_team': team,
            'stints': stint_list,
            'cached_date': str(datetime.now())
        }
        _save_cache(cache, cache_path)

        return stint_list

    # Multi-team player - determine chronological order
    stint_list = build_stint_cache(
        playerid_fg=playerid_fg,
        year=year,
        player_type=player_type,
        stint_stats=player_stints
    )

    # Save to cache
    cache[cache_key] = {
        'player_name': player_stints.iloc[0].get('Name', 'Unknown'),
        'player_type': player_type,
        'current_team': stint_list[-1]['team'] if stint_list else None,
        'stints': stint_list,
        'cached_date': str(datetime.now())
    }
    _save_cache(cache, cache_path)

    return stint_list


def build_stint_cache(
    playerid_fg: int,
    year: int,
    player_type: str,
    stint_stats: pd.DataFrame
) -> List[Dict]:
    """
    Determine chronological order of stints using Statcast (ONE-TIME QUERY).

    Args:
        playerid_fg: FanGraphs player ID
        year: Season year
        player_type: 'hitter' or 'pitcher'
        stint_stats: FanGraphs team="0,to" data for this player (multiple rows)

    Returns:
        List of stints in chronological order with stats

    Process:
        1. Convert FanGraphs ID to MLBAM ID via playerid_lookup()
        2. Query Statcast for last 7 days of season (to find current team)
        3. Extract batting/pitching team from most recent game
        4. Match current team to stint_stats to determine order
        5. Build chronological stint list
        6. Return stint list
    """
    # Convert FanGraphs ID to MLBAM ID
    try:
        # Try to get player name from stint_stats
        player_name = stint_stats.iloc[0].get('Name', '')
        if player_name:
            name_parts = player_name.split()
            if len(name_parts) >= 2:
                last_name = name_parts[-1]
                first_name = name_parts[0]
            else:
                # Single name? Use what we have
                last_name = player_name
                first_name = ''

            player_lookup = pybaseball.playerid_lookup(last_name, first_name)

            if len(player_lookup) == 0:
                print(f"Warning: Could not find player in ID lookup for {player_name}")
                # Fall back to using stint games as proxy for order (most games = earliest stint)
                return _fallback_stint_ordering(stint_stats, player_type)

            # Find matching FanGraphs ID
            matching = player_lookup[player_lookup['key_fangraphs'] == playerid_fg]
            if len(matching) == 0:
                print(f"Warning: No matching FanGraphs ID in lookup for {player_name}")
                return _fallback_stint_ordering(stint_stats, player_type)

            mlbam_id = matching.iloc[0]['key_mlbam']
            print(f"Found MLBAM ID: {mlbam_id} for {player_name}")
        else:
            print(f"Warning: No player name available for {playerid_fg}")
            return _fallback_stint_ordering(stint_stats, player_type)

    except Exception as e:
        print(f"Error in player ID lookup: {e}")
        return _fallback_stint_ordering(stint_stats, player_type)

    # Query Statcast for last 7 days to find current team
    try:
        end_date = f"{year}-10-01"
        start_date = f"{year}-09-24"

        if player_type == 'hitter':
            statcast_data = pybaseball.statcast_batter(start_date, end_date, mlbam_id)
        else:
            statcast_data = pybaseball.statcast_pitcher(start_date, end_date, mlbam_id)

        if statcast_data is None or len(statcast_data) == 0:
            print(f"Warning: No Statcast data for player {playerid_fg} in last 7 days of {year}")
            return _fallback_stint_ordering(stint_stats, player_type)

        # Extract current team from most recent game
        statcast_data = statcast_data.sort_values('game_date')
        most_recent_game = statcast_data.iloc[-1]

        if player_type == 'hitter':
            # Determine batting team from inning_topbot
            if most_recent_game['inning_topbot'] == 'Top':
                current_team = most_recent_game['away_team']
            else:
                current_team = most_recent_game['home_team']
        else:
            # Pitching team is opposite of batting team
            if most_recent_game['inning_topbot'] == 'Top':
                current_team = most_recent_game['home_team']
            else:
                current_team = most_recent_game['away_team']

        print(f"Current team identified: {current_team}")

    except Exception as e:
        print(f"Error querying Statcast: {e}")
        return _fallback_stint_ordering(stint_stats, player_type)

    # Build chronological stint list by matching current team
    stint_list = []
    usage_key = 'PA' if player_type == 'hitter' else 'IP'

    # Find current team in stint_stats
    current_stint_row = stint_stats[stint_stats['Team'] == current_team]

    if len(current_stint_row) == 0:
        print(f"Warning: Current team {current_team} not found in FanGraphs stints")
        return _fallback_stint_ordering(stint_stats, player_type)

    # Assign order: current team is last, others come before
    # For simplicity, order other teams by games played (more games = earlier)
    other_stints = stint_stats[stint_stats['Team'] != current_team]
    other_stints = other_stints.sort_values('G', ascending=False)  # Most games first

    order = 0
    for idx, row in other_stints.iterrows():
        stint_list.append({
            'team': row['Team'],
            'G': int(row['G']),
            usage_key: float(row[usage_key]),
            'order': order
        })
        order += 1

    # Add current team as last stint
    current_row = current_stint_row.iloc[0]
    stint_list.append({
        'team': current_row['Team'],
        'G': int(current_row['G']),
        usage_key: float(current_row[usage_key]),
        'order': order
    })

    return stint_list


def _fallback_stint_ordering(stint_stats: pd.DataFrame, player_type: str) -> List[Dict]:
    """
    Fallback method to order stints when Statcast is unavailable.

    Uses games played as proxy: more games = earlier stint (usually)

    Args:
        stint_stats: FanGraphs team="0,to" data for player
        player_type: 'hitter' or 'pitcher'

    Returns:
        List of stints ordered by games (descending)
    """
    print("Using fallback stint ordering (by games played)")

    stint_list = []
    usage_key = 'PA' if player_type == 'hitter' else 'IP'

    # Sort by games descending (most games = earliest stint, usually)
    sorted_stints = stint_stats.sort_values('G', ascending=False)

    for order, (idx, row) in enumerate(sorted_stints.iterrows()):
        stint_list.append({
            'team': row['Team'],
            'G': int(row['G']),
            usage_key: float(row[usage_key]),
            'order': order
        })

    return stint_list


def calculate_weighted_usage(
    stints: List[Dict],
    usage_key: str,
    games_key: str = 'G',
    weighting_params: Optional[Dict] = None
) -> Dict:
    """
    Calculate weighted usage rate using exponential recency decay + transition curves.

    Weighting Strategy:
        Each stint gets two weights multiplied together:
        1. Recency weight: historical_decay_rate ^ stints_ago
        2. Transition weight: Based on current stint size and curve

    Args:
        stints: List from get_player_chronological_stints()
        usage_key: Column name for usage ('IP' or 'PA')
        games_key: Column name for games
        weighting_params: Dict with keys:
            - transition_midpoint: Game/appearance count for 50/50 weight
            - transition_complete: Game/appearance count for 100% current
            - historical_decay: Decay rate per older stint (e.g., 0.7)
            - curve_type: 'sigmoid', 'linear', or 'exponential'

    Returns:
        {
            'weighted_usage_per_game': float,
            'total_games': int,
            'total_usage': float,
            'current_team': str,
            'stint_weights': List[float]  # For debugging
        }
    """
    if len(stints) == 0:
        return {
            'weighted_usage_per_game': 0.0,
            'total_games': 0,
            'total_usage': 0.0,
            'current_team': None,
            'stint_weights': []
        }

    # Use default params if not provided
    if weighting_params is None:
        # Guess player type from usage_key
        player_type = 'hitter' if usage_key == 'PA' else 'pitcher'
        weighting_params = DEFAULT_WEIGHTING_PARAMS[player_type]

    # Extract parameters
    transition_midpoint = weighting_params.get('transition_midpoint', 5)
    transition_complete = weighting_params.get('transition_complete', 10)
    historical_decay = weighting_params.get('historical_decay', 0.7)
    curve_type = weighting_params.get('curve_type', 'sigmoid')

    # Get current stint info
    current_stint = stints[-1]
    current_games = current_stint.get(games_key, 0)
    current_team = current_stint['team']

    # Calculate transition progress (0 = just traded, 1 = fully transitioned)
    if curve_type == 'sigmoid':
        # Sigmoid curve from transition_start to transition_complete
        # midpoint = 50% transition
        transition_progress = _sigmoid_curve(
            current_games,
            midpoint=transition_midpoint,
            steepness=4.0
        )
    elif curve_type == 'linear':
        # Linear interpolation
        transition_progress = min(1.0, max(0.0,
            (current_games - 1) / max(1, transition_complete - 1)
        ))
    else:  # exponential
        # Exponential approach to 1.0
        transition_progress = 1.0 - np.exp(-current_games / transition_midpoint)

    # Calculate weights for each stint
    stint_weights = []
    weighted_sum_usage = 0.0
    weighted_sum_games = 0.0

    for i, stint in enumerate(stints):
        stints_ago = len(stints) - 1 - i  # 0 for current, 1 for previous, etc.

        # Recency weight (exponential decay)
        recency_weight = historical_decay ** stints_ago

        # Transition weight
        if stints_ago == 0:
            # Current stint: weight increases as player settles in
            transition_weight = transition_progress
        else:
            # Old stints: weight decreases as player settles into new team
            transition_weight = 1.0 - transition_progress

        # Final weight is product of both
        final_weight = recency_weight * transition_weight
        stint_weights.append(final_weight)

        # Weighted accumulation
        games = stint.get(games_key, 0)
        usage = stint.get(usage_key, 0)

        weighted_sum_usage += usage * final_weight
        weighted_sum_games += games * final_weight

    # Calculate weighted usage per game
    if weighted_sum_games > 0:
        weighted_usage_per_game = weighted_sum_usage / weighted_sum_games
    else:
        weighted_usage_per_game = 0.0

    # Total stats (unweighted)
    total_games = sum(stint.get(games_key, 0) for stint in stints)
    total_usage = sum(stint.get(usage_key, 0) for stint in stints)

    return {
        'weighted_usage_per_game': weighted_usage_per_game,
        'total_games': int(total_games),
        'total_usage': float(total_usage),
        'current_team': current_team,
        'stint_weights': stint_weights
    }
