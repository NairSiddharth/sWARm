"""
Season splitting utilities for ROS training data.

Creates multiple training samples per player-season at different completion points,
enabling the ROS model to learn from any point in the season (not just fixed splits).
"""

import pandas as pd
import numpy as np
from typing import List, Tuple
from ..constants import FULL_SEASON_GAMES


def calculate_season_completion(
    games_played: float,
    team_games_played: float,
    season_length: int = FULL_SEASON_GAMES
) -> float:
    """
    Calculate season completion percentage based on games played.

    Uses team games played (not player games) for accurate season timing.

    Args:
        games_played: Player's games played
        team_games_played: Team's total games played so far
        season_length: Total games in season (default: 162)

    Returns:
        Season completion percentage (0.0 to 1.0)

    Example:
        >>> calculate_season_completion(70, 81, 162)
        0.5  # Midseason (All-Star break)
        >>> calculate_season_completion(40, 40, 162)
        0.247  # ~25% through season
    """
    return team_games_played / season_length


def create_multipoint_splits(
    full_season_df: pd.DataFrame,
    split_points: List[float] = [0.25, 0.5, 0.75],
    player_type: str = 'hitter',
    season_length: int = FULL_SEASON_GAMES
) -> pd.DataFrame:
    """
    Create multiple training samples per player-season at different completion points.

    For each player-season, generates N samples (one per split point) representing:
    "At X% of season completion, player had Y stats, then produced Z remaining WAR"

    This enables flexible ROS predictions at any point in the season (not just fixed splits).

    Args:
        full_season_df: Historical full-season data with columns:
            Required: playerid, Name, Team, Year, G, WAR
            Hitters: PA, H, HR, RBI, SB, BB, SO (counting stats)
                     AVG, OBP, SLG, wOBA, K%, BB% (rate stats)
            Pitchers: IP, W, SV, SO, BB, ER (counting stats)
                      ERA, K%, BB%, WHIP, FIP (rate stats)
        split_points: List of season completion percentages (default: [0.25, 0.5, 0.75])
        player_type: 'hitter' or 'pitcher'
        season_length: Total games in season (default: 162)

    Returns:
        DataFrame with N times more rows, each representing a split point:
        - Original columns (preserved for context)
        - split_point: Which split this row represents (0.25, 0.5, 0.75)
        - season_completion_pct: Same as split_point
        - games_played: Player games at this split point
        - team_games_played: Team games at this split point
        - current_[stat]: Player's stats through this point (counting stats scaled)
        - remaining_WAR: Actual WAR produced in remaining games (TARGET)
        - remaining_usage: Remaining PA/IP (for projection calculations)

    Example:
        >>> # Bobby Witt Jr. 2024: 161G, 709PA, 10.49 WAR (full season)
        >>> splits = create_multipoint_splits(df_2024, [0.5], 'hitter')
        >>> splits[splits['Name'] == 'Bobby Witt Jr.']
        playerid  Year  split_point  games_played  current_PA  remaining_WAR  ...
        594834    2024  0.5          80.5          354.5       5.25          ...

        # Interpretation: At midseason, Witt had ~355 PA, then produced 5.25 WAR ROS

    Note:
        - Counting stats (G, PA, IP, WAR, H, HR, etc.) are split proportionally
        - Rate stats (AVG, K%, ERA, etc.) remain constant (assumes uniform performance)
        - This is a simplification; real within-season variation is captured in features
    """
    # Normalize player ID column name (handle PlayerId, playerid, MLBAMID variations)
    player_id_col = None
    for col in ['playerid', 'PlayerId', 'MLBAMID']:
        if col in full_season_df.columns:
            player_id_col = col
            break

    if player_id_col is None:
        raise ValueError("No player ID column found. Expected one of: playerid, PlayerId, MLBAMID")

    # If not already 'playerid', rename for consistency
    if player_id_col != 'playerid':
        full_season_df = full_season_df.copy()
        full_season_df['playerid'] = full_season_df[player_id_col]

    # Validate required columns
    required_base = ['Year', 'G', 'WAR']
    usage_col = 'PA' if player_type == 'hitter' else 'IP'
    required_cols = required_base + [usage_col]

    missing = [col for col in required_cols if col not in full_season_df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Define counting stats to split (proportionally)
    if player_type == 'hitter':
        counting_stats = ['G', 'PA', 'WAR', 'H', 'HR', 'RBI', 'SB', 'BB', 'SO', '2B', '3B']
    else:
        counting_stats = ['G', 'IP', 'WAR', 'W', 'SV', 'SO', 'BB', 'ER', 'H', 'HR']

    # Only use counting stats that exist in the DataFrame
    counting_stats = [col for col in counting_stats if col in full_season_df.columns]

    # Identify rate stats (everything else that's not ID/meta cols)
    id_cols = ['playerid', 'Name', 'Team', 'Year', 'Pos', 'Position']
    rate_stats = [col for col in full_season_df.columns
                  if col not in counting_stats and col not in id_cols]

    # Create split rows
    split_rows = []

    for idx, row in full_season_df.iterrows():
        full_season_games = row['G']
        full_season_usage = row[usage_col]
        full_season_war = row['WAR']

        # Skip if insufficient data
        if pd.isna(full_season_games) or full_season_games < 20:
            continue
        if pd.isna(full_season_usage) or full_season_usage < (50 if player_type == 'hitter' else 20):
            continue
        if pd.isna(full_season_war):
            continue

        for split_point in split_points:
            # Calculate team games at this split point
            team_games_at_split = season_length * split_point

            # Calculate player games at this split point (proportional to team games)
            # Player might not play every game, so scale accordingly
            player_games_at_split = full_season_games * split_point

            # Calculate usage at this split point
            usage_at_split = full_season_usage * split_point

            # Calculate WAR at this split point
            war_at_split = full_season_war * split_point

            # Calculate remaining values
            remaining_games = full_season_games - player_games_at_split
            remaining_usage = full_season_usage - usage_at_split
            remaining_war = full_season_war - war_at_split

            # Create new row
            new_row = {}

            # Copy ID columns
            for col in id_cols:
                if col in row.index:
                    new_row[col] = row[col]

            # Add split metadata
            new_row['split_point'] = split_point
            new_row['season_completion_pct'] = split_point
            new_row['team_games_played'] = team_games_at_split

            # Scale counting stats to split point
            for stat in counting_stats:
                if stat in row.index and not pd.isna(row[stat]):
                    # Current value (at split point)
                    new_row[f'current_{stat}'] = row[stat] * split_point
                    # Keep original for reference
                    new_row[f'full_{stat}'] = row[stat]

            # Copy rate stats unchanged (assumes uniform distribution)
            for stat in rate_stats:
                if stat in row.index:
                    new_row[stat] = row[stat]

            # Add key derived values
            new_row['games_played'] = player_games_at_split
            new_row[f'current_{usage_col}'] = usage_at_split
            new_row['current_WAR'] = war_at_split
            new_row['remaining_games'] = remaining_games
            new_row[f'remaining_{usage_col}'] = remaining_usage
            new_row['remaining_WAR'] = remaining_war  # This is the TARGET

            # Add WAR rate (needed by ROSFeatureBuilder)
            if player_type == 'hitter' and usage_at_split > 0:
                new_row['WAR_per_600'] = (war_at_split / usage_at_split) * 600
            elif player_type == 'pitcher' and usage_at_split > 0:
                new_row['WAR_per_162'] = (war_at_split / usage_at_split) * 162
            else:
                new_row['WAR_per_600' if player_type == 'hitter' else 'WAR_per_162'] = 0.0

            split_rows.append(new_row)

    # Create DataFrame
    result_df = pd.DataFrame(split_rows)

    # Sort by player, year, split point
    if 'playerid' in result_df.columns:
        result_df = result_df.sort_values(['playerid', 'Year', 'split_point'])

    return result_df


def validate_split_data(split_df: pd.DataFrame, player_type: str = 'hitter') -> Tuple[bool, List[str]]:
    """
    Validate split data for training.

    Args:
        split_df: Output from create_multipoint_splits()
        player_type: 'hitter' or 'pitcher'

    Returns:
        Tuple of (is_valid, list of error messages)

    Example:
        >>> valid, errors = validate_split_data(splits, 'hitter')
        >>> if not valid:
        ...     print("Validation errors:", errors)
    """
    errors = []

    # Check required columns
    required_cols = [
        'playerid', 'Year', 'split_point', 'season_completion_pct',
        'games_played', 'team_games_played', 'remaining_WAR'
    ]

    usage_col = 'PA' if player_type == 'hitter' else 'IP'
    required_cols.extend([f'current_{usage_col}', f'remaining_{usage_col}'])

    missing = [col for col in required_cols if col not in split_df.columns]
    if missing:
        errors.append(f"Missing required columns: {missing}")

    # Check for NaN in critical columns
    if 'remaining_WAR' in split_df.columns:
        nan_count = split_df['remaining_WAR'].isna().sum()
        if nan_count > 0:
            errors.append(f"Found {nan_count} NaN values in remaining_WAR (target)")

    # Check split_point values are valid
    if 'split_point' in split_df.columns:
        invalid_splits = split_df[
            (split_df['split_point'] < 0) | (split_df['split_point'] > 1)
        ]
        if len(invalid_splits) > 0:
            errors.append(f"Found {len(invalid_splits)} invalid split_point values (must be 0-1)")

    # Check remaining_WAR is reasonable
    if 'remaining_WAR' in split_df.columns:
        extreme_values = split_df[
            (split_df['remaining_WAR'] < -3) | (split_df['remaining_WAR'] > 10)
        ]
        if len(extreme_values) > 0:
            errors.append(f"Found {len(extreme_values)} extreme remaining_WAR values (check data)")

    return len(errors) == 0, errors
