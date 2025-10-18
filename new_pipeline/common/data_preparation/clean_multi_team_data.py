"""
Multi-Team Player Data Cleaning

Pre-processes raw FanGraphs data to clean multi-team players before ROS projections.
Replaces "- - -" team designation with comma-separated team list and attaches
stint metadata for weighted usage calculations.
"""

from typing import Dict, List
import pandas as pd
import json
from pybaseball import playerid_reverse_lookup
from .team_stint_tracker import get_player_chronological_stints


def clean_multi_team_players(
    df: pd.DataFrame,
    year: int,
    player_type: str
) -> pd.DataFrame:
    """
    Clean multi-team player data by replacing "- - -" with team list and adding metadata.

    Process:
        1. Identify multi-team players (Team == "- - -" in input df)
        2. For each multi-team player:
           a. Get chronological stints from team_stint_tracker
           b. Build comma-separated team list: "MIA, NYY"
           c. Replace "- - -" in Team column with team list
           d. Extract current team (last in stint list)
           e. Attach stint metadata to row for later use in projections
        3. Return cleaned DataFrame

    Args:
        df: Input DataFrame from FanGraphs (may contain "- - -" teams)
        year: Season year
        player_type: 'hitter' or 'pitcher'

    Returns:
        Cleaned DataFrame with:
        - 'Team': Modified (replaces "- - -" with "MIA, NYY")
        - '_multi_team_current': Current team abbreviation (e.g., "NYY")
        - '_multi_team_stints': JSON string of stint data (for weighted calculation)

    Example transformation:
    BEFORE:
    | MLBAMID | Name              | Team  | G   | IP   |
    |---------|-------------------|-------|-----|------|
    | 592351  | Sean Newcomb      | - - - | 27  | 64.0 |

    AFTER:
    | MLBAMID | Name          | Team      | G   | IP   | _multi_team_current | _multi_team_stints |
    |---------|---------------|-----------|-----|------|---------------------|-------------------|
    | 592351  | Sean Newcomb  | MIA, ATL  | 27  | 64.0 | ATL                 | [{"team":"MIA",...}] |
    """
    # Make a copy to avoid modifying original
    df_clean = df.copy()

    # Initialize new columns if they don't exist
    if '_multi_team_current' not in df_clean.columns:
        df_clean['_multi_team_current'] = None
    if '_multi_team_stints' not in df_clean.columns:
        df_clean['_multi_team_stints'] = None

    # Find multi-team players
    multi_team_mask = df_clean['Team'] == '- - -'
    multi_team_players = df_clean[multi_team_mask]

    if len(multi_team_players) == 0:
        print(f"No multi-team players found in {player_type} data for {year}")
        return df_clean

    print(f"Found {len(multi_team_players)} multi-team {player_type}(s) to clean")

    # Process each multi-team player
    for idx, player in multi_team_players.iterrows():
        player_name = player.get('Name', 'Unknown')
        mlbam_id = player.get('MLBAMID', None)

        # Try to get FanGraphs playerid directly from the data first
        playerid_fg = player.get('PlayerId', None)

        if playerid_fg is not None and not pd.isna(playerid_fg):
            # FanGraphs CSV includes playerid - use it directly
            try:
                playerid_fg = int(playerid_fg)
            except (ValueError, TypeError):
                playerid_fg = None  # Invalid format, try fallback

        # Fallback: Convert MLBAM ID to FanGraphs ID via pybaseball
        if playerid_fg is None:
            if mlbam_id is None or pd.isna(mlbam_id):
                print(f"  Warning: No playerid or MLBAM ID for {player_name}, skipping")
                continue

            try:
                id_lookup = playerid_reverse_lookup([mlbam_id], key_type='mlbam')
                if len(id_lookup) == 0:
                    print(f"  Warning: Could not find FanGraphs ID for {player_name} (MLBAM {mlbam_id}), skipping")
                    continue

                playerid_fg = int(id_lookup.iloc[0]['key_fangraphs'])

            except Exception as e:
                print(f"  Error converting MLBAM to FanGraphs ID for {player_name}: {e}")
                continue

        try:
            # Get chronological stints
            stints = get_player_chronological_stints(
                playerid_fg=playerid_fg,
                year=year,
                player_type=player_type
            )

            if len(stints) == 0:
                print(f"  Warning: No stints found for {player_name} (ID {playerid_fg})")
                continue

            # Build comma-separated team list (chronological order)
            team_list = ', '.join([stint['team'] for stint in stints])

            # Get current team (last in stint list)
            current_team = stints[-1]['team']

            # Update DataFrame
            df_clean.at[idx, 'Team'] = team_list
            df_clean.at[idx, '_multi_team_current'] = current_team
            df_clean.at[idx, '_multi_team_stints'] = json.dumps(stints)

            print(f"  {player_name}: {team_list} (current: {current_team})")

        except Exception as e:
            print(f"  Error processing {player_name} (ID {playerid_fg}): {e}")
            continue

    # Verify cleaning worked
    remaining_multi_team = sum(df_clean['Team'] == '- - -')
    if remaining_multi_team > 0:
        print(f"Warning: {remaining_multi_team} players still have '- - -' team designation")

    return df_clean
