"""
Positional Adjustment Module for WAR Models
==========================================

Legacy compatibility module for positional adjustments.
Provides functions needed by temp_modeling.py and other legacy components.
"""

import pandas as pd
import numpy as np
import glob
import os
from typing import Dict, List, Optional, Tuple

# Standard positional WAR adjustments per 600 PA (FanGraphs scale)
POSITION_WAR_ADJUSTMENTS = {
    'C': +1.25,   # Catcher: highest positive adjustment
    'SS': +0.75,  # Shortstop: high positive adjustment
    'CF': +0.25,  # Center field: small positive adjustment
    '3B': +0.25,  # Third base: small positive adjustment
    '2B': +0.3,   # Second base: small positive adjustment
    'LF': -0.7,   # Left field: negative adjustment
    'RF': -0.75,  # Right field: negative adjustment
    '1B': -1.25,  # First base: large negative adjustment
    'DH': -1.75,  # Designated hitter: largest negative adjustment
    'P': 0.0      # Pitcher: neutral (they rarely hit)
}

def load_positional_adjustments_for_war_models(data_dir=None):
    """
    Load positional data for WAR models.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: BP positions, FG positions
    """
    if data_dir is None:
        data_dir = r"C:\Users\nairs\Documents\GithubProjects\oWAR\MLB Player Data"

    print("Loading positional adjustments for WAR models...")

    # Load FanGraphs defensive data
    fg_positions = load_fangraphs_defensive_data(data_dir)

    # Load BP fielding data
    bp_positions = load_bp_fielding_data(data_dir)

    print(f"Loaded positional data:")
    print(f"  FG positions: {len(fg_positions) if fg_positions is not None else 0}")
    print(f"  BP positions: {len(bp_positions) if bp_positions is not None else 0}")

    return bp_positions, fg_positions

def load_fangraphs_defensive_data(data_dir):
    """Load FanGraphs defensive data."""
    defensive_files = glob.glob(
        os.path.join(data_dir, "FanGraphs_Data", "defensive", "fangraphs_defensive_standard_*.csv")
    )

    if not defensive_files:
        print("  Warning: No FanGraphs defensive files found")
        return pd.DataFrame()

    all_defensive_data = []

    for file in sorted(defensive_files):
        try:
            year = int(os.path.basename(file).split('_')[-1].replace('.csv', ''))
            df = pd.read_csv(file, encoding='utf-8-sig')

            df['Season'] = year
            defensive_cols = ['MLBAMID', 'Name', 'Season', 'Pos', 'Inn']

            # Only include columns that exist
            available_cols = [col for col in defensive_cols if col in df.columns]
            if len(available_cols) < 3:  # Need at least MLBAMID, Name, Season
                continue

            df_filtered = df[available_cols].copy()
            df_filtered = df_filtered.rename(columns={'Pos': 'Primary_Position', 'Inn': 'Total_Innings'})
            df_filtered = df_filtered.dropna(subset=['MLBAMID'])

            all_defensive_data.append(df_filtered)

        except Exception as e:
            print(f"    Error loading {file}: {e}")

    if all_defensive_data:
        combined_defensive = pd.concat(all_defensive_data, ignore_index=True)
        print(f"  FG defensive data: {len(combined_defensive)} records")
        return combined_defensive

    return pd.DataFrame()

def load_bp_fielding_data(data_dir):
    """Load BP fielding data."""
    fielding_files = glob.glob(
        os.path.join(data_dir, "BP_Data", "fielding", "bp_fielding_*.csv")
    )

    if not fielding_files:
        print("  Warning: No BP fielding files found")
        return pd.DataFrame()

    all_fielding_data = []

    for file in sorted(fielding_files):
        try:
            year = int(os.path.basename(file).split('_')[-1].replace('.csv', ''))
            df = pd.read_csv(file, encoding='utf-8-sig')

            df['Season'] = year
            fielding_cols = ['mlbid', 'Name', 'Season', 'Position', 'Games', 'Innings']

            # Only include columns that exist
            available_cols = [col for col in fielding_cols if col in df.columns]
            if len(available_cols) < 3:  # Need at least mlbid, Name, Season
                continue

            df_filtered = df[available_cols].copy()
            df_filtered = df_filtered.dropna(subset=['mlbid'])

            all_fielding_data.append(df_filtered)

        except Exception as e:
            print(f"    Error loading {file}: {e}")

    if all_fielding_data:
        combined_fielding = pd.concat(all_fielding_data, ignore_index=True)
        # Determine primary positions
        primary_positions = determine_primary_positions(combined_fielding)
        print(f"  BP fielding data: {len(primary_positions)} player-seasons")
        return primary_positions

    return pd.DataFrame()

def determine_primary_positions(fielding_df):
    """Determine primary position for each player-year based on innings played."""
    if len(fielding_df) == 0:
        return pd.DataFrame()

    player_year_positions = []

    for (mlbid, season), group in fielding_df.groupby(['mlbid', 'Season']):
        group_clean = group.copy()

        # Handle innings
        if 'Innings' in group_clean.columns:
            group_clean['Innings'] = group_clean['Innings'].fillna(0)
            if group_clean['Innings'].sum() > 0:
                primary_pos_row = group_clean.loc[group_clean['Innings'].idxmax()]
                total_innings = group_clean['Innings'].sum()
            else:
                # Fall back to games
                if 'Games' in group_clean.columns:
                    group_clean['Games'] = group_clean['Games'].fillna(0)
                    primary_pos_row = group_clean.loc[group_clean['Games'].idxmax()]
                    total_innings = group_clean['Games'].sum()
                else:
                    primary_pos_row = group_clean.iloc[0]
                    total_innings = 0
        else:
            # No innings, use games
            if 'Games' in group_clean.columns:
                group_clean['Games'] = group_clean['Games'].fillna(0)
                primary_pos_row = group_clean.loc[group_clean['Games'].idxmax()]
                total_innings = group_clean['Games'].sum()
            else:
                primary_pos_row = group_clean.iloc[0]
                total_innings = 0

        player_year_positions.append({
            'mlbid': mlbid,
            'Name': primary_pos_row['Name'],
            'Season': season,
            'Primary_Position': primary_pos_row.get('Position', 'OF'),
            'Total_Innings': total_innings
        })

    return pd.DataFrame(player_year_positions)

def calculate_positional_adjustment(primary_position, playing_time_pa, full_season_pa=600):
    """
    Calculate positional WAR adjustment based on position and playing time.

    Args:
        primary_position: Player's primary position (e.g., 'SS', '1B')
        playing_time_pa: Player's plate appearances
        full_season_pa: Plate appearances for full season (default 600)

    Returns:
        Positional WAR adjustment value
    """
    if pd.isna(primary_position) or primary_position not in POSITION_WAR_ADJUSTMENTS:
        return 0.0

    base_adjustment = POSITION_WAR_ADJUSTMENTS[primary_position]
    playing_time_ratio = playing_time_pa / full_season_pa
    playing_time_ratio = min(playing_time_ratio, 1.5)  # Cap at reasonable maximum

    return base_adjustment * playing_time_ratio

def merge_positional_data_with_offensive(df_enhanced, bp_positions, fg_positions):
    """
    Merge positional adjustments with offensive data.

    Args:
        df_enhanced: DataFrame with offensive stats
        bp_positions: BP positional data
        fg_positions: FanGraphs positional data

    Returns:
        DataFrame with added Positional_WAR column
    """
    print("Merging positional data with offensive stats...")

    # Make a copy to avoid modifying original
    result_df = df_enhanced.copy()

    # Determine column names
    year_col = 'Season' if 'Season' in result_df.columns else 'Year'
    id_col = 'mlbid' if 'mlbid' in result_df.columns else 'MLBAMID'

    # Initialize position columns
    result_df['Primary_Position'] = None
    result_df['Total_Innings'] = None

    # Try FanGraphs data first (comprehensive coverage)
    if fg_positions is not None and len(fg_positions) > 0:
        if id_col == 'MLBAMID' and 'MLBAMID' in fg_positions.columns:
            merged = result_df.merge(
                fg_positions[['MLBAMID', 'Season', 'Primary_Position', 'Total_Innings']],
                left_on=[id_col, year_col],
                right_on=['MLBAMID', 'Season'],
                how='left',
                suffixes=('', '_fg')
            )
            result_df['Primary_Position'] = merged['Primary_Position_fg']
            result_df['Total_Innings'] = merged['Total_Innings_fg']
        elif id_col == 'mlbid':
            # MLBAMID = mlbid, just rename for merge
            if 'MLBAMID' in fg_positions.columns:
                fg_renamed = fg_positions.copy()
                fg_renamed = fg_renamed.rename(columns={'MLBAMID': 'mlbid'})

                merged = result_df.merge(
                    fg_renamed[['mlbid', 'Season', 'Primary_Position', 'Total_Innings']],
                    left_on=[id_col, year_col],
                    right_on=['mlbid', 'Season'],
                    how='left',
                    suffixes=('', '_fg')
                )
                result_df['Primary_Position'] = merged['Primary_Position_fg']
                result_df['Total_Innings'] = merged['Total_Innings_fg']

    # Fill missing positions with BP data
    if bp_positions is not None and len(bp_positions) > 0 and id_col == 'mlbid':
        missing_positions = result_df['Primary_Position'].isna()

        if missing_positions.sum() > 0:
            bp_merge = result_df[missing_positions].merge(
                bp_positions[['mlbid', 'Season', 'Primary_Position', 'Total_Innings']],
                left_on=[id_col, year_col],
                right_on=['mlbid', 'Season'],
                how='left',
                suffixes=('', '_bp')
            )

            result_df.loc[missing_positions, 'Primary_Position'] = bp_merge['Primary_Position_bp'].values
            result_df.loc[missing_positions, 'Total_Innings'] = bp_merge['Total_Innings_bp'].values

    # Calculate positional WAR adjustments
    result_df['Positional_WAR'] = result_df.apply(
        lambda row: calculate_positional_adjustment(
            row['Primary_Position'],
            row.get('PA', 600)  # Default to 600 if PA missing
        ), axis=1
    )

    # Report success rate
    total_with_positions = result_df['Primary_Position'].notna().sum()
    print(f"  Position assignment: {total_with_positions}/{len(result_df)} ({total_with_positions/len(result_df)*100:.1f}%)")

    if total_with_positions > 0:
        adj_stats = result_df['Positional_WAR'].describe()
        print(f"  Positional WAR range: {adj_stats['min']:.3f} to {adj_stats['max']:.3f} (mean: {adj_stats['mean']:.3f})")

    return result_df