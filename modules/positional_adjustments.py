"""
Positional adjustment calculations for WAR prediction models

This module loads fielding/defensive data to determine player positions
and calculates positional WAR adjustments as features for ML models.
"""

import pandas as pd
import numpy as np
import glob
import os

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

def load_bp_fielding_data(data_dir=None):
    """
    Load BP fielding data with primary position determination

    Args:
        data_dir: Directory containing BP fielding CSV files

    Returns:
        DataFrame with columns: ['mlbid', 'Name', 'Season', 'Primary_Position', 'Total_Innings']
    """
    print("Loading BP fielding data for positional adjustments...")

    if data_dir is None:
        data_dir = r"C:\Users\nairs\Documents\GithubProjects\oWAR\MLB Player Data"

    # Load all BP fielding files
    fielding_files = glob.glob(os.path.join(data_dir, "BP_Data", "fielding", "bp_fielding_*.csv"))

    all_fielding_data = []

    for file in sorted(fielding_files):
        year = int(os.path.basename(file).split('_')[-1].replace('.csv', ''))

        try:
            df = pd.read_csv(file, encoding='utf-8-sig')

            # Add year info
            df['Season'] = year

            # Keep relevant columns
            fielding_cols = ['mlbid', 'Name', 'Season', 'Position', 'Games', 'Innings']
            df_filtered = df[fielding_cols].copy()

            # Filter out rows with missing critical data
            df_filtered = df_filtered.dropna(subset=['mlbid', 'Position'])

            all_fielding_data.append(df_filtered)

            print(f"   SUCCESS {year}: {len(df_filtered)} fielding records loaded")

        except Exception as e:
            print(f"   ERROR {year}: Error loading - {e}")

    if not all_fielding_data:
        print("ERROR: No BP fielding data loaded")
        return pd.DataFrame()

    # Combine all years
    combined_fielding = pd.concat(all_fielding_data, ignore_index=True)
    print(f"Combined BP fielding data: {len(combined_fielding)} total records")

    # Determine primary position for each player-year
    primary_positions = determine_primary_positions(combined_fielding)

    return primary_positions

def determine_primary_positions(fielding_df):
    """
    Determine primary position for each player-year based on innings played

    Args:
        fielding_df: DataFrame with fielding data

    Returns:
        DataFrame with primary position for each player-year
    """
    print("Determining primary positions based on innings played...")

    # Group by player-year and find position with most innings
    player_year_positions = []

    for (mlbid, season), group in fielding_df.groupby(['mlbid', 'Season']):
        # Handle missing innings data
        group_clean = group.copy()
        group_clean['Innings'] = group_clean['Innings'].fillna(0)

        # Find position with most innings (or games if innings missing)
        if group_clean['Innings'].sum() > 0:
            primary_pos_row = group_clean.loc[group_clean['Innings'].idxmax()]
            total_innings = group_clean['Innings'].sum()
        else:
            # Fall back to games if innings not available
            group_clean['Games'] = group_clean['Games'].fillna(0)
            primary_pos_row = group_clean.loc[group_clean['Games'].idxmax()]
            total_innings = group_clean['Games'].sum()  # Use games as proxy

        player_year_positions.append({
            'mlbid': mlbid,
            'Name': primary_pos_row['Name'],
            'Season': season,
            'Primary_Position': primary_pos_row['Position'],
            'Total_Innings': total_innings,
            'Positions_Played': group['Position'].tolist()
        })

    primary_positions_df = pd.DataFrame(player_year_positions)

    # Report summary
    position_counts = primary_positions_df['Primary_Position'].value_counts()
    print(f"Primary position distribution:")
    for pos, count in position_counts.head(10).items():
        print(f"   {pos}: {count}")

    multi_position_players = primary_positions_df[
        primary_positions_df['Positions_Played'].apply(len) > 1
    ]
    print(f"Multi-position players: {len(multi_position_players)}/{len(primary_positions_df)} ({len(multi_position_players)/len(primary_positions_df)*100:.1f}%)")

    return primary_positions_df

def load_fangraphs_defensive_data(data_dir=None):
    """
    Load FanGraphs defensive data as fallback for missing BP data

    Args:
        data_dir: Directory containing FanGraphs defensive CSV files

    Returns:
        DataFrame with columns: ['MLBAMID', 'Name', 'Season', 'Primary_Position']
    """
    print("Loading FanGraphs defensive data as fallback...")

    if data_dir is None:
        data_dir = r"C:\Users\nairs\Documents\GithubProjects\oWAR\MLB Player Data"

    # Load all FanGraphs defensive files
    defensive_files = glob.glob(os.path.join(data_dir, "FanGraphs_Data", "defensive", "fangraphs_defensive_standard_*.csv"))

    all_defensive_data = []

    for file in sorted(defensive_files):
        year = int(os.path.basename(file).split('_')[-1].replace('.csv', ''))

        try:
            df = pd.read_csv(file, encoding='utf-8-sig')

            # Add year info
            df['Season'] = year

            # Keep relevant columns
            defensive_cols = ['MLBAMID', 'Name', 'Season', 'Pos', 'Inn']
            df_filtered = df[defensive_cols].copy()

            # Rename for consistency
            df_filtered = df_filtered.rename(columns={'Pos': 'Primary_Position', 'Inn': 'Total_Innings'})

            # Filter out rows with missing critical data
            df_filtered = df_filtered.dropna(subset=['MLBAMID', 'Primary_Position'])

            all_defensive_data.append(df_filtered)

            print(f"   SUCCESS {year}: {len(df_filtered)} defensive records loaded")

        except Exception as e:
            print(f"   ERROR {year}: Error loading - {e}")

    if not all_defensive_data:
        print("WARNING: No FanGraphs defensive data loaded")
        return pd.DataFrame()

    # Combine all years
    combined_defensive = pd.concat(all_defensive_data, ignore_index=True)
    print(f"Combined FanGraphs defensive data: {len(combined_defensive)} total records")

    return combined_defensive

def calculate_positional_war_adjustment(primary_position, playing_time_pa, full_season_pa=600):
    """
    Calculate positional WAR adjustment based on position and playing time

    Args:
        primary_position: Player's primary position (e.g., 'SS', '1B')
        playing_time_pa: Player's plate appearances
        full_season_pa: Plate appearances for full season (default 600)

    Returns:
        Positional WAR adjustment value
    """
    if pd.isna(primary_position) or primary_position not in POSITION_WAR_ADJUSTMENTS:
        return 0.0

    # Get base adjustment for position
    base_adjustment = POSITION_WAR_ADJUSTMENTS[primary_position]

    # Pro-rate by playing time
    playing_time_ratio = playing_time_pa / full_season_pa

    # Cap at reasonable maximum (don't over-adjust for players with huge PA)
    playing_time_ratio = min(playing_time_ratio, 1.5)

    return base_adjustment * playing_time_ratio

def merge_positional_data_with_offensive(offensive_df, bp_positions, fg_positions=None):
    """
    Merge positional data with offensive statistics
    NOW PRIORITIZING COMPREHENSIVE FANGRAPHS DATA OVER BP DATA

    Args:
        offensive_df: DataFrame with offensive stats (must have mlbid/MLBAMID, Season/Year, PA)
        bp_positions: BP positional data (fallback)
        fg_positions: FanGraphs positional data (PRIMARY - comprehensive 0+ PA coverage)

    Returns:
        DataFrame with added Positional_WAR column
    """
    print("Merging positional data with offensive statistics...")

    df_enhanced = offensive_df.copy()

    # Determine year column name and ID column
    year_col = 'Season' if 'Season' in df_enhanced.columns else 'Year'
    id_col = 'mlbid' if 'mlbid' in df_enhanced.columns else 'MLBAMID'

    print(f"Using {year_col} as year column and {id_col} as ID column")

    # PRIMARY: Try to merge with FanGraphs comprehensive data first
    if fg_positions is not None and len(fg_positions) > 0:
        if id_col == 'MLBAMID':
            # Direct merge for WAR data
            merged = df_enhanced.merge(
                fg_positions[['MLBAMID', 'Season', 'Primary_Position', 'Total_Innings']],
                left_on=[id_col, year_col],
                right_on=['MLBAMID', 'Season'],
                how='left',
                suffixes=('', '_fg')
            )
            fg_merges = merged['Primary_Position'].notna().sum()
            print(f"PRIMARY FG position merges: {fg_merges}/{len(merged)} ({fg_merges/len(merged)*100:.1f}%)")
        elif id_col == 'mlbid':
            # For WARP data, mlbid = MLBAMID (same values, different column names)
            # Create a temporary copy of FG data with renamed column for merging
            fg_positions_renamed = fg_positions.copy()
            fg_positions_renamed = fg_positions_renamed.rename(columns={'MLBAMID': 'mlbid'})

            merged = df_enhanced.merge(
                fg_positions_renamed[['mlbid', 'Season', 'Primary_Position', 'Total_Innings']],
                left_on=[id_col, year_col],
                right_on=['mlbid', 'Season'],
                how='left',
                suffixes=('', '_fg')
            )
            fg_merges = merged['Primary_Position'].notna().sum()
            print(f"PRIMARY FG position merges: {fg_merges}/{len(merged)} ({fg_merges/len(merged)*100:.1f}%)")
        else:
            merged = df_enhanced.copy()
            merged['Primary_Position'] = None
            fg_merges = 0
            print(f"PRIMARY FG position merges: SKIPPED (unknown ID column: {id_col})")
    else:
        merged = df_enhanced.copy()
        merged['Primary_Position'] = None
        fg_merges = 0

    # FALLBACK: Fill missing positions with BP data
    if bp_positions is not None and len(bp_positions) > 0:
        missing_positions = merged['Primary_Position'].isna()

        if id_col == 'mlbid' and missing_positions.sum() > 0:
            # Merge BP data for missing positions
            bp_merge = merged[missing_positions].merge(
                bp_positions[['mlbid', 'Season', 'Primary_Position', 'Total_Innings']],
                left_on=[id_col, year_col],
                right_on=['mlbid', 'Season'],
                how='left',
                suffixes=('', '_bp')
            )

            # Update the main dataframe with BP data
            merged.loc[missing_positions, 'Primary_Position'] = bp_merge['Primary_Position'].values
            merged.loc[missing_positions, 'Total_Innings'] = bp_merge['Total_Innings'].values

            bp_merges = bp_merge['Primary_Position'].notna().sum()
            print(f"FALLBACK BP position merges: {bp_merges}/{missing_positions.sum()} missing records ({bp_merges/missing_positions.sum()*100:.1f}% of missing)")
        else:
            bp_merges = 0
            print(f"FALLBACK BP position merges: SKIPPED (no missing positions or ID mismatch)")

    # Calculate positional WAR adjustments
    merged['Positional_WAR'] = merged.apply(
        lambda row: calculate_positional_war_adjustment(
            row['Primary_Position'],
            row.get('PA', 600)  # Default to 600 if PA missing
        ), axis=1
    )

    # Report on adjustments
    total_with_positions = merged['Primary_Position'].notna().sum()
    print(f"TOTAL successful position merges: {total_with_positions}/{len(merged)} ({total_with_positions/len(merged)*100:.1f}%)")

    adjustment_stats = merged['Positional_WAR'].describe()
    print(f"Positional WAR adjustment stats:")
    print(f"   Mean: {adjustment_stats['mean']:.3f}")
    print(f"   Std: {adjustment_stats['std']:.3f}")
    print(f"   Range: {adjustment_stats['min']:.3f} to {adjustment_stats['max']:.3f}")

    # Clean up merge columns
    columns_to_keep = [col for col in merged.columns if not col.endswith('_fg') and not col.endswith('_bp')]
    merged_clean = merged[columns_to_keep]

    return merged_clean

def load_positional_adjustments_for_war_models(data_dir=None):
    """
    Complete pipeline to load and process positional data for WAR models

    Args:
        data_dir: Directory containing fielding/defensive CSV files

    Returns:
        Tuple of (bp_positions, fg_positions) DataFrames ready for merging
    """
    print("LOADING POSITIONAL DATA FOR WAR MODELS")
    print("=" * 60)

    # Load FanGraphs defensive data (NOW COMPREHENSIVE - 0+ PA coverage)
    fg_positions = load_fangraphs_defensive_data(data_dir)

    # Load BP fielding data (fallback for any gaps)
    bp_positions = load_bp_fielding_data(data_dir)

    print(f"\nPositional data summary (UPDATED PRIORITY):")
    print(f"   FG positions: {len(fg_positions)} player-seasons (PRIMARY - comprehensive)")
    print(f"   BP positions: {len(bp_positions)} player-seasons (FALLBACK)")

    return bp_positions, fg_positions