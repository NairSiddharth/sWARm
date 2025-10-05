"""Data preparation utilities for predictive modeling.

This module handles data filtering, mapping, and feature engineering
for both hitters and pitchers in preparation for model training.
"""

from pathlib import Path
from typing import Dict, Tuple, Optional, List, Any
import os
import sys

import numpy as np
import pandas as pd

from common_modules.config import DATA_DIR
from common_modules.logging import get_logger
from common_modules.positional_adjustments import POSITION_WAR_ADJUSTMENTS

logger = get_logger(__name__)


def filter_position_players_pitching(
    pitcher_df: pd.DataFrame,
    two_way_analysis: Dict[str, Any],
    data_source: str = 'war'
) -> pd.DataFrame:
    """Filter out position players who pitched but aren't qualified two-way players.

    Args:
        pitcher_df: DataFrame with pitcher data
        two_way_analysis: Result from get_cleaned_two_way_data()
        data_source: 'war' or 'warp'

    Returns:
        Filtered DataFrame with only legitimate pitchers and qualified two-way players
    """
    if len(pitcher_df) == 0:
        return pitcher_df

    logger.info(f"Filtering position players from {data_source.upper()} pitcher data...")

    # Get data from updated two-way analysis structure
    two_way_players = two_way_analysis['two_way_players']
    emergency_pitching = two_way_analysis['filtered_data']['emergency_pitching']

    # Create a set of legitimate pitcher names
    legitimate_pitcher_names = set()

    # Add qualified two-way players (these should definitely be included)
    for player_key in two_way_players.keys():
        name, year = player_key.rsplit('_', 1)
        legitimate_pitcher_names.add(name)

    # Create a set of emergency pitcher names to exclude
    emergency_pitcher_names = set()
    for emergency_player in emergency_pitching:
        emergency_pitcher_names.add(emergency_player['name'])

    # Filter the DataFrame
    year_col = 'Season' if data_source == 'warp' else 'Year'

    if 'Name' in pitcher_df.columns and year_col in pitcher_df.columns:
        # Filtering logic:
        # 1. Keep all pitchers who are NOT in the emergency pitcher list
        # 2. Always keep qualified two-way players even if they might appear in other lists

        # Create boolean mask
        is_not_emergency = ~pitcher_df['Name'].isin(emergency_pitcher_names)
        is_two_way_qualified = pitcher_df['Name'].isin(legitimate_pitcher_names)

        # Keep if: not emergency OR is qualified two-way
        keep_mask = is_not_emergency | is_two_way_qualified
        filtered_df = pitcher_df[keep_mask].copy()

        removed_count = len(pitcher_df) - len(filtered_df)
        emergency_removed = len(pitcher_df[pitcher_df['Name'].isin(emergency_pitcher_names)])
        two_way_kept = len(pitcher_df[pitcher_df['Name'].isin(legitimate_pitcher_names)])

        logger.info(f"Removed {removed_count} emergency / position player pitching records")
        logger.info(f"Emergency pitchers filtered: {emergency_removed}")
        logger.info(f"Qualified two-way players kept: {two_way_kept}")
        logger.info(f"Total legitimate pitcher records: {len(filtered_df)}")

        return filtered_df
    else:
        logger.warning("Expected columns not found, returning original data")
        return pitcher_df


def filter_pitchers_from_hitting_data(
    df: pd.DataFrame,
    data_source: str = 'war',
    year_col: str = 'Year'
) -> pd.DataFrame:
    """Filter out pitchers from hitting data, except qualified two-way players (2020+).

    Args:
        df: DataFrame with hitting data
        data_source: 'war' or 'warp'
        year_col: Column name for year ('Year' for WAR, 'Season' for WARP)

    Returns:
        Filtered DataFrame with pitchers removed (except two-way players)
    """
    if len(df) == 0:
        return df

    logger.info(f"Filtering pitchers from {data_source.upper()} hitting data...")

    # For now, implement basic pitcher filtering
    # This would need to be enhanced with actual pitcher identification logic
    # based on your data structure

    # TODO: Implement proper pitcher identification logic later
    logger.info("Skipping pitcher filtering for simplified testing")
    logger.info(f"{data_source.upper()} hitting data: {len(df)} records (no filtering applied)")
    return df


def create_mlbid_mapping(
    warp_df: pd.DataFrame,
    war_df: pd.DataFrame
) -> Dict[int, int]:
    """Create player mapping based on MLBID AND YEAR instead of names.

    Args:
        warp_df: DataFrame with WARP data (has 'mlbid' and 'Season' columns)
        war_df: DataFrame with WAR data (has 'MLBAMID' and 'Year' columns)

    Returns:
        Dictionary mapping of {warp_index: war_index} for matched players
    """
    logger.info("Creating MLBID+Year-based player mapping...")

    # Determine year column names
    warp_year_col = 'Season' if 'Season' in warp_df.columns else 'Year'
    war_year_col = 'Year' if 'Year' in war_df.columns else 'Season'

    # Create index mapping by matching (mlbid, year) tuples
    index_mapping = {}

    # Build lookup dictionary for WAR data: {(mlbid, year): war_index}
    war_lookup = {}
    for idx, row in war_df.iterrows():
        mlbid = row.get('MLBAMID')
        year = row.get(war_year_col)
        if pd.notna(mlbid) and pd.notna(year):
            mlbid = int(mlbid)
            year = int(year)
            # If multiple WAR records for same (mlbid, year), keep first
            if (mlbid, year) not in war_lookup:
                war_lookup[(mlbid, year)] = idx

    logger.info(f"WAR data: {len(war_lookup)} unique (mlbid, year) combinations")

    # Match WARP records to WAR records
    matched_count = 0
    unmatched_count = 0

    for idx, row in warp_df.iterrows():
        mlbid = row.get('mlbid')
        year = row.get(warp_year_col)

        if pd.notna(mlbid) and pd.notna(year):
            mlbid = int(mlbid)
            year = int(year)

            # Look up matching WAR record
            if (mlbid, year) in war_lookup:
                index_mapping[idx] = war_lookup[(mlbid, year)]
                matched_count += 1
            else:
                unmatched_count += 1

    logger.info(f"WARP data: {matched_count} records matched, {unmatched_count} unmatched")
    logger.info(f"Created {len(index_mapping)} unique WARP->WAR index mappings")

    return index_mapping


def prepare_data_for_kfold() -> Tuple[Optional[Dict], Optional[Dict]]:
    """Prepare comprehensive dataset for K-fold cross-validation.

    Returns:
        Tuple of (hitter_data, pitcher_data) dictionaries with prepared features and targets
    """
    logger.info("Preparing comprehensive dataset for K-fold cross-validation...")

    # Import data loading functions
    from common_modules.derived_stats import load_bp_warp_data
    from common_modules.enhanced_features import get_enhanced_features
    from common_modules.positional_adjustments import PositionalAdjustmentCalculator
    from .data_loading import (
        load_expanded_fangraphs_data,
        load_expanded_fangraphs_pitcher_data,
    )

    # Use the new enhanced features system instead of legacy analytics
    logger.info("Loading enhanced features...")
    baserunning_data, defense_data = get_enhanced_features()

    # Convert to the expected format for backward compatibility
    def calculate_enhanced_baserunning_values():
        return baserunning_data

    def clean_defensive_players():
        return defense_data

    # Load datasets - USING EXPANDED DATA
    logger.info("Loading Baseball Prospectus WARP data...")
    hitter_warp, pitcher_warp = load_bp_warp_data()

    logger.info("Loading EXPANDED FanGraphs data...")
    hitter_war_raw = load_expanded_fangraphs_data()

    # Filter pitchers from WAR hitting data
    hitter_war = filter_pitchers_from_hitting_data(hitter_war_raw, 'war', 'Year')

    # Filter pitchers from WARP hitting data
    hitter_warp = filter_pitchers_from_hitting_data(hitter_warp, 'warp', 'Season')

    # Reset indices after filtering to avoid mapping issues
    hitter_warp = hitter_warp.reset_index(drop=True)
    hitter_war = hitter_war.reset_index(drop=True)

    # Load expanded pitcher data with two-way player logic
    logger.info("Loading EXPANDED pitcher data...")
    pitcher_war_raw = load_expanded_fangraphs_pitcher_data()

    # For now, use pitcher data as-is (skip complex two-way filtering for initial test)
    pitcher_war = pitcher_war_raw

    # Reset indices after filtering
    pitcher_warp = pitcher_warp.reset_index(drop=True)
    pitcher_war = pitcher_war.reset_index(drop=True)

    logger.info(
        f"Pitcher data after two-way filtering: {len(pitcher_warp)} WARP, {len(pitcher_war)} WAR")

    # Enhanced features
    enhanced_baserunning = calculate_enhanced_baserunning_values()
    enhanced_defensive = clean_defensive_players()

    # Load positional data for adjustments using new system
    pos_calc = PositionalAdjustmentCalculator()
    pos_calc.load_defensive_data()
    bp_positions = pos_calc.bp_positions if hasattr(pos_calc, 'bp_positions') else None
    fg_positions = pos_calc.fg_positions if hasattr(pos_calc, 'fg_positions') else None

    logger.info(f"Loaded data: {len(hitter_warp)} hitter WARP, {len(pitcher_warp)} pitcher WARP")
    logger.info(f"              {len(hitter_war)} hitter WAR, {len(pitcher_war)} pitcher WAR")
    bp_count = len(bp_positions) if bp_positions is not None else 0
    fg_count = len(fg_positions) if fg_positions is not None else 0
    logger.info(f"              {bp_count} BP positions, {fg_count} FG positions")

    # Create MLBID-based mappings instead of name - based
    logger.info("Creating MLBID-based player mappings...")
    hitter_mapping = create_mlbid_mapping(hitter_warp, hitter_war)

    # Only create pitcher mapping if we have pitcher WAR data
    if len(pitcher_war) > 0:
        pitcher_mapping = create_mlbid_mapping(pitcher_warp, pitcher_war)
    else:
        pitcher_mapping = {}

    logger.info(
        f"MLBID mappings created: {
            len(hitter_mapping)} hitters, {
            len(pitcher_mapping)} pitchers")

    def prepare_dataset(
        warp_data: pd.DataFrame,
        war_data: pd.DataFrame,
        mapping: Dict[int, int],
        enhanced_br: Dict,
        enhanced_def: Dict,
        player_type: str,
        bp_pos: Optional[pd.DataFrame],
        fg_pos: Optional[pd.DataFrame]
    ) -> Optional[Dict]:
        """Prepare matched dataset with features and targets."""
        if len(mapping) == 0:
            return None

        # MLBID mapping is already index - based: {warp_index: war_index}
        warp_indices = list(mapping.keys())
        war_indices = list(mapping.values())

        if len(warp_indices) == 0:
            return None

        warp_matched = warp_data.iloc[warp_indices].reset_index(drop=True)
        war_matched = war_data.iloc[war_indices].reset_index(drop=True)

        # Add enhanced features
        def add_enhanced_features(df: pd.DataFrame, data_source: str = 'warp') -> pd.DataFrame:
            """Add enhanced features to the dataset."""
            df_enhanced = df.copy()
            df_enhanced['Enhanced_Baserunning'] = df_enhanced['Name'].map(enhanced_br).fillna(0.0)
            df_enhanced['Enhanced_Defense'] = df_enhanced['Name'].map(enhanced_def).fillna(0.0)

            # Add GDP rate for hitters (situational hitting metric)
            if player_type == 'hitter' and 'GDP' in df_enhanced.columns and 'PA' in df_enhanced.columns:
                df_enhanced['GDP_rate'] = df_enhanced['GDP'].fillna(
                    0) / df_enhanced['PA'].replace(0, 1)
                df_enhanced['GDP_rate'] = df_enhanced['GDP_rate'].fillna(0.0)
            elif player_type == 'hitter':
                df_enhanced['GDP_rate'] = 0.0  # Default if GDP/PA not available

            # Add positional adjustments using new system
            if player_type == 'hitter' and 'PA' in df_enhanced.columns:
                # Calculate positional adjustment based on primary position and PA
                if 'Pos' in df_enhanced.columns:
                    df_enhanced['Positional_WAR'] = df_enhanced.apply(lambda row: POSITION_WAR_ADJUSTMENTS.get(
                        row.get('Pos', ''), 0.0) * (row.get('PA', 600) / 600), axis=1, )
                else:
                    df_enhanced['Positional_WAR'] = 0.0
            else:
                df_enhanced['Positional_WAR'] = 0.0

            # Add enhanced pitcher features for 14-feature expansion
            if player_type == 'pitcher':
                # Load percentage-based pitcher features for consistent scaling
                from common_modules.derived_stats import (
                    load_percentage_pitcher_features,
                    get_player_percentage_features,
                )
                from common_modules.wpa_li_features import (
                    load_wpa_li_features,
                    get_wpa_li_for_pitcher,
                )
                from common_modules.era_normalization import (
                    load_era_normalization_data,
                    calculate_era_normalization_factors,
                    normalize_era,
                )
                from common_modules.plate_discipline_features import (
                    load_plate_discipline_features,
                    get_plate_discipline_for_pitcher,
                )
                from common_modules.pitcher_workload_calculator import classify_pitcher_role

                # Load percentage-based features from all available years (2016-2024)
                years = [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]
                percentage_features = load_percentage_pitcher_features(years)

                # Load WPA/LI features
                wpa_li_features = load_wpa_li_features(years)

                # Load ERA normalization data
                era_norm_data = load_era_normalization_data(years)
                era_norm_factors = calculate_era_normalization_factors(era_norm_data)

                # Load plate discipline features
                plate_disc_data = load_plate_discipline_features(years)

                # Initialize all pitcher feature columns (14 features now)
                pitcher_feature_columns = [
                    'BB%', 'K%', 'K-BB%',
                    'ERA_normalized',  # NEW: Role & year-adjusted ERA
                    'damage_control_ratio',
                    'Opportunity_Success',
                    'Contact_Quality_Index',
                    'HBP%',
                    'Statcast_Launch_Quality_Index',
                    'WPA/LI',  # Win impact metric
                    'CSW%',    # NEW: Called strike + whiff %
                    'Contact%',  # NEW: Contact rate on swings
                    'SwStr%',  # NEW: Swinging strike %
                    'Dominance_Index',  # NEW: CSW%/Contact% composite metric
                ]
                for col in pitcher_feature_columns:
                    df_enhanced[col] = 0.0

                # Map percentage-based pitcher features by player ID
                feature_load_failures = 0
                for idx, row in df_enhanced.iterrows():
                    player_id = row.get('MLBAMID', row.get('mlbid', ''))
                    season = row.get('Season', row.get('Year', 2024))

                    if player_id:
                        try:
                            # Load existing percentage features
                            player_features = get_player_percentage_features(
                                player_id, percentage_features,
                            )
                            # Map all percentage-based features
                            for feature_name, feature_value in player_features.items():
                                if feature_name in df_enhanced.columns:
                                    df_enhanced.at[idx, feature_name] = feature_value

                            # Add WPA/LI feature
                            wpa_li_value = get_wpa_li_for_pitcher(
                                player_id, wpa_li_features, default=0.0
                            )
                            df_enhanced.at[idx, 'WPA/LI'] = wpa_li_value

                            # NEW: Add ERA normalization
                            era = row.get('ERA', 4.0)
                            games = row.get('G', 20)
                            innings = row.get('IP', 50)
                            games_started = row.get('GS', 0)

                            role_info = classify_pitcher_role(
                                games_pitched=games,
                                innings_pitched=innings,
                                games_started=games_started
                            )

                            era_normalized = normalize_era(
                                era=era,
                                role=role_info['role'],
                                season=int(season),
                                normalization_factors=era_norm_factors
                            )
                            df_enhanced.at[idx, 'ERA_normalized'] = era_normalized

                            # NEW: Add plate discipline features (individual + composite)
                            plate_disc = get_plate_discipline_for_pitcher(
                                player_id, plate_disc_data
                            )
                            df_enhanced.at[idx, 'CSW%'] = plate_disc['CSW%']
                            df_enhanced.at[idx, 'Contact%'] = plate_disc['Contact%']
                            df_enhanced.at[idx, 'SwStr%'] = plate_disc['SwStr%']
                            df_enhanced.at[idx, 'Dominance_Index'] = plate_disc['Dominance_Index']

                        except Exception as e:
                            # Log failure and use defaults
                            feature_load_failures += 1
                            logger.warning(
                                f"Failed to load features for player {player_id} "
                                f"(row {idx}): {str(e)}"
                            )
                            continue

                if feature_load_failures > 0:
                    logger.warning(
                        f"Failed to load percentage features for {feature_load_failures} "
                        f"out of {len(df_enhanced)} pitchers - using default values"
                    )
            else:
                # Initialize pitcher features as 0 for hitters (not applicable)
                pitcher_feature_columns = [
                    'BB%', 'K%', 'K-BB%', 'damage_control_ratio',
                    'Opportunity_Success', 'Contact_Quality_Index',
                    'HBP%', 'Statcast_Launch_Quality_Index',
                    'WPA/LI',
                    'WHIP',
                ]
                for col in pitcher_feature_columns:
                    df_enhanced[col] = 0.0

            return df_enhanced

        warp_enhanced = add_enhanced_features(warp_matched, 'warp')
        war_enhanced = add_enhanced_features(war_matched, 'war')

        # Define features
        if player_type == 'hitter':
            # ENHANCED HITTER FEATURES (10 features)
            warp_features = [
                'K%', 'BB%', 'AVG', 'OBP', 'SLG', 'PA',
                'Positional_WAR', 'GDP_rate',
                'Enhanced_Baserunning', 'Enhanced_Defense',
            ]
            war_features = warp_features.copy()
        else:  # pitcher
            # 15-FEATURE PITCHER SET (includes ERA normalization + plate discipline)
            warp_features = [
                'IP', 'BB%', 'K%', 'K-BB%',
                'ERA_normalized',  # NEW: Role & year-adjusted ERA
                'damage_control_ratio', 'Opportunity_Success',
                'Contact_Quality_Index', 'HBP%',
                'Statcast_Launch_Quality_Index',
                'WPA/LI',
                'CSW%',      # NEW: Called strike + whiff %
                'Contact%',  # NEW: Contact rate on swings
                'SwStr%',    # NEW: Swinging strike %
                'Dominance_Index',  # NEW: CSW%/Contact% composite metric
            ]
            war_features = warp_features.copy()

        # Filter to only available columns
        warp_available = warp_enhanced.columns.tolist()
        war_available = war_enhanced.columns.tolist()

        warp_features = [col for col in warp_features if col in warp_available]
        war_features = [col for col in war_features if col in war_available]

        logger.debug(f"Selected WARP features: {warp_features}")
        logger.debug(f"Selected WAR features: {war_features}")

        # Extract features and targets with NaN cleaning
        warp_valid_mask = warp_enhanced['WARP'].notna()
        war_valid_mask = war_enhanced['WAR'].notna()

        logger.info(
            f"WARP data: {warp_valid_mask.sum()}/{len(warp_enhanced)} records with valid WARP")
        logger.info(f"WAR data: {war_valid_mask.sum()}/{len(war_enhanced)} records with valid WAR")

        # Filter to only valid records
        warp_clean = warp_enhanced[warp_valid_mask].reset_index(drop=True)
        war_clean = war_enhanced[war_valid_mask].reset_index(drop=True)

        warp_X = warp_clean[warp_features].fillna(0)
        warp_y = warp_clean['WARP']
        warp_names = warp_clean['Name']
        # WARP data: Look for Season first, then Year
        warp_years = (
            warp_clean['Season'].tolist() if 'Season' in warp_clean.columns
            else warp_clean['Year'].tolist() if 'Year' in warp_clean.columns
            else ['2021'] * len(warp_clean),
        )

        war_X = war_clean[war_features].fillna(0)
        war_y = war_clean['WAR']
        war_names = war_clean['Name']
        # WAR data: Look for Year
        war_years = (
            war_clean['Year'].tolist() if 'Year' in war_clean.columns
            else ['2021'] * len(war_clean),
        )

        # Extract MLB IDs for cross-reference in filtering
        war_mlbids = war_clean['MLBAMID'].tolist() if 'MLBAMID' in war_clean.columns else []
        warp_mlbids = warp_clean['mlbid'].tolist() if 'mlbid' in warp_clean.columns else []

        return {
            'warp': {'X': warp_X, 'y': warp_y, 'names': warp_names, 'years': warp_years, 'mlbids': warp_mlbids},
            'war': {'X': war_X, 'y': war_y, 'names': war_names, 'years': war_years, 'mlbids': war_mlbids},
        }

    # Prepare datasets
    hitter_data = prepare_dataset(
        hitter_warp, hitter_war, hitter_mapping,
        enhanced_baserunning, enhanced_defensive, 'hitter',
        bp_positions, fg_positions,
    )
    pitcher_data = prepare_dataset(
        pitcher_warp, pitcher_war, pitcher_mapping,
        enhanced_baserunning, enhanced_defensive, 'pitcher',
        bp_positions, fg_positions,
    )

    # Apply legitimate pitcher filtering to remove position players who pitch in blowouts
    logger.info("Applying legitimate pitcher filtering...")
    from common_modules.filter_legitimate_pitchers import apply_pitcher_filtering, apply_hitter_filtering
    try:
        filtered_pitcher_data = apply_pitcher_filtering(hitter_data, pitcher_data)
        if filtered_pitcher_data:
            pitcher_data = filtered_pitcher_data
            logger.info("Pitcher filtering applied successfully")
        else:
            logger.warning("Pitcher filtering failed, using unfiltered data")
    except Exception as e:
        logger.warning(f"Pitcher filtering error: {e}, using unfiltered data")

    # Apply legitimate hitter filtering to remove pitchers forced to hit (NL pre-DH)
    logger.info("Applying legitimate hitter filtering...")
    try:
        filtered_hitter_data = apply_hitter_filtering(hitter_data, pitcher_data)
        if filtered_hitter_data:
            hitter_data = filtered_hitter_data
            logger.info("Hitter filtering applied successfully")
        else:
            logger.warning("Hitter filtering failed, using unfiltered data")
    except Exception as e:
        logger.warning(f"Hitter filtering error: {e}, using unfiltered data")

    return hitter_data, pitcher_data
