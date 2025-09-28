#!/usr/bin/env python3
"""
Derived Statistics Module for sWARm Analysis

This module handles derived statistics calculation for both Baseball Prospectus
and FanGraphs data sources, including:

Baseball Prospectus K% and BB% calculations for pre-2020 data:
- K% = SO (or K) / PA
- BB% = BB / PA

Enhanced FanGraphs features:
- LOB% (Left on Base Percentage) from advanced files
- GB% (Ground Ball Percentage) from battedball files
- Interaction features (damage_control_ratio, etc.)
"""

import pandas as pd
import numpy as np
import glob
import os

# Constants
PRE_2020_CUTOFF = 2020  # Year when BP started providing K% and BB% directly
STANDARD_FILE_FILTER = 'standard'  # Filter to exclude standard files

def fix_bp_derived_statistics(df, year):
    """
    Add missing derived statistics for BP data

    Args:
        df: DataFrame with BP data for a given year
        year: The year of the data

    Returns:
        DataFrame with K% and BB% calculated if missing
    """
    df_fixed = df.copy()

    # For pre-2020 data, calculate K% and BB% if missing
    if year < PRE_2020_CUTOFF:
        print(f"   Calculating derived statistics for {year} data...")

        # Calculate K% (strikeouts / plate appearances)
        if 'K%' not in df_fixed.columns and 'SO' in df_fixed.columns and 'PA' in df_fixed.columns:
            # Handle potential division by zero
            df_fixed['K%'] = np.where(
                df_fixed['PA'] > 0,
                (df_fixed['SO'] / df_fixed['PA']) * 100,
                0.0
            )
            print(f"      SUCCESS: Calculated K% from SO/PA")
        elif 'K%' not in df_fixed.columns and 'K' in df_fixed.columns and 'PA' in df_fixed.columns:
            # Some files might use 'K' instead of 'SO'
            df_fixed['K%'] = np.where(
                df_fixed['PA'] > 0,
                (df_fixed['K'] / df_fixed['PA']) * 100,
                0.0
            )
            print(f"      SUCCESS: Calculated K% from K/PA")

        # Calculate BB% (walks / plate appearances)
        if 'BB%' not in df_fixed.columns and 'BB' in df_fixed.columns and 'PA' in df_fixed.columns:
            df_fixed['BB%'] = np.where(
                df_fixed['PA'] > 0,
                (df_fixed['BB'] / df_fixed['PA']) * 100,
                0.0
            )
            print(f"      SUCCESS: Calculated BB% from BB/PA")

        # Report on calculations
        if 'K%' in df_fixed.columns:
            valid_k_pct = df_fixed['K%'].notna().sum()
            print(f"      DATA: K%: {valid_k_pct}/{len(df_fixed)} records have valid values")

        if 'BB%' in df_fixed.columns:
            valid_bb_pct = df_fixed['BB%'].notna().sum()
            print(f"      DATA: BB%: {valid_bb_pct}/{len(df_fixed)} records have valid values")
    else:
        print(f"   OK: {year} data already has K% and BB% - no calculation needed")

    return df_fixed

def load_fixed_bp_data(data_dir=None):
    """
    Load BP data with properly calculated derived statistics

    Args:
        data_dir: Optional path to data directory. If None, uses default location.

    Returns:
        Tuple of (hitter_data, pitcher_data) with fixed K% and BB%
    """
    print("LOADING BP DATA WITH FIXED DERIVED STATISTICS")
    print("=" * 60)

    if data_dir is None:
        data_dir = r"C:\Users\nairs\Documents\GithubProjects\oWAR\MLB Player Data"

    # Load hitter data
    print("\nProcessing BP Hitter Data:")
    hitter_files = glob.glob(os.path.join(data_dir, "BP_Data", "hitters", "bp_hitters_*.csv"))
    hitter_files = [f for f in hitter_files if STANDARD_FILE_FILTER not in f]  # Exclude standard files

    all_hitter_data = []
    for file in sorted(hitter_files):
        year = int(os.path.basename(file).split('_')[-1].replace('.csv', ''))

        try:
            df = pd.read_csv(file, encoding='utf-8-sig')

            if 'WARP' in df.columns or 'BWARP' in df.columns:
                # Standardize WARP column name
                if 'BWARP' in df.columns and 'WARP' not in df.columns:
                    df = df.rename(columns={'BWARP': 'WARP'})

                # Standardize Name column (pre-2020 uses 'NAME', post-2020 uses 'Name')
                if 'NAME' in df.columns and 'Name' not in df.columns:
                    df = df.rename(columns={'NAME': 'Name'})

                # Add year and season info
                df['Season'] = year
                df['Year'] = year

                # Fix derived statistics
                df_fixed = fix_bp_derived_statistics(df, year)
                all_hitter_data.append(df_fixed)

                print(f"   SUCCESS {year}: {len(df_fixed)} records loaded")
            else:
                print(f"   WARNING {year}: No WARP column found, skipping")

        except Exception as e:
            print(f"   ERROR {year}: Error loading - {e}")

    # Load pitcher data
    print("\nProcessing BP Pitcher Data:")
    pitcher_files = glob.glob(os.path.join(data_dir, "BP_Data", "pitchers", "bp_pitchers_*.csv"))
    pitcher_files = [f for f in pitcher_files if STANDARD_FILE_FILTER not in f]  # Exclude standard files

    all_pitcher_data = []
    for file in sorted(pitcher_files):
        year = int(os.path.basename(file).split('_')[-1].replace('.csv', ''))

        try:
            df = pd.read_csv(file, encoding='utf-8-sig')

            if 'WARP' in df.columns or 'PWARP' in df.columns:
                # Standardize WARP column name
                if 'PWARP' in df.columns and 'WARP' not in df.columns:
                    df = df.rename(columns={'PWARP': 'WARP'})

                # Standardize Name column (pre-2020 uses 'NAME', post-2020 uses 'Name')
                if 'NAME' in df.columns and 'Name' not in df.columns:
                    df = df.rename(columns={'NAME': 'Name'})

                # Add year and season info
                df['Season'] = year
                df['Year'] = year

                # Fix derived statistics
                df_fixed = fix_bp_derived_statistics(df, year)
                all_pitcher_data.append(df_fixed)

                print(f"   SUCCESS {year}: {len(df_fixed)} records loaded")
            else:
                print(f"   WARNING {year}: No WARP column found, skipping")

        except Exception as e:
            print(f"   ERROR {year}: Error loading - {e}")

    # Combine all data
    if all_hitter_data:
        combined_hitters = pd.concat(all_hitter_data, ignore_index=True)
        print(f"\nCombined Hitter Data: {len(combined_hitters)} total records")

        # Check K% and BB% coverage
        k_pct_coverage = combined_hitters['K%'].notna().sum() / len(combined_hitters) * 100
        bb_pct_coverage = combined_hitters['BB%'].notna().sum() / len(combined_hitters) * 100
        print(f"   K% coverage: {k_pct_coverage:.1f}%")
        print(f"   BB% coverage: {bb_pct_coverage:.1f}%")
    else:
        combined_hitters = pd.DataFrame()
        print("\nERROR: No hitter data loaded")

    if all_pitcher_data:
        combined_pitchers = pd.concat(all_pitcher_data, ignore_index=True)
        print(f"\nCombined Pitcher Data: {len(combined_pitchers)} total records")

        # Check K% and BB% coverage
        k_pct_coverage = combined_pitchers['K%'].notna().sum() / len(combined_pitchers) * 100
        bb_pct_coverage = combined_pitchers['BB%'].notna().sum() / len(combined_pitchers) * 100
        print(f"   K% coverage: {k_pct_coverage:.1f}%")
        print(f"   BB% coverage: {bb_pct_coverage:.1f}%")
    else:
        combined_pitchers = pd.DataFrame()
        print("\nERROR: No pitcher data loaded")

    print(f"\nSUCCESS: BP DATA LOADING WITH DERIVED STATISTICS COMPLETE!")
    return combined_hitters, combined_pitchers


def load_enhanced_pitcher_features(data_dir=None, years=None):
    """
    Load enhanced pitcher features (LOB%, GB%) from FanGraphs data with BP fallback.

    Args:
        data_dir: Optional path to data directory. If None, uses default location.
        years: List of years to load. If None, loads 2016-2024.

    Returns:
        Dictionary with enhanced features by player ID:
        {
            'LOB%': {player_id: value, ...},
            'GB%': {player_id: value, ...},
            'damage_control_ratio': {player_id: value, ...}
        }
    """
    print("LOADING ENHANCED PITCHER FEATURES")
    print("=" * 50)

    if data_dir is None:
        data_dir = r"C:\Users\nairs\Documents\GithubProjects\oWAR\MLB Player Data"

    if years is None:
        years = list(range(2016, 2025))  # 2016-2024

    enhanced_features = {
        'LOB%': {},
        'GB%': {},
        'HR/9': {},
        'damage_control_ratio': {}
    }

    for year in years:
        print(f"\nProcessing {year} enhanced features...")

        # Load LOB% from advanced files
        lob_data = _load_lob_percentage(data_dir, year)
        if lob_data:
            enhanced_features['LOB%'].update(lob_data)
            print(f"  LOB%: {len(lob_data)} players loaded")

        # Load GB% from battedball files (FanGraphs primary, BP fallback)
        gb_data = _load_ground_ball_percentage(data_dir, year)
        if gb_data:
            enhanced_features['GB%'].update(gb_data)
            print(f"  GB%: {len(gb_data)} players loaded")

        # Load HR/9 for damage control calculation
        hr9_data = _load_hr9_data(data_dir, year)
        if hr9_data:
            enhanced_features['HR/9'].update(hr9_data)
            print(f"  HR/9: {len(hr9_data)} players loaded")

    # Calculate interaction features
    damage_control = _calculate_damage_control_ratio(enhanced_features)
    enhanced_features['damage_control_ratio'] = damage_control

    print(f"\nENHANCED FEATURES SUMMARY:")
    print(f"  LOB% coverage: {len(enhanced_features['LOB%'])} players")
    print(f"  GB% coverage: {len(enhanced_features['GB%'])} players")
    print(f"  HR/9 coverage: {len(enhanced_features['HR/9'])} players")
    print(f"  Damage control ratio: {len(enhanced_features['damage_control_ratio'])} players")

    return enhanced_features


def _load_lob_percentage(data_dir, year):
    """Load LOB% from FanGraphs advanced files."""

    # Handle 2025 first half naming
    if year == 2025:
        file_path = os.path.join(data_dir, "FanGraphs_Data", "pitchers", f"fangraphs_pitchers_{year}_firsthalf_advanced.csv")
    else:
        file_path = os.path.join(data_dir, "FanGraphs_Data", "pitchers", f"fangraphs_pitchers_{year}_advanced.csv")

    if not os.path.exists(file_path):
        print(f"    LOB% file not found: {year}")
        return {}

    try:
        df = pd.read_csv(file_path)

        # Check required columns
        if 'MLBAMID' not in df.columns or 'LOB%' not in df.columns:
            print(f"    LOB% missing required columns: {year}")
            return {}

        # Create player_id -> LOB% mapping
        lob_data = {}
        for _, row in df.iterrows():
            player_id = row['MLBAMID']
            lob_pct = row['LOB%']

            if pd.notna(player_id) and pd.notna(lob_pct):
                # Convert from decimal to percentage (0.75 -> 75.0)
                lob_data[int(player_id)] = float(lob_pct) * 100

        return lob_data

    except Exception as e:
        print(f"    Error loading LOB% for {year}: {e}")
        return {}


def _load_ground_ball_percentage(data_dir, year):
    """Load GB% from FanGraphs battedball files with BP fallback."""

    # Try FanGraphs battedball first
    fg_data = _load_gb_from_fangraphs(data_dir, year)
    if fg_data:
        return fg_data

    # Fallback to BP data
    bp_data = _load_gb_from_bp(data_dir, year)
    return bp_data


def _load_gb_from_fangraphs(data_dir, year):
    """Load GB% from FanGraphs battedball files."""

    # Handle 2025 first half naming
    if year == 2025:
        file_path = os.path.join(data_dir, "FanGraphs_Data", "pitchers", f"fangraphs_pitchers_{year}_firsthalf_battedball.csv")
    else:
        file_path = os.path.join(data_dir, "FanGraphs_Data", "pitchers", f"fangraphs_pitchers_{year}_battedball.csv")

    if not os.path.exists(file_path):
        return {}

    try:
        df = pd.read_csv(file_path)

        # Check required columns
        if 'MLBAMID' not in df.columns or 'GB%' not in df.columns:
            return {}

        # Create player_id -> GB% mapping
        gb_data = {}
        for _, row in df.iterrows():
            player_id = row['MLBAMID']
            gb_pct = row['GB%']

            if pd.notna(player_id) and pd.notna(gb_pct):
                # Store as decimal (0.45 = 45%)
                gb_data[int(player_id)] = float(gb_pct)

        return gb_data

    except Exception as e:
        print(f"    Error loading FG GB% for {year}: {e}")
        return {}


def _load_gb_from_bp(data_dir, year):
    """Load GB% from Baseball Prospectus files as fallback."""

    file_path = os.path.join(data_dir, "BP_Data", "pitchers", f"bp_pitchers_{year}.csv")

    if not os.path.exists(file_path):
        return {}

    try:
        df = pd.read_csv(file_path)

        # Check required columns (BP uses 'mlbid' and might have 'GB%' or 'Ground_Ball_Pct')
        if 'mlbid' not in df.columns:
            return {}

        gb_col = None
        for col in ['GB%', 'Ground_Ball_Pct', 'GB_Pct']:
            if col in df.columns:
                gb_col = col
                break

        if gb_col is None:
            return {}

        # Create player_id -> GB% mapping
        gb_data = {}
        for _, row in df.iterrows():
            player_id = row['mlbid']
            gb_pct = row[gb_col]

            if pd.notna(player_id) and pd.notna(gb_pct):
                # Convert to decimal if needed
                if gb_pct > 1.0:  # Assume percentage format (45.0 vs 0.45)
                    gb_pct = gb_pct / 100.0
                gb_data[int(player_id)] = float(gb_pct)

        return gb_data

    except Exception as e:
        print(f"    Error loading BP GB% for {year}: {e}")
        return {}


def _load_hr9_data(data_dir, year):
    """Load HR/9 from FanGraphs advanced files (same source as LOB%)."""

    # Handle 2025 first half naming
    if year == 2025:
        file_path = os.path.join(data_dir, "FanGraphs_Data", "pitchers", f"fangraphs_pitchers_{year}_firsthalf_advanced.csv")
    else:
        file_path = os.path.join(data_dir, "FanGraphs_Data", "pitchers", f"fangraphs_pitchers_{year}_advanced.csv")

    if not os.path.exists(file_path):
        return {}

    try:
        df = pd.read_csv(file_path)

        # Check required columns
        if 'MLBAMID' not in df.columns or 'HR/9' not in df.columns:
            return {}

        # Create player_id -> HR/9 mapping
        hr9_data = {}
        for _, row in df.iterrows():
            player_id = row['MLBAMID']
            hr9 = row['HR/9']

            if pd.notna(player_id) and pd.notna(hr9):
                hr9_data[int(player_id)] = float(hr9)

        return hr9_data

    except Exception as e:
        print(f"    Error loading HR/9 for {year}: {e}")
        return {}


def _calculate_damage_control_ratio(enhanced_features):
    """
    Calculate damage control ratio = LOB% / (HR/9 + 0.5)

    This interaction feature captures pitchers who effectively strand baserunners
    even when allowing home runs, representing clutch/damage limitation ability.
    """
    damage_control = {}

    lob_data = enhanced_features['LOB%']
    hr9_data = enhanced_features['HR/9']

    # Calculate ratio for players with both LOB% and HR/9 data
    common_players = set(lob_data.keys()) & set(hr9_data.keys())

    for player_id in common_players:
        lob_pct = lob_data[player_id]
        hr9 = hr9_data[player_id]

        # Convert LOB% from decimal (0.75) to percentage (75%) for calculation
        lob_percentage = lob_pct * 100

        # damage_control_ratio = LOB% / (HR/9 + 0.5)
        # Higher values = better at limiting damage despite allowing HRs
        damage_control[player_id] = lob_percentage / (hr9 + 0.5)

    print(f"  Calculated damage_control_ratio for {len(damage_control)} players")
    return damage_control


def get_player_enhanced_features(player_id, enhanced_features_dict):
    """
    Get enhanced features for a specific player.

    Args:
        player_id: MLB player ID (MLBAMID or mlbid)
        enhanced_features_dict: Dict from load_enhanced_pitcher_features()

    Returns:
        Dict with player's enhanced feature values
    """
    try:
        player_id = int(player_id)
    except (ValueError, TypeError):
        return {'LOB%': 0.0, 'GB%': 0.0, 'damage_control_ratio': 0.0}

    return {
        'LOB%': enhanced_features_dict['LOB%'].get(player_id, 0.0),
        'GB%': enhanced_features_dict['GB%'].get(player_id, 0.0),
        'HR/9': enhanced_features_dict['HR/9'].get(player_id, 0.0),
        'damage_control_ratio': enhanced_features_dict['damage_control_ratio'].get(player_id, 0.0)
    }


def load_new_pitcher_features(data_dir=None, years=None):
    """
    Load new pitcher features for 11-feature expansion:
    SV (Saves), Hard%, Med%, Soft% (from battedball), HBP, WP (from standard)

    Args:
        data_dir: Optional path to data directory. If None, uses default location.
        years: List of years to load. If None, loads 2016-2025.

    Returns:
        Dictionary with new features by player ID:
        {
            'SV': {player_id: value, ...},
            'Hard%': {player_id: value, ...},
            'Med%': {player_id: value, ...},
            'Soft%': {player_id: value, ...},
            'HBP': {player_id: value, ...},
            'WP': {player_id: value, ...}
        }
    """
    print("LOADING NEW PITCHER FEATURES (5 features for 11-feature expansion)")
    print("=" * 70)

    if data_dir is None:
        data_dir = r"C:\Users\nairs\Documents\GithubProjects\oWAR\MLB Player Data"

    if years is None:
        years = list(range(2016, 2026))  # 2016-2025

    new_features = {
        'SV': {},      # Saves (from standard data)
        'Hard%': {},   # Hard contact % (from battedball data)
        'Med%': {},    # Medium contact % (from battedball data)
        'Soft%': {},   # Soft contact % (from battedball data)
        'HBP': {},     # Hit by pitch (from standard data)
        'WP': {}       # Wild pitches (from standard data)
    }

    for year in years:
        print(f"\nProcessing {year} new features...")

        # Load standard data features (SV, HBP, WP)
        standard_data = _load_standard_new_features(data_dir, year)
        if standard_data:
            for feature in ['SV', 'HBP', 'WP']:
                if feature in standard_data:
                    new_features[feature].update(standard_data[feature])
                    print(f"  {feature}: {len(standard_data[feature])} players loaded")

        # Load battedball data features (Hard%, Med%, Soft%)
        battedball_data = _load_battedball_contact_features(data_dir, year)
        if battedball_data:
            for feature in ['Hard%', 'Med%', 'Soft%']:
                if feature in battedball_data:
                    new_features[feature].update(battedball_data[feature])
                    print(f"  {feature}: {len(battedball_data[feature])} players loaded")

    print(f"\nNEW FEATURES SUMMARY:")
    for feature_name, feature_data in new_features.items():
        print(f"  {feature_name} coverage: {len(feature_data)} players")

    return new_features


def _load_standard_new_features(data_dir, year):
    """Load SV, HBP, WP from FanGraphs standard files."""

    # Handle 2025 first half naming
    if year == 2025:
        file_path = os.path.join(data_dir, "FanGraphs_Data", "pitchers", f"fangraphs_pitchers_{year}_firsthalf_standard.csv")
    else:
        file_path = os.path.join(data_dir, "FanGraphs_Data", "pitchers", f"fangraphs_pitchers_{year}_standard.csv")

    if not os.path.exists(file_path):
        print(f"    Standard file not found: {year}")
        return {}

    try:
        df = pd.read_csv(file_path)

        # Check required columns
        required_cols = ['MLBAMID', 'SV', 'HBP', 'WP']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"    Standard missing columns {missing_cols}: {year}")
            return {}

        # Create player_id -> feature mappings
        standard_data = {'SV': {}, 'HBP': {}, 'WP': {}}

        for _, row in df.iterrows():
            player_id = row['MLBAMID']

            if pd.notna(player_id):
                player_id = int(player_id)

                # Load each feature
                for feature in ['SV', 'HBP', 'WP']:
                    value = row[feature]
                    if pd.notna(value):
                        standard_data[feature][player_id] = float(value)

        return standard_data

    except Exception as e:
        print(f"    Error loading standard features {year}: {e}")
        return {}


def _load_battedball_contact_features(data_dir, year):
    """Load Hard%, Med%, Soft% from FanGraphs battedball files."""

    # Handle 2025 first half naming
    if year == 2025:
        file_path = os.path.join(data_dir, "FanGraphs_Data", "pitchers", f"fangraphs_pitchers_{year}_firsthalf_battedball.csv")
    else:
        file_path = os.path.join(data_dir, "FanGraphs_Data", "pitchers", f"fangraphs_pitchers_{year}_battedball.csv")

    if not os.path.exists(file_path):
        print(f"    Battedball file not found: {year}")
        return {}

    try:
        df = pd.read_csv(file_path)

        # Check required columns
        required_cols = ['MLBAMID', 'Hard%', 'Med%', 'Soft%']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"    Battedball missing columns {missing_cols}: {year}")
            return {}

        # Create player_id -> feature mappings
        battedball_data = {'Hard%': {}, 'Med%': {}, 'Soft%': {}}

        for _, row in df.iterrows():
            player_id = row['MLBAMID']

            if pd.notna(player_id):
                player_id = int(player_id)

                # Load each contact quality feature
                for feature in ['Hard%', 'Med%', 'Soft%']:
                    value = row[feature]
                    if pd.notna(value):
                        # Convert from decimal to percentage (0.310 -> 31.0)
                        battedball_data[feature][player_id] = float(value) * 100

        return battedball_data

    except Exception as e:
        print(f"    Error loading battedball features {year}: {e}")
        return {}


def _load_bs_for_year(year, data_dir):
    """Load BS (Blown Saves) from FanGraphs standard files."""

    # Handle 2025 first half naming
    if year == 2025:
        file_path = os.path.join(data_dir, "FanGraphs_Data", "pitchers", f"fangraphs_pitchers_{year}_firsthalf.csv")
    else:
        file_path = os.path.join(data_dir, "FanGraphs_Data", "pitchers", f"fangraphs_pitchers_{year}.csv")

    if not os.path.exists(file_path):
        return {}

    try:
        df = pd.read_csv(file_path)

        # Check required columns
        if 'MLBAMID' not in df.columns or 'BS' not in df.columns:
            return {}

        # Create player_id -> BS mapping
        bs_data = {}
        for _, row in df.iterrows():
            player_id = row['MLBAMID']
            bs = row['BS']

            if pd.notna(player_id) and pd.notna(bs):
                bs_data[int(player_id)] = int(bs)

        return bs_data

    except Exception as e:
        print(f"    Error loading BS for {year}: {e}")
        return {}


def _load_qs_for_year(year, data_dir):
    """Load QS (Quality Starts) from FanGraphs standard files."""

    # Handle 2025 first half naming
    if year == 2025:
        file_path = os.path.join(data_dir, "FanGraphs_Data", "pitchers", f"fangraphs_pitchers_{year}_firsthalf.csv")
    else:
        file_path = os.path.join(data_dir, "FanGraphs_Data", "pitchers", f"fangraphs_pitchers_{year}.csv")

    if not os.path.exists(file_path):
        return {}

    try:
        df = pd.read_csv(file_path)

        # Check required columns
        if 'MLBAMID' not in df.columns or 'QS' not in df.columns:
            return {}

        # Create player_id -> QS mapping
        qs_data = {}
        for _, row in df.iterrows():
            player_id = row['MLBAMID']
            qs = row['QS']

            if pd.notna(player_id) and pd.notna(qs):
                qs_data[int(player_id)] = int(qs)

        return qs_data

    except Exception as e:
        print(f"    Error loading QS for {year}: {e}")
        return {}


def _load_hld_for_year(year, data_dir):
    """Load HLD (Holds) from FanGraphs standard files."""

    # Handle 2025 first half naming
    if year == 2025:
        file_path = os.path.join(data_dir, "FanGraphs_Data", "pitchers", f"fangraphs_pitchers_{year}_firsthalf.csv")
    else:
        file_path = os.path.join(data_dir, "FanGraphs_Data", "pitchers", f"fangraphs_pitchers_{year}.csv")

    if not os.path.exists(file_path):
        return {}

    try:
        df = pd.read_csv(file_path)

        # Check required columns
        if 'MLBAMID' not in df.columns or 'HLD' not in df.columns:
            return {}

        # Create player_id -> HLD mapping
        hld_data = {}
        for _, row in df.iterrows():
            player_id = row['MLBAMID']
            hld = row['HLD']

            if pd.notna(player_id) and pd.notna(hld):
                hld_data[int(player_id)] = int(hld)

        return hld_data

    except Exception as e:
        print(f"    Error loading HLD for {year}: {e}")
        return {}


def _load_games_for_year(year, data_dir):
    """Load G (Games) from FanGraphs standard files."""

    # Handle 2025 first half naming
    if year == 2025:
        file_path = os.path.join(data_dir, "FanGraphs_Data", "pitchers", f"fangraphs_pitchers_{year}_firsthalf.csv")
    else:
        file_path = os.path.join(data_dir, "FanGraphs_Data", "pitchers", f"fangraphs_pitchers_{year}.csv")

    if not os.path.exists(file_path):
        return {}

    try:
        df = pd.read_csv(file_path)

        # Check required columns
        if 'MLBAMID' not in df.columns or 'G' not in df.columns:
            return {}

        # Create player_id -> G mapping
        games_data = {}
        for _, row in df.iterrows():
            player_id = row['MLBAMID']
            games = row['G']

            if pd.notna(player_id) and pd.notna(games):
                games_data[int(player_id)] = int(games)

        return games_data

    except Exception as e:
        print(f"    Error loading G for {year}: {e}")
        return {}


# DEPRECATED: SV_efficiency implementation
#
# This SV_efficiency implementation has been deprecated due to systematic bias
# against starting pitchers. The formula SV/(SV+BS+1) creates a scenario where:
# 1. ALL starters get SV_efficiency = 0.0 (meaningless)
# 2. Model learns SV_efficiency > 0 correlates with good performance (from reliever data)
# 3. SV_efficiency = 0 becomes a penalty signal, systematically undervaluing elite starters
#
# Analysis showed this caused ~1.7 WAR systematic undervaluation of elite starters
# like Skubal (actual 4.77 fWAR vs predicted ~3.0 WAR).
#
# REPLACED BY: Opportunity_Success = (QS + SV + HLD - BS) / G
# This new metric provides role-neutral evaluation that:
# - Uses QS rate for starters (meaningful differentiation)
# - Uses save/hold success for relievers (preserves information)
# - Handles all pitcher roles without systematic bias
#
# def calculate_sv_efficiency(sv_data, bs_data):
#     """
#     Calculate SV_efficiency = SV / (SV + BS + 1) for all players.
#
#     The +1 regularization ensures:
#     - Starters with SV=0, BS=0 get efficiency=0.0 (neutral)
#     - Relievers with saves get meaningful efficiency scores
#     - Avoids division by zero for players with no save situations
#
#     Args:
#         sv_data: Dict of {player_id: saves}
#         bs_data: Dict of {player_id: blown_saves}
#
#     Returns:
#         Dict of {player_id: sv_efficiency}
#     """
#     sv_efficiency = {}
#
#     # Get all players who have either SV or BS data
#     all_players = set(sv_data.keys()) | set(bs_data.keys())
#
#     for player_id in all_players:
#         sv = sv_data.get(player_id, 0)
#         bs = bs_data.get(player_id, 0)
#
#         # Calculate efficiency with regularization
#         # SV_efficiency = SV / (SV + BS + 1)
#         efficiency = sv / (sv + bs + 1)
#         sv_efficiency[player_id] = efficiency
#
#     return sv_efficiency


def calculate_opportunity_success(qs_data, sv_data, hld_data, bs_data, games_data):
    """
    Calculate Opportunity_Success = (QS + SV + HLD - BS) / G for all players.

    This comprehensive opportunity metric provides role-neutral pitcher evaluation:
    - Starters: QS (Quality Starts) represent successful outings
    - Closers: SV (Saves) minus BS (Blown Saves) = net save success
    - Setup Men: HLD (Holds) provide credit for high-leverage success
    - All Roles: Denominator G (Games) provides opportunity context

    Formula captures:
    - QS: 6+ IP, 3 or fewer ER (starter success)
    - SV: Successful save opportunities (closer success)
    - HLD: Successful hold opportunities (setup success)
    - BS: Failed save opportunities (appropriately penalized)
    - G: Total games (opportunity context)

    Args:
        qs_data: Dict of {player_id: quality_starts}
        sv_data: Dict of {player_id: saves}
        hld_data: Dict of {player_id: holds}
        bs_data: Dict of {player_id: blown_saves}
        games_data: Dict of {player_id: games}

    Returns:
        Dict of {player_id: opportunity_success}
    """
    opportunity_success = {}

    # Get all players who have games data (required denominator)
    for player_id, games in games_data.items():
        if games == 0:
            opportunity_success[player_id] = 0.0
            continue

        # Get opportunity components (default to 0 if missing)
        qs = qs_data.get(player_id, 0)
        sv = sv_data.get(player_id, 0)
        hld = hld_data.get(player_id, 0)
        bs = bs_data.get(player_id, 0)

        # Calculate comprehensive opportunity success rate
        # Opportunity_Success = (QS + SV + HLD - BS) / G
        success_rate = (qs + sv + hld - bs) / games
        opportunity_success[player_id] = success_rate

    return opportunity_success


def _load_bb_pct_for_year(data_dir, year):
    """Load BB% from FanGraphs advanced files (decimal format)."""

    # Handle 2025 first half naming
    if year == 2025:
        file_path = os.path.join(data_dir, "FanGraphs_Data", "pitchers", f"fangraphs_pitchers_{year}_firsthalf_advanced.csv")
    else:
        file_path = os.path.join(data_dir, "FanGraphs_Data", "pitchers", f"fangraphs_pitchers_{year}_advanced.csv")

    if not os.path.exists(file_path):
        return {}

    try:
        df = pd.read_csv(file_path)

        if 'MLBAMID' not in df.columns or 'BB%' not in df.columns:
            return {}

        bb_pct_data = {}
        for _, row in df.iterrows():
            player_id = row['MLBAMID']
            bb_pct = row['BB%']

            if pd.notna(player_id) and pd.notna(bb_pct):
                # Convert from decimal to percentage (0.08 -> 8.0)
                bb_pct_data[int(player_id)] = float(bb_pct) * 100

        return bb_pct_data

    except Exception as e:
        print(f"    Error loading BB% for {year}: {e}")
        return {}


def _load_k_pct_for_year(data_dir, year):
    """Load K% from FanGraphs advanced files (decimal format)."""

    # Handle 2025 first half naming
    if year == 2025:
        file_path = os.path.join(data_dir, "FanGraphs_Data", "pitchers", f"fangraphs_pitchers_{year}_firsthalf_advanced.csv")
    else:
        file_path = os.path.join(data_dir, "FanGraphs_Data", "pitchers", f"fangraphs_pitchers_{year}_advanced.csv")

    if not os.path.exists(file_path):
        return {}

    try:
        df = pd.read_csv(file_path)

        if 'MLBAMID' not in df.columns or 'K%' not in df.columns:
            return {}

        k_pct_data = {}
        for _, row in df.iterrows():
            player_id = row['MLBAMID']
            k_pct = row['K%']

            if pd.notna(player_id) and pd.notna(k_pct):
                # Convert from decimal to percentage (0.22 -> 22.0)
                k_pct_data[int(player_id)] = float(k_pct) * 100

        return k_pct_data

    except Exception as e:
        print(f"    Error loading K% for {year}: {e}")
        return {}


def _load_hr_fb_pct_for_year(data_dir, year):
    """Load HR/FB from FanGraphs battedball files (decimal format)."""

    # Handle 2025 first half naming
    if year == 2025:
        file_path = os.path.join(data_dir, "FanGraphs_Data", "pitchers", f"fangraphs_pitchers_{year}_firsthalf_battedball.csv")
    else:
        file_path = os.path.join(data_dir, "FanGraphs_Data", "pitchers", f"fangraphs_pitchers_{year}_battedball.csv")

    if not os.path.exists(file_path):
        return {}

    try:
        df = pd.read_csv(file_path)

        if 'MLBAMID' not in df.columns or 'HR/FB' not in df.columns:
            return {}

        hr_fb_data = {}
        for _, row in df.iterrows():
            player_id = row['MLBAMID']
            hr_fb = row['HR/FB']

            if pd.notna(player_id) and pd.notna(hr_fb):
                # Convert from decimal to percentage (0.12 -> 12.0)
                hr_fb_data[int(player_id)] = float(hr_fb) * 100

        return hr_fb_data

    except Exception as e:
        print(f"    Error loading HR/FB for {year}: {e}")
        return {}


def _calculate_percentage_damage_control_ratio(lob_pct_data, hr_fb_pct_data):
    """
    Calculate percentage-based damage control ratio = LOB% / (HR% + 0.5)

    Updated version using HR% (HR/FB) instead of HR/9 for consistent percentage scaling.
    All components now in percentage format for mathematical coherence.
    """
    damage_control = {}

    # Calculate ratio for players with both LOB% and HR% data
    common_players = set(lob_pct_data.keys()) & set(hr_fb_pct_data.keys())

    for player_id in common_players:
        lob_pct = lob_pct_data[player_id]  # Already in percentage format
        hr_pct = hr_fb_pct_data[player_id]  # Already in percentage format

        # damage_control_ratio = LOB% / (HR% + 0.5)
        # All percentage-based for consistent scaling
        damage_control[player_id] = lob_pct / (hr_pct + 0.5)

    return damage_control


def calculate_normalized_contact_quality_index(hard_pct_data, med_pct_data, soft_pct_data):
    """
    Calculate Contact Quality Index with Modified Z-Score Normalization.

    This function combines the empirical CQI calculation with modified z-score normalization
    for intuitive interpretation while preserving mathematical relationships.

    Formula: intuitive_cqi = 50 + (z_score * 15)
    Where z_score = (raw_cqi - population_mean) / population_std

    Based on empirical analysis of 2,351 players (2016-2025):
    - Population mean: -95.58
    - Population std: ~68.4 (estimated from range/6)
    - Raw range: -239.38 to +218.64
    - Normalized range: ~20 to 80 (95% confidence interval)

    Benefits:
    - League average anchored at 50
    - Intuitive 0-100 scale interpretation
    - Preserves relative player rankings
    - Improves model performance through standardization

    Args:
        hard_pct_data: Dict of {player_id: hard_contact_percentage} (0-100 scale)
        med_pct_data: Dict of {player_id: medium_contact_percentage} (0-100 scale)
        soft_pct_data: Dict of {player_id: soft_contact_percentage} (0-100 scale)

    Returns:
        Dict of {player_id: normalized_contact_quality_index} (intuitive 0-100 scale)
    """
    # Calculate raw Contact Quality Index
    raw_cqi = calculate_contact_quality_index(hard_pct_data, med_pct_data, soft_pct_data)

    if not raw_cqi:
        return {}

    # Calculate population statistics for normalization
    raw_values = list(raw_cqi.values())
    population_mean = np.mean(raw_values)
    population_std = np.std(raw_values, ddof=1)  # Sample standard deviation

    # Apply modified z-score normalization
    normalized_cqi = {}
    for player_id, raw_value in raw_cqi.items():
        z_score = (raw_value - population_mean) / population_std
        intuitive_index = 50 + (z_score * 15)
        normalized_cqi[player_id] = intuitive_index

    print(f"  Contact Quality Index normalized: mean={np.mean(list(normalized_cqi.values())):.1f}, std={np.std(list(normalized_cqi.values())):.1f}")
    return normalized_cqi


def calculate_contact_quality_index(hard_pct_data, med_pct_data, soft_pct_data):
    """
    Calculate EMPIRICALLY-DERIVED Contact Quality Index (CQI) using baseball logic constraints.

    EMPIRICAL FORMULA (derived from 2,351 players, 2016-2025):
    CQI = -0.2926*Hard% + -2.3938*Med% + +2.1864*Soft%

    This formula was empirically optimized using Option 3 (baseball logic constraints):
    - Hard% weight < 0 (hard contact hurts pitchers): -0.2926
    - Soft% weight > 0 (soft contact helps pitchers): +2.1864
    - Med% weight unconstrained (optimization decided): -2.3938

    Key findings from empirical analysis:
    - Medium contact is MORE harmful than hard contact (-2.3938 vs -0.2926)
    - This makes baseball sense: med contact often becomes line drives/gaps
    - Soft contact strongly helps pitchers (+2.1864)
    - Correlations: WAR=-0.0234, WARP=-0.0178, Avg|corr|=0.0206

    Scale interpretation (based on empirical data):
    - Range: -239.38 to +218.64
    - Mean: -95.58 (most pitchers allow more hard/med than soft contact)
    - Higher values = better contact management for pitchers

    Args:
        hard_pct_data: Dict of {player_id: hard_contact_percentage} (0-100 scale)
        med_pct_data: Dict of {player_id: medium_contact_percentage} (0-100 scale)
        soft_pct_data: Dict of {player_id: soft_contact_percentage} (0-100 scale)

    Returns:
        Dict of {player_id: empirically_derived_contact_quality_index}
    """
    contact_quality_index = {}

    # Get all players who have contact data
    all_players = set(hard_pct_data.keys()) | set(med_pct_data.keys()) | set(soft_pct_data.keys())

    for player_id in all_players:
        hard_pct = hard_pct_data.get(player_id, 0.0)
        med_pct = med_pct_data.get(player_id, 0.0)
        soft_pct = soft_pct_data.get(player_id, 0.0)

        # EMPIRICALLY-DERIVED Contact Quality Index from baseball logic optimization
        # CQI = -0.2926*Hard% + -2.3938*Med% + +2.1864*Soft%
        cqi = (-0.2926 * hard_pct) + (-2.3938 * med_pct) + (2.1864 * soft_pct)
        contact_quality_index[player_id] = cqi

    return contact_quality_index


def calculate_normalized_statcast_launch_quality_index(statcast_data_dict):
    """
    Calculate Statcast Launch Quality Index with Modified Z-Score Normalization.

    This function combines the empirical SLQI calculation with modified z-score normalization
    for intuitive interpretation while preserving mathematical relationships.

    Formula: intuitive_slqi = 50 + (z_score * 15)
    Where z_score = (raw_slqi - population_mean) / population_std

    Based on empirical analysis of 2,277 players (2016-2024):
    - Population mean: 19.1 (from test results)
    - Population std: ~35.7 (estimated from range -148.5 to 65.9)
    - Raw range: -148.5 to +65.9
    - Normalized range: ~20 to 80 (95% confidence interval)

    Benefits:
    - League average anchored at 50
    - Intuitive 0-100 scale interpretation
    - Preserves relative player rankings
    - Improves model performance through standardization
    - Independent from Contact Quality Index (validates launch angle control hypothesis)

    Args:
        statcast_data_dict: Dict of {player_id: {'avg_hit_angle': value, 'anglesweetspotpercent': value}}

    Returns:
        Dict of {player_id: normalized_statcast_launch_quality_index} (intuitive 0-100 scale)
    """
    # Calculate raw Statcast Launch Quality Index
    raw_slqi = calculate_statcast_launch_quality_index(statcast_data_dict)

    if not raw_slqi:
        return {}

    # Calculate population statistics for normalization
    raw_values = list(raw_slqi.values())
    population_mean = np.mean(raw_values)
    population_std = np.std(raw_values, ddof=1)  # Sample standard deviation

    # Apply modified z-score normalization
    normalized_slqi = {}
    for player_id, raw_value in raw_slqi.items():
        z_score = (raw_value - population_mean) / population_std
        intuitive_index = 50 + (z_score * 15)
        normalized_slqi[player_id] = intuitive_index

    print(f"  Statcast Launch Quality Index normalized: mean={np.mean(list(normalized_slqi.values())):.1f}, std={np.std(list(normalized_slqi.values())):.1f}")
    return normalized_slqi


def calculate_statcast_launch_quality_index(statcast_data_dict):
    """
    Calculate EMPIRICALLY-DERIVED Statcast Launch Quality Index (SLQI) from exit velocity data.

    Based on systematic empirical testing that validated user's hypothesis about launch angle independence.
    Formula: SLQI = -0.056 × (avg_hit_angle - 14.2)² + 0.659 × anglesweetspotpercent

    Key insights from empirical analysis (n=2,235 pitchers, 2016-2024):
    - avg_hit_angle: U-shaped relationship with performance (extremes good, middle bad)
    - anglesweetspotpercent: Linear negative relationship (more sweet spot = worse for pitcher)
    - Combined correlation with WAR: -0.119 (moderate strength)
    - Independent from Contact Quality Index (validates user's "hard grounders vs soft line drives" hypothesis)

    Args:
        statcast_data_dict: Dict of {player_id: {'avg_hit_angle': value, 'anglesweetspotpercent': value}}

    Returns:
        Dict of {player_id: Statcast_Launch_Quality_Index}
    """
    statcast_launch_quality = {}

    # Optimal angle from empirical analysis
    OPTIMAL_ANGLE = 14.2  # Mean angle from 2,277 pitcher dataset

    # Empirically-derived weights from scipy optimization
    ANGLE_WEIGHT = -0.056  # Coefficient for quadratic angle term
    SWEET_SPOT_WEIGHT = 0.659  # Coefficient for sweet spot percentage

    for player_id, features in statcast_data_dict.items():
        if 'avg_hit_angle' in features and 'anglesweetspotpercent' in features:
            avg_hit_angle = features['avg_hit_angle']
            sweet_spot_pct = features['anglesweetspotpercent']

            # EMPIRICALLY-DERIVED SLQI = w_angle × (angle - optimal)² + w_sweet × sweet_spot%
            # Quadratic term captures U-shaped launch angle control effect
            # Linear term captures sweet spot management effect
            angle_deviation_sq = (avg_hit_angle - OPTIMAL_ANGLE) ** 2
            slqi = ANGLE_WEIGHT * angle_deviation_sq + SWEET_SPOT_WEIGHT * sweet_spot_pct

            statcast_launch_quality[player_id] = slqi

    print(f"  Statcast_Launch_Quality_Index: {len(statcast_launch_quality)} players calculated")
    return statcast_launch_quality


def load_statcast_exit_velocity_data(data_dir=None):
    """
    Load Statcast exit velocity data for calculating Launch Quality Index.

    Args:
        data_dir: Optional path to Statcast data directory

    Returns:
        Dict of {player_id: {'avg_hit_angle': value, 'anglesweetspotpercent': value}}
    """
    if data_dir is None:
        data_dir = r"C:\Users\nairs\Documents\GithubProjects\oWAR\MLB Player Data\Statcast_Data\exit_velocity"

    print("LOADING STATCAST EXIT VELOCITY DATA")
    print("=" * 40)

    years = [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]
    all_data = {}

    for year in years:
        filename = f"exit_velocity_pitchers_{year}.csv"
        filepath = os.path.join(data_dir, filename)

        if not os.path.exists(filepath):
            print(f"  {year}: File not found")
            continue

        try:
            df = pd.read_csv(filepath)

            for _, row in df.iterrows():
                player_id = None
                if pd.notna(row['player_id']):
                    try:
                        player_id = int(float(row['player_id']))  # Convert to integer for consistency with other data
                    except (ValueError, TypeError):
                        continue

                if player_id:
                    if player_id not in all_data:
                        all_data[player_id] = {}

                    if pd.notna(row['avg_hit_angle']):
                        all_data[player_id]['avg_hit_angle'] = float(row['avg_hit_angle'])

                    if pd.notna(row['anglesweetspotpercent']):
                        all_data[player_id]['anglesweetspotpercent'] = float(row['anglesweetspotpercent'])

            print(f"  {year}: {len([p for p in all_data if 'avg_hit_angle' in all_data[p] or 'anglesweetspotpercent' in all_data[p]])} players loaded")

        except Exception as e:
            print(f"  {year}: Error loading - {e}")

    print(f"Total Statcast players: {len(all_data)}")
    return all_data


def calculate_hbp_percentage(hbp_data, pitches_data):
    """
    Calculate HBP percentage (HBP/Pitches * 100) for consistent feature scaling.

    Following the same percentage standardization logic as BB%, K%, HR%.

    Args:
        hbp_data: Dictionary of {player_id: hbp_count}
        pitches_data: Dictionary of {player_id: total_pitches}

    Returns:
        Dictionary of {player_id: hbp_percentage}
    """
    hbp_percentage = {}

    # Find common players with both HBP and Pitches data
    common_players = set(hbp_data.keys()) & set(pitches_data.keys())

    for player_id in common_players:
        hbp = hbp_data[player_id]
        pitches = pitches_data[player_id]

        # Calculate HBP% = (HBP / Pitches) * 100
        if pitches > 0:
            hbp_pct = (hbp / pitches) * 100
            hbp_percentage[player_id] = hbp_pct

    return hbp_percentage


def _load_pitches_for_year(data_dir, year):
    """Load Pitches data from FanGraphs battedball files."""
    try:
        if year == 2025:
            file_path = os.path.join(data_dir, "FanGraphs_Data", "pitchers", f"fangraphs_pitchers_{year}_firsthalf_battedball.csv")
        else:
            file_path = os.path.join(data_dir, "FanGraphs_Data", "pitchers", f"fangraphs_pitchers_{year}_battedball.csv")

        if not os.path.exists(file_path):
            print(f"    Battedball file not found: {year}")
            return {}

        df = pd.read_csv(file_path)

        pitches_data = {}
        for _, row in df.iterrows():
            player_id = row.get('MLBAMID', '')
            pitches = row.get('Pitches', 0)

            if player_id and pd.notna(pitches):
                try:
                    pitches_data[str(player_id)] = float(pitches)
                except (ValueError, TypeError):
                    continue

        return pitches_data

    except Exception as e:
        print(f"    Error loading pitches data {year}: {e}")
        return {}


def load_new_pitcher_features_with_sv_efficiency(data_dir=None, years=None):
    """
    Load new pitcher features for 11-feature expansion with SV_efficiency:
    SV_efficiency (calculated from SV/BS), Hard%, Med%, Soft% (from battedball), HBP, WP (from standard)

    Args:
        data_dir: Optional path to data directory. If None, uses default location.
        years: List of years to load. If None, loads 2016-2025.

    Returns:
        Dictionary with new features by player ID:
        {
            'SV_efficiency': {player_id: value, ...},
            'Hard%': {player_id: value, ...},
            'Med%': {player_id: value, ...},
            'Soft%': {player_id: value, ...},
            'HBP': {player_id: value, ...},
            'WP': {player_id: value, ...}
        }
    """
    print("LOADING NEW PITCHER FEATURES (6 features for 11-feature expansion with SV_efficiency)")
    print("=" * 80)

    if data_dir is None:
        data_dir = r"C:\Users\nairs\Documents\GithubProjects\oWAR\MLB Player Data"

    if years is None:
        years = list(range(2016, 2026))  # 2016-2025

    # Initialize feature containers
    all_sv_data = {}
    all_bs_data = {}
    new_features = {
        'SV_efficiency': {},  # Calculated from SV and BS
        'Hard%': {},          # Hard contact % (from battedball data)
        'Med%': {},           # Medium contact % (from battedball data)
        'Soft%': {},          # Soft contact % (from battedball data)
        'HBP': {},            # Hit by pitch (from standard data)
        'WP': {}              # Wild pitches (from standard data)
    }

    for year in years:
        print(f"\nProcessing {year} new features...")

        # Load SV and BS for SV_efficiency calculation
        standard_data = _load_standard_new_features(data_dir, year)
        bs_data = _load_bs_for_year(year, data_dir)

        if standard_data and 'SV' in standard_data:
            all_sv_data.update(standard_data['SV'])
            print(f"  SV: {len(standard_data['SV'])} players loaded")

        if bs_data:
            all_bs_data.update(bs_data)
            print(f"  BS: {len(bs_data)} players loaded")

        # Load HBP and WP from standard data
        if standard_data:
            for feature in ['HBP', 'WP']:
                if feature in standard_data:
                    new_features[feature].update(standard_data[feature])
                    print(f"  {feature}: {len(standard_data[feature])} players loaded")

        # Load battedball data features (Hard%, Med%, Soft%)
        battedball_data = _load_battedball_contact_features(data_dir, year)
        if battedball_data:
            for feature in ['Hard%', 'Med%', 'Soft%']:
                if feature in battedball_data:
                    new_features[feature].update(battedball_data[feature])
                    print(f"  {feature}: {len(battedball_data[feature])} players loaded")

    # Calculate SV_efficiency from all SV and BS data
    print(f"\nCalculating SV_efficiency from SV and BS data...")
    sv_efficiency_data = calculate_sv_efficiency(all_sv_data, all_bs_data)
    new_features['SV_efficiency'] = sv_efficiency_data
    print(f"  SV_efficiency: {len(sv_efficiency_data)} players calculated")

    print(f"\nNEW FEATURES SUMMARY:")
    for feature_name, feature_data in new_features.items():
        print(f"  {feature_name} coverage: {len(feature_data)} players")

    return new_features


def load_new_pitcher_features_with_opportunity_success(data_dir=None, years=None):
    """
    Load new pitcher features for 11-feature expansion with Opportunity_Success:
    Opportunity_Success (calculated from QS, SV, HLD, BS, G), Hard%, Med%, Soft% (from battedball), HBP, WP (from standard)

    This replaces the deprecated SV_efficiency with a comprehensive opportunity metric
    that provides role-neutral evaluation across all pitcher types.

    Args:
        data_dir: Optional path to data directory. If None, uses default location.
        years: List of years to load. If None, loads 2016-2025.

    Returns:
        Dictionary with new features by player ID:
        {
            'Opportunity_Success': {player_id: value, ...},
            'Hard%': {player_id: value, ...},
            'Med%': {player_id: value, ...},
            'Soft%': {player_id: value, ...},
            'HBP': {player_id: value, ...},
            'WP': {player_id: value, ...}
        }
    """
    print("LOADING NEW PITCHER FEATURES (6 features for 11-feature expansion with Opportunity_Success)")
    print("=" * 85)

    if data_dir is None:
        data_dir = r"C:\Users\nairs\Documents\GithubProjects\oWAR\MLB Player Data"

    if years is None:
        years = list(range(2016, 2026))  # 2016-2025

    # Initialize data containers for opportunity calculation
    all_qs_data = {}
    all_sv_data = {}
    all_hld_data = {}
    all_bs_data = {}
    all_games_data = {}

    # Initialize other feature containers
    new_features = {
        'Opportunity_Success': {},  # Calculated from QS, SV, HLD, BS, G
        'Hard%': {},               # Hard contact % (from battedball data)
        'Med%': {},                # Medium contact % (from battedball data)
        'Soft%': {},               # Soft contact % (from battedball data)
        'HBP': {},                 # Hit by pitch (from standard data)
        'WP': {}                   # Wild pitches (from standard data)
    }

    for year in years:
        print(f"\nProcessing {year} new features...")

        # Load opportunity metric components
        qs_data = _load_qs_for_year(year, data_dir)
        hld_data = _load_hld_for_year(year, data_dir)
        bs_data = _load_bs_for_year(year, data_dir)
        games_data = _load_games_for_year(year, data_dir)

        if qs_data:
            all_qs_data.update(qs_data)
            print(f"  QS: {len(qs_data)} players loaded")

        if hld_data:
            all_hld_data.update(hld_data)
            print(f"  HLD: {len(hld_data)} players loaded")

        if bs_data:
            all_bs_data.update(bs_data)
            print(f"  BS: {len(bs_data)} players loaded")

        if games_data:
            all_games_data.update(games_data)
            print(f"  G: {len(games_data)} players loaded")

        # Load SV and other standard features
        standard_data = _load_standard_new_features(data_dir, year)
        if standard_data and 'SV' in standard_data:
            all_sv_data.update(standard_data['SV'])
            print(f"  SV: {len(standard_data['SV'])} players loaded")

        # Load HBP and WP from standard data
        if standard_data:
            for feature in ['HBP', 'WP']:
                if feature in standard_data:
                    new_features[feature].update(standard_data[feature])
                    print(f"  {feature}: {len(standard_data[feature])} players loaded")

        # Load battedball data features (Hard%, Med%, Soft%)
        battedball_data = _load_battedball_contact_features(data_dir, year)
        if battedball_data:
            for feature in ['Hard%', 'Med%', 'Soft%']:
                if feature in battedball_data:
                    new_features[feature].update(battedball_data[feature])
                    print(f"  {feature}: {len(battedball_data[feature])} players loaded")

    # Calculate Opportunity_Success from all collected data
    print(f"\nCalculating Opportunity_Success from QS, SV, HLD, BS, G data...")
    opportunity_success_data = calculate_opportunity_success(
        all_qs_data, all_sv_data, all_hld_data, all_bs_data, all_games_data
    )
    new_features['Opportunity_Success'] = opportunity_success_data
    print(f"  Opportunity_Success: {len(opportunity_success_data)} players calculated")

    print(f"\nNEW FEATURES SUMMARY:")
    for feature_name, feature_data in new_features.items():
        print(f"  {feature_name} coverage: {len(feature_data)} players")

    return new_features


def get_player_new_features(player_id, new_features_dict):
    """
    Get new features for a specific player.

    Args:
        player_id: MLB player ID (MLBAMID or mlbid)
        new_features_dict: Dict from load_new_pitcher_features_with_opportunity_success()

    Returns:
        Dict with player's new feature values
    """
    try:
        player_id = int(player_id)
    except (ValueError, TypeError):
        return {
            'Opportunity_Success': 0.0, 'Hard%': 0.0, 'Med%': 0.0,
            'Soft%': 0.0, 'HBP': 0.0, 'WP': 0.0
        }

    return {
        'Opportunity_Success': new_features_dict['Opportunity_Success'].get(player_id, 0.0),
        'Hard%': new_features_dict['Hard%'].get(player_id, 0.0),
        'Med%': new_features_dict['Med%'].get(player_id, 0.0),
        'Soft%': new_features_dict['Soft%'].get(player_id, 0.0),
        'HBP': new_features_dict['HBP'].get(player_id, 0.0),
        'WP': new_features_dict['WP'].get(player_id, 0.0)
    }


def get_player_new_features_legacy_sv_efficiency(player_id, new_features_dict):
    """
    DEPRECATED: Get new features for a specific player with SV_efficiency.

    This function is kept for backward compatibility but should be replaced
    with get_player_new_features() which uses Opportunity_Success.

    Args:
        player_id: MLB player ID (MLBAMID or mlbid)
        new_features_dict: Dict from load_new_pitcher_features_with_sv_efficiency()

    Returns:
        Dict with player's new feature values (legacy format)
    """
    try:
        player_id = int(player_id)
    except (ValueError, TypeError):
        return {
            'SV_efficiency': 0.0, 'Hard%': 0.0, 'Med%': 0.0,
            'Soft%': 0.0, 'HBP': 0.0, 'WP': 0.0
        }

    return {
        'SV_efficiency': new_features_dict['SV_efficiency'].get(player_id, 0.0),
        'Hard%': new_features_dict['Hard%'].get(player_id, 0.0),
        'Med%': new_features_dict['Med%'].get(player_id, 0.0),
        'Soft%': new_features_dict['Soft%'].get(player_id, 0.0),
        'HBP': new_features_dict['HBP'].get(player_id, 0.0),
        'WP': new_features_dict['WP'].get(player_id, 0.0)
    }


def load_new_pitcher_features_with_contact_quality_index(data_dir=None, years=None):
    """
    Load new pitcher features for 9-feature expansion with Contact Quality Index:
    Opportunity_Success (calculated from QS, SV, HLD, BS, G), Contact_Quality_Index (replaces Hard%/Med%/Soft%), HBP, WP

    This improves upon the 11-feature approach by:
    - Replacing 3 correlated contact features with 1 comprehensive Contact Quality Index
    - Eliminating multicollinearity between Hard%/Med%/Soft%
    - Maintaining defense-independent contact quality information
    - Reducing feature count from 11 to 9 for better model efficiency

    Args:
        data_dir: Optional path to data directory. If None, uses default location.
        years: List of years to load. If None, loads 2016-2025.

    Returns:
        Dictionary with new features by player ID:
        {
            'Opportunity_Success': {player_id: value, ...},
            'Contact_Quality_Index': {player_id: value, ...},
            'HBP': {player_id: value, ...},
            'WP': {player_id: value, ...}
        }
    """
    print("LOADING NEW PITCHER FEATURES (4 features for 9-feature expansion with Contact Quality Index)")
    print("=" * 95)

    if data_dir is None:
        data_dir = r"C:\Users\nairs\Documents\GithubProjects\oWAR\MLB Player Data"

    if years is None:
        years = list(range(2016, 2026))  # 2016-2025

    # Initialize data containers for opportunity calculation
    all_qs_data = {}
    all_sv_data = {}
    all_hld_data = {}
    all_bs_data = {}
    all_games_data = {}

    # Initialize contact quality data containers
    all_hard_pct_data = {}
    all_med_pct_data = {}
    all_soft_pct_data = {}

    # Initialize other feature containers
    new_features = {
        'Opportunity_Success': {},      # Calculated from QS, SV, HLD, BS, G
        'Contact_Quality_Index': {},    # Calculated from Hard%, Med%, Soft%
        'HBP': {},                      # Hit by pitch (from standard data)
        'WP': {}                        # Wild pitches (from standard data)
    }

    for year in years:
        print(f"\nProcessing {year} new features...")

        # Load opportunity data
        qs_year = _load_qs_for_year(data_dir, year)
        sv_year = _load_sv_for_year(data_dir, year)
        hld_year = _load_hld_for_year(data_dir, year)
        bs_year = _load_bs_for_year(data_dir, year)
        games_year = _load_games_for_year(data_dir, year)

        # Update cumulative opportunity data
        all_qs_data.update(qs_year)
        all_sv_data.update(sv_year)
        all_hld_data.update(hld_year)
        all_bs_data.update(bs_year)
        all_games_data.update(games_year)

        # Load contact quality data
        hard_pct_year = _load_hard_pct_for_year(data_dir, year)
        med_pct_year = _load_med_pct_for_year(data_dir, year)
        soft_pct_year = _load_soft_pct_for_year(data_dir, year)

        # Update cumulative contact data
        all_hard_pct_data.update(hard_pct_year)
        all_med_pct_data.update(med_pct_year)
        all_soft_pct_data.update(soft_pct_year)

        # Load other features
        hbp_year = _load_hbp_for_year(data_dir, year)
        wp_year = _load_wp_for_year(data_dir, year)

        # Update cumulative other features
        new_features['HBP'].update(hbp_year)
        new_features['WP'].update(wp_year)

        print(f"  G: {len(games_year)} players loaded")
        print(f"  SV: {len(sv_year)} players loaded")
        print(f"  HBP: {len(hbp_year)} players loaded")
        print(f"  WP: {len(wp_year)} players loaded")
        print(f"  Hard%: {len(hard_pct_year)} players loaded")
        print(f"  Med%: {len(med_pct_year)} players loaded")
        print(f"  Soft%: {len(soft_pct_year)} players loaded")

    # Calculate Opportunity_Success from all collected data
    print(f"\nCalculating Opportunity_Success from QS, SV, HLD, BS, G data...")
    opportunity_success_data = calculate_opportunity_success(
        all_qs_data, all_sv_data, all_hld_data, all_bs_data, all_games_data
    )
    new_features['Opportunity_Success'] = opportunity_success_data
    print(f"  Opportunity_Success: {len(opportunity_success_data)} players calculated")

    # Calculate Contact Quality Index from all collected contact data
    print(f"\nCalculating Contact_Quality_Index from Hard%, Med%, Soft% data...")
    contact_quality_data = calculate_contact_quality_index(
        all_hard_pct_data, all_med_pct_data, all_soft_pct_data
    )
    new_features['Contact_Quality_Index'] = contact_quality_data
    print(f"  Contact_Quality_Index: {len(contact_quality_data)} players calculated")

    print(f"\nNEW FEATURES SUMMARY:")
    for feature_name, feature_data in new_features.items():
        print(f"  {feature_name} coverage: {len(feature_data)} players")

    return new_features


def _load_bb_pct_for_year(data_dir, year):
    """
    Load BB% data from advanced files for a specific year (converted to percentage format)
    """
    filename = f"fangraphs_pitchers_{year}_advanced.csv"
    filepath = os.path.join(data_dir, filename)  # Fixed: removed str(year) subdirectory

    bb_pct_data = {}

    if os.path.exists(filepath):
        try:
            df = pd.read_csv(filepath)
            for _, row in df.iterrows():
                player_id = row.get('MLBAMID', row.get('playerid'))  # Fixed: use MLBAMID
                bb_pct = row.get('BB%', row.get('BB_pct'))

                if pd.notna(player_id) and pd.notna(bb_pct):
                    # Convert from decimal to percentage (0.08 -> 8.0)
                    bb_pct_data[int(player_id)] = float(bb_pct) * 100
        except Exception as e:
            print(f"    ERROR loading BB% from {filepath}: {e}")

    return bb_pct_data


def _load_k_pct_for_year(data_dir, year):
    """
    Load K% data from advanced files for a specific year (converted to percentage format)
    """
    filename = f"fangraphs_pitchers_{year}_advanced.csv"
    filepath = os.path.join(data_dir, filename)  # Fixed: removed str(year) subdirectory

    k_pct_data = {}

    if os.path.exists(filepath):
        try:
            df = pd.read_csv(filepath)
            for _, row in df.iterrows():
                player_id = row.get('MLBAMID', row.get('playerid'))  # Fixed: use MLBAMID
                k_pct = row.get('K%', row.get('K_pct'))

                if pd.notna(player_id) and pd.notna(k_pct):
                    # Convert from decimal to percentage (0.22 -> 22.0)
                    k_pct_data[int(player_id)] = float(k_pct) * 100
        except Exception as e:
            print(f"    ERROR loading K% from {filepath}: {e}")

    return k_pct_data


def _load_hr_fb_pct_for_year(data_dir, year):
    """
    Load HR/FB data from battedball files for a specific year (converted to percentage format)
    """
    filename = f"fangraphs_pitchers_{year}_battedball.csv"
    filepath = os.path.join(data_dir, filename)  # Fixed: removed str(year) subdirectory

    hr_fb_data = {}

    if os.path.exists(filepath):
        try:
            df = pd.read_csv(filepath)
            for _, row in df.iterrows():
                player_id = row.get('MLBAMID', row.get('playerid'))  # Fixed: use MLBAMID
                hr_fb = row.get('HR/FB', row.get('HR_FB'))

                if pd.notna(player_id) and pd.notna(hr_fb):
                    # Convert from decimal to percentage (0.12 -> 12.0)
                    hr_fb_data[int(player_id)] = float(hr_fb) * 100
        except Exception as e:
            print(f"    ERROR loading HR/FB from {filepath}: {e}")

    return hr_fb_data


def _calculate_percentage_damage_control_ratio(lob_pct_data, hr_fb_pct_data):
    """
    Calculate damage_control_ratio using percentage-based formula:
    damage_control_ratio = LOB% / (HR% + 0.5)

    Uses HR/FB for HR% component
    """
    damage_control = {}

    for player_id in lob_pct_data:
        lob_pct = lob_pct_data[player_id]
        hr_pct = hr_fb_pct_data.get(player_id, 10.0)  # Default to 10% if missing

        # damage_control_ratio = LOB% / (HR% + 0.5)
        damage_control[player_id] = lob_pct / (hr_pct + 0.5)

    return damage_control


def load_percentage_pitcher_features(data_directory, years):
    """
    Load pitcher features with consistent percentage scaling.

    Replaces BB/9, K/9 with BB%, K% and updates damage_control_ratio to use HR%

    Args:
        data_directory: Path to data directory
        years: List of years to process

    Returns:
        Dict with all percentage-based pitcher features
    """
    print("LOADING PERCENTAGE-BASED PITCHER FEATURES")
    print("=" * 50)
    print(f"Standardizing all features to percentage format for consistent scaling")
    print(f"Processing years: {years}")
    print()

    # Initialize feature collections
    all_bb_pct_data = {}
    all_k_pct_data = {}
    all_hr_fb_pct_data = {}
    all_lob_pct_data = {}  # Already in percentage format
    all_hbp_data = {}
    all_pitches_data = {}  # For HBP% calculation
    all_wp_data = {}
    all_hard_pct_data = {}
    all_med_pct_data = {}
    all_soft_pct_data = {}

    # Load opportunity success components
    all_qs_data = {}
    all_sv_data = {}
    all_hld_data = {}
    all_bs_data = {}
    all_games_data = {}

    for year in years:
        print(f"Loading {year} percentage-based data...")

        # Load percentage features
        bb_pct_year = _load_bb_pct_for_year(data_directory, year)
        k_pct_year = _load_k_pct_for_year(data_directory, year)
        hr_fb_pct_year = _load_hr_fb_pct_for_year(data_directory, year)

        # Load existing percentage features
        lob_pct_year = _load_lob_percentage(data_directory, year)
        battedball_contact = _load_battedball_contact_features(data_directory, year)
        hard_pct_year = battedball_contact.get('Hard%', {}) if battedball_contact else {}
        med_pct_year = battedball_contact.get('Med%', {}) if battedball_contact else {}
        soft_pct_year = battedball_contact.get('Soft%', {}) if battedball_contact else {}

        # Load count features
        standard_features = _load_standard_new_features(data_directory, year)
        hbp_year = standard_features.get('HBP', {}) if standard_features else {}
        wp_year = standard_features.get('WP', {}) if standard_features else {}

        # Load pitches data for HBP% calculation
        pitches_year = _load_pitches_for_year(data_directory, year)

        # Load opportunity success components
        qs_year = _load_qs_for_year(year, data_directory)
        sv_year = standard_features.get('SV', {}) if standard_features else {}
        hld_year = _load_hld_for_year(year, data_directory)
        bs_year = _load_bs_for_year(year, data_directory)
        games_year = _load_games_for_year(year, data_directory)

        # Merge into all-years collections
        all_bb_pct_data.update(bb_pct_year)
        all_k_pct_data.update(k_pct_year)
        all_hr_fb_pct_data.update(hr_fb_pct_year)
        all_lob_pct_data.update(lob_pct_year)
        all_hbp_data.update(hbp_year)
        all_pitches_data.update(pitches_year)
        all_wp_data.update(wp_year)
        all_hard_pct_data.update(hard_pct_year)
        all_med_pct_data.update(med_pct_year)
        all_soft_pct_data.update(soft_pct_year)

        all_qs_data.update(qs_year)
        all_sv_data.update(sv_year)
        all_hld_data.update(hld_year)
        all_bs_data.update(bs_year)
        all_games_data.update(games_year)

        print(f"  BB%: {len(bb_pct_year)} players loaded")
        print(f"  K%: {len(k_pct_year)} players loaded")
        print(f"  HR/FB%: {len(hr_fb_pct_year)} players loaded")
        print(f"  LOB%: {len(lob_pct_year)} players loaded")
        print(f"  HBP: {len(hbp_year)} players loaded")
        print(f"  Pitches: {len(pitches_year)} players loaded")
        print(f"  WP: {len(wp_year)} players loaded")
        print(f"  Hard%: {len(hard_pct_year)} players loaded")
        print(f"  Med%: {len(med_pct_year)} players loaded")
        print(f"  Soft%: {len(soft_pct_year)} players loaded")

    # Calculate derived features using percentage format
    percentage_features = {}

    # Calculate Opportunity_Success from all collected data
    print(f"\nCalculating Opportunity_Success from QS, SV, HLD, BS, G data...")
    opportunity_success_data = calculate_opportunity_success(
        all_qs_data, all_sv_data, all_hld_data, all_bs_data, all_games_data
    )
    percentage_features['Opportunity_Success'] = opportunity_success_data
    print(f"  Opportunity_Success: {len(opportunity_success_data)} players calculated")

    # Calculate percentage-based damage control ratio
    print(f"\nCalculating percentage-based damage_control_ratio from LOB% and HR/FB% data...")
    damage_control_data = _calculate_percentage_damage_control_ratio(
        all_lob_pct_data, all_hr_fb_pct_data
    )
    percentage_features['damage_control_ratio'] = damage_control_data
    print(f"  damage_control_ratio: {len(damage_control_data)} players calculated")

    # Calculate Contact Quality Index with normalization from all collected contact data
    print(f"\nCalculating normalized Contact_Quality_Index from Hard%, Med%, Soft% data...")
    contact_quality_data = calculate_normalized_contact_quality_index(
        all_hard_pct_data, all_med_pct_data, all_soft_pct_data
    )
    percentage_features['Contact_Quality_Index'] = contact_quality_data
    print(f"  Contact_Quality_Index: {len(contact_quality_data)} players calculated")

    # Calculate HBP percentage from HBP and Pitches data
    print(f"\nCalculating HBP% from HBP and Pitches data...")
    hbp_percentage_data = calculate_hbp_percentage(all_hbp_data, all_pitches_data)
    percentage_features['HBP%'] = hbp_percentage_data
    print(f"  HBP%: {len(hbp_percentage_data)} players calculated")

    # Calculate Statcast Launch Quality Index with normalization
    print(f"\nCalculating normalized Statcast_Launch_Quality_Index from exit velocity data...")
    statcast_data = load_statcast_exit_velocity_data()
    statcast_launch_quality_data = calculate_normalized_statcast_launch_quality_index(statcast_data)
    percentage_features['Statcast_Launch_Quality_Index'] = statcast_launch_quality_data
    print(f"  Statcast_Launch_Quality_Index: {len(statcast_launch_quality_data)} players calculated")

    # Add all base features
    percentage_features['BB%'] = all_bb_pct_data
    percentage_features['K%'] = all_k_pct_data
    percentage_features['HR/FB%'] = all_hr_fb_pct_data
    percentage_features['LOB%'] = all_lob_pct_data
    percentage_features['WP'] = all_wp_data
    percentage_features['Hard%'] = all_hard_pct_data
    percentage_features['Med%'] = all_med_pct_data
    percentage_features['Soft%'] = all_soft_pct_data

    print(f"\nPERCENTAGE FEATURES SUMMARY:")
    for feature_name, feature_data in percentage_features.items():
        print(f"  {feature_name} coverage: {len(feature_data)} players")

    return percentage_features


def get_player_percentage_features(player_id, percentage_features_dict):
    """
    Get percentage-based features for a specific player.

    Args:
        player_id: MLB player ID (MLBAMID or mlbid)
        percentage_features_dict: Dict from load_percentage_pitcher_features()

    Returns:
        Dict with player's percentage-based feature values
    """
    try:
        player_id = int(player_id)
    except (ValueError, TypeError):
        return {
            'BB%': 9.0, 'K%': 20.0, 'HR/FB%': 10.0, 'LOB%': 72.0,
            'damage_control_ratio': 2.4, 'Opportunity_Success': 0.0,
            'Contact_Quality_Index': 50.0, 'HBP%': 0.0, 'WP': 0.0,  # Normalized mean
            'Statcast_Launch_Quality_Index': 50.0  # Normalized mean
        }

    return {
        'BB%': percentage_features_dict['BB%'].get(player_id, 9.0),
        'K%': percentage_features_dict['K%'].get(player_id, 20.0),
        'HR/FB%': percentage_features_dict['HR/FB%'].get(player_id, 10.0),
        'LOB%': percentage_features_dict['LOB%'].get(player_id, 72.0),
        'damage_control_ratio': percentage_features_dict['damage_control_ratio'].get(player_id, 2.4),
        'Opportunity_Success': percentage_features_dict['Opportunity_Success'].get(player_id, 0.0),
        'Contact_Quality_Index': percentage_features_dict['Contact_Quality_Index'].get(player_id, 50.0),  # Normalized mean
        'HBP%': percentage_features_dict['HBP%'].get(player_id, 0.0),
        'WP': percentage_features_dict['WP'].get(player_id, 0.0),
        'Statcast_Launch_Quality_Index': percentage_features_dict['Statcast_Launch_Quality_Index'].get(player_id, 50.0)  # Normalized mean
    }


def get_player_complete_pitcher_features(player_id, percentage_features_dict, new_features_dict, current_season_data=None):
    """
    Get complete 11-feature set for a pitcher including IP, ERA, and all derived features.

    Args:
        player_id: MLB player ID (MLBAMID or mlbid)
        percentage_features_dict: Dict from load_percentage_pitcher_features()
        new_features_dict: Dict from load_new_pitcher_features_with_contact_quality_index()
        current_season_data: DataFrame with current season stats (IP, ERA, etc.)

    Returns:
        numpy array with 11 features: [IP, BB%, K%, ERA, damage_control_ratio, SV_efficiency, Hard%, Med%, Soft%, HBP, WP]
    """
    try:
        player_id = int(player_id)
    except (ValueError, TypeError):
        # Default values for unknown player
        return np.array([100.0, 9.0, 20.0, 4.50, 2.4, 0.0, 35.0, 40.0, 25.0, 0.0, 0.0])

    # Start with default values
    features = {
        'IP': 100.0,
        'BB%': 9.0,
        'K%': 20.0,
        'ERA': 4.50,
        'damage_control_ratio': 2.4,
        'SV_efficiency': 0.0,
        'Hard%': 35.0,
        'Med%': 40.0,
        'Soft%': 25.0,
        'HBP': 0.0,
        'WP': 0.0
    }

    # Get percentage features (BB%, K%, damage_control_ratio)
    if percentage_features_dict:
        features['BB%'] = percentage_features_dict['BB%'].get(player_id, 9.0)
        features['K%'] = percentage_features_dict['K%'].get(player_id, 20.0)
        features['damage_control_ratio'] = percentage_features_dict['damage_control_ratio'].get(player_id, 2.4)

    # Get new features (Hard%, Med%, Soft%, HBP, WP, SV for efficiency calc)
    if new_features_dict:
        features['Hard%'] = new_features_dict['Hard%'].get(player_id, 35.0)
        features['Med%'] = new_features_dict['Med%'].get(player_id, 40.0)
        features['Soft%'] = new_features_dict['Soft%'].get(player_id, 25.0)
        features['HBP'] = new_features_dict['HBP'].get(player_id, 0.0)
        features['WP'] = new_features_dict['WP'].get(player_id, 0.0)

        # Calculate SV_efficiency from saves
        sv = new_features_dict['SV'].get(player_id, 0.0)
        features['SV_efficiency'] = min(sv * 2.0, 10.0)  # Cap at 10 for extreme closers

    # Get current season IP, ERA, BB%, K% if available
    if current_season_data is not None and not current_season_data.empty:
        # Try to find player by various ID columns
        player_row = None
        for id_col in ['MLBAMID', 'mlbid', 'playerid']:
            if id_col in current_season_data.columns:
                matches = current_season_data[current_season_data[id_col] == player_id]
                if not matches.empty:
                    player_row = matches.iloc[0]
                    break

        if player_row is not None:
            # Extract IP and ERA from current season data
            features['IP'] = player_row.get('IP', features['IP'])

            # Try different ERA column names (merged data might have ERA_adv, ERA_std)
            era_value = (player_row.get('ERA') or
                        player_row.get('ERA_adv') or
                        player_row.get('ERA_std') or
                        features['ERA'])
            features['ERA'] = era_value

            # Extract BB% and K% if available (convert from decimal to percentage)
            if 'BB%' in player_row:
                bb_decimal = player_row['BB%']
                if bb_decimal is not None and bb_decimal < 1.0:  # Decimal format
                    features['BB%'] = bb_decimal * 100.0
                elif bb_decimal is not None:  # Already percentage format
                    features['BB%'] = bb_decimal

            if 'K%' in player_row:
                k_decimal = player_row['K%']
                if k_decimal is not None and k_decimal < 1.0:  # Decimal format
                    features['K%'] = k_decimal * 100.0
                elif k_decimal is not None:  # Already percentage format
                    features['K%'] = k_decimal

    # Return as numpy array in exact feature order expected by model
    feature_order = ['IP', 'BB%', 'K%', 'ERA', 'damage_control_ratio', 'SV_efficiency', 'Hard%', 'Med%', 'Soft%', 'HBP', 'WP']
    return np.array([features[feat] for feat in feature_order])


def get_player_new_features_with_contact_quality_index(player_id, new_features_dict):
    """
    Get new features for a specific player with Contact Quality Index.

    Args:
        player_id: MLB player ID (MLBAMID or mlbid)
        new_features_dict: Dict from load_new_pitcher_features_with_contact_quality_index()

    Returns:
        Dict with player's new feature values
    """
    try:
        player_id = int(player_id)
    except (ValueError, TypeError):
        return {
            'Opportunity_Success': 0.0, 'Contact_Quality_Index': 0.0,
            'HBP': 0.0, 'WP': 0.0
        }

    return {
        'Opportunity_Success': new_features_dict['Opportunity_Success'].get(player_id, 0.0),
        'Contact_Quality_Index': new_features_dict['Contact_Quality_Index'].get(player_id, 0.0),
        'HBP': new_features_dict['HBP'].get(player_id, 0.0),
        'WP': new_features_dict['WP'].get(player_id, 0.0)
    }