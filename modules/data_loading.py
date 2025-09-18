"""
Data Loading Module for oWAR Analysis

This module handles all data loading operations including:
- Primary dataset loading (hitters, pitchers, fielding, etc.)
- Cache management for mappings and computed data
- Specialized data loaders for external datasets (WARP, OAA, framing)
"""

import os
import pandas as pd
import json
import glob
from pathlib import Path

# Import configuration from parent module
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Constants (will be imported from main module)
DATA_DIR = r"C:\Users\nairs\Documents\GithubProjects\oWAR\MLB Player Data"
CACHE_DIR = r"C:\Users\nairs\Documents\GithubProjects\oWAR\cache"

# Ensure cache directory exists
os.makedirs(CACHE_DIR, exist_ok=True)

__all__ = [
    'load_primary_datasets',
    'load_mapping_from_file',
    'clear_mapping_cache',
    'clear_all_cache',
    'load_official_oaa_data',
    'load_yearly_bp_data',
    'load_yearly_catcher_framing_data',
    'get_primary_dataframes'
]

# Global variables to store loaded data
_primary_dataframes = {}

def load_primary_datasets():
    """
    Load all primary CSV datasets used in oWAR analysis.

    Returns:
        dict: Dictionary containing all primary dataframes
    """
    global _primary_dataframes

    print("Loading primary datasets...")

    try:
        _primary_dataframes = {
            'hitter_by_game_df': pd.read_csv(os.path.join(DATA_DIR, "hittersByGame(player_offense_data).csv"), low_memory=False),
            'pitcher_by_game_df': pd.read_csv(os.path.join(DATA_DIR, "pitchersByGame(pitcher_data).csv"), low_memory=False),
            'baserunning_by_game_df': pd.read_csv(os.path.join(DATA_DIR, "baserunningNotes(player_offense_data).csv")),
            'fielding_by_game_df': pd.read_csv(os.path.join(DATA_DIR, "fieldingNotes(player_defensive_data).csv")),
            'warp_hitter_df': pd.read_csv(os.path.join(DATA_DIR, "bp_hitters_2021.csv")),
            'warp_pitcher_df': pd.read_csv(os.path.join(DATA_DIR, "bp_pitchers_2021.csv")),
            'oaa_hitter_df': pd.read_csv(os.path.join(DATA_DIR, "outs_above_average.csv")),
            'fielding_df': pd.read_csv(os.path.join(DATA_DIR, "fieldingNotes(player_defensive_data).csv")),
            'baserunning_df': pd.read_csv(os.path.join(DATA_DIR, "baserunningNotes(player_offense_data).csv")),
            'war_df': pd.read_csv(os.path.join(DATA_DIR, "FanGraphs Leaderboard.csv"))
        }

        print(f"Successfully loaded {len(_primary_dataframes)} primary datasets:")
        for name, df in _primary_dataframes.items():
            print(f"  {name}: {len(df):,} rows")

        return _primary_dataframes

    except FileNotFoundError as e:
        print(f"Error: Could not find required data file - {e}")
        return {}
    except Exception as e:
        print(f"Error loading primary datasets: {e}")
        return {}

def get_primary_dataframes():
    """
    Get the loaded primary dataframes. Loads them if not already loaded.

    Returns:
        dict: Dictionary containing all primary dataframes
    """
    global _primary_dataframes

    if not _primary_dataframes:
        return load_primary_datasets()

    return _primary_dataframes

def load_mapping_from_file(source_names, target_names):
    """
    Load name mapping from persistent file

    Args:
        source_names: List of source names
        target_names: List of target names

    Returns:
        dict or None: Cached mapping if valid, None otherwise
    """
    from cleanedDataParser import get_cache_filename  # Import helper function

    filename = get_cache_filename(source_names, target_names)
    filepath = os.path.join(CACHE_DIR, filename)

    if not os.path.exists(filepath):
        return None

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            cache_data = json.load(f)

        # Validate cache is still current
        metadata = cache_data.get('metadata', {})
        source_count = len([x for x in source_names if pd.notna(x)])
        target_count = len([x for x in target_names if pd.notna(x)])

        if (metadata.get('source_count') == source_count and
            metadata.get('target_count') == target_count):

            print(f"Loaded cached mapping from {filename}")
            print(f"   Created: {metadata.get('created_timestamp', 'Unknown')}")
            print(f"   Mappings: {metadata.get('mapping_count', 0)}")
            return cache_data['mapping']
        else:
            print(f"Cache invalid (data changed), will regenerate mapping")
            return None

    except Exception as e:
        print(f"Warning: Could not load mapping cache: {e}")
        return None

def clear_mapping_cache():
    """Clear all cached name mappings (useful when data changes)"""
    try:
        cache_files = glob.glob(os.path.join(CACHE_DIR, "name_mapping_*.json"))
        for file in cache_files:
            os.remove(file)
        print(f"Cleared {len(cache_files)} cached mapping files")
    except Exception as e:
        print(f"Warning: Could not clear cache: {e}")

def clear_all_cache():
    """Clear all cached data (mappings, baserunning, defensive)"""
    try:
        cache_files = glob.glob(os.path.join(CACHE_DIR, "*.json"))
        for file in cache_files:
            os.remove(file)
        print(f"Cleared {len(cache_files)} cached files")
    except Exception as e:
        print(f"Warning: Could not clear cache: {e}")

def load_official_oaa_data():
    """
    Load and clean the official OAA data for comparison

    Returns:
        dict: {player_name: {official_oaa, position, fielding_runs_prevented}}
    """
    # Get OAA dataframe from primary datasets
    dataframes = get_primary_dataframes()
    oaa_hitter_df = dataframes.get('oaa_hitter_df')

    if oaa_hitter_df is None:
        print("Warning: OAA hitter dataframe not loaded")
        return {}

    oaa_data = {}

    for _, row in oaa_hitter_df.iterrows():
        last_name = str(row.get('last_name', '')).strip()
        first_name = str(row.get(' first_name', '')).strip()

        if last_name == 'nan' or first_name == 'nan' or not last_name or not first_name:
            continue

        player_name = f"{first_name} {last_name}"
        oaa_value = row.get('outs_above_average', 0)
        position = str(row.get('primary_pos_formatted', '')).strip()

        if pd.notna(oaa_value):
            oaa_data[player_name] = {
                'official_oaa': float(oaa_value),
                'position': position,
                'fielding_runs_prevented': row.get('fielding_runs_prevented', 0)
            }

    return oaa_data

def load_yearly_bp_data():
    """
    Load and unify Baseball Prospectus WARP data from 2016-2024 (hitters) and 2016-2021 (pitchers)

    Returns:
        dict: {
            'hitters': {player_name_year: warp_value},
            'pitchers': {player_name_year: warp_value}
        }
    """
    cache_file = os.path.join(CACHE_DIR, "yearly_bp_data.json")

    # Check cache first
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                cached_data = json.load(f)
            print(f"Loaded cached yearly BP data ({len(cached_data.get('hitters', {}))} hitter-seasons, {len(cached_data.get('pitchers', {}))} pitcher-seasons)")
            return cached_data
        except:
            pass

    print("=== LOADING YEARLY BP DATA (2016-2024) ===")

    bp_data = {'hitters': {}, 'pitchers': {}}

    # Load hitters data (2016-2024)
    for year in range(2016, 2025):  # 2016-2024
        filename = os.path.join(DATA_DIR, f'bp_hitters_{year}.csv')
        if not os.path.exists(filename):
            continue

        try:
            df = pd.read_csv(filename)
            df.columns = df.columns.str.strip().str.strip('"')

            # Handle different formats
            if year <= 2019:
                # Format: NAME, YEAR, BWARP
                name_col = 'NAME'
                warp_col = 'BWARP'
            else:
                # Format: Name, WARP (2020+)
                name_col = 'Name'
                warp_col = 'WARP'

            # Process each player
            for _, row in df.iterrows():
                player_name = str(row.get(name_col, '')).strip()
                warp_value = row.get(warp_col, 0)

                if player_name and pd.notna(warp_value) and player_name != 'nan':
                    key = f"{player_name}_{year}"
                    bp_data['hitters'][key] = float(warp_value)

            print(f"  {year} hitters: {len(df)} players loaded")

        except Exception as e:
            print(f"  Error loading {filename}: {e}")

    # Load pitchers data (2016-2021)
    for year in range(2016, 2022):  # 2016-2021
        filename = os.path.join(DATA_DIR, f'bp_pitchers_{year}.csv')
        if not os.path.exists(filename):
            continue

        try:
            df = pd.read_csv(filename)
            df.columns = df.columns.str.strip().str.strip('"')

            # Handle different formats
            if year <= 2019:
                name_col = 'NAME'
                warp_col = 'PWARP'
            else:
                name_col = 'Name'
                warp_col = 'WARP'

            # Process each player
            for _, row in df.iterrows():
                player_name = str(row.get(name_col, '')).strip()
                warp_value = row.get(warp_col, 0)

                if player_name and pd.notna(warp_value) and player_name != 'nan':
                    key = f"{player_name}_{year}"
                    bp_data['pitchers'][key] = float(warp_value)

            print(f"  {year} pitchers: {len(df)} players loaded")

        except Exception as e:
            print(f"  Error loading {filename}: {e}")

    # Cache the results
    try:
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(bp_data, f, indent=2)
        print(f"Cached BP data to {cache_file}")
    except Exception as e:
        print(f"Warning: Could not cache BP data: {e}")

    print(f"Loaded BP data: {len(bp_data['hitters'])} hitter-seasons, {len(bp_data['pitchers'])} pitcher-seasons")
    return bp_data

def load_yearly_catcher_framing_data():
    """
    Load and unify catcher framing data from 2016-2021 with yearly breakdown

    Returns:
        dict: {player_name_year: framing_runs} e.g. {'Buster Posey_2016': 31.0}
    """
    cache_file = os.path.join(CACHE_DIR, "yearly_catcher_framing_data.json")

    # Check cache first
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                cached_data = json.load(f)
            print(f"Loaded cached yearly catcher framing data ({len(cached_data)} player-seasons)")
            return cached_data
        except:
            pass

    print("=== LOADING YEARLY CATCHER FRAMING DATA (2016-2021) ===")

    yearly_framing_data = {}

    # Data format patterns by year
    years_with_formats = {
        # Format 1: separate last_name/first_name columns, player_id, runs_extra_strikes
        2016: 'format1',
        2017: 'format1',
        # Format 2: single name column (id), rv_tot for run value
        2018: 'format2',
        2019: 'format2',
        2020: 'format2',
        2021: 'format1'  # But with fielder_2 instead of player_id
    }

    for year in range(2016, 2022):
        filename = os.path.join(DATA_DIR, f'catcher_framing_{year}.csv')
        if not os.path.exists(filename):
            print(f"  Missing file: {filename}")
            continue

        try:
            df = pd.read_csv(filename)
            df.columns = df.columns.str.strip()  # Clean column names
            print(f"Processing {year}: {len(df)} records")

            format_type = years_with_formats.get(year, 'format1')

            if format_type == 'format1':
                # Format with separate name columns
                for _, row in df.iterrows():
                    if year == 2021:
                        # 2021 uses 'fielder_2' instead of 'player_id'
                        last_name = str(row.get('fielder_2', '')).strip()
                        first_name = str(row.get('first_name', '')).strip()
                    else:
                        last_name = str(row.get('last_name', '')).strip()
                        first_name = str(row.get('first_name', '')).strip()

                    framing_runs = row.get('runs_extra_strikes', 0)

                    if last_name and first_name and pd.notna(framing_runs):
                        player_name = f"{first_name} {last_name}"
                        key = f"{player_name}_{year}"
                        yearly_framing_data[key] = float(framing_runs)

            elif format_type == 'format2':
                # Format with single name column
                for _, row in df.iterrows():
                    player_name = str(row.get('id', '')).strip()
                    framing_runs = row.get('rv_tot', 0)

                    if player_name and pd.notna(framing_runs) and player_name != 'nan':
                        key = f"{player_name}_{year}"
                        yearly_framing_data[key] = float(framing_runs)

            print(f"  {year}: Added {len([k for k in yearly_framing_data.keys() if k.endswith(f'_{year}')])} player records")

        except Exception as e:
            print(f"  Error processing {filename}: {e}")

    # Cache the results
    try:
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(yearly_framing_data, f, indent=2)
        print(f"Cached yearly catcher framing data to {cache_file}")
    except Exception as e:
        print(f"Warning: Could not cache framing data: {e}")

    print(f"Loaded yearly catcher framing data: {len(yearly_framing_data)} player-seasons")
    return yearly_framing_data