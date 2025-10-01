"""
DEPRECATED: Historical Data Loading Module for oWAR Analysis
=============================================================

**MOVED TO LEGACY: December 2024**
**Reason**: No longer actively used by main application modules. Functionality
           has been integrated into other data loading modules or simplified.

This module was originally created to load comprehensive historical data from
FanGraphs and Baseball Prospectus for complex two-way player analysis. Since
the two-way player identification has been simplified to use threshold-based
filtering directly on existing data, this separate historical data loading
is no longer needed.

**Original Purpose**:
- Load FanGraphs comprehensive data (hitters, pitchers, defensive) from 2016-2024
- Load Baseball Prospectus WARP data for historical analysis
- Support complex two-way player identification in two_way_players.py

**Current Status**:
- NOT used by any active modules in current_season_modules or future_season_modules
- Only imported by legacy_modules/two_way_players.py (also deprecated)
- Referenced in comments in current_season_modules/current_season_data_loading.py

**Replacement**:
- For current season data: Use current_season_modules/current_season_data_loading.py
- For model training data: Use current_season_modules/modeling/data_loading.py
- For pitcher filtering: Use common_modules/filter_legitimate_pitchers.py

**Note**: The functions load_yearly_catcher_framing_data() and
         load_yearly_bp_baserunning_data() that were originally here have been
         moved to current_season_modules/current_season_data_loading.py where
         they are actively used.
"""

# Standard library imports
import json
import os
from pathlib import Path

# Third-party imports
import pandas as pd

# Local imports
from common_modules.config import DATA_DIR, CACHE_DIR
from common_modules.logging import get_logger

# Module logger
logger = get_logger(__name__)

# Ensure cache directory exists
CACHE_DIR.mkdir(exist_ok=True, parents=True)

__all__ = [
    'load_comprehensive_fangraphs_data',
    'load_yearly_bp_data'
]


def load_comprehensive_fangraphs_data():
    """
    Load and unify comprehensive FanGraphs data from 2016-2024 across 5 data types:

    1. Hitters: Basic (WAR, wRC+, wOBA), Advanced (sabermetrics), Standard (counting stats)
    2. Pitchers: Basic (WAR, FIP, xERA), Advanced (rates, ERA-), Standard (counting stats)
    3. Defensive: Advanced and Standard fielding metrics

    This provides rich feature sets for enhanced WAR/WARP prediction and future season forecasting.

    Returns:
        dict: {
            'hitters': {player_name_year: {combined_features_dict}},
            'pitchers': {player_name_year: {combined_features_dict}},
            'defensive': {player_name_year: {defensive_features_dict}}
        }
    """
    cache_file = os.path.join(CACHE_DIR, "comprehensive_fangraphs_data.json")

    # Check cache first
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                cached_data = json.load(f)
            print(f"Loaded cached comprehensive FanGraphs data:")
            print(f"  Hitters: {len(cached_data.get('hitters', {}))} player-seasons")
            print(f"  Pitchers: {len(cached_data.get('pitchers', {}))} player-seasons")
            print(f"  Defensive: {len(cached_data.get('defensive', {}))} player-seasons")
            return cached_data
        except:
            pass

    print("=== LOADING COMPREHENSIVE FANGRAPHS DATA (2016-2024) ===")
    print("Combining 5 data types: Hitters (3), Pitchers (3), Defensive (2)")

    fangraphs_data = {'hitters': {}, 'pitchers': {}, 'defensive': {}}

    # Load hitters data (3 types per year: basic, advanced, standard)
    for year in range(2016, 2025):  # 2016-2024
        print(f"\nProcessing {year}...")

        # Hitters - combine 3 data types
        hitter_files = {
            'basic': os.path.join(DATA_DIR, "FanGraphs_Data", "hitters", f'fangraphs_hitters_{year}.csv'),
            'advanced': os.path.join(DATA_DIR, "FanGraphs_Data", "hitters", f'fangraphs_hitters_{year}_advanced.csv'),
            'standard': os.path.join(DATA_DIR, "FanGraphs_Data", "hitters", f'fangraphs_hitters_{year}_standard.csv'),
            'battedball': os.path.join(DATA_DIR, "FanGraphs_Data", "hitters", f'fangraphs_hitters_{year}_standard.csv'),
        }

        hitter_data = {}
        for data_type, filename in hitter_files.items():
            if os.path.exists(filename):
                try:
                    df = pd.read_csv(filename, encoding='utf-8-sig')  # Handle BOM
                    df.columns = df.columns.str.strip()

                    for _, row in df.iterrows():
                        name = str(row.get('Name', '')).strip()
                        if name and name != 'nan':
                            key = f"{name}_{year}"
                            if key not in hitter_data:
                                hitter_data[key] = {'name': name, 'year': year, 'team': row.get('Team', 'UNK')}

                            # Add all columns with prefix for data type (preserve MLBAMID)
                            for col, val in row.items():
                                if col not in ['Name', 'Team', 'NameASCII', 'PlayerId']:
                                    prefixed_col = f"{data_type}_{col}" if data_type != 'basic' else col
                                    hitter_data[key][prefixed_col] = val
                                elif col == 'MLBAMID':
                                    # Always preserve MLBAMID for player identification
                                    hitter_data[key]['MLBAMID'] = val

                    print(f"  Hitters {data_type}: {len(df)} players loaded")
                except Exception as e:
                    print(f"  Error loading {filename}: {e}")

        # Store hitter data
        for key, data in hitter_data.items():
            fangraphs_data['hitters'][key] = data

        # Pitchers - combine 3 data types
        pitcher_files = {
            'basic': os.path.join(DATA_DIR, "FanGraphs_Data", "pitchers", f'fangraphs_pitchers_{year}.csv'),
            'advanced': os.path.join(DATA_DIR, "FanGraphs_Data", "pitchers", f'fangraphs_pitchers_{year}_advanced.csv'),
            'standard': os.path.join(DATA_DIR, "FanGraphs_Data", "pitchers", f'fangraphs_pitchers_{year}_standard.csv'),
            'battedball': os.path.join(DATA_DIR, "FanGraphs_Data", "pitchers", f'fangraphs_pitchers_{year}_battedball.csv')
        }

        pitcher_data = {}
        for data_type, filename in pitcher_files.items():
            if os.path.exists(filename):
                try:
                    df = pd.read_csv(filename, encoding='utf-8-sig')
                    df.columns = df.columns.str.strip()

                    for _, row in df.iterrows():
                        name = str(row.get('Name', '')).strip()
                        if name and name != 'nan':
                            key = f"{name}_{year}"
                            if key not in pitcher_data:
                                pitcher_data[key] = {'name': name, 'year': year, 'team': row.get('Team', 'UNK')}

                            # Add all columns with prefix for data type (preserve MLBAMID)
                            for col, val in row.items():
                                if col not in ['Name', 'Team', 'NameASCII', 'PlayerId']:
                                    prefixed_col = f"{data_type}_{col}" if data_type != 'basic' else col
                                    pitcher_data[key][prefixed_col] = val
                                elif col == 'MLBAMID':
                                    # Always preserve MLBAMID for player identification
                                    pitcher_data[key]['MLBAMID'] = val

                    print(f"  Pitchers {data_type}: {len(df)} players loaded")
                except Exception as e:
                    print(f"  Error loading {filename}: {e}")

        # Store pitcher data
        for key, data in pitcher_data.items():
            fangraphs_data['pitchers'][key] = data

        # Defensive data - combine advanced and standard
        defensive_files = {
            'advanced': os.path.join(DATA_DIR, "FanGraphs_Data", "defensive", f'fangraphs_defensive_advanced_{year}.csv'),
            'standard': os.path.join(DATA_DIR, "FanGraphs_Data", "defensive", f'fangraphs_defensive_standard_{year}.csv')
        }

        defensive_data = {}
        for data_type, filename in defensive_files.items():
            if os.path.exists(filename):
                try:
                    df = pd.read_csv(filename, encoding='utf-8-sig')
                    df.columns = df.columns.str.strip()

                    for _, row in df.iterrows():
                        name = str(row.get('Name', '')).strip()
                        if name and name != 'nan':
                            key = f"{name}_{year}"
                            if key not in defensive_data:
                                defensive_data[key] = {'name': name, 'year': year, 'team': row.get('Team', 'UNK')}

                            # Add defensive columns with prefix
                            for col, val in row.items():
                                if col not in ['Name', 'Team', 'NameASCII', 'PlayerId', 'MLBAMID']:
                                    prefixed_col = f"def_{data_type}_{col}"
                                    defensive_data[key][prefixed_col] = val

                    print(f"  Defensive {data_type}: {len(df)} players loaded")
                except Exception as e:
                    print(f"  Error loading {filename}: {e}")

        # Store defensive data
        for key, data in defensive_data.items():
            fangraphs_data['defensive'][key] = data

    # Cache the comprehensive dataset
    try:
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(fangraphs_data, f, indent=2, default=str)  # default=str handles pandas dtypes
        print(f"\nCached comprehensive FanGraphs data:")
        print(f"  Hitters: {len(fangraphs_data['hitters'])} player-seasons")
        print(f"  Pitchers: {len(fangraphs_data['pitchers'])} player-seasons")
        print(f"  Defensive: {len(fangraphs_data['defensive'])} player-seasons")
    except Exception as e:
        print(f"Warning: Could not cache comprehensive data: {e}")

    return fangraphs_data


def load_yearly_bp_data():
    """
    Load and unify Baseball Prospectus WARP data from 2016-2024 (hitters) and 2016-2024 (pitchers)

    NOTE: 2022-2024 pitcher data uses dual CSV format (regular + standard versions)

    Returns:
        dict: {
            'hitters': {player_name_year: warp_value},
            'pitchers': {player_name_year: warp_value}
        }
    """
    cache_file = os.path.join(CACHE_DIR, "yearly_bp_data_v2.json")

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
        filename = os.path.join(DATA_DIR, "BP_Data", "hitters", f'bp_hitters_{year}.csv')
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

    # Load pitchers data (2016-2024) - Extended to include 2022-2024 with dual CSV support
    for year in range(2016, 2025):  # 2016-2024
       
        filenames = [
            os.path.join(DATA_DIR, "BP_Data", "pitchers", f'bp_pitchers_{year}.csv'),
            os.path.join(DATA_DIR, "BP_Data", "pitchers", f'bp_pitchers_{year}_standard.csv')
        ]

        players_loaded = 0
        for filename in filenames:
            if not os.path.exists(filename):
                continue

            try:
                df = pd.read_csv(filename)
                df.columns = df.columns.str.strip().str.strip('"')

                # Handle different formats across years
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
                        # For dual CSV years, combine data (later files may have additional players)
                        if key not in bp_data['pitchers'] or bp_data['pitchers'][key] == 0:
                            bp_data['pitchers'][key] = float(warp_value)

                players_loaded += len(df)

            except Exception as e:
                print(f"  Error loading {filename}: {e}")

        if players_loaded > 0:
            # Count unique players for this year
            year_players = len([k for k in bp_data['pitchers'].keys() if k.endswith(f'_{year}')])
            csv_info = " (dual CSV)" if year >= 2022 else ""
            print(f"  {year} pitchers: {year_players} unique players loaded{csv_info}")

    # Cache the results
    try:
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(bp_data, f, indent=2)
        print(f"Cached BP data to {cache_file}")
    except Exception as e:
        print(f"Warning: Could not cache BP data: {e}")

    print(f"Loaded BP data: {len(bp_data['hitters'])} hitter-seasons, {len(bp_data['pitchers'])} pitcher-seasons")
    return bp_data