"""
Enhanced Data Loading Module for oWAR Analysis

This module handles the comprehensive multi-file data structure including:
- FanGraphs data (3 CSV types per year: main, advanced, standard)
- Baseball Prospectus data (2 CSV types per year: main, standard)
- Season-aware data preparation for multi-year modeling
- Unified data access with backward compatibility

Data Structure:
- FanGraphs: 54 files (2016-2024, hitters/pitchers, 3 types each)
- Baseball Prospectus: 36 files (2016-2024, hitters/pitchers, 2 types each)
- Total: 90 comprehensive data files with rich metrics
"""

import os
import pandas as pd
import numpy as np
import json
from pathlib import Path
import glob
from typing import Dict, List, Tuple, Optional

# Import configuration from parent module
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Constants
DATA_DIR = r"C:\Users\nairs\Documents\GithubProjects\oWAR\MLB Player Data"
CACHE_DIR = r"C:\Users\nairs\Documents\GithubProjects\oWAR\cache"

# Ensure cache directory exists
os.makedirs(CACHE_DIR, exist_ok=True)

__all__ = [
    'load_fangraphs_data_comprehensive',
    'load_bp_data_comprehensive',
    'load_unified_dataset',
    'get_available_years',
    'validate_data_consistency',
    'create_season_aware_splits',
    'load_enhanced_yearly_data'
]

def get_available_years() -> Dict[str, List[int]]:
    """
    Discover available years for each data source and type

    Returns:
        Dict with data source availability by year
    """
    availability = {
        'fangraphs_hitters': [],
        'fangraphs_pitchers': [],
        'bp_hitters': [],
        'bp_pitchers': []
    }

    # Check FanGraphs files
    for player_type in ['hitters', 'pitchers']:
        pattern = os.path.join(DATA_DIR, f"fangraphs_{player_type}_*.csv")
        files = glob.glob(pattern)
        years = []
        for file in files:
            basename = os.path.basename(file)
            # Extract year from filename like "fangraphs_hitters_2024.csv"
            if not ('advanced' in basename or 'standard' in basename):
                try:
                    year = int(basename.split('_')[-1].replace('.csv', ''))
                    years.append(year)
                except ValueError:
                    continue
        availability[f'fangraphs_{player_type}'] = sorted(years)

    # Check BP files
    for player_type in ['hitters', 'pitchers']:
        pattern = os.path.join(DATA_DIR, f"bp_{player_type}_*.csv")
        files = glob.glob(pattern)
        years = []
        for file in files:
            basename = os.path.basename(file)
            # Extract year from filename like "bp_hitters_2024.csv"
            if not 'standard' in basename:
                try:
                    year = int(basename.split('_')[-1].replace('.csv', ''))
                    years.append(year)
                except ValueError:
                    continue
        availability[f'bp_{player_type}'] = sorted(years)

    return availability

def load_fangraphs_data_comprehensive(player_type: str = 'hitters', years: Optional[List[int]] = None) -> pd.DataFrame:
    """
    Load comprehensive FanGraphs data combining all 3 file types per year

    Args:
        player_type: 'hitters' or 'pitchers'
        years: List of years to load (None = all available)

    Returns:
        Combined DataFrame with all FanGraphs metrics and season information
    """
    cache_file = os.path.join(CACHE_DIR, f"fangraphs_{player_type}_comprehensive.json")

    # Check cache first
    if os.path.exists(cache_file):
        try:
            cached_df = pd.read_json(cache_file, orient='records')
            print(f"Loaded cached FanGraphs {player_type} data ({len(cached_df)} player-seasons)")
            return cached_df
        except:
            pass

    print(f"Loading comprehensive FanGraphs {player_type} data...")

    if years is None:
        availability = get_available_years()
        years = availability[f'fangraphs_{player_type}']

    all_data = []

    for year in years:
        print(f"  Processing FanGraphs {player_type} {year}...")

        # Load the 3 file types for this year
        main_file = os.path.join(DATA_DIR, f"fangraphs_{player_type}_{year}.csv")
        advanced_file = os.path.join(DATA_DIR, f"fangraphs_{player_type}_advanced_{year}.csv")
        standard_file = os.path.join(DATA_DIR, f"fangraphs_{player_type}_standard_{year}.csv")

        dataframes = {}

        # Load main file (has WAR, wOBA, wRC+, etc.)
        if os.path.exists(main_file):
            try:
                df_main = pd.read_csv(main_file, encoding='utf-8-sig')
                df_main['Year'] = year
                df_main['DataSource'] = 'FanGraphs'
                dataframes['main'] = df_main
            except Exception as e:
                print(f"    Warning: Could not load {main_file}: {e}")
                continue

        # Load advanced file (has UBR, wSB, wRAA, etc.)
        if os.path.exists(advanced_file):
            try:
                df_advanced = pd.read_csv(advanced_file, encoding='utf-8-sig')
                # Merge with main on common identifiers
                merge_cols = ['Name', 'Team', 'PlayerId', 'MLBAMID']
                available_merge_cols = [col for col in merge_cols if col in df_main.columns and col in df_advanced.columns]
                if available_merge_cols:
                    # Select only unique columns from advanced (avoid duplicates)
                    advanced_unique_cols = available_merge_cols + [col for col in df_advanced.columns if col not in df_main.columns]
                    df_advanced_unique = df_advanced[advanced_unique_cols]
                    dataframes['main'] = dataframes['main'].merge(df_advanced_unique, on=available_merge_cols, how='left')
            except Exception as e:
                print(f"    Warning: Could not merge advanced file: {e}")

        # Load standard file (has traditional counting stats)
        if os.path.exists(standard_file):
            try:
                df_standard = pd.read_csv(standard_file, encoding='utf-8-sig')
                # Merge with main on common identifiers
                merge_cols = ['Name', 'Team', 'PlayerId', 'MLBAMID']
                available_merge_cols = [col for col in merge_cols if col in dataframes['main'].columns and col in df_standard.columns]
                if available_merge_cols:
                    # Select only unique columns from standard
                    standard_unique_cols = available_merge_cols + [col for col in df_standard.columns if col not in dataframes['main'].columns]
                    df_standard_unique = df_standard[standard_unique_cols]
                    dataframes['main'] = dataframes['main'].merge(df_standard_unique, on=available_merge_cols, how='left')
            except Exception as e:
                print(f"    Warning: Could not merge standard file: {e}")

        if 'main' in dataframes:
            all_data.append(dataframes['main'])

    if not all_data:
        print(f"No FanGraphs {player_type} data found!")
        return pd.DataFrame()

    # Combine all years
    combined_df = pd.concat(all_data, ignore_index=True)

    # Add metadata
    combined_df['PlayerType'] = player_type.rstrip('s').title()  # 'hitters' -> 'Hitter'

    # Cache the result
    try:
        combined_df.to_json(cache_file, orient='records', indent=2)
        print(f"Cached FanGraphs {player_type} data ({len(combined_df)} player-seasons)")
    except Exception as e:
        print(f"Warning: Could not cache data: {e}")

    print(f"Loaded {len(combined_df)} {player_type} player-seasons from {len(years)} years ({min(years)}-{max(years)})")
    return combined_df

def load_bp_data_comprehensive(player_type: str = 'hitters', years: Optional[List[int]] = None) -> pd.DataFrame:
    """
    Load comprehensive Baseball Prospectus data combining both file types per year

    Args:
        player_type: 'hitters' or 'pitchers'
        years: List of years to load (None = all available)

    Returns:
        Combined DataFrame with all BP metrics and season information
    """
    cache_file = os.path.join(CACHE_DIR, f"bp_{player_type}_comprehensive.json")

    # Check cache first
    if os.path.exists(cache_file):
        try:
            cached_df = pd.read_json(cache_file, orient='records')
            print(f"Loaded cached BP {player_type} data ({len(cached_df)} player-seasons)")
            return cached_df
        except:
            pass

    print(f"Loading comprehensive BP {player_type} data...")

    if years is None:
        availability = get_available_years()
        years = availability[f'bp_{player_type}']

    all_data = []

    for year in years:
        print(f"  Processing BP {player_type} {year}...")

        # Load the 2 file types for this year
        main_file = os.path.join(DATA_DIR, f"bp_{player_type}_{year}.csv")
        standard_file = os.path.join(DATA_DIR, f"bp_{player_type}_{year}_standard.csv")

        # Load main file (has WARP, DRC+, DRA, etc.)
        if os.path.exists(main_file):
            try:
                df_main = pd.read_csv(main_file, encoding='utf-8-sig')
                df_main['Year'] = year
                df_main['DataSource'] = 'BaseballProspectus'

                # Load standard file if available and merge
                if os.path.exists(standard_file):
                    try:
                        df_standard = pd.read_csv(standard_file, encoding='utf-8-sig')
                        # Merge on common identifiers
                        merge_cols = ['bpid', 'mlbid', 'Name', 'Team']
                        available_merge_cols = [col for col in merge_cols if col in df_main.columns and col in df_standard.columns]
                        if available_merge_cols:
                            # Select only unique columns from standard
                            standard_unique_cols = available_merge_cols + [col for col in df_standard.columns if col not in df_main.columns]
                            df_standard_unique = df_standard[standard_unique_cols]
                            df_main = df_main.merge(df_standard_unique, on=available_merge_cols, how='left')
                    except Exception as e:
                        print(f"    Warning: Could not merge standard file: {e}")

                all_data.append(df_main)
            except Exception as e:
                print(f"    Warning: Could not load {main_file}: {e}")
                continue

    if not all_data:
        print(f"No BP {player_type} data found!")
        return pd.DataFrame()

    # Combine all years
    combined_df = pd.concat(all_data, ignore_index=True)

    # Add metadata
    combined_df['PlayerType'] = player_type.rstrip('s').title()  # 'hitters' -> 'Hitter'

    # Cache the result
    try:
        combined_df.to_json(cache_file, orient='records', indent=2)
        print(f"Cached BP {player_type} data ({len(combined_df)} player-seasons)")
    except Exception as e:
        print(f"Warning: Could not cache data: {e}")

    print(f"Loaded {len(combined_df)} {player_type} player-seasons from {len(years)} years ({min(years)}-{max(years)})")
    return combined_df

def load_unified_dataset(player_type: str = 'hitters',
                        include_fangraphs: bool = True,
                        include_bp: bool = True,
                        years: Optional[List[int]] = None) -> pd.DataFrame:
    """
    Load and combine FanGraphs + BP data into unified dataset

    Args:
        player_type: 'hitters' or 'pitchers'
        include_fangraphs: Include FanGraphs data
        include_bp: Include BP data
        years: List of years to load (None = all available)

    Returns:
        Unified DataFrame ready for modeling with season information
    """
    datasets = []

    if include_fangraphs:
        fg_data = load_fangraphs_data_comprehensive(player_type, years)
        if not fg_data.empty:
            datasets.append(fg_data)

    if include_bp:
        bp_data = load_bp_data_comprehensive(player_type, years)
        if not bp_data.empty:
            datasets.append(bp_data)

    if not datasets:
        print(f"No data loaded for {player_type}!")
        return pd.DataFrame()

    # If we have both datasets, we could join them on Name+Team+Year
    # For now, let's concatenate them (treating as separate observations)
    unified_df = pd.concat(datasets, ignore_index=True)

    print(f"Unified dataset: {len(unified_df)} {player_type} player-seasons")
    print(f"  Data sources: {unified_df['DataSource'].value_counts().to_dict()}")
    print(f"  Years covered: {sorted(unified_df['Year'].unique())}")

    return unified_df

def validate_data_consistency(df: pd.DataFrame) -> Dict[str, any]:
    """
    Validate data consistency across years and sources

    Args:
        df: DataFrame to validate

    Returns:
        Dict with validation results
    """
    validation = {
        'total_rows': len(df),
        'unique_players': df['Name'].nunique() if 'Name' in df.columns else 0,
        'years_covered': sorted(df['Year'].unique()) if 'Year' in df.columns else [],
        'missing_data': {},
        'data_sources': df['DataSource'].value_counts().to_dict() if 'DataSource' in df.columns else {},
        'warnings': []
    }

    # Check for missing critical columns
    critical_cols = ['Name', 'Team', 'Year']
    for col in critical_cols:
        if col not in df.columns:
            validation['warnings'].append(f"Missing critical column: {col}")
        else:
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                validation['missing_data'][col] = missing_count

    # Check year consistency
    if 'Year' in df.columns:
        year_range = df['Year'].max() - df['Year'].min() + 1
        expected_years = set(range(df['Year'].min(), df['Year'].max() + 1))
        actual_years = set(df['Year'].unique())
        missing_years = expected_years - actual_years
        if missing_years:
            validation['warnings'].append(f"Missing years in sequence: {sorted(missing_years)}")

    return validation

def create_season_aware_splits(df: pd.DataFrame,
                              features: List[str],
                              target: str,
                              test_size: float = 0.25,
                              random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, List, List, List, List]:
    """
    Create train/test splits while preserving season information

    Args:
        df: DataFrame with season data
        features: List of feature column names
        target: Target column name
        test_size: Proportion for test set
        random_state: Random state for reproducibility

    Returns:
        (X_train, X_test, y_train, y_test, seasons_train, seasons_test)
    """
    from sklearn.model_selection import train_test_split

    # Prepare feature matrix and target
    X = df[features].copy()
    y = df[target].copy()
    seasons = df['Year'].copy() if 'Year' in df.columns else ['2021'] * len(df)

    # Handle missing values
    X = X.fillna(X.median(numeric_only=True))

    # Split while preserving season information
    X_train, X_test, y_train, y_test, seasons_train, seasons_test = train_test_split(
        X, y, seasons, test_size=test_size, random_state=random_state
    )

    return X_train, X_test, y_train, y_test, list(seasons_train), list(seasons_test)

def load_enhanced_yearly_data() -> Dict[str, pd.DataFrame]:
    """
    Load all enhanced yearly data for backward compatibility with existing code

    Returns:
        Dict with DataFrames for each data type
    """
    print("Loading enhanced yearly datasets...")

    datasets = {}

    # Load FanGraphs data
    datasets['fangraphs_hitters'] = load_fangraphs_data_comprehensive('hitters')
    datasets['fangraphs_pitchers'] = load_fangraphs_data_comprehensive('pitchers')

    # Load BP data
    datasets['bp_hitters'] = load_bp_data_comprehensive('hitters')
    datasets['bp_pitchers'] = load_bp_data_comprehensive('pitchers')

    # Create unified datasets
    datasets['unified_hitters'] = load_unified_dataset('hitters')
    datasets['unified_pitchers'] = load_unified_dataset('pitchers')

    # Validation
    for name, df in datasets.items():
        if not df.empty:
            validation = validate_data_consistency(df)
            print(f"\n{name}: {validation['total_rows']} rows, {validation['unique_players']} unique players")
            if validation['warnings']:
                print(f"  Warnings: {validation['warnings']}")

    return datasets