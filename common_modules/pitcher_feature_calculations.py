"""
Pitcher feature calculations module for oWAR project.

This module handles calculation and processing of pitcher-specific features and metrics
from various sources including Baseball Prospectus, FanGraphs, and Statcast.
"""

# Standard library imports
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Third-party imports
import numpy as np
import pandas as pd

# Local imports
from .config import (
    BP_DATA_DIR,
    FANGRAPHS_DATA_DIR,
    STATCAST_DATA_DIR,
    PRE_2020_CUTOFF,
    STANDARD_FILE_FILTER,
    DEFAULT_DATA_YEARS,
    STATCAST_AVAILABLE_YEARS
)
from .logging import get_logger

# Initialize logger
logger = get_logger(__name__)


def fix_bp_derived_statistics(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """
    Add missing derived statistics for BP data.

    For pre-2020 data, calculates K% and BB% if missing since BP
    started providing these directly only from 2020 onwards.

    Args:
            df: DataFrame with BP data for a given year
            year: The year of the data

    Returns:
            DataFrame with K% and BB% calculated if missing

    Raises:
            ValueError: If required columns for calculation are missing
    """
    try:
        df_fixed = df.copy()

        # For pre-2020 data, calculate K% and BB% if missing
        if year < PRE_2020_CUTOFF:
            logger.info(f"Calculating derived statistics for {year} data")

            # Calculate K% (strikeouts / plate appearances)
            if 'K%' not in df_fixed.columns:
                if 'SO' in df_fixed.columns and 'PA' in df_fixed.columns:
                    df_fixed['K%'] = np.where(
                        df_fixed['PA'] > 0,
                        (df_fixed['SO'] / df_fixed['PA']) * 100,
                        0.0
                    )
                    logger.debug(f"Calculated K% from SO/PA for {year}")
                elif 'K' in df_fixed.columns and 'PA' in df_fixed.columns:
                    df_fixed['K%'] = np.where(
                        df_fixed['PA'] > 0,
                        (df_fixed['K'] / df_fixed['PA']) * 100,
                        0.0
                    )
                    logger.debug(f"Calculated K% from K/PA for {year}")

            # Calculate BB% (walks / plate appearances)
            if 'BB%' not in df_fixed.columns and 'BB' in df_fixed.columns and 'PA' in df_fixed.columns:
                df_fixed['BB%'] = np.where(
                    df_fixed['PA'] > 0,
                    (df_fixed['BB'] / df_fixed['PA']) * 100,
                    0.0
                )
                logger.debug(f"Calculated BB% from BB/PA for {year}")

        return df_fixed

    except Exception as e:
        logger.error(f"Error fixing BP derived statistics for {year}: {e}")
        raise


def load_fixed_bp_data(data_dir: Optional[Path] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load BP data with properly calculated derived statistics.

    Args:
            data_dir: Optional path to data directory. If None, uses default from config.

    Returns:
            Tuple of (hitter_data, pitcher_data) DataFrames with fixed K% and BB%

    Raises:
            FileNotFoundError: If BP data directory doesn't exist
            ValueError: If no valid data files are found
    """
    logger.info("Loading BP data with fixed derived statistics")

    if data_dir is None:
        data_dir = BP_DATA_DIR

    if not data_dir.exists():
        raise FileNotFoundError(f"BP data directory not found: {data_dir}")

    # Load hitter data
    hitter_dir = data_dir / "hitters"
    all_hitter_data = []

    if hitter_dir.exists():
        hitter_files = list(hitter_dir.glob("bp_hitters_*.csv"))
        hitter_files = [f for f in hitter_files if STANDARD_FILE_FILTER not in f.name]

        for file in sorted(hitter_files):
            try:
                year = int(file.stem.split('_')[-1])
                df = pd.read_csv(file, encoding='utf-8-sig')

                if 'WARP' in df.columns or 'BWARP' in df.columns:
                    # Standardize column names
                    if 'BWARP' in df.columns and 'WARP' not in df.columns:
                        df = df.rename(columns={'BWARP': 'WARP'})
                    if 'NAME' in df.columns and 'Name' not in df.columns:
                        df = df.rename(columns={'NAME': 'Name'})

                    df['Season'] = year
                    df['Year'] = year
                    df_fixed = fix_bp_derived_statistics(df, year)
                    all_hitter_data.append(df_fixed)
                    logger.debug(f"Loaded {len(df_fixed)} hitter records for {year}")

            except Exception as e:
                logger.error(f"Error loading hitter file {file}: {e}")

    # Load pitcher data
    pitcher_dir = data_dir / "pitchers"
    all_pitcher_data = []

    if pitcher_dir.exists():
        pitcher_files = list(pitcher_dir.glob("bp_pitchers_*.csv"))
        pitcher_files = [f for f in pitcher_files if STANDARD_FILE_FILTER not in f.name]

        for file in sorted(pitcher_files):
            try:
                year = int(file.stem.split('_')[-1])
                df = pd.read_csv(file, encoding='utf-8-sig')

                if 'WARP' in df.columns or 'PWARP' in df.columns:
                    # Standardize column names
                    if 'PWARP' in df.columns and 'WARP' not in df.columns:
                        df = df.rename(columns={'PWARP': 'WARP'})
                    if 'NAME' in df.columns and 'Name' not in df.columns:
                        df = df.rename(columns={'NAME': 'Name'})

                    df['Season'] = year
                    df['Year'] = year
                    df_fixed = fix_bp_derived_statistics(df, year)
                    all_pitcher_data.append(df_fixed)
                    logger.debug(f"Loaded {len(df_fixed)} pitcher records for {year}")

            except Exception as e:
                logger.error(f"Error loading pitcher file {file}: {e}")

    # Combine all data
    if not all_hitter_data and not all_pitcher_data:
        raise ValueError("No valid BP data files found")

    combined_hitters = pd.concat(all_hitter_data,
                                 ignore_index=True) if all_hitter_data else pd.DataFrame()
    combined_pitchers = pd.concat(all_pitcher_data,
                                  ignore_index=True) if all_pitcher_data else pd.DataFrame()

    logger.info(
        f"Loaded {
            len(combined_hitters)} hitter and {
            len(combined_pitchers)} pitcher records")
    return combined_hitters, combined_pitchers


def load_fangraphs_feature(year: int, feature_type: str, column_name: str,
                           data_dir: Optional[Path] = None) -> Dict[int, float]:
    """
    Load a specific feature from FanGraphs files.

    Args:
            year: Year to load data for
            feature_type: Type of FanGraphs file ('advanced', 'standard', 'battedball')
            column_name: Name of the column to extract
            data_dir: Optional data directory path

    Returns:
            Dictionary mapping player_id to feature value

    Example:
            >>> load_fangraphs_feature(2024, 'advanced', 'LOB%')
            {123456: 75.2, 234567: 68.9, ...}
    """
    if data_dir is None:
        data_dir = FANGRAPHS_DATA_DIR

    feature_data = {}

    # Try multiple filename patterns
    filename_patterns = [
        f"fangraphs_pitchers_{year}_firsthalf_{feature_type}.csv",
        f"fangraphs_pitchers_{year}_{feature_type}.csv",
    ]

    for filename in filename_patterns:
        filepath = data_dir / "pitchers" / filename

        if filepath.exists():
            try:
                df = pd.read_csv(filepath)

                if 'MLBAMID' in df.columns and column_name in df.columns:
                    for _, row in df.iterrows():
                        player_id = row['MLBAMID']
                        value = row[column_name]

                        if pd.notna(player_id) and pd.notna(value):
                            feature_data[int(player_id)] = float(value)

                    logger.debug(
                        f"Loaded {column_name} from {filename}: {
                            len(feature_data)} players")
                    break

            except Exception as e:
                logger.error(f"Error loading {column_name} from {filepath}: {e}")

    return feature_data


def load_lob_percentage(year: int, data_dir: Optional[Path] = None) -> Dict[int, float]:
    """
    Load LOB% from FanGraphs advanced files.

    Args:
            year: Year to load
            data_dir: Optional data directory

    Returns:
            Dictionary mapping player_id to LOB% (in percentage format)
    """
    lob_data = load_fangraphs_feature(year, 'advanced', 'LOB%', data_dir)

    # Convert from decimal to percentage if needed
    converted_data = {}
    for player_id, value in lob_data.items():
        # FanGraphs stores as decimal (0.75 = 75%)
        converted_data[player_id] = value * 100 if value <= 1.0 else value

    return converted_data


def load_ground_ball_percentage(year: int, data_dir: Optional[Path] = None) -> Dict[int, float]:
    """
    Load GB% from FanGraphs battedball files with BP fallback.

    Args:
            year: Year to load
            data_dir: Optional data directory

    Returns:
            Dictionary mapping player_id to GB% (as decimal)
    """
    # Try FanGraphs first
    gb_data = load_fangraphs_feature(year, 'battedball', 'GB%', data_dir)

    # Fallback to BP if no FanGraphs data
    if not gb_data:
        gb_data = load_gb_from_bp(year, data_dir)

    return gb_data


def load_gb_from_bp(year: int, data_dir: Optional[Path] = None) -> Dict[int, float]:
    """
    Load GB% from Baseball Prospectus files as fallback.

    Args:
            year: Year to load
            data_dir: Optional data directory

    Returns:
            Dictionary mapping player_id to GB% (as decimal)
    """
    if data_dir is None:
        data_dir = BP_DATA_DIR

    filepath = data_dir / "pitchers" / f"bp_pitchers_{year}.csv"

    if not filepath.exists():
        return {}

    try:
        df = pd.read_csv(filepath)

        if 'mlbid' not in df.columns:
            return {}

        # Find GB% column
        gb_col = None
        for col in ['GB%', 'Ground_Ball_Pct', 'GB_Pct']:
            if col in df.columns:
                gb_col = col
                break

        if gb_col is None:
            return {}

        gb_data = {}
        for _, row in df.iterrows():
            player_id = row['mlbid']
            gb_pct = row[gb_col]

            if pd.notna(player_id) and pd.notna(gb_pct):
                # Convert to decimal if needed
                if gb_pct > 1.0:
                    gb_pct = gb_pct / 100.0
                gb_data[int(player_id)] = float(gb_pct)

        logger.debug(f"Loaded GB% from BP for {year}: {len(gb_data)} players")
        return gb_data

    except Exception as e:
        logger.error(f"Error loading BP GB% for {year}: {e}")
        return {}


def load_statcast_exit_velocity_data(
        data_dir: Optional[Path] = None) -> Dict[int, Dict[str, float]]:
    """
    Load Statcast exit velocity data for calculating Launch Quality Index.

    Args:
            data_dir: Optional path to Statcast data directory

    Returns:
            Dictionary of {player_id: {'avg_hit_angle': value, 'anglesweetspotpercent': value}}
    """
    if data_dir is None:
        data_dir = STATCAST_DATA_DIR / "exit_velocity"

    logger.info("Loading Statcast exit velocity data")
    all_data = {}

    for year in STATCAST_AVAILABLE_YEARS:
        filepath = data_dir / f"exit_velocity_pitchers_{year}.csv"

        if not filepath.exists():
            logger.debug(f"Statcast file not found for {year}")
            continue

        try:
            df = pd.read_csv(filepath)

            for _, row in df.iterrows():
                if pd.notna(row.get('player_id')):
                    try:
                        player_id = int(float(row['player_id']))

                        if player_id not in all_data:
                            all_data[player_id] = {}

                        if pd.notna(row.get('avg_hit_angle')):
                            all_data[player_id]['avg_hit_angle'] = float(row['avg_hit_angle'])

                        if pd.notna(row.get('anglesweetspotpercent')):
                            all_data[player_id]['anglesweetspotpercent'] = float(
                                row['anglesweetspotpercent'])

                    except (ValueError, TypeError):
                        continue

            logger.debug(f"Loaded Statcast data for {year}")

        except Exception as e:
            logger.error(f"Error loading Statcast data for {year}: {e}")

    logger.info(f"Loaded Statcast data for {len(all_data)} total players")
    return all_data


def load_opportunity_components(years: Optional[List[int]] = None,
                                data_dir: Optional[Path] = None) -> Dict[str, Dict[int, float]]:
    """
    Load all components needed for Opportunity_Success calculation.

    Args:
            years: Years to load (defaults to 2016-2025)
            data_dir: Optional data directory

    Returns:
            Dictionary with keys: 'QS', 'SV', 'HLD', 'BS', 'G'
    """
    if years is None:
        years = DEFAULT_DATA_YEARS

    logger.info(f"Loading opportunity components for years {years[0]}-{years[-1]}")

    components = {
        'QS': {},
        'SV': {},
        'HLD': {},
        'BS': {},
        'G': {}
    }

    for year in years:
        try:
            # Load each component
            qs_data = load_fangraphs_feature(year, 'standard', 'QS', data_dir)
            components['QS'].update(qs_data)

            sv_data = load_fangraphs_feature(year, 'standard', 'SV', data_dir)
            components['SV'].update(sv_data)

            hld_data = load_fangraphs_feature(year, 'standard', 'HLD', data_dir)
            components['HLD'].update(hld_data)

            bs_data = load_fangraphs_feature(year, 'standard', 'BS', data_dir)
            components['BS'].update(bs_data)

            games_data = load_fangraphs_feature(year, 'standard', 'G', data_dir)
            components['G'].update(games_data)

        except Exception as e:
            logger.error(f"Error loading opportunity components for {year}: {e}")

    return components


def load_contact_quality_data(years: Optional[List[int]] = None,
                              data_dir: Optional[Path] = None) -> Dict[str, Dict[int, float]]:
    """
    Load contact quality data (Hard%, Med%, Soft%) from FanGraphs.

    Args:
            years: Years to load
            data_dir: Optional data directory

    Returns:
            Dictionary with keys: 'Hard%', 'Med%', 'Soft%'
    """
    if years is None:
        years = DEFAULT_DATA_YEARS

    logger.info(f"Loading contact quality data for years {years[0]}-{years[-1]}")

    contact_data = {
        'Hard%': {},
        'Med%': {},
        'Soft%': {}
    }

    for year in years:
        try:
            for feature in ['Hard%', 'Med%', 'Soft%']:
                feature_data = load_fangraphs_feature(year, 'battedball', feature, data_dir)

                # Convert from decimal to percentage
                for player_id, value in feature_data.items():
                    contact_data[feature][player_id] = value * 100 if value <= 1.0 else value

        except Exception as e:
            logger.error(f"Error loading contact quality data for {year}: {e}")

    return contact_data


def load_percentage_features(years: Optional[List[int]] = None,
                             data_dir: Optional[Path] = None) -> Dict[str, Dict[int, float]]:
    """
    Load all percentage-based features (BB%, K%, K-BB%, HR/FB%, LOB%).

    Args:
            years: Years to load
            data_dir: Optional data directory

    Returns:
            Dictionary with percentage feature data
    """
    if years is None:
        years = DEFAULT_DATA_YEARS

    logger.info(f"Loading percentage features for years {years[0]}-{years[-1]}")

    percentage_features = {
        'BB%': {},
        'K%': {},
        'K-BB%': {},
        'HR/FB%': {},
        'LOB%': {}
    }

    for year in years:
        try:
            # Load BB% and K%
            bb_data = load_fangraphs_feature(year, 'advanced', 'BB%', data_dir)
            k_data = load_fangraphs_feature(year, 'advanced', 'K%', data_dir)
            k_bb_data = load_fangraphs_feature(year, 'advanced', 'K-BB%', data_dir)

            # Convert to percentage format
            for player_id, value in bb_data.items():
                percentage_features['BB%'][player_id] = value * 100 if value <= 1.0 else value

            for player_id, value in k_data.items():
                percentage_features['K%'][player_id] = value * 100 if value <= 1.0 else value

            for player_id, value in k_bb_data.items():
                percentage_features['K-BB%'][player_id] = value * 100 if value <= 1.0 else value

            # Load HR/FB%
            hr_fb_data = load_fangraphs_feature(year, 'battedball', 'HR/FB', data_dir)
            for player_id, value in hr_fb_data.items():
                percentage_features['HR/FB%'][player_id] = value * 100 if value <= 1.0 else value

            # Load LOB%
            lob_data = load_lob_percentage(year, data_dir)
            percentage_features['LOB%'].update(lob_data)

        except Exception as e:
            logger.error(f"Error loading percentage features for {year}: {e}")

    return percentage_features
