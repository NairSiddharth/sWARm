"""Data loading functionality for predictive modeling.

This module handles loading and initial processing of FanGraphs and WARP data
for both hitters and pitchers, providing a unified interface for model training.
"""

from pathlib import Path
from typing import Optional
import glob
import os

import pandas as pd

from common_modules.config import DATA_DIR
from common_modules.logging import get_logger

logger = get_logger(__name__)


def load_expanded_fangraphs_data(
    data_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Load expanded FanGraphs data with full PA spectrum (0 to max PA).

    Uses new file structure: fangraphs_hitters_xxxx.csv (main files with WAR)

    Args:
        data_dir: Optional directory path for data files

    Returns:
        DataFrame containing combined hitter data from all years
    """
    logger.info("Loading expanded FanGraphs hitter data...")

    if data_dir is None:
        data_dir = DATA_DIR

    # Load main hitter files (these contain WAR)
    hitter_files = glob.glob(
        os.path.join(data_dir, "FanGraphs_Data", "hitters", "fangraphs_hitters_*.csv"),
    )

    # Exclude the suffixed files, we only want the main files with years
    hitter_files = [
        f for f in hitter_files
        if not any(
            suffix in f
            for suffix in ['_standard', '_advanced', '_battedball', '_firsthalf']
        )
    ]

    all_hitter_data = []
    for file in sorted(hitter_files):
        year = int(os.path.basename(file).split('_')[-1].replace('.csv', ''))

        try:
            df = pd.read_csv(file, encoding='utf-8-sig')

            if 'WAR' in df.columns:
                # Add year info
                df['Year'] = year
                df['Type'] = 'Hitter'
                all_hitter_data.append(df)
                logger.info(f"SUCCESS {year}: {len(df)} hitter records loaded")
            else:
                logger.warning(f"{year}: No WAR column found in {file}")

        except Exception as e:
            logger.error(f"{year}: Error loading {file} - {e}")

    if all_hitter_data:
        combined_hitters = pd.concat(all_hitter_data, ignore_index=True)

        # Check PA distribution to confirm expansion
        if 'PA' in combined_hitters.columns:
            logger.info(
                f"Expanded FanGraphs hitter data: {len(combined_hitters)} total records",
            )
            logger.info(
                f"PA range: {combined_hitters['PA'].min()} to {combined_hitters['PA'].max()}",
            )
            logger.debug(f"Players with 0 PA: {(combined_hitters['PA'] == 0).sum()}")
            logger.debug(f"Players with <100 PA: {(combined_hitters['PA'] < 100).sum()}")
            logger.debug(f"Players with 400+ PA: {(combined_hitters['PA'] >= 400).sum()}")

        return combined_hitters
    else:
        logger.warning("No hitter data loaded")
        return pd.DataFrame()


def load_expanded_fangraphs_pitcher_data(
    data_dir: Optional[Path] = None
) -> pd.DataFrame:
    """Load expanded FanGraphs pitcher data with full appearance spectrum.

    Uses new file structure: fangraphs_pitchers_xxxx.csv (main files with WAR)

    Args:
        data_dir: Optional directory path for data files

    Returns:
        DataFrame containing combined pitcher data from all years
    """
    logger.info("Loading expanded FanGraphs pitcher data...")

    if data_dir is None:
        data_dir = DATA_DIR

    # Load main pitcher files (these contain WAR)
    pitcher_files = glob.glob(
        os.path.join(data_dir, "FanGraphs_Data", "pitchers", "fangraphs_pitchers_*.csv"),
    )

    # Exclude the suffixed files, we only want the main files with years
    pitcher_files = [
        f for f in pitcher_files
        if not any(
            suffix in f
            for suffix in ['_standard', '_advanced', '_battedball', '_firsthalf',
                          '_winprobability', '_stuff', '_plate_discipline']
        )
    ]

    all_pitcher_data = []
    for file in sorted(pitcher_files):
        year = int(os.path.basename(file).split('_')[-1].replace('.csv', ''))

        try:
            df = pd.read_csv(file, encoding='utf-8-sig')

            if 'WAR' in df.columns:
                # Add year column
                df['Year'] = year
                all_pitcher_data.append(df)
                logger.info(f"SUCCESS {year}: {len(df)} pitcher records loaded")
            else:
                logger.warning(f"{year}: No WAR column found, skipping")

        except Exception as e:
            logger.error(f"{year}: Error loading - {e}")

    if all_pitcher_data:
        combined_pitchers = pd.concat(all_pitcher_data, ignore_index=True)

        # Check appearance distribution to confirm expansion
        if 'G' in combined_pitchers.columns:
            logger.info(
                f"Expanded FanGraphs pitcher data: {len(combined_pitchers)} total records",
            )
            logger.info(
                f"Games range: {combined_pitchers['G'].min()} to {combined_pitchers['G'].max()}",
            )
            logger.debug(f"Pitchers with 0 G: {(combined_pitchers['G'] == 0).sum()}")
            logger.debug(f"Pitchers with <10 G: {(combined_pitchers['G'] < 10).sum()}")
            logger.debug(f"Pitchers with 30+ G: {(combined_pitchers['G'] >= 30).sum()}")

        return combined_pitchers
    else:
        logger.warning("No pitcher data loaded")
        return pd.DataFrame()


def load_comprehensive_warp_hitter_data() -> Optional[pd.DataFrame]:
    """Load comprehensive WARP hitter data with multi-source integration.

    Returns:
        DataFrame with WARP hitter data or None if loading fails
    """
    logger.info("Loading comprehensive WARP hitter data")
    from .data_preparation import prepare_data_for_kfold

    try:
        hitter_data, _ = prepare_data_for_kfold()
        return hitter_data['warp'] if hitter_data else None
    except Exception as e:
        logger.error(f"Error loading WARP hitter data: {e}")
        return None


def load_comprehensive_fangraphs_hitter_data() -> Optional[pd.DataFrame]:
    """Load comprehensive FanGraphs hitter WAR data with multi-source integration.

    Returns:
        DataFrame with FanGraphs hitter data or None if loading fails
    """
    logger.info("Loading comprehensive FanGraphs hitter data")
    from .data_preparation import prepare_data_for_kfold

    try:
        hitter_data, _ = prepare_data_for_kfold()
        return hitter_data['war'] if hitter_data else None
    except Exception as e:
        logger.error(f"Error loading FanGraphs hitter data: {e}")
        return None


def load_comprehensive_warp_pitcher_data() -> Optional[pd.DataFrame]:
    """Load comprehensive WARP pitcher data with multi-source integration.

    Returns:
        DataFrame with WARP pitcher data or None if loading fails
    """
    logger.info("Loading comprehensive WARP pitcher data")
    from .data_preparation import prepare_data_for_kfold

    try:
        _, pitcher_data = prepare_data_for_kfold()
        return pitcher_data['warp'] if pitcher_data else None
    except Exception as e:
        logger.error(f"Error loading WARP pitcher data: {e}")
        return None


def load_comprehensive_fangraphs_pitcher_data() -> Optional[pd.DataFrame]:
    """Load comprehensive FanGraphs pitcher WAR data with multi-source integration.

    Returns:
        DataFrame with FanGraphs pitcher data or None if loading fails
    """
    logger.info("Loading comprehensive FanGraphs pitcher data")
    from .data_preparation import prepare_data_for_kfold

    try:
        _, pitcher_data = prepare_data_for_kfold()
        return pitcher_data['war'] if pitcher_data else None
    except Exception as e:
        logger.error(f"Error loading FanGraphs pitcher data: {e}")
        return None


# Backward compatibility functions for existing code
def load_and_prepare_hitter_data() -> Optional[pd.DataFrame]:
    """Load and prepare hitter WARP data for modeling.

    This is a backward compatibility wrapper around load_comprehensive_warp_hitter_data.

    Returns:
        DataFrame with prepared hitter WARP data or None if loading fails
    """
    logger.info("Loading and preparing hitter WARP data")
    try:
        return load_comprehensive_warp_hitter_data()
    except Exception as e:
        logger.error(f"Error loading hitter WARP data: {e}")
        return None


def load_and_prepare_hitter_war_data() -> Optional[pd.DataFrame]:
    """Load and prepare hitter WAR data for modeling.

    This is a backward compatibility wrapper around load_comprehensive_fangraphs_hitter_data.

    Returns:
        DataFrame with prepared hitter WAR data or None if loading fails
    """
    logger.info("Loading and preparing hitter WAR data")
    try:
        return load_comprehensive_fangraphs_hitter_data()
    except Exception as e:
        logger.error(f"Error loading hitter WAR data: {e}")
        return None


def load_and_prepare_pitcher_data() -> Optional[pd.DataFrame]:
    """Load and prepare pitcher WARP data for modeling.

    This is a backward compatibility wrapper around load_comprehensive_warp_pitcher_data.

    Returns:
        DataFrame with prepared pitcher WARP data or None if loading fails
    """
    logger.info("Loading and preparing pitcher WARP data")
    try:
        return load_comprehensive_warp_pitcher_data()
    except Exception as e:
        logger.error(f"Error loading pitcher WARP data: {e}")
        return None


def load_and_prepare_pitcher_war_data() -> Optional[pd.DataFrame]:
    """Load and prepare pitcher WAR data for modeling.

    This is a backward compatibility wrapper around load_comprehensive_fangraphs_pitcher_data.

    Returns:
        DataFrame with prepared pitcher WAR data or None if loading fails
    """
    logger.info("Loading and preparing pitcher WAR data")
    try:
        return load_comprehensive_fangraphs_pitcher_data()
    except Exception as e:
        logger.error(f"Error loading pitcher WAR data: {e}")
        return None
