"""
WPA/LI feature loading and preprocessing for pitcher predictions.

Win Probability Added per Leverage Index captures clutch performance
and game impact, complementing traditional rate and result metrics.
"""

import glob
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from .logging import get_logger
from .config import DATA_DIR

logger = get_logger(__name__)

# Default value for pitchers without WPA/LI data
DEFAULT_WPA_LI = 0.0  # Neutral impact


def load_wpa_li_features(years: Optional[List[int]] = None) -> Dict[int, float]:
    """
    Load WPA/LI features from FanGraphs win probability files.

    Uses glob patterns to find files, extendable to any year without code changes.
    Matches players by MLBAMID (not name) to avoid duplicates and ensure accuracy.

    Args:
        years: Years to load (default: 2016-2024 for training)

    Returns:
        Dictionary mapping MLBAMID to WPA/LI value

    Raises:
        FileNotFoundError: If no data files found for any year

    Example:
        >>> wpa_li = load_wpa_li_features([2023, 2024])
        >>> wpa_li[660271]  # Skubal's MLBAMID
        1.54
    """
    if years is None:
        years = list(range(2016, 2025))

    wpa_li_data = {}
    files_found = 0

    for year in years:
        # Construct glob pattern (extendable, no hardcoded paths)
        pattern = str(DATA_DIR / "FanGraphs_Data" / "pitchers" /
                     f"fangraphs_pitchers_{year}_winprobability*.csv")

        files = glob.glob(pattern)

        # Filter for appropriate season type
        if year < 2025:
            # Historical: exclude partial seasons
            files = [f for f in files if "firsthalf" not in f and "secondhalf" not in f]
        else:
            # Current year: prefer firsthalf if available
            firsthalf = [f for f in files if "firsthalf" in f]
            files = firsthalf if firsthalf else files

        if not files:
            logger.warning(f"No WPA/LI data found for {year}")
            continue

        # Load data
        try:
            df = pd.read_csv(files[0])

            # Validate required columns
            if 'MLBAMID' not in df.columns or 'WPA/LI' not in df.columns:
                logger.error(f"Missing required columns in {files[0]}")
                continue

            # Extract and store
            valid_count = 0
            for _, row in df.iterrows():
                mlbamid = row.get('MLBAMID')
                wpa_li = row.get('WPA/LI')

                if pd.notna(mlbamid) and pd.notna(wpa_li):
                    wpa_li_data[int(mlbamid)] = float(wpa_li)
                    valid_count += 1

            logger.info(f"Loaded WPA/LI for {valid_count} pitchers from {year}")
            files_found += 1

        except Exception as e:
            logger.error(f"Error loading {files[0]}: {e}")
            continue

    if files_found == 0:
        raise FileNotFoundError(
            f"No WPA/LI data files found for years {years}. "
            f"Check data directory: {DATA_DIR / 'FanGraphs_Data' / 'pitchers'}"
        )

    logger.info(f"Total WPA/LI features loaded: {len(wpa_li_data)} unique pitchers")
    return wpa_li_data


def get_wpa_li_for_pitcher(mlbamid: int,
                            wpa_li_data: Dict[int, float],
                            default: float = DEFAULT_WPA_LI) -> float:
    """
    Get WPA/LI value for a specific pitcher.

    Args:
        mlbamid: Player's MLB ID
        wpa_li_data: Dictionary of WPA/LI values
        default: Value to return if not found

    Returns:
        WPA/LI value or default
    """
    return wpa_li_data.get(mlbamid, default)


def normalize_wpa_li(wpa_li_data: Dict[int, float]) -> Dict[int, float]:
    """
    Normalize WPA/LI values to z-scores for model stability.

    Args:
        wpa_li_data: Raw WPA/LI values

    Returns:
        Normalized WPA/LI values (mean=0, std=1)

    Note:
        Handles outliers by clipping to [-3, 3] sigma
    """
    values = np.array(list(wpa_li_data.values()))

    mean = np.mean(values)
    std = np.std(values)

    normalized = {}
    for mlbamid, value in wpa_li_data.items():
        z_score = (value - mean) / std if std > 0 else 0
        # Clip outliers
        z_score = np.clip(z_score, -3, 3)
        normalized[mlbamid] = z_score

    logger.info(f"Normalized WPA/LI: mean={mean:.3f}, std={std:.3f}")
    return normalized


def validate_wpa_li_data(df: pd.DataFrame) -> bool:
    """
    Validate WPA/LI data quality.

    Checks:
    - Required columns present
    - No excessive missing values
    - Values in reasonable range
    - No duplicate MLBAMIDs

    Args:
        df: DataFrame with WPA/LI data

    Returns:
        True if validation passes, False otherwise
    """
    # Required columns
    if 'MLBAMID' not in df.columns or 'WPA/LI' not in df.columns:
        logger.error("Missing required columns")
        return False

    # Check for duplicates
    duplicates = df['MLBAMID'].duplicated().sum()
    if duplicates > 0:
        logger.warning(f"{duplicates} duplicate MLBAMIDs found")

    # Check missing values
    missing_pct = df['WPA/LI'].isna().sum() / len(df) * 100
    if missing_pct > 20:
        logger.error(f"{missing_pct:.1f}% missing WPA/LI values")
        return False

    # Check value range (typical: -2 to 3)
    wpa_li_values = df['WPA/LI'].dropna()
    if wpa_li_values.min() < -5 or wpa_li_values.max() > 5:
        logger.warning(
            f"Unusual WPA/LI range: [{wpa_li_values.min():.2f}, {wpa_li_values.max():.2f}]"
        )

    return True
