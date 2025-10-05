"""
Plate Discipline Feature Loading for Pitcher Predictions.

Loads CSW%, Contact%, and SwStr% from FanGraphs plate discipline files.
These metrics capture command (called strikes) and stuff (whiffs) beyond
simple K%.
"""

import glob
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from .logging import get_logger
from .config import DATA_DIR
from .feature_calculations import calculate_dominance_index

logger = get_logger(__name__)

# Default values (league average approximations)
DEFAULT_CSW = 28.0           # League average ~28%
DEFAULT_CONTACT = 78.0       # League average ~78%
DEFAULT_SWSTR = 10.5         # League average ~10.5%
DEFAULT_DOMINANCE = 0.359    # League average CSW/Contact (28/78)


def load_plate_discipline_features(
    years: Optional[List[int]] = None
) -> Dict[str, Dict[int, float]]:
    """
    Load plate discipline features from FanGraphs files.

    Args:
        years: Years to load (default: 2016-2024)

    Returns:
        Dictionary: {
            'CSW%': {mlbamid: value},
            'Contact%': {mlbamid: value},
            'SwStr%': {mlbamid: value}
        }

    Raises:
        FileNotFoundError: If no data files found for any year

    Example:
        >>> plate_disc = load_plate_discipline_features([2024])
        >>> plate_disc['CSW%'][660271]  # Skubal's MLBAMID
        31.2
    """
    if years is None:
        years = list(range(2016, 2025))

    csw_data = {}
    contact_data = {}
    swstr_data = {}
    files_found = 0

    for year in years:
        # Plate discipline data is in the _stuff.csv files
        if year >= 2025:
            # For 2025, use firsthalf_stuff.csv
            pattern = str(DATA_DIR / "FanGraphs_Data" / "pitchers" /
                         f"fangraphs_pitchers_{year}_firsthalf_stuff.csv")
            files = glob.glob(pattern)
            if not files:
                # Try regular stuff file if firsthalf doesn't exist
                pattern = str(DATA_DIR / "FanGraphs_Data" / "pitchers" /
                             f"fangraphs_pitchers_{year}_stuff.csv")
                files = glob.glob(pattern)
        else:
            pattern = str(DATA_DIR / "FanGraphs_Data" / "pitchers" /
                         f"fangraphs_pitchers_{year}_stuff.csv")
            files = glob.glob(pattern)

        if not files:
            logger.warning(f"No plate discipline data found for {year}")
            continue

        try:
            df = pd.read_csv(files[0])

            # Handle player ID column name variations
            if 'MLBAMID' in df.columns:
                id_col = 'MLBAMID'
            elif 'PlayerId' in df.columns:
                id_col = 'PlayerId'
            elif 'playerid' in df.columns:
                id_col = 'playerid'
            else:
                logger.error(f"No player ID column found in {files[0]}")
                continue

            # Map column names (handle percentage signs, underscores, etc.)
            # CSW% column variations
            csw_col = None
            for col in df.columns:
                if 'csw' in col.lower():
                    csw_col = col
                    break

            # Contact% column variations (exclude Z-Contact%)
            contact_col = None
            for col in df.columns:
                if 'contact' in col.lower() and 'z-' not in col.lower() and 'zone' not in col.lower():
                    contact_col = col
                    break

            # SwStr% column variations
            swstr_col = None
            for col in df.columns:
                if 'swstr' in col.lower():
                    swstr_col = col
                    break

            if not all([csw_col, contact_col, swstr_col]):
                missing = []
                if not csw_col:
                    missing.append('CSW%')
                if not contact_col:
                    missing.append('Contact%')
                if not swstr_col:
                    missing.append('SwStr%')
                logger.error(
                    f"Missing required plate discipline columns in {files[0]}: "
                    f"{missing}"
                )
                continue

            # Extract and store (convert from decimal to percentage)
            valid_count = 0
            for _, row in df.iterrows():
                mlbamid = row.get(id_col)
                csw = row.get(csw_col)
                contact = row.get(contact_col)
                swstr = row.get(swstr_col)

                if pd.notna(mlbamid):
                    try:
                        mlbamid_int = int(mlbamid)

                        # Values in _stuff.csv are decimals (0.35 = 35%)
                        # Convert to percentages for consistency
                        if pd.notna(csw):
                            csw_data[mlbamid_int] = float(csw) * 100
                        if pd.notna(contact):
                            contact_data[mlbamid_int] = float(contact) * 100
                        if pd.notna(swstr):
                            swstr_data[mlbamid_int] = float(swstr) * 100

                        valid_count += 1
                    except (ValueError, TypeError) as e:
                        logger.debug(
                            f"Error converting values for MLBAMID {mlbamid}: {e}"
                        )
                        continue

            logger.info(
                f"Loaded plate discipline for {valid_count} pitchers from {year}"
            )
            files_found += 1

        except Exception as e:
            logger.error(f"Error loading {files[0]}: {e}")
            continue

    if files_found == 0:
        raise FileNotFoundError(
            f"No plate discipline data files found for years {years}. "
            f"Check data directory: {DATA_DIR / 'FanGraphs_Data' / 'pitchers'}"
        )

    # Calculate Dominance Index (CSW% / Contact%)
    dominance_data = calculate_dominance_index(csw_data, contact_data)

    logger.info(
        f"Plate discipline features loaded: "
        f"CSW% ({len(csw_data)}), Contact% ({len(contact_data)}), "
        f"SwStr% ({len(swstr_data)}), Dominance Index ({len(dominance_data)})"
    )

    return {
        'CSW%': csw_data,
        'Contact%': contact_data,
        'SwStr%': swstr_data,
        'Dominance_Index': dominance_data
    }


def get_plate_discipline_for_pitcher(
    mlbamid: int,
    plate_disc_data: Dict[str, Dict[int, float]]
) -> Dict[str, float]:
    """
    Get plate discipline features for a specific pitcher.

    Args:
        mlbamid: Player's MLB ID
        plate_disc_data: Dictionary from load_plate_discipline_features()

    Returns:
        Dictionary: {
            'CSW%': float,
            'Contact%': float,
            'SwStr%': float,
            'Dominance_Index': float
        }

    Default values (league average approximations):
        - CSW%: 28.0
        - Contact%: 78.0
        - SwStr%: 10.5
        - Dominance_Index: 0.359

    Example:
        >>> plate_disc_data = load_plate_discipline_features([2024])
        >>> get_plate_discipline_for_pitcher(660271, plate_disc_data)
        {'CSW%': 31.2, 'Contact%': 73.1, 'SwStr%': 12.9, 'Dominance_Index': 0.427}
    """
    return {
        'CSW%': plate_disc_data['CSW%'].get(mlbamid, DEFAULT_CSW),
        'Contact%': plate_disc_data['Contact%'].get(mlbamid, DEFAULT_CONTACT),
        'SwStr%': plate_disc_data['SwStr%'].get(mlbamid, DEFAULT_SWSTR),
        'Dominance_Index': plate_disc_data.get('Dominance_Index', {}).get(mlbamid, DEFAULT_DOMINANCE)
    }


def normalize_plate_discipline_features(
    plate_disc_data: Dict[str, Dict[int, float]]
) -> Dict[str, Dict[int, float]]:
    """
    Normalize plate discipline features to z-scores.

    Use if raw features show high variance or scale issues during testing.

    Args:
        plate_disc_data: Dictionary from load_plate_discipline_features()

    Returns:
        Normalized features: {
            'CSW%': {mlbamid: z_score},
            'Contact%': {mlbamid: z_score},
            'SwStr%': {mlbamid: z_score}
        }

    Note:
        This is optional - only use if raw features perform poorly.
        Per user request, test raw features first.
    """
    normalized = {}

    for feature_name in ['CSW%', 'Contact%', 'SwStr%']:
        values = np.array(list(plate_disc_data[feature_name].values()))

        if len(values) == 0:
            logger.warning(f"No values to normalize for {feature_name}")
            normalized[feature_name] = {}
            continue

        mean = np.mean(values)
        std = np.std(values)

        normalized[feature_name] = {}
        for mlbamid, value in plate_disc_data[feature_name].items():
            if std > 0:
                z_score = (value - mean) / std
                # Clip outliers to [-3, 3] sigma
                z_score = np.clip(z_score, -3, 3)
            else:
                z_score = 0.0

            normalized[feature_name][mlbamid] = float(z_score)

        logger.info(
            f"Normalized {feature_name}: mean={mean:.2f}, std={std:.2f}"
        )

    return normalized
