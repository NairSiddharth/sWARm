#!/usr/bin/env python3
"""
Derived Features Module for sWARm Analysis.

This module provides the main interface for calculating enhanced baserunning
and defensive features using comprehensive data sources. It coordinates
between data loading and feature calculation modules.

Baserunning:
- BP Data: SB, CS, SB%, PO, XBT%
- Statcast: seconds_since_hit_090 for speed calculations

Defense:
- FanGraphs Standard: Pos, Inn, PO, A, E, DPS, DPT, DPF, Scp
- FanGraphs Statcast: Throwing, Blocking, Framing, Arm (catchers)
- Statcast Catch Probability: 5-star rating system
"""

__version__ = '2.0.0'
__author__ = 'oWAR Development Team'

# Standard library imports
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Third-party imports
import pandas as pd

# Local imports
from .config import DEFAULT_DATA_YEARS
from .feature_calculations import (
    calculate_contact_quality_index,
    calculate_damage_control_ratio,
    calculate_hbp_percentage,
    calculate_normalized_contact_quality_index,
    calculate_normalized_statcast_launch_quality_index,
    calculate_opportunity_success,
    calculate_percentage_damage_control_ratio,
    calculate_normalized_damage_control_ratio,
    get_player_complete_pitcher_features,
    get_player_percentage_features
)
from .logging import get_logger
from .pitcher_feature_calculations import (
    load_contact_quality_data,
    load_opportunity_components,
    load_percentage_features,
    load_statcast_exit_velocity_data
)
# Note: load_fixed_bp_data is now only used internally by load_bp_warp_data()

# Initialize logger
logger = get_logger(__name__)

# Public API exports
# Note: Park factor functions removed - use park_factors module instead
__all__ = [
    'load_enhanced_pitcher_features',
    'load_percentage_pitcher_features',
    'get_player_enhanced_features',
    'get_player_percentage_features',
    'get_player_complete_pitcher_features',
    'load_bp_warp_data'  # Public API for Baseball Prospectus WARP data
]


def load_bp_warp_data(data_dir: Optional[Path] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load Baseball Prospectus WARP data for hitters and pitchers.

    This is the public API for accessing BP data with properly calculated
    derived statistics (K%, BB%, etc.).

    Args:
            data_dir: Optional path to BP data directory. If None, uses default.

    Returns:
            Tuple containing:
            - DataFrame with hitter WARP data including Name, Season, PA, AVG, OBP, SLG, WARP
            - DataFrame with pitcher WARP data including Name, Season, IP, ERA, WHIP, K%, BB%, WARP

    Raises:
            FileNotFoundError: If BP data directory doesn't exist
            ValueError: If no valid data files are found

    Example:
            >>> hitters, pitchers = load_bp_warp_data()
            >>> print(f"Loaded {len(hitters)} hitter-seasons and {len(pitchers)} pitcher-seasons")
            >>> print(f"Hitters WARP range: {hitters['WARP'].min():.1f} to {hitters['WARP'].max():.1f}")
    """
    logger.info("Loading Baseball Prospectus WARP data via public API")

    # Delegate to internal implementation
    # This provides a stable public interface while allowing the internal
    # implementation to change without breaking downstream code
    from .pitcher_feature_calculations import load_fixed_bp_data

    try:
        hitters, pitchers = load_fixed_bp_data(data_dir)
        logger.info(
            f"Successfully loaded {
                len(hitters)} hitter and {
                len(pitchers)} pitcher records")
        return hitters, pitchers
    except Exception as e:
        logger.error(f"Failed to load BP WARP data: {str(e)}", exc_info=True)
        raise


# Note: Park factor functionality has been moved to park_factors.py
# Import from there if you need park factor adjustments:
# from .park_factors import load_park_factors, apply_park_factor_adjustments


def load_enhanced_pitcher_features(
        years: Optional[List[int]] = None) -> Dict[str, Dict[int, float]]:
    """
    Load enhanced pitcher features including LOB%, GB%, HR/9, and damage_control_ratio.

    Args:
            years: Years to load (defaults to 2016-2025)

    Returns:
            Dictionary with enhanced features by player ID:
            {
                    'LOB%': {player_id: value, ...},
                    'GB%': {player_id: value, ...},
                    'HR/9': {player_id: value, ...},
                    'damage_control_ratio': {player_id: value, ...}
            }

    Raises:
            ValueError: If no data can be loaded
    """
    if years is None:
        years = DEFAULT_DATA_YEARS

    logger.info(f"Loading enhanced pitcher features for {len(years)} years")

    try:
        # Load percentage features (includes LOB%)
        percentage_features = load_percentage_features(years)

        # For backward compatibility, return subset of features
        # Note: GB% and HR/9 would need additional implementation
        enhanced_features = {
            'LOB%': percentage_features.get('LOB%', {}),
            'GB%': {},  # Not implemented in refactored version
            'HR/9': {},  # Not implemented in refactored version
            'damage_control_ratio': {}
        }

        # Calculate NORMALIZED damage control ratio if we have the data
        if 'LOB%' in percentage_features and 'HR/FB%' in percentage_features:
            damage_control = calculate_normalized_damage_control_ratio(
                percentage_features['LOB%'],
                percentage_features['HR/FB%']
            )
            enhanced_features['damage_control_ratio'] = damage_control

        logger.info(f"Loaded enhanced features for {len(enhanced_features['LOB%'])} players")
        return enhanced_features

    except Exception as e:
        logger.error(f"Error loading enhanced pitcher features: {e}")
        raise ValueError(f"Failed to load enhanced pitcher features: {e}")


def get_player_enhanced_features(
        player_id: int, enhanced_features_dict: Dict[str, Dict[int, float]]) -> Dict[str, float]:
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
        logger.warning(f"Invalid player_id: {player_id}")
        return {'LOB%': 0.0, 'GB%': 0.0, 'damage_control_ratio': 0.0}

    return {
        'LOB%': enhanced_features_dict.get(
            'LOB%',
            {}).get(
            player_id,
            0.0),
        'GB%': enhanced_features_dict.get(
            'GB%',
                {}).get(
                    player_id,
                    0.0),
        'HR/9': enhanced_features_dict.get(
            'HR/9',
            {}).get(
            player_id,
            0.0),
        'damage_control_ratio': enhanced_features_dict.get(
            'damage_control_ratio',
            {}).get(
            player_id,
            0.0)}


def load_percentage_pitcher_features(
        years: Optional[List[int]] = None) -> Dict[str, Dict[int, float]]:
    """
    Load pitcher features with consistent percentage scaling.

    Replaces BB/9, K/9 with BB%, K% and updates damage_control_ratio to use HR%.

    Args:
            years: List of years to process (defaults to 2016-2025)

    Returns:
            Dictionary with all percentage-based pitcher features

    Raises:
            ValueError: If data loading fails
    """
    if years is None:
        years = DEFAULT_DATA_YEARS

    logger.info(f"Loading percentage-based pitcher features for years {years[0]}-{years[-1]}")

    try:
        # Load all base percentage features
        percentage_features = load_percentage_features(years)

        # Load opportunity components and calculate Opportunity_Success
        opportunity_components = load_opportunity_components(years)
        opportunity_success = calculate_opportunity_success(
            opportunity_components['QS'],
            opportunity_components['SV'],
            opportunity_components['HLD'],
            opportunity_components['BS'],
            opportunity_components['G']
        )
        percentage_features['Opportunity_Success'] = opportunity_success

        # Calculate NORMALIZED damage control ratio
        if 'LOB%' in percentage_features and 'HR/FB%' in percentage_features:
            damage_control = calculate_normalized_damage_control_ratio(
                percentage_features['LOB%'],
                percentage_features['HR/FB%']
            )
            percentage_features['damage_control_ratio'] = damage_control

        # Load and calculate Contact Quality Index
        contact_data = load_contact_quality_data(years)
        if all(k in contact_data for k in ['Hard%', 'Med%', 'Soft%']):
            contact_quality_index = calculate_normalized_contact_quality_index(
                contact_data['Hard%'],
                contact_data['Med%'],
                contact_data['Soft%']
            )
            percentage_features['Contact_Quality_Index'] = contact_quality_index

            # Also include raw contact percentages
            percentage_features.update(contact_data)

        # Load and calculate Statcast Launch Quality Index
        statcast_data = load_statcast_exit_velocity_data()
        if statcast_data:
            statcast_index = calculate_normalized_statcast_launch_quality_index(statcast_data)
            percentage_features['Statcast_Launch_Quality_Index'] = statcast_index

        # Calculate HBP% (would need additional implementation for pitches data)
        percentage_features['HBP%'] = {}

        logger.info("Percentage features loaded successfully")
        for feature_name, feature_data in percentage_features.items():
            logger.debug(f"{feature_name} coverage: {len(feature_data)} players")

        return percentage_features

    except Exception as e:
        logger.error(f"Error loading percentage pitcher features: {e}")
        raise ValueError(f"Failed to load percentage pitcher features: {e}")
