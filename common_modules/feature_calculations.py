"""
Feature calculation module for oWAR project.

This module contains all feature calculation and transformation functions
for pitcher statistics, including derived metrics and indices.
"""

# Standard library imports
from typing import Dict, Optional

# Third-party imports
import numpy as np
import pandas as pd

# Local imports
from .config import (
    DAMAGE_CONTROL_RATIO_MAX,
    DAMAGE_CONTROL_RATIO_MIN,
    OPTIMAL_LAUNCH_ANGLE,
    SV_EFFICIENCY_CAP,
    Z_SCORE_CENTER,
    Z_SCORE_SCALE,
    DEFAULT_BB_PCT,
    DEFAULT_K_PCT,
    DEFAULT_ERA,
    DEFAULT_LOB_PCT,
    DEFAULT_HR_FB_PCT,
    DEFAULT_DAMAGE_CONTROL_RATIO,
    DEFAULT_OPPORTUNITY_SUCCESS,
    DEFAULT_CONTACT_QUALITY_INDEX,
    DEFAULT_STATCAST_LAUNCH_QUALITY_INDEX,
    DEFAULT_HARD_PCT,
    DEFAULT_MED_PCT,
    DEFAULT_SOFT_PCT
)
from .logging import get_logger

# Initialize logger
logger = get_logger(__name__)


def calculate_damage_control_ratio(lob_pct_data: Dict[int, float],
                                   hr9_data: Dict[int, float]) -> Dict[int, float]:
    """
    Calculate damage control ratio = LOB% / (HR/9 + 0.5).

    This interaction feature captures pitchers who effectively strand baserunners
    even when allowing home runs, representing clutch/damage limitation ability.

    Args:
            lob_pct_data: Dictionary mapping player_id to LOB%
            hr9_data: Dictionary mapping player_id to HR/9

    Returns:
            Dictionary mapping player_id to damage_control_ratio

    Example:
            >>> calculate_damage_control_ratio({123: 75.0}, {123: 1.2})
            {123: 53.57}
    """
    damage_control = {}

    try:
        # Calculate ratio for players with both LOB% and HR/9 data
        common_players = set(lob_pct_data.keys()) & set(hr9_data.keys())

        for player_id in common_players:
            lob_pct = lob_pct_data[player_id]
            hr9 = hr9_data[player_id]

            # damage_control_ratio = LOB% / (HR/9 + 0.5)
            ratio = lob_pct / (hr9 + 0.5)

            # Apply bounds checking
            if ratio > DAMAGE_CONTROL_RATIO_MAX:
                ratio = DAMAGE_CONTROL_RATIO_MAX
                logger.debug(
                    f"Capped damage_control_ratio at {DAMAGE_CONTROL_RATIO_MAX} for player {player_id}")
            elif ratio < DAMAGE_CONTROL_RATIO_MIN:
                ratio = DAMAGE_CONTROL_RATIO_MIN
                logger.debug(
                    f"Floored damage_control_ratio at {DAMAGE_CONTROL_RATIO_MIN} for player {player_id}")

            damage_control[player_id] = ratio

        logger.info(f"Calculated damage_control_ratio for {len(damage_control)} players")

    except Exception as e:
        logger.error(f"Error calculating damage control ratio: {e}")
        raise

    return damage_control


def calculate_percentage_damage_control_ratio(lob_pct_data: Dict[int, float],
                                              hr_fb_pct_data: Dict[int, float]) -> Dict[int, float]:
    """
    Calculate percentage-based damage control ratio = LOB% / (HR% + 0.5).

    Updated version using HR% (HR/FB) instead of HR/9 for consistent percentage scaling.
    All components now in percentage format for mathematical coherence.

    Args:
            lob_pct_data: Dictionary mapping player_id to LOB% (percentage format)
            hr_fb_pct_data: Dictionary mapping player_id to HR/FB% (percentage format)

    Returns:
            Dictionary mapping player_id to damage_control_ratio

    Raises:
            ValueError: If input data is empty
    """
    if not lob_pct_data or not hr_fb_pct_data:
        raise ValueError("Input data cannot be empty")

    damage_control = {}

    try:
        # Calculate ratio for players with both LOB% and HR% data
        common_players = set(lob_pct_data.keys()) & set(hr_fb_pct_data.keys())

        for player_id in common_players:
            lob_pct = lob_pct_data[player_id]
            hr_pct = hr_fb_pct_data[player_id]

            # damage_control_ratio = LOB% / (HR% + 0.5)
            damage_control[player_id] = lob_pct / (hr_pct + 0.5)

        logger.info(f"Calculated percentage damage_control_ratio for {len(damage_control)} players")

    except Exception as e:
        logger.error(f"Error calculating percentage damage control ratio: {e}")
        raise

    return damage_control


def calculate_normalized_damage_control_ratio(lob_pct_data: Dict[int, float],
                                             hr_fb_pct_data: Dict[int, float]) -> Dict[int, float]:
    """
    Calculate RAW damage control ratio (NO normalization).

    PHASE 1 UPDATE: Removed normalization to fix double-normalization bug.
    Tree models (RandomForest, XGBoost) don't require normalization.
    StandardScaler will handle normalization for Keras if needed.

    Formula: LOB% / (HR/FB% + 0.5)

    Args:
        lob_pct_data: Dictionary mapping player_id to LOB% (percentage format)
        hr_fb_pct_data: Dictionary mapping player_id to HR/FB% (percentage format)

    Returns:
        Dictionary mapping player_id to RAW damage_control_ratio
    """
    # Calculate raw damage control ratio (no normalization)
    raw_damage_control = calculate_percentage_damage_control_ratio(lob_pct_data, hr_fb_pct_data)

    if not raw_damage_control:
        logger.warning("No raw damage control data")
        return {}

    try:
        # Log statistics for monitoring
        raw_values = list(raw_damage_control.values())
        mean_raw = np.mean(raw_values)
        std_raw = np.std(raw_values, ddof=1)
        logger.info(
            f"Damage Control Ratio (RAW): mean={mean_raw:.1f}, std={std_raw:.1f}"
        )

        return raw_damage_control

    except Exception as e:
        logger.error(f"Error calculating damage control ratio: {e}")
        raise


def calculate_opportunity_success(qs_data: Dict[int, int],
                                  sv_data: Dict[int, int],
                                  hld_data: Dict[int, int],
                                  bs_data: Dict[int, int],
                                  games_data: Dict[int, int]) -> Dict[int, float]:
    """
    Calculate Opportunity_Success = (QS + SV + HLD - BS) / G for all players.

    This comprehensive opportunity metric provides role-neutral pitcher evaluation:
    - Starters: QS (Quality Starts) represent successful outings
    - Closers: SV (Saves) minus BS (Blown Saves) = net save success
    - Setup Men: HLD (Holds) provide credit for high-leverage success
    - All Roles: Denominator G (Games) provides opportunity context

    Args:
            qs_data: Dictionary mapping player_id to quality starts
            sv_data: Dictionary mapping player_id to saves
            hld_data: Dictionary mapping player_id to holds
            bs_data: Dictionary mapping player_id to blown saves
            games_data: Dictionary mapping player_id to games

    Returns:
            Dictionary mapping player_id to opportunity_success rate

    Example:
            >>> calculate_opportunity_success({1: 20}, {1: 0}, {1: 0}, {1: 0}, {1: 32})
            {1: 0.625}
    """
    opportunity_success = {}

    try:
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
            success_rate = (qs + sv + hld - bs) / games
            opportunity_success[player_id] = success_rate

        logger.info(f"Calculated opportunity_success for {len(opportunity_success)} players")

    except Exception as e:
        logger.error(f"Error calculating opportunity success: {e}")
        raise

    return opportunity_success


def calculate_contact_quality_index(hard_pct_data: Dict[int, float],
                                    med_pct_data: Dict[int, float],
                                    soft_pct_data: Dict[int, float]) -> Dict[int, float]:
    """
    Calculate EMPIRICALLY-DERIVED Contact Quality Index (CQI) using baseball logic constraints.

    EMPIRICAL FORMULA (derived from 2,351 players, 2016-2025):
    CQI = -0.2926*Hard% + -2.3938*Med% + +2.1864*Soft%

    Key findings from empirical analysis:
    - Medium contact is MORE harmful than hard contact (-2.3938 vs -0.2926)
    - This makes baseball sense: med contact often becomes line drives/gaps
    - Soft contact strongly helps pitchers (+2.1864)

    Args:
            hard_pct_data: Dictionary mapping player_id to hard contact % (0-100 scale)
            med_pct_data: Dictionary mapping player_id to medium contact % (0-100 scale)
            soft_pct_data: Dictionary mapping player_id to soft contact % (0-100 scale)

    Returns:
            Dictionary mapping player_id to raw contact quality index
    """
    contact_quality_index = {}

    try:
        # Get all players who have contact data
        all_players = set(
            hard_pct_data.keys()) | set(
            med_pct_data.keys()) | set(
            soft_pct_data.keys())

        for player_id in all_players:
            hard_pct = hard_pct_data.get(player_id, 0.0)
            med_pct = med_pct_data.get(player_id, 0.0)
            soft_pct = soft_pct_data.get(player_id, 0.0)

            # EMPIRICALLY-DERIVED Contact Quality Index
            cqi = (-0.2926 * hard_pct) + (-2.3938 * med_pct) + (2.1864 * soft_pct)
            contact_quality_index[player_id] = cqi

        logger.debug(f"Calculated raw CQI for {len(contact_quality_index)} players")

    except Exception as e:
        logger.error(f"Error calculating contact quality index: {e}")
        raise

    return contact_quality_index


def calculate_normalized_contact_quality_index(hard_pct_data: Dict[int, float],
                                               med_pct_data: Dict[int, float],
                                               soft_pct_data: Dict[int, float]) -> Dict[int, float]:
    """
    Calculate RAW Contact Quality Index (NO normalization).

    PHASE 1 UPDATE: Removed normalization to fix double-normalization bug.
    Tree models (RandomForest, XGBoost) don't require normalization.
    StandardScaler will handle normalization for Keras if needed.

    Formula: -0.2926*Hard% + -2.3938*Med% + +2.1864*Soft%

    Args:
            hard_pct_data: Dictionary mapping player_id to hard contact % (0-100 scale)
            med_pct_data: Dictionary mapping player_id to medium contact % (0-100 scale)
            soft_pct_data: Dictionary mapping player_id to soft contact % (0-100 scale)

    Returns:
            Dictionary mapping player_id to RAW contact quality index
    """
    # Calculate raw Contact Quality Index (no normalization)
    raw_cqi = calculate_contact_quality_index(hard_pct_data, med_pct_data, soft_pct_data)

    if not raw_cqi:
        logger.warning("No raw CQI data")
        return {}

    try:
        # Log statistics for monitoring
        raw_values = list(raw_cqi.values())
        mean_raw = np.mean(raw_values)
        std_raw = np.std(raw_values, ddof=1)
        logger.info(
            f"Contact Quality Index (RAW): mean={mean_raw:.1f}, std={std_raw:.1f}")

    except Exception as e:
        logger.error(f"Error calculating contact quality index: {e}")
        raise

    return raw_cqi


def calculate_statcast_launch_quality_index(
        statcast_data_dict: Dict[int, Dict[str, float]]) -> Dict[int, float]:
    """
    Calculate EMPIRICALLY-DERIVED Statcast Launch Quality Index (SLQI) from exit velocity data.

    Formula: SLQI = -0.056 × (avg_hit_angle - 14.2)² + 0.659 × anglesweetspotpercent

    Key insights from empirical analysis (n=2,235 pitchers, 2016-2024):
    - avg_hit_angle: U-shaped relationship with performance (extremes good, middle bad)
    - anglesweetspotpercent: Linear negative relationship (more sweet spot = worse for pitcher)

    Args:
            statcast_data_dict: Dictionary mapping player_id to {'avg_hit_angle': value, 'anglesweetspotpercent': value}

    Returns:
            Dictionary mapping player_id to Statcast Launch Quality Index
    """
    statcast_launch_quality = {}

    # Empirically-derived weights
    ANGLE_WEIGHT = -0.056
    SWEET_SPOT_WEIGHT = 0.659

    try:
        for player_id, features in statcast_data_dict.items():
            if 'avg_hit_angle' in features and 'anglesweetspotpercent' in features:
                avg_hit_angle = features['avg_hit_angle']
                sweet_spot_pct = features['anglesweetspotpercent']

                # Calculate SLQI
                angle_deviation_sq = (avg_hit_angle - OPTIMAL_LAUNCH_ANGLE) ** 2
                slqi = ANGLE_WEIGHT * angle_deviation_sq + SWEET_SPOT_WEIGHT * sweet_spot_pct

                statcast_launch_quality[player_id] = slqi

        logger.debug(f"Calculated raw SLQI for {len(statcast_launch_quality)} players")

    except Exception as e:
        logger.error(f"Error calculating Statcast launch quality index: {e}")
        raise

    return statcast_launch_quality


def calculate_normalized_statcast_launch_quality_index(
        statcast_data_dict: Dict[int, Dict[str, float]]) -> Dict[int, float]:
    """
    Calculate RAW Statcast Launch Quality Index (NO normalization).

    PHASE 1 UPDATE: Removed normalization to fix double-normalization bug.
    Tree models (RandomForest, XGBoost) don't require normalization.
    StandardScaler will handle normalization for Keras if needed.

    Formula: (avg_hit_angle weight) + (anglesweetspotpercent weight)

    Args:
            statcast_data_dict: Dictionary mapping player_id to exit velocity features

    Returns:
            Dictionary mapping player_id to RAW SLQI
    """
    # Calculate raw Statcast Launch Quality Index (no normalization)
    raw_slqi = calculate_statcast_launch_quality_index(statcast_data_dict)

    if not raw_slqi:
        logger.warning("No raw SLQI data")
        return {}

    try:
        # Log statistics for monitoring (no normalization)
        raw_values = list(raw_slqi.values())
        mean_raw = np.mean(raw_values)
        std_raw = np.std(raw_values, ddof=1)
        logger.info(
            f"Statcast Launch Quality Index (RAW): mean={mean_raw:.1f}, std={std_raw:.1f}")

    except Exception as e:
        logger.error(f"Error calculating Statcast launch quality index: {e}")
        raise

    return raw_slqi


def calculate_dominance_index(csw_pct_data: Dict[int, float],
                              contact_pct_data: Dict[int, float]) -> Dict[int, float]:
    """
    Calculate Dominance Index = CSW% / Contact%.

    This composite metric captures pitcher's ability to dominate plate appearances:
    - High CSW% (Called Strike + Whiff %) = pitcher controls the strike zone
    - Low Contact% = pitcher prevents balls in play
    - Higher ratio = more dominant pitcher

    Args:
            csw_pct_data: Dictionary mapping player_id to CSW% (percentage format)
            contact_pct_data: Dictionary mapping player_id to Contact% (percentage format)

    Returns:
            Dictionary mapping player_id to dominance_index

    Example:
            >>> calculate_dominance_index({123: 32.0}, {123: 75.0})
            {123: 0.427}
    """
    dominance_index = {}

    try:
        # Calculate ratio for players with both CSW% and Contact% data
        common_players = set(csw_pct_data.keys()) & set(contact_pct_data.keys())

        for player_id in common_players:
            csw_pct = csw_pct_data[player_id]
            contact_pct = contact_pct_data[player_id]

            # Avoid division by zero
            if contact_pct == 0:
                logger.warning(f"Contact% is 0 for player {player_id}, skipping")
                continue

            # dominance_index = CSW% / Contact%
            dominance_index[player_id] = csw_pct / contact_pct

        logger.info(f"Calculated dominance_index for {len(dominance_index)} players")

    except Exception as e:
        logger.error(f"Error calculating dominance index: {e}")
        raise

    return dominance_index


def calculate_hbp_percentage(hbp_data: Dict[int, int],
                             pitches_data: Dict[int, float]) -> Dict[str, float]:
    """
    Calculate HBP percentage (HBP/Pitches * 100) for consistent feature scaling.

    Args:
            hbp_data: Dictionary mapping player_id to HBP count
            pitches_data: Dictionary mapping player_id to total pitches

    Returns:
            Dictionary mapping player_id (as string) to HBP percentage

    Raises:
            ValueError: If input data is invalid
    """
    hbp_percentage = {}

    try:
        # Convert both datasets to string keys for consistent matching
        hbp_data_str = {str(k): v for k, v in hbp_data.items()}
        pitches_data_str = {str(k): v for k, v in pitches_data.items()}

        # Find common players with both HBP and Pitches data
        common_players = set(hbp_data_str.keys()) & set(pitches_data_str.keys())

        for player_id in common_players:
            hbp = hbp_data_str[player_id]
            pitches = pitches_data_str[player_id]

            # Calculate HBP% = (HBP / Pitches) * 100
            if pitches > 0:
                hbp_pct = (hbp / pitches) * 100
                hbp_percentage[player_id] = hbp_pct

        logger.info(f"Calculated HBP% for {len(hbp_percentage)} players")

    except Exception as e:
        logger.error(f"Error calculating HBP percentage: {e}")
        raise

    return hbp_percentage


def get_player_percentage_features(
        player_id: int, percentage_features_dict: Dict[str, Dict[int, float]]) -> Dict[str, float]:
    """
    Get percentage-based features for a specific player.

    Args:
            player_id: MLB player ID (MLBAMID or mlbid)
            percentage_features_dict: Dictionary of percentage features

    Returns:
            Dictionary with player's percentage-based feature values
    """
    try:
        player_id = int(player_id)
    except (ValueError, TypeError):
        logger.warning(f"Invalid player_id: {player_id}, returning defaults")
        return {
            'BB%': DEFAULT_BB_PCT,
            'K%': DEFAULT_K_PCT,
            'K-BB%': DEFAULT_K_PCT - DEFAULT_BB_PCT,
            'HR/FB%': DEFAULT_HR_FB_PCT,
            'LOB%': DEFAULT_LOB_PCT,
            'damage_control_ratio': DEFAULT_DAMAGE_CONTROL_RATIO,
            'Opportunity_Success': DEFAULT_OPPORTUNITY_SUCCESS,
            'Contact_Quality_Index': DEFAULT_CONTACT_QUALITY_INDEX,
            'HBP%': 0.0,
            'Statcast_Launch_Quality_Index': DEFAULT_STATCAST_LAUNCH_QUALITY_INDEX
        }

    return {
        'BB%': percentage_features_dict.get('BB%', {}).get(player_id, DEFAULT_BB_PCT),
        'K%': percentage_features_dict.get('K%', {}).get(player_id, DEFAULT_K_PCT),
        'K-BB%': percentage_features_dict.get('K-BB%', {}).get(
            player_id, DEFAULT_K_PCT - DEFAULT_BB_PCT
        ),
        'HR/FB%': percentage_features_dict.get('HR/FB%', {}).get(player_id, DEFAULT_HR_FB_PCT),
        'LOB%': percentage_features_dict.get('LOB%', {}).get(player_id, DEFAULT_LOB_PCT),
        'damage_control_ratio': percentage_features_dict.get('damage_control_ratio', {}).get(
            player_id, DEFAULT_DAMAGE_CONTROL_RATIO
        ),
        'Opportunity_Success': percentage_features_dict.get('Opportunity_Success', {}).get(
            player_id, DEFAULT_OPPORTUNITY_SUCCESS
        ),
        'Contact_Quality_Index': percentage_features_dict.get('Contact_Quality_Index', {}).get(
            player_id, DEFAULT_CONTACT_QUALITY_INDEX
        ),
        'HBP%': percentage_features_dict.get('HBP%', {}).get(str(player_id), 0.0),
        'Statcast_Launch_Quality_Index': percentage_features_dict.get('Statcast_Launch_Quality_Index', {}).get(
            player_id, DEFAULT_STATCAST_LAUNCH_QUALITY_INDEX
        )
    }


def get_player_complete_pitcher_features(player_id: int,
                                         percentage_features_dict: Dict[str, Dict[int, float]],
                                         current_season_data: Optional[pd.DataFrame] = None) -> np.ndarray:
    """
    Get complete feature set for a pitcher including all derived features.

    Args:
            player_id: MLB player ID
            percentage_features_dict: Dictionary from load_percentage_pitcher_features()
            current_season_data: DataFrame with current season stats (IP, ERA, etc.)

    Returns:
            numpy array with complete feature set

    Example:
            >>> features = get_player_complete_pitcher_features(123456, features_dict)
            array([100.0, 9.0, 20.0, 4.50, ...])
    """
    try:
        player_id = int(player_id)
    except (ValueError, TypeError):
        logger.warning(f"Invalid player_id: {player_id}, returning defaults")
        return np.array([100.0, DEFAULT_BB_PCT, DEFAULT_K_PCT, DEFAULT_ERA,
                         DEFAULT_DAMAGE_CONTROL_RATIO, DEFAULT_OPPORTUNITY_SUCCESS,
                         DEFAULT_HARD_PCT, DEFAULT_MED_PCT, DEFAULT_SOFT_PCT, 0.0, 0.0])

    # Start with default values
    features = {
        'IP': 100.0,
        'BB%': DEFAULT_BB_PCT,
        'K%': DEFAULT_K_PCT,
        'ERA': DEFAULT_ERA,
        'damage_control_ratio': DEFAULT_DAMAGE_CONTROL_RATIO,
        'Opportunity_Success': DEFAULT_OPPORTUNITY_SUCCESS,
        'Hard%': DEFAULT_HARD_PCT,
        'Med%': DEFAULT_MED_PCT,
        'Soft%': DEFAULT_SOFT_PCT,
        'HBP': 0.0,
        'WP': 0.0
    }

    # Get percentage features
    if percentage_features_dict:
        player_features = get_player_percentage_features(player_id, percentage_features_dict)
        features.update(player_features)

    # Get current season data if available
    if current_season_data is not None and not current_season_data.empty:
        try:
            # Try to find player by various ID columns
            player_row = None
            for id_col in ['MLBAMID', 'mlbid', 'playerid']:
                if id_col in current_season_data.columns:
                    matches = current_season_data[current_season_data[id_col] == player_id]
                    if not matches.empty:
                        player_row = matches.iloc[0]
                        break

            if player_row is not None:
                # Extract IP and ERA
                features['IP'] = player_row.get('IP', features['IP'])
                features['ERA'] = (player_row.get('ERA') or
                                   player_row.get('ERA_adv') or
                                   player_row.get('ERA_std') or
                                   features['ERA'])

                # Extract BB% and K% if available
                for stat in ['BB%', 'K%']:
                    if stat in player_row:
                        value = player_row[stat]
                        if value is not None:
                            # Convert from decimal to percentage if needed
                            features[stat] = value * 100.0 if value < 1.0 else value

        except Exception as e:
            logger.error(f"Error extracting current season data for player {player_id}: {e}")

    # Return as numpy array in expected order
    feature_order = ['IP', 'BB%', 'K%', 'ERA', 'damage_control_ratio',
                     'Opportunity_Success', 'Hard%', 'Med%', 'Soft%', 'HBP', 'WP']
    return np.array([features[feat] for feat in feature_order])
