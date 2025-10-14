"""
Pitcher Composite Feature Calculations.

Pure functions that calculate composite features from base stats.
All inputs must be in percentage format (23.2, not 0.232).

Active Composites (from feature_sets.py SKILL_FEATURES):
1. damage_control_ratio = LOB% / (HR/FB% + 0.5)
2. Opportunity_Success = (K% - BB%) * LOB%
3. strikeout_efficiency = K% × (100 - BB%)
4. contact_management = GB% × (100 - BB%)
5. strikeout_contact_quality = K% × (100 - Hard%)
6. Statcast_Launch_Quality_Index = empirical formula from Statcast components
7. SD_MD_Net = SD - MD (reliever-specific high-leverage success)

Interaction Features Note:
Interaction features (3-5) produce values in range 1,500-4,000 due to multiplication.
StandardScaler in Phase 2 will normalize all features to equal scale.

Phase 1 Note: All composites return RAW values (no normalization).
StandardScaler will handle normalization later.
"""

from typing import Dict


# Empirical constants for Statcast Launch Quality Index
# From feature_calculations.py - empirically derived from 2016-2024 data
OPTIMAL_LAUNCH_ANGLE = 14.2
ANGLE_WEIGHT = -0.056
SWEET_SPOT_WEIGHT = 0.659


def calculate_damage_control_ratio(
    lob_pct_data: Dict[int, float],
    hr_fb_pct_data: Dict[int, float]
) -> Dict[int, float]:
    """
    Calculate damage_control_ratio = LOB% / (HR/FB% + 0.5).

    Measures pitcher's ability to strand runners relative to home run tendency.

    CRITICAL: Inputs MUST be percentages (72.2, not 0.722).
    The scale mismatch bug in realtime_composite_calculator.py was caused
    by receiving decimals here, resulting in values off by 10,000x.

    Args:
        lob_pct_data: {MLBAMID: LOB% in percentage format}
        hr_fb_pct_data: {MLBAMID: HR/FB% in percentage format}

    Returns:
        dict: {MLBAMID: damage_control_ratio}

    Example:
        LOB% = 75.0, HR/FB% = 10.5
        >>> damage_control_ratio = 75.0 / (10.5 + 0.5) = 6.82
    """
    common_players = set(lob_pct_data.keys()) & set(hr_fb_pct_data.keys())

    return {
        pid: lob_pct_data[pid] / (hr_fb_pct_data[pid] + 0.5)
        for pid in common_players
    }


def calculate_opportunity_success(
    k_pct_data: Dict[int, float],
    bb_pct_data: Dict[int, float],
    lob_pct_data: Dict[int, float]
) -> Dict[int, float]:
    """
    Calculate Opportunity_Success = (K% - BB%) * LOB%.

    Combines strikeout-walk differential with strand rate.
    Keep percentages consistent - StandardScaler will normalize later.

    Args:
        k_pct_data: {MLBAMID: K% in percentage format}
        bb_pct_data: {MLBAMID: BB% in percentage format}
        lob_pct_data: {MLBAMID: LOB% in percentage format}

    Returns:
        dict: {MLBAMID: Opportunity_Success}

    Example:
        K% = 28.0, BB% = 7.0, LOB% = 75.0
        >>> opportunity_success = (28.0 - 7.0) * 75.0 = 1,575
    """
    common_players = set(k_pct_data.keys()) & set(bb_pct_data.keys()) & set(lob_pct_data.keys())

    return {
        pid: (k_pct_data[pid] - bb_pct_data[pid]) * lob_pct_data[pid]
        for pid in common_players
    }


def calculate_strikeout_efficiency(
    k_pct_data: Dict[int, float],
    bb_pct_data: Dict[int, float]
) -> Dict[int, float]:
    """
    Calculate strikeout_efficiency = K% * (100 - BB%).

    Strikeout ability is more valuable with low walk rates.

    Args:
        k_pct_data: {MLBAMID: K% in percentage format}
        bb_pct_data: {MLBAMID: BB% in percentage format}

    Returns:
        dict: {MLBAMID: strikeout_efficiency}

    Example:
        K% = 28.0, BB% = 7.0
        >>> strikeout_efficiency = 28.0 * (100 - 7.0) = 2604.0
    """
    common_players = set(k_pct_data.keys()) & set(bb_pct_data.keys())

    return {
        pid: k_pct_data[pid] * (100.0 - bb_pct_data[pid])
        for pid in common_players
    }


def calculate_contact_management(
    gb_pct_data: Dict[int, float],
    bb_pct_data: Dict[int, float]
) -> Dict[int, float]:
    """
    Calculate contact_management = GB% × (100 - BB%).

    Ground ball ability is more valuable with low walk rates.
    Interaction feature - will be normalized by StandardScaler.

    Args:
        gb_pct_data: {MLBAMID: GB% in percentage format}
        bb_pct_data: {MLBAMID: BB% in percentage format}

    Returns:
        dict: {MLBAMID: contact_management}

    Example:
        GB% = 45.0, BB% = 7.0
        >>> contact_management = 45.0 * (100 - 7.0) = 4,185
    """
    common_players = set(gb_pct_data.keys()) & set(bb_pct_data.keys())

    return {
        pid: gb_pct_data[pid] * (100.0 - bb_pct_data[pid])
        for pid in common_players
    }


def calculate_strikeout_contact_quality(
    k_pct_data: Dict[int, float],
    hard_pct_data: Dict[int, float]
) -> Dict[int, float]:
    """
    Calculate strikeout_contact_quality = K% × (100 - Hard%).

    Strikeout ability is more valuable when batters make weak contact.
    Interaction feature - will be normalized by StandardScaler.

    Args:
        k_pct_data: {MLBAMID: K% in percentage format}
        hard_pct_data: {MLBAMID: Hard% in percentage format}

    Returns:
        dict: {MLBAMID: strikeout_contact_quality}

    Example:
        K% = 28.0, Hard% = 35.0
        >>> strikeout_contact_quality = 28.0 * (100 - 35.0) = 1,820
    """
    common_players = set(k_pct_data.keys()) & set(hard_pct_data.keys())

    return {
        pid: k_pct_data[pid] * (100.0 - hard_pct_data[pid])
        for pid in common_players
    }


def calculate_statcast_launch_quality_index(
    statcast_data: Dict[int, Dict[str, float]]
) -> Dict[int, float]:
    """
    Calculate Statcast Launch Quality Index from exit velocity components.

    EMPIRICALLY-DERIVED formula (from feature_calculations.py):
    SLQI = -0.056 × (avg_hit_angle - 14.2)² + 0.659 × anglesweetspotpercent

    Key insights:
    - avg_hit_angle: U-shaped relationship (extremes good, middle bad)
    - anglesweetspotpercent: Linear negative (more sweet spot = worse for pitcher)

    Args:
        statcast_data: {
            MLBAMID: {
                'avg_hit_angle': float,
                'anglesweetspotpercent': float
            }
        }

    Returns:
        dict: {MLBAMID: Statcast_Launch_Quality_Index}

    Example:
        avg_hit_angle = 10.0, anglesweetspotpercent = 30.0
        >>> angle_dev_sq = (10.0 - 14.2)² = 17.64
        >>> slqi = -0.056 * 17.64 + 0.659 * 30.0 = 18.78
    """
    result = {}

    for pid, data in statcast_data.items():
        if 'avg_hit_angle' in data and 'anglesweetspotpercent' in data:
            avg_hit_angle = data['avg_hit_angle']
            sweet_spot_pct = data['anglesweetspotpercent']

            # Calculate angle deviation squared
            angle_deviation_sq = (avg_hit_angle - OPTIMAL_LAUNCH_ANGLE) ** 2

            # Apply empirical formula
            slqi = (ANGLE_WEIGHT * angle_deviation_sq) + (SWEET_SPOT_WEIGHT * sweet_spot_pct)

            result[pid] = slqi

    return result


def calculate_all_pitcher_composites(
    bb_pct: Dict[int, float],
    k_pct: Dict[int, float],
    gb_pct: Dict[int, float],
    hard_pct: Dict[int, float],
    lob_pct: Dict[int, float],
    hr_fb_pct: Dict[int, float],
    statcast_data: Dict[int, Dict[str, float]],
    sd: Dict[int, int],
    md: Dict[int, int]
) -> Dict[str, Dict[int, float]]:
    """
    Calculate all pitcher composite features at once.

    Convenience function for getting all composites.

    Args:
        bb_pct: Walk percentage data
        k_pct: Strikeout percentage data
        gb_pct: Ground ball percentage data (park-adjusted)
        hard_pct: Hard contact percentage data
        lob_pct: Left on base percentage data
        hr_fb_pct: Home run to fly ball percentage data (park-adjusted)
        statcast_data: Statcast components
        sd: Shutdown count data
        md: Meltdown count data

    Returns:
        dict: {
            'damage_control_ratio': {...},
            'Opportunity_Success': {...},
            'strikeout_efficiency': {...},
            'contact_management': {...},
            'strikeout_contact_quality': {...},
            'Statcast_Launch_Quality_Index': {...},
            'SD_MD_Net': {...}
        }
    """
    return {
        'damage_control_ratio': calculate_damage_control_ratio(lob_pct, hr_fb_pct),
        'Opportunity_Success': calculate_opportunity_success(k_pct, bb_pct, lob_pct),
        'strikeout_efficiency': calculate_strikeout_efficiency(k_pct, bb_pct),
        'contact_management': calculate_contact_management(gb_pct, bb_pct),
        'strikeout_contact_quality': calculate_strikeout_contact_quality(k_pct, hard_pct),
        'Statcast_Launch_Quality_Index': calculate_statcast_launch_quality_index(statcast_data),
        'SD_MD_Net': calculate_shutdown_success(sd, md)
    }


# ============================================================================
# Running Game Control
# ============================================================================

# Caps for running control value
RUNNING_CONTROL_CAP_MIN = -4.0
RUNNING_CONTROL_CAP_MAX = 3.0


def calculate_running_control(n_cs: int, n_pk: int, n_sb: int, n_bk: int) -> float:
    """
    Calculate pitcher's running game control value.

    Measures ability to prevent steals via pickoffs, quick delivery, and deception.

    Split attribution with catchers:
    - CS/SB: 50% to pitcher, 50% to catcher (catcher gets other half via Throwing)
    - PK/BK: 100% to pitcher (pitcher-only events)

    Weights:
    - CS: +0.25 (half of 0.50, split with catcher)
    - PK: +0.60 (slightly more than CS, pure pitcher skill)
    - SB: -0.125 (half of 0.25, split with catcher)
    - BK: -0.50 (pure pitcher error, full penalty)

    Args:
        n_cs: Caught stealing (pitcher credited)
        n_pk: Pickoffs
        n_sb: Stolen bases allowed
        n_bk: Balks

    Returns:
        float: Running control run value

    Example:
        Good pickoff artist: 4 CS, 3 PK, 8 SB, 0 BK
        >>> calculate_running_control(4, 3, 8, 0)
        1.8  # (4×0.25) + (3×0.60) - (8×0.125) - (0×0.50)

        Poor delivery (Burnes 2024): 4 CS, 1 PK, 35 SB, 0 BK
        >>> calculate_running_control(4, 1, 35, 0)
        -2.775  # (4×0.25) + (1×0.60) - (35×0.125) - (0×0.50)

    Notes:
        - Inspired by but not matching Statcast's runs_prevented metric
        - Transparent weights, fully explainable
        - Avoids double-counting with catcher Throwing component
    """
    cs_value = n_cs * 0.25
    pk_value = n_pk * 0.60
    sb_value = n_sb * -0.125
    bk_value = n_bk * -0.50

    return cs_value + pk_value + sb_value + bk_value


# ============================================================================
# Shutdown Success (Reliever-Specific Signal)
# ============================================================================

def calculate_shutdown_success(
    sd_data: Dict[int, int],
    md_data: Dict[int, int]
) -> Dict[int, float]:
    """
    Calculate SD_MD_Net = Shutdowns - Meltdowns.

    Measures reliever's high-leverage success rate.
    Starters get 0 (no SD/MD recorded).

    From FanGraphs:
    - SD (Shutdown): Entering high-leverage, getting outs without damage
    - MD (Meltdown): Entering high-leverage, allowing inherited runs/blowing save

    Args:
        sd_data: {MLBAMID: Shutdown count}
        md_data: {MLBAMID: Meltdown count}

    Returns:
        dict: {MLBAMID: SD_MD_Net}

    Example:
        Elite closer: SD=29, MD=6
        >>> sd_md_net = 29 - 6 = 23

        Starter: SD=0, MD=0
        >>> sd_md_net = 0 - 0 = 0
    """
    common_players = set(sd_data.keys()) & set(md_data.keys())

    return {
        pid: float(sd_data[pid] - md_data[pid])
        for pid in common_players
    }
