"""
Future Season Hitter Feature Loaders

Optimized for year-to-year prediction with additional power and batted ball features.
Keeps all existing features, adds: ISO, GB%, HR/FB, Hard%, Pull%
"""

from typing import Dict, List, Tuple

# Reuse helpers from common
from new_pipeline.common.loaders.helpers import (
    _load_fangraphs_feature,
    _convert_decimal_to_percentage,
    validate_percentage_scale
)

# Reuse existing loaders for retained features
from new_pipeline.common.loaders.hitter_loaders import (
    load_k_pct_all_years,              # REUSE
    load_bb_pct_all_years,             # REUSE
    load_pa_all_years,                 # REUSE
    load_gdp_all_years,                # REUSE
    load_avg_park_adjusted,            # REUSE
    load_obp_park_adjusted,            # REUSE
    load_slg_park_adjusted,            # REUSE
    load_positional_war,               # REUSE
    load_positions_all_years,          # REUSE
    load_enhanced_baserunning,         # REUSE
    load_enhanced_defense              # REUSE
)


# ============================================================================
# NEW LOADERS - Hitter Batted Ball & Power Metrics
# ============================================================================

def load_iso_all_years(years: List[int]) -> Dict[Tuple[int, int], float]:
    """
    Load ISO (Isolated Power = SLG - AVG) for hitters - year-specific.

    ISO measures extra bases per at-bat, isolating power from batting average.
    Higher is better (more power).

    Source: FanGraphs_Data/hitters/fangraphs_hitters_YYYY_advanced.csv
    Column: ISO
    Scale: FanGraphs stores as decimal (0.240 = .240 ISO)

    Args:
        years: Years to load

    Returns:
        dict: {(MLBAMID, Year): ISO in decimal format}

    Example:
        Aaron Judge 2023: ISO = 0.328 (62 HR, elite power)
        Luis Arraez 2023: ISO = 0.076 (4 HR, pure contact)
    """
    # Load raw data (already decimal format - no conversion needed for ISO)
    iso_data = _load_fangraphs_feature(
        years,
        'advanced',
        'ISO',
        player_type='hitters'
    )

    # ISO stays in decimal format (like AVG, OBP, SLG)
    # No conversion needed

    return iso_data


def load_gb_pct_all_years(years: List[int]) -> Dict[Tuple[int, int], float]:
    """
    Load GB% (ground ball percentage) for hitters - year-specific.

    GB% measures percentage of batted balls hit on the ground.
    Neutral (neither universally good nor bad).

    Source: FanGraphs_Data/hitters/fangraphs_hitters_YYYY_battedball.csv
    Column: GB%
    Scale: FanGraphs stores as decimal (0.482 = 48.2%), convert to percentage

    Args:
        years: Years to load

    Returns:
        dict: {(MLBAMID, Year): GB% in percentage format}

    Example:
        Steven Kwan 2023: GB% = 54.2% (ground ball contact hitter)
        Pete Alonso 2023: GB% = 34.1% (fly ball power hitter)
    """
    # Load raw data (decimals) - GB% is in battedball file
    gb_raw = _load_fangraphs_feature(
        years,
        'battedball',
        'GB%',
        player_type='hitters'
    )

    # Convert decimal → percentage
    gb_pct = {
        (pid, year): _convert_decimal_to_percentage(val)
        for (pid, year), val in gb_raw.items()
    }

    # Validate
    if gb_pct:
        validate_percentage_scale(
            {pid: val for (pid, year), val in gb_pct.items()},
            'GB% (Hitters)',
            expected_range=(25, 65)
        )

    return gb_pct


def load_hr_fb_pct_all_years(years: List[int]) -> Dict[Tuple[int, int], float]:
    """
    Load HR/FB (home runs per fly ball) for hitters - year-specific.

    HR/FB measures power on fly balls specifically.
    Higher is better (more fly ball power).

    Source: FanGraphs_Data/hitters/fangraphs_hitters_YYYY_battedball.csv
    Column: HR/FB
    Scale: FanGraphs stores as decimal (0.182 = 18.2%), convert to percentage

    Args:
        years: Years to load

    Returns:
        dict: {(MLBAMID, Year): HR/FB in percentage format}

    Example:
        Aaron Judge 2023: HR/FB = 30.2% (extreme power on fly balls)
        Luis Arraez 2023: HR/FB = 5.6% (no fly ball power)
    """
    # Load raw data (decimals) - HR/FB is in battedball file
    hr_fb_raw = _load_fangraphs_feature(
        years,
        'battedball',
        'HR/FB',
        player_type='hitters'
    )

    # Convert decimal → percentage
    hr_fb_pct = {
        (pid, year): _convert_decimal_to_percentage(val)
        for (pid, year), val in hr_fb_raw.items()
    }

    # Validate
    if hr_fb_pct:
        validate_percentage_scale(
            {pid: val for (pid, year), val in hr_fb_pct.items()},
            'HR/FB (Hitters)',
            expected_range=(0, 40)
        )

    return hr_fb_pct


def load_hard_pct_all_years(years: List[int]) -> Dict[Tuple[int, int], float]:
    """
    Load Hard% (hard contact percentage) for hitters - year-specific.

    Hard% measures percentage of batted balls hit hard (95+ mph exit velocity).
    Higher is better (more hard contact).

    Source: FanGraphs_Data/hitters/fangraphs_hitters_YYYY_battedball.csv
    Column: Hard%
    Scale: FanGraphs stores as decimal (0.418 = 41.8%), convert to percentage

    Args:
        years: Years to load

    Returns:
        dict: {(MLBAMID, Year): Hard% in percentage format}

    Example:
        Juan Soto 2023: Hard% = 48.6% (elite contact quality)
        Nick Madrigal 2023: Hard% = 27.1% (weak contact)
    """
    # Load raw data (decimals) - Hard% is in battedball file
    hard_raw = _load_fangraphs_feature(
        years,
        'battedball',
        'Hard%',
        player_type='hitters'
    )

    # Convert decimal → percentage
    hard_pct = {
        (pid, year): _convert_decimal_to_percentage(val)
        for (pid, year), val in hard_raw.items()
    }

    # Validate
    if hard_pct:
        validate_percentage_scale(
            {pid: val for (pid, year), val in hard_pct.items()},
            'Hard% (Hitters)',
            expected_range=(20, 55)
        )

    return hard_pct


def load_pull_pct_all_years(years: List[int]) -> Dict[Tuple[int, int], float]:
    """
    Load Pull% (pull field percentage) for hitters - year-specific.

    Pull% measures percentage of batted balls hit to pull field.
    Neutral (neither universally good nor bad).

    Source: FanGraphs_Data/hitters/fangraphs_hitters_YYYY_battedball.csv
    Column: Pull%
    Scale: FanGraphs stores as decimal (0.428 = 42.8%), convert to percentage

    Args:
        years: Years to load

    Returns:
        dict: {(MLBAMID, Year): Pull% in percentage format}

    Example:
        Yandy Diaz 2023: Pull% = 34.2% (uses all fields)
        Kyle Schwarber 2023: Pull% = 47.8% (extreme pull, power focused)
    """
    # Load raw data (decimals) - Pull% is in battedball file
    pull_raw = _load_fangraphs_feature(
        years,
        'battedball',
        'Pull%',
        player_type='hitters'
    )

    # Convert decimal → percentage
    pull_pct = {
        (pid, year): _convert_decimal_to_percentage(val)
        for (pid, year), val in pull_raw.items()
    }

    # Validate
    if pull_pct:
        validate_percentage_scale(
            {pid: val for (pid, year), val in pull_pct.items()},
            'Pull% (Hitters)',
            expected_range=(25, 55)
        )

    return pull_pct
