"""
Baserunning Metric Helper Functions.

Pure calculation functions for Enhanced_Baserunning composite.
Used by hitter_loaders.py to compute baserunning value from multiple sources.

Key Concepts:
- Steal runs: SB creates value, CS/PO destroys value
- Extra-base taking: Aggressive baserunning (1st→3rd on singles, etc.)
- Sprint speed: Raw athleticism measurement

Design Philosophy:
- SB weight (0.25) captures strategic/threat value beyond run expectancy
- CS/PO penalties (-0.50) based on empirical run expectancy research
- Yearly median baselines adjust for era changes and remove outliers
- Billy Hamilton types get proper credit for elite baserunning identity

References:
- FanGraphs THT: "Stolen Base Attempts: An Algorithm for Allocating Run Value"
  https://tht.fangraphs.com/stolen-base-attempts-an-algorithm-for-allocating-run-value/
"""

from typing import Dict, Tuple
import pandas as pd
import numpy as np


# ============================================================================
# Run Value Weights
# ============================================================================

# Steal component weights (research-informed)
SB_RUN_VALUE = 0.25   # Successful steal (includes threat/strategic value)
CS_RUN_VALUE = -0.50  # Caught stealing (run expectancy + out cost)
PO_RUN_VALUE = -0.50  # Picked off (same outcome as CS)

# Scaling factors for XBT and sprint speed
XBT_SCALING = 10.0    # Extra-base taking aggressiveness
SPEED_SCALING = 0.5   # Sprint speed impact

# Caps for total baserunning value
BASERUNNING_CAP_MIN = -7.0
BASERUNNING_CAP_MAX = 15.0


# ============================================================================
# Component Calculations
# ============================================================================

def calculate_steal_runs(sb: float, cs: float, po: float) -> float:
    """
    Calculate baserunning value from stolen bases, caught stealing, and pickoffs.

    Weights:
    - SB: +0.25 runs (includes threat value beyond run expectancy)
    - CS: -0.50 runs (run expectancy research: ~-0.595 in 0-out situations)
    - PO: -0.50 runs (same outcome as CS)

    Args:
        sb: Stolen bases
        cs: Caught stealing
        po: Picked off

    Returns:
        float: Net steal run value

    Example:
        Elite base stealer: 50 SB, 10 CS, 5 PO
        >>> calculate_steal_runs(50, 10, 5)
        5.0  # (50×0.25) - (10×0.50) - (5×0.50)

        Billy Hamilton 2014: 56 SB, 13 CS, 4 PO
        >>> calculate_steal_runs(56, 13, 4)
        5.5  # (56×0.25) - (13×0.50) - (4×0.50)

    Notes:
        - Break-even steal rate: 66.7% (0.50 / (0.25 + 0.50))
        - League average steal rate: ~70-75%
        - Elite base stealers: 75-80%+
    """
    return (sb * SB_RUN_VALUE) + (cs * CS_RUN_VALUE) + (po * PO_RUN_VALUE)


def calculate_xbt_runs(player_xbt_pct: float, median_xbt_pct: float) -> float:
    """
    Calculate extra-base taking value relative to league median.

    XBT% measures aggressive baserunning:
    - 1st → 3rd on singles
    - 1st → Home on doubles
    - 2nd → Home on singles

    Args:
        player_xbt_pct: Player's XBT% (stored as percentage: 55.0 = 55%)
        median_xbt_pct: League median XBT% for that year

    Returns:
        float: XBT run value relative to median

    Example:
        Aggressive runner: 55% XBT, league median 40%
        >>> calculate_xbt_runs(55.0, 40.0)
        1.5  # (55 - 40) × 0.10 (scaled)

        Conservative runner: 30% XBT, league median 40%
        >>> calculate_xbt_runs(30.0, 40.0)
        -1.0  # (30 - 40) × 0.10

    Notes:
        - Typical XBT% range: 25% to 60%
        - Median is more robust than mean (not skewed by outliers)
        - Scaling factor (10.0) converts percentage difference to run value
    """
    # XBT% is stored as percentage (not decimal), so no conversion needed
    xbt_diff = player_xbt_pct - median_xbt_pct
    return xbt_diff * (XBT_SCALING / 100.0)  # Convert pct points to run value


def calculate_speed_value(player_speed_ft_per_sec: float, median_speed_ft_per_sec: float) -> float:
    """
    Calculate sprint speed value relative to league median.

    Sprint speed measured as feet per second (90ft / time from home to 1st).

    Args:
        player_speed_ft_per_sec: Player's sprint speed (ft/s)
        median_speed_ft_per_sec: League median sprint speed for that year

    Returns:
        float: Speed run value relative to median

    Example:
        Elite runner: 29.5 ft/s, league median 27.0 ft/s
        >>> calculate_speed_value(29.5, 27.0)
        1.25  # (29.5 - 27.0) × 0.5

        Slow runner: 25.0 ft/s, league median 27.0 ft/s
        >>> calculate_speed_value(25.0, 27.0)
        -1.0  # (25.0 - 27.0) × 0.5

    Notes:
        - Typical sprint speed range: 24-31 ft/s
        - Elite: 29-31 ft/s (Elly De La Cruz, Bobby Witt Jr.)
        - Slow: 24-26 ft/s (catchers, corner infielders)
        - Scaling factor (0.5) converts speed difference to run value
    """
    speed_diff = player_speed_ft_per_sec - median_speed_ft_per_sec
    return speed_diff * SPEED_SCALING


def calculate_sprint_speed(seconds_to_90ft: float) -> float:
    """
    Convert Statcast time to sprint speed in ft/s.

    Args:
        seconds_to_90ft: Time from contact to reaching 1st base (seconds)

    Returns:
        float: Sprint speed in feet per second

    Example:
        Elite runner: 4.0 seconds to 90ft
        >>> calculate_sprint_speed(4.0)
        22.5  # 90 / 4.0

        Slow runner: 4.5 seconds to 90ft
        >>> calculate_sprint_speed(4.5)
        20.0  # 90 / 4.5
    """
    if seconds_to_90ft <= 0:
        return 0.0
    return 90.0 / seconds_to_90ft


def apply_baserunning_cap(baserunning_runs: float) -> float:
    """
    Apply symmetric caps to baserunning value.

    Caps: [-7, 15]
    - Lower cap prevents outliers from small samples (injured players, etc.)
    - Upper cap prevents single-season anomalies (but allows elite 50+ SB seasons)

    Args:
        baserunning_runs: Raw baserunning value

    Returns:
        float: Capped baserunning value

    Example:
        Elite runner with +13 runs (Ohtani 59 SB):
        >>> apply_baserunning_cap(13.0)
        13.0  # Not capped

        Extreme outlier with +18 runs:
        >>> apply_baserunning_cap(18.0)
        15.0  # Capped at +15

        Injured player with -9 runs:
        >>> apply_baserunning_cap(-9.0)
        -7.0  # Capped at -7

    Notes:
        - No position-specific caps (baserunning independent of position)
        - Elite base stealers (Ohtani, Elly, Billy Hamilton) can hit +12-15
        - Catchers can hit -7 cap (slow + no stealing)
    """
    return max(BASERUNNING_CAP_MIN, min(BASERUNNING_CAP_MAX, baserunning_runs))


# ============================================================================
# Yearly Baseline Calculations
# ============================================================================

def calculate_yearly_baselines(
    bp_df: pd.DataFrame,
    statcast_df: pd.DataFrame = None
) -> Dict[str, float]:
    """
    Calculate league median baselines for a given year.

    Uses median (not mean) to be robust against outliers.

    Args:
        bp_df: BP baserunning data for one year
        statcast_df: Statcast running splits for one year (optional)

    Returns:
        dict: {'xbt_median': float, 'speed_median': float}

    Example:
        >>> baselines = calculate_yearly_baselines(bp_df_2024, statcast_df_2024)
        >>> baselines
        {'xbt_median': 9.8, 'speed_median': 27.2}

    Notes:
        - XBT% stored as percentage (9.8 = 9.8%, not 0.098)
        - Sprint speed calculated from seconds_since_hit_090
        - If Statcast unavailable, speed_median = 0 (no speed component)
    """
    baselines = {}

    # XBT% median from BP data
    if 'XBT%' in bp_df.columns:
        xbt_values = pd.to_numeric(bp_df['XBT%'], errors='coerce').dropna()
        baselines['xbt_median'] = xbt_values.median()
    else:
        baselines['xbt_median'] = 0.0

    # Sprint speed median from Statcast
    if statcast_df is not None and 'seconds_since_hit_090' in statcast_df.columns:
        seconds = pd.to_numeric(statcast_df['seconds_since_hit_090'], errors='coerce').dropna()
        speeds = 90.0 / seconds
        baselines['speed_median'] = speeds.median()
    else:
        baselines['speed_median'] = 0.0

    return baselines
