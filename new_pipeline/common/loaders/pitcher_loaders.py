"""
Pitcher Feature Loaders - Clean Implementation

Each loader is declarative and focuses on:
1. What to load (file type, column name)
2. How to convert (decimal→percentage or keep raw)
3. What to validate (expected range)

All file handling, error handling, and repetitive logic is in helpers._load_fangraphs_feature()

Active Features:
1. BB% - Walk percentage
2. K% - Strikeout percentage
3. SwStr% - Swinging strike percentage
4. WPA/LI - Win probability added / Leverage index
5. SD - Shutdowns (high-leverage success count)
6. MD - Meltdowns (high-leverage failure count)
7. LOB% - Left on base percentage
8. Hard% - Hard contact percentage

Park-adjusted features (implemented separately):
7. ERA - Earned run average (park-adjusted, 3yr factor)
8. GB% - Ground ball percentage (park-adjusted, GB factor)
9. HR/FB% - Home run to fly ball percentage (park-adjusted, HR factor)

Statcast composites:
10. Statcast Launch Quality Index - avg_hit_angle + anglesweetspotpercent
11. Running_Control - CS/PK/SB/BK running game control (3yr weighted)

See: pitcher_feature_pipeline_design.md for specifications
"""

from typing import Dict, List
from pathlib import Path
import glob
import pandas as pd
from .helpers import (
    _load_fangraphs_feature,
    _load_park_adjusted_fangraphs_feature,
    _convert_decimal_to_percentage,
    validate_percentage_scale
)
from ..transformers.pitcher_composites import (
    calculate_running_control,
    RUNNING_CONTROL_CAP_MIN,
    RUNNING_CONTROL_CAP_MAX
)
from ..constants import STATCAST_DIR


def load_bb_pct_all_years(years: List[int]) -> Dict[int, float]:
    """
    Load BB% (walk percentage).

    FanGraphs stores as decimal (0.082 = 8.2%), we convert to percentage.

    Args:
        years: Years to load

    Returns:
        dict: {MLBAMID: BB% in percentage format}
    """
    # Load raw data (decimals)
    bb_raw = _load_fangraphs_feature(years, 'advanced', 'BB%')

    # Convert decimal → percentage
    bb_pct = {pid: _convert_decimal_to_percentage(val) for pid, val in bb_raw.items()}

    # Validate
    if bb_pct:
        validate_percentage_scale(bb_pct, 'BB%', expected_range=(0, 100))

    return bb_pct


def load_k_pct_all_years(years: List[int]) -> Dict[int, float]:
    """
    Load K% (strikeout percentage).

    FanGraphs stores as decimal (0.232 = 23.2%), we convert to percentage.

    Args:
        years: Years to load

    Returns:
        dict: {MLBAMID: K% in percentage format}
    """
    # Load raw data (decimals)
    k_raw = _load_fangraphs_feature(years, 'advanced', 'K%')

    # Convert decimal → percentage
    k_pct = {pid: _convert_decimal_to_percentage(val) for pid, val in k_raw.items()}

    # Validate
    if k_pct:
        validate_percentage_scale(k_pct, 'K%', expected_range=(0, 100))

    return k_pct


def load_swstr_all_years(years: List[int]) -> Dict[int, float]:
    """
    Load SwStr% (swinging strike percentage).

    FanGraphs stores as decimal (0.125 = 12.5%), we convert to percentage.

    Args:
        years: Years to load

    Returns:
        dict: {MLBAMID: SwStr% in percentage format}
    """
    # Load raw data (decimals)
    swstr_raw = _load_fangraphs_feature(years, 'stuff', 'SwStr%')

    # Convert decimal → percentage
    swstr_pct = {pid: _convert_decimal_to_percentage(val) for pid, val in swstr_raw.items()}

    # Validate
    if swstr_pct:
        validate_percentage_scale(swstr_pct, 'SwStr%', expected_range=(0, 50))

    return swstr_pct


def load_wpa_li_all_years(years: List[int]) -> Dict[int, float]:
    """
    Load WPA/LI (win probability added per leverage index).

    This is a RATIO, not a percentage. No conversion needed.
    Typical range: -0.5 to +1.5

    Args:
        years: Years to load

    Returns:
        dict: {MLBAMID: WPA/LI ratio}
    """
    # Load raw data (already correct scale - it's a ratio)
    wpa_li = _load_fangraphs_feature(years, 'winprobability', 'WPA/LI')

    # No conversion, no validation (ratio can be negative)
    return wpa_li


def load_sd_all_years(years: List[int]) -> Dict[int, int]:
    """
    Load SD (Shutdowns) - high-leverage success events.

    This is a COUNT, not a percentage. No conversion needed.
    Typical range: 0-30 (starters get 0, elite closers ~25+)

    Args:
        years: Years to load

    Returns:
        dict: {MLBAMID: SD count}
    """
    # Load raw data (integer counts)
    sd_raw = _load_fangraphs_feature(years, 'winprobability', 'SD')

    # Convert to int, default to 0 for missing
    sd = {}
    for pid, val in sd_raw.items():
        try:
            sd[pid] = int(val) if val else 0
        except (ValueError, TypeError):
            sd[pid] = 0

    return sd


def load_md_all_years(years: List[int]) -> Dict[int, int]:
    """
    Load MD (Meltdowns) - high-leverage failure events.

    This is a COUNT, not a percentage. No conversion needed.
    Typical range: 0-10 (starters get 0, struggling closers ~8-10)

    Args:
        years: Years to load

    Returns:
        dict: {MLBAMID: MD count}
    """
    # Load raw data (integer counts)
    md_raw = _load_fangraphs_feature(years, 'winprobability', 'MD')

    # Convert to int, default to 0 for missing
    md = {}
    for pid, val in md_raw.items():
        try:
            md[pid] = int(val) if val else 0
        except (ValueError, TypeError):
            md[pid] = 0

    return md


def load_lob_pct_all_years(years: List[int]) -> Dict[int, float]:
    """
    Load LOB% (left on base percentage).

    FanGraphs stores as decimal (0.722 = 72.2%), we convert to percentage.

    Args:
        years: Years to load

    Returns:
        dict: {MLBAMID: LOB% in percentage format}
    """
    # Load raw data (decimals)
    lob_raw = _load_fangraphs_feature(years, 'advanced', 'LOB%')

    # Convert decimal → percentage
    lob_pct = {pid: _convert_decimal_to_percentage(val) for pid, val in lob_raw.items()}

    # Validate (LOB% is bounded - cannot physically exceed 100%)
    if lob_pct:
        validate_percentage_scale(lob_pct, 'LOB%', expected_range=(50, 100), is_bounded=True)

    return lob_pct


def load_hard_pct_all_years(years: List[int]) -> Dict[int, float]:
    """
    Load Hard% (hard contact percentage).

    FanGraphs stores as decimal (0.358 = 35.8%), we convert to percentage.

    Args:
        years: Years to load

    Returns:
        dict: {MLBAMID: Hard% in percentage format}
    """
    # Load raw data (decimals)
    hard_raw = _load_fangraphs_feature(years, 'battedball', 'Hard%')

    # Convert decimal → percentage
    hard_pct = {pid: _convert_decimal_to_percentage(val) for pid, val in hard_raw.items()}

    # Validate (Hard% is bounded - cannot physically exceed 100%)
    if hard_pct:
        validate_percentage_scale(hard_pct, 'Hard%', expected_range=(0, 100), is_bounded=True)

    return hard_pct


# ============================================================================
# Park-Adjusted Loaders
# ============================================================================


def load_era_park_adjusted(years: List[int]) -> Dict[int, float]:
    """
    Load ERA with 3-year park factor adjustment.

    FanGraphs stores ERA as raw value (not decimal).
    Park adjustment uses 3yr factor (already halved by FanGraphs).

    Args:
        years: Years to load

    Returns:
        dict: {MLBAMID: park-adjusted ERA}

    Example:
        Coors pitcher with raw ERA 4.50, park factor 108:
        - Adjusted ERA = 4.50 * (100/108) = 4.17
    """
    # Load ERA with park adjustment (no conversion - ERA is not a percentage)
    era_adjusted = _load_park_adjusted_fangraphs_feature(
        years, 'advanced', 'ERA', '3yr'
    )

    return era_adjusted


def load_gb_pct_park_adjusted(years: List[int]) -> Dict[int, float]:
    """
    Load GB% with GB park factor adjustment.

    FanGraphs stores as decimal (0.44 = 44%), we convert to percentage.
    Park adjustment uses GB factor.

    GB% is capped at 100% because it represents ground balls / balls in play,
    which cannot physically exceed 100%. Park adjustment can push values above
    100% in extreme cases (e.g., 87.5% at park factor 85 -> 102.94%).

    Args:
        years: Years to load

    Returns:
        dict: {MLBAMID: park-adjusted GB% in percentage format, capped at 100%}
    """
    # Load GB% with park adjustment (returns raw decimals, park-adjusted)
    # Note: cap_at_100=True caps at 1.0 for decimal format
    gb_raw_adjusted = _load_park_adjusted_fangraphs_feature(
        years, 'battedball', 'GB%', 'GB', cap_at_100=True
    )

    # Convert decimal → percentage (0.44 -> 44.0)
    # Since we capped at 1.0, this will be at most 100.0
    gb_pct = {pid: _convert_decimal_to_percentage(val) for pid, val in gb_raw_adjusted.items()}

    # Validate (GB% is bounded - cannot physically exceed 100%)
    if gb_pct:
        validate_percentage_scale(gb_pct, 'GB%', expected_range=(0, 100), is_bounded=True)

    return gb_pct


def load_hr_fb_pct_park_adjusted(years: List[int]) -> Dict[int, float]:
    """
    Load HR/FB% with HR park factor adjustment.

    CRITICAL: This was MISSING in old pipeline!
    See: hr_fb_park_adjustment_integration.md

    FanGraphs stores as decimal (0.125 = 12.5%), we convert to percentage.
    Park adjustment uses HR factor.

    Without this adjustment:
    - Coors pitchers penalized (inflated HR/FB%)
    - Oracle pitchers rewarded (deflated HR/FB%)

    Args:
        years: Years to load

    Returns:
        dict: {MLBAMID: park-adjusted HR/FB% in percentage format}
    """
    # Load HR/FB with park adjustment (returns raw decimals, park-adjusted)
    hr_fb_raw_adjusted = _load_park_adjusted_fangraphs_feature(
        years, 'battedball', 'HR/FB', 'HR'
    )

    # Convert decimal → percentage
    hr_fb_pct = {pid: _convert_decimal_to_percentage(val) for pid, val in hr_fb_raw_adjusted.items()}

    # Validate
    if hr_fb_pct:
        validate_percentage_scale(hr_fb_pct, 'HR/FB%', expected_range=(0, 50))

    return hr_fb_pct


# ============================================================================
# Statcast Loader
# ============================================================================


def load_statcast_data(years: List[int]) -> Dict[int, Dict[str, float]]:
    """
    Load Statcast exit velocity data for Launch Quality Index calculation.

    Returns RAW components - composite calculation happens in transformer.

    Source: Statcast_Data/exit_velocity/exit_velocity_pitchers_{year}.csv
    Columns: avg_hit_angle, anglesweetspotpercent

    Args:
        years: Years to load

    Returns:
        dict: {
            MLBAMID: {
                'avg_hit_angle': float,
                'anglesweetspotpercent': float
            }
        }

    Note:
        - player_id column used (not MLBAMID, but same ID)
        - Most recent year's data overwrites earlier years
        - Empty dict if no data available
    """
    statcast_data = {}

    for year in years:
        csv_path = STATCAST_DIR / "exit_velocity" / f"exit_velocity_pitchers_{year}.csv"

        if not csv_path.exists():
            continue

        try:
            df = pd.read_csv(csv_path)

            # player_id is the MLBAMID equivalent in Statcast data
            if 'player_id' not in df.columns:
                continue

            for _, row in df.iterrows():
                if pd.notna(row['player_id']):
                    try:
                        # player_id is sometimes float, convert to int
                        mlbamid = int(float(row['player_id']))

                        # Initialize dict for this player if needed
                        if mlbamid not in statcast_data:
                            statcast_data[mlbamid] = {}

                        # Store raw components (no conversion - these are actual angles/percentages)
                        if pd.notna(row.get('avg_hit_angle')):
                            statcast_data[mlbamid]['avg_hit_angle'] = float(row['avg_hit_angle'])

                        if pd.notna(row.get('anglesweetspotpercent')):
                            statcast_data[mlbamid]['anglesweetspotpercent'] = float(row['anglesweetspotpercent'])

                    except (ValueError, TypeError):
                        continue

        except Exception as e:
            continue

    return statcast_data


# ============================================================================
# Running Control Loader
# ============================================================================


def load_running_control_all_years(years: List[int]) -> Dict[int, float]:
    """
    Load pitcher Running Control from Statcast running game data.

    Running Control measures pitcher's ability to prevent steals via:
    - CS (caught stealing): +0.25 runs
    - PK (pickoffs): +0.60 runs
    - SB (stolen bases): -0.125 runs
    - BK (balks): -0.50 runs

    Attribution split with catchers:
    - CS/SB: 50% to pitcher, 50% to catcher (catcher gets other half via Throwing)
    - PK/BK: 100% to pitcher (pitcher-only events)

    3-year weighted average: 50% most recent, 30% year-1, 20% year-2
    Caps: [-4, +3]

    Source: Statcast_Data/pitcher_running_against/pitcher_running_game_statcast_{year}.csv

    Args:
        years: Years to load (typically 3 years for weighted average)

    Returns:
        dict: {MLBAMID: running_control value in runs}

    Example:
        >>> load_running_control_all_years([2022, 2023, 2024])
        {519144: 1.25, 605483: -2.10, ...}  # Pitcher MLBAMIDs with values

    Notes:
        - Uses 3-year weighted average to smooth small-sample noise
        - Most pitchers face 20-40 steal attempts per season
        - Elite: +1.5 to +2.5 runs (Bibee, Morton, Skenes)
        - Poor: -2.5 to -3.5 runs (Burnes, Diaz, Gore)
    """
    yearly_data = {}

    # Load data for each year
    for year in years:
        csv_path = STATCAST_DIR / "pitcher_running_against" / f"pitcher_running_game_statcast_{year}.csv"

        if not csv_path.exists():
            continue

        try:
            df = pd.read_csv(csv_path)

            # Check required columns exist
            required_cols = ['player_id', 'n_cs', 'n_pk', 'n_sb', 'n_bk']
            if not all(col in df.columns for col in required_cols):
                continue

            year_values = {}

            for _, row in df.iterrows():
                if pd.notna(row['player_id']):
                    try:
                        # player_id is the MLBAMID equivalent in Statcast
                        mlbamid = int(float(row['player_id']))

                        # Extract steal/pickoff/balk counts
                        n_cs = 0 if pd.isna(row['n_cs']) else int(row['n_cs'])
                        n_pk = 0 if pd.isna(row['n_pk']) else int(row['n_pk'])
                        n_sb = 0 if pd.isna(row['n_sb']) else int(row['n_sb'])
                        n_bk = 0 if pd.isna(row['n_bk']) else int(row['n_bk'])

                        # Calculate running control value
                        running_control = calculate_running_control(n_cs, n_pk, n_sb, n_bk)

                        # Apply caps
                        running_control = max(RUNNING_CONTROL_CAP_MIN,
                                            min(RUNNING_CONTROL_CAP_MAX, running_control))

                        year_values[mlbamid] = running_control

                    except (ValueError, TypeError):
                        continue

            yearly_data[year] = year_values

        except Exception as e:
            continue

    # Apply 3-year weighted average (50%/30%/20%)
    if not yearly_data:
        return {}

    # Sort years descending (most recent first)
    sorted_years = sorted(yearly_data.keys(), reverse=True)

    # Get all unique pitchers across all years
    all_pitchers = set()
    for year_values in yearly_data.values():
        all_pitchers.update(year_values.keys())

    # Apply weighting
    weighted_values = {}
    weights = [0.5, 0.3, 0.2]  # Most recent → oldest

    for pitcher in all_pitchers:
        weighted_sum = 0.0
        total_weight = 0.0

        for i, year in enumerate(sorted_years[:3]):  # Max 3 years
            if pitcher in yearly_data[year]:
                weight = weights[i]
                weighted_sum += yearly_data[year][pitcher] * weight
                total_weight += weight

        # Normalize by actual weights used (handles missing years)
        if total_weight > 0:
            weighted_values[pitcher] = weighted_sum / total_weight

    return weighted_values
