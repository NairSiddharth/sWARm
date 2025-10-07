"""
Helper functions for feature loaders.

All loaders use these helpers for consistency.
"""

import glob
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

# Configuration - Import directory paths from constants
from ..constants import FANGRAPHS_PITCHER_DIR, FANGRAPHS_HITTER_DIR, CACHE_DIR

# Map player_type string to corresponding directory
_PLAYER_TYPE_DIRS = {
    'pitchers': FANGRAPHS_PITCHER_DIR,
    'hitters': FANGRAPHS_HITTER_DIR
}


def get_current_season_year() -> int:
    """
    Get current season year (calendar year).

    Used to detect which years should check for partial season files.

    Returns:
        int: Current year (e.g., 2025)

    Example:
        >>> get_current_season_year()
        2025
    """
    return datetime.now().year


def _load_fangraphs_feature(
    years: List[int],
    file_type: str,
    column_name: str,
    player_type: str = 'pitchers'
) -> Dict[int, float]:
    """
    Generic loader for FanGraphs features.

    Handles all repetitive logic:
    - File pattern matching (with firsthalf for 2025+)
    - MLBAMID column variations
    - Column name variations (if needed)
    - Error handling

    Returns RAW data - conversion is the caller's responsibility.

    Args:
        years: List of years to load
        file_type: Type of FanGraphs file ('advanced', 'stuff', 'battedball', 'winprobability', etc.)
        column_name: Name of column to extract
        player_type: 'pitchers' or 'hitters'

    Returns:
        dict: {MLBAMID: raw_value} - most recent value per player

    Example:
        >>> # Load raw BB% (as decimals)
        >>> bb_raw = _load_fangraphs_feature([2024], 'advanced', 'BB%')
        >>> bb_raw[660271]
        0.062  # Raw decimal, not converted yet
    """
    feature_dict = {}

    for year in years:
        # For recent seasons (current and previous year), try partial season files first
        # Matches any partial season: *_firsthalf_*, *_quarter_*, *_2weeks_*, *_month_*, etc.
        files_found = []

        if year >= get_current_season_year() - 1:
            # Try partial season pattern (glob for ANY partial season file)
            partial_pattern = f"fangraphs_{player_type}_{year}_*_{file_type}.csv"
            partial_matches = sorted(glob.glob(str(_PLAYER_TYPE_DIRS[player_type] / partial_pattern)))

            # Filter to only actual partial season files (exclude full season)
            full_season_file = f"fangraphs_{player_type}_{year}_{file_type}.csv"
            partial_matches = [f for f in partial_matches if not f.endswith(full_season_file)]

            if partial_matches:
                # Use first alphabetical match (firsthalf < month < quarter < 2weeks)
                files_found = [partial_matches[0]]

        # Fall back to full season file if no partial season found
        if not files_found:
            # Handle empty file_type (base comprehensive file)
            if file_type:
                full_season_pattern = f"fangraphs_{player_type}_{year}_{file_type}.csv"
            else:
                full_season_pattern = f"fangraphs_{player_type}_{year}.csv"
            files_found = glob.glob(str(_PLAYER_TYPE_DIRS[player_type] / full_season_pattern))

        if not files_found:
            continue

        try:
            df = pd.read_csv(files_found[0])

            # Find player ID column (handle variations)
            id_col = None
            for col in ['MLBAMID', 'PlayerId', 'playerid']:
                if col in df.columns:
                    id_col = col
                    break

            if id_col is None:
                continue

            # Find target column (handle case variations)
            target_col = None
            if column_name in df.columns:
                target_col = column_name
            else:
                # Try case-insensitive match
                for col in df.columns:
                    if col.lower() == column_name.lower():
                        target_col = col
                        break

            if target_col is None:
                continue

            # Extract data
            for _, row in df.iterrows():
                if pd.notna(row[id_col]) and pd.notna(row[target_col]):
                    mlbamid = int(row[id_col])
                    value = float(row[target_col])
                    # Most recent year overwrites earlier years
                    feature_dict[mlbamid] = value

        except Exception as e:
            # Silent failure - let caller decide if empty dict is a problem
            continue

    return feature_dict


def _load_park_adjusted_fangraphs_feature(
    years: List[int],
    file_type: str,
    column_name: str,
    park_factor_type: str,
    player_type: str = 'pitchers'
) -> Dict[int, float]:
    """
    Load FanGraphs feature with park adjustment applied.

    Handles:
    - Loading raw stat + team from CSV
    - Loading park factors
    - Applying corrected park adjustment (no double-blending)
    - Decimal → percentage conversion (caller decides)

    Args:
        years: Years to load
        file_type: FanGraphs file type ('advanced', 'battedball', etc.)
        column_name: Stat column name
        park_factor_type: Park factor to use ('3yr', 'HR', 'GB', etc.)
        player_type: 'pitchers' or 'hitters'

    Returns:
        dict: {MLBAMID: park-adjusted value}

    Example:
        >>> # Load ERA with 3yr park adjustment (returns raw ERA, not percentage)
        >>> era = _load_park_adjusted_fangraphs_feature([2024], 'advanced', 'ERA', '3yr')
    """
    feature_dict = {}

    for year in years:
        # For recent seasons, try partial season files first (same logic as _load_fangraphs_feature)
        files_found = []

        if year >= get_current_season_year() - 1:
            # Try partial season pattern
            partial_pattern = f"fangraphs_{player_type}_{year}_*_{file_type}.csv"
            partial_matches = sorted(glob.glob(str(_PLAYER_TYPE_DIRS[player_type] / partial_pattern)))

            # Filter to only actual partial season files
            full_season_file = f"fangraphs_{player_type}_{year}_{file_type}.csv"
            partial_matches = [f for f in partial_matches if not f.endswith(full_season_file)]

            if partial_matches:
                files_found = [partial_matches[0]]

        # Fall back to full season file
        if not files_found:
            # Handle empty file_type (base comprehensive file)
            if file_type:
                full_season_pattern = f"fangraphs_{player_type}_{year}_{file_type}.csv"
            else:
                full_season_pattern = f"fangraphs_{player_type}_{year}.csv"
            files_found = glob.glob(str(_PLAYER_TYPE_DIRS[player_type] / full_season_pattern))

        if not files_found:
            continue

        # Load park factors for this year
        try:
            park_factors = _load_park_factors(year)
        except FileNotFoundError:
            # No park factors for this year - skip park adjustment
            continue

        try:
            df = pd.read_csv(files_found[0])

            # Find player ID column
            id_col = None
            for col in ['MLBAMID', 'PlayerId', 'playerid']:
                if col in df.columns:
                    id_col = col
                    break

            if id_col is None:
                continue

            # Find stat column
            target_col = None
            if column_name in df.columns:
                target_col = column_name
            else:
                for col in df.columns:
                    if col.lower() == column_name.lower():
                        target_col = col
                        break

            if target_col is None or 'Team' not in df.columns:
                continue

            # Extract and adjust
            for _, row in df.iterrows():
                if pd.notna(row[id_col]) and pd.notna(row[target_col]) and pd.notna(row['Team']):
                    mlbamid = int(row[id_col])
                    raw_value = float(row[target_col])
                    team = str(row['Team']).strip().upper()

                    # Get park factor for this team
                    team_factors = park_factors.get(team, {})
                    park_factor = team_factors.get(park_factor_type, 100.0)

                    # Apply park adjustment (CORRECTED - no double blending)
                    adjusted_value = _apply_park_adjustment(raw_value, park_factor)

                    feature_dict[mlbamid] = adjusted_value

        except Exception as e:
            continue

    return feature_dict


def _convert_decimal_to_percentage(value):
    """
    Convert FanGraphs decimal format to percentage.

    FanGraphs CSVs store: 0.232 (means 23.2%)
    We need: 23.2 (for formulas to work correctly)

    CRITICAL: See CRITICAL_SCALE_MISMATCH_ISSUE.md
    Without this conversion, composite features are off by 5-10,000x!

    Args:
        value (float): Decimal value (0.232)

    Returns:
        float: Percentage value (23.2)

    Example:
        >>> _convert_decimal_to_percentage(0.232)
        23.2
    """
    return value * 100.0


def _load_park_factors(year):
    """
    Load cached park factors for a specific year.

    Park factors are cached as JSON files in cache/ directory.
    Format: {'Team': {'3yr': 105, 'GB': 98, 'HR': 108, ...}, ...}

    IMPORTANT: FanGraphs park factors are ALREADY HALVED for full season stats.
    See: https://www.fangraphs.com/tools/guts?type=pf
    "All Park Factors have already been halved for use on full season stats."

    This means the 50% home/road blend has already been applied.
    Do NOT blend again or you'll under-adjust!

    Args:
        year (int): Year to load park factors for

    Returns:
        dict: {team_abbrev: {factor_type: value}}

    Raises:
        FileNotFoundError: If park factors not cached for year

    Example:
        >>> factors = _load_park_factors(2024)
        >>> factors['COL']['HR']  # Coors Field HR factor (already halved)
        108
    """
    cache_file = CACHE_DIR / f"fangraphs_park_factors_{year}.json"

    if cache_file.exists():
        with open(cache_file, 'r') as f:
            return json.load(f)
    else:
        raise FileNotFoundError(
            f"Park factors not cached for {year}. "
            f"Expected: {cache_file}"
        )


def _apply_park_adjustment(value, park_factor, stat_type='percentage'):
    """
    Apply park adjustment using pre-halved FanGraphs factors.

    CRITICAL: FanGraphs park factors are ALREADY HALVED for 50% home/road split.
    See: https://www.fangraphs.com/tools/guts?type=pf

    The implementation guide was WRONG - it suggested blending again:
        effective_park = (park_factor + 100) / 2  # DON'T DO THIS!

    This would blend TWICE (once by FanGraphs, once by us), under-adjusting stats.

    CORRECT formula (direct division):
        adjustment = 100 / park_factor
        adjusted_value = value * adjustment

    Args:
        value (float): Raw stat value
        park_factor (float): FanGraphs park factor (100 = neutral, ALREADY halved)
        stat_type (str): 'era' or 'percentage' (both use same formula)

    Returns:
        float: Park-adjusted value

    Example:
        ERA 3.00 at Coors (park factor 108, already halved):
        >>> _apply_park_adjustment(3.00, 108, stat_type='era')
        2.78  # Correct: 3.00 * (100/108)

        WRONG (if we blended again):
        effective_park = (108 + 100) / 2 = 104
        3.00 * (100/104) = 2.88  # Under-adjusted!

        HR/FB% 15.0 at Oracle (park factor 85, already halved):
        >>> _apply_park_adjustment(15.0, 85, stat_type='percentage')
        17.65  # Correct: 15.0 * (100/85)
    """
    # Direct division - FanGraphs already did the 50% blend!
    adjustment = 100.0 / park_factor
    return value * adjustment


def _load_counting_stat_with_proration(
    years: List[int],
    column_name: str,
    reference_usage_dict: Dict[int, float],
    source_dir: Path,
    source_file_pattern: str,
    usage_column: str = 'PA',
    id_column: str = 'MLBAMID'
) -> Dict[int, float]:
    """
    Load counting stat with PA/Inn-based proration.

    Use for stats that accumulate over playing time (SB, CS, PO, DPS, DPT, DPF, Scp, Framing).
    Prorates only when source timeframe doesn't match reference timeframe.

    Alignment tolerance: 10% (ratio 0.9-1.1)
    - Aligned: Use raw value directly (no proration)
    - Source > Reference: Prorate down (partial season in reference)
    - Source < Reference: Use raw value with warning

    Args:
        years: Years to load
        column_name: Counting stat column name (e.g., 'SB', 'DPS', 'Framing')
        reference_usage_dict: {MLBAMID: usage} from FanGraphs (PA or Inn)
        source_dir: Path to source data directory
        source_file_pattern: File pattern with {year} placeholder (e.g., 'bp_baserunning_{year}.csv')
        usage_column: Column name for usage in source file ('PA' or 'Inn')
        id_column: Player ID column name (varies by source: 'MLBAMID', 'mlbid', 'player_id')

    Returns:
        dict: {MLBAMID: prorated_value}

    Example:
        >>> # Load SB with PA-based proration
        >>> pa_dict = _load_fangraphs_feature([2025], 'standard', 'PA', 'hitters')  # 300 PA (firsthalf)
        >>> sb_dict = _load_counting_stat_with_proration(
        ...     years=[2025],
        ...     column_name='SB',
        ...     reference_usage_dict=pa_dict,
        ...     source_dir=Path('MLB Player Data/BP_Data/baserunning'),
        ...     source_file_pattern='bp_baserunning_{year}.csv',
        ...     usage_column='PA',
        ...     id_column='mlbid'
        ... )
        >>> # If BP has 600 PA (full season) with 20 SB:
        >>> # Prorated: 20 * (300/600) = 10 SB
    """
    feature_dict = {}

    for year in years:
        file_path = source_dir / source_file_pattern.format(year=year)

        if not file_path.exists():
            continue

        try:
            df = pd.read_csv(file_path)

            # Check required columns exist
            if id_column not in df.columns or column_name not in df.columns or usage_column not in df.columns:
                continue

            for _, row in df.iterrows():
                if pd.notna(row[id_column]) and pd.notna(row[column_name]) and pd.notna(row[usage_column]):
                    try:
                        mlbamid = int(float(row[id_column]))
                        raw_value = float(row[column_name])
                        source_usage = float(row[usage_column])

                        # Get reference usage (from FanGraphs)
                        reference_usage = reference_usage_dict.get(mlbamid)

                        if reference_usage is None or reference_usage <= 0:
                            continue

                        # Calculate alignment ratio
                        usage_ratio = reference_usage / source_usage

                        # Check if aligned (within 10% tolerance)
                        if 0.9 <= usage_ratio <= 1.1:
                            # Aligned - use directly, no proration
                            feature_dict[mlbamid] = raw_value

                        elif usage_ratio < 1.0:
                            # Source has MORE data (full season > partial)
                            # Prorate DOWN to match reference timeframe
                            prorated = raw_value * usage_ratio
                            feature_dict[mlbamid] = prorated

                        else:
                            # Source has LESS data (shouldn't happen often)
                            # Use raw value with warning
                            feature_dict[mlbamid] = raw_value

                    except (ValueError, TypeError):
                        continue

        except Exception as e:
            continue

    return feature_dict


def validate_percentage_scale(feature_dict, feature_name, expected_range=(0, 100)):
    """
    Validate that percentage features are on 0-100 scale, not 0-1.

    Common bug: Forgetting to convert decimals to percentages.
    This catches it immediately.

    Args:
        feature_dict (dict): {MLBAMID: value}
        feature_name (str): Name of feature for error message
        expected_range (tuple): (min, max) expected values

    Raises:
        ValueError: If >50% of values are outside expected range

    Example:
        >>> bb_dict = {660271: 0.082}  # WRONG! Still in decimal
        >>> validate_percentage_scale(bb_dict, 'BB%')
        ValueError: BB%: Likely still in DECIMAL format (should be PERCENTAGE)!
    """
    import numpy as np

    values = np.array(list(feature_dict.values()))

    # Check for decimal format (common bug)
    if expected_range == (0, 100):  # Percentage
        decimal_count = np.sum(values < 1.0)
        if decimal_count > len(values) * 0.5:
            raise ValueError(
                f"{feature_name}: {decimal_count}/{len(values)} values are <1.0. "
                f"Likely still in DECIMAL format (should be PERCENTAGE)!"
            )

    # Check range
    min_val, max_val = values.min(), values.max()
    exp_min, exp_max = expected_range

    if min_val < exp_min or max_val > exp_max * 1.5:  # Allow 50% over max
        print(f"Warning: {feature_name} range [{min_val:.2f}, {max_val:.2f}] "
              f"outside expected [{exp_min}, {exp_max}]")
