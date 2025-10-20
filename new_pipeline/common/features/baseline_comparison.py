"""
Baseline Comparison Features for ROS Prediction

Compares current performance to healthy baselines (career peak, recent, etc) across ALL model features.
Includes composite feature calculation for historical data.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional

# Import composite calculators
from ..transformers.pitcher_composites import (
    calculate_damage_control_ratio,
    calculate_opportunity_success,
    calculate_strikeout_efficiency,
    calculate_contact_management,
    calculate_strikeout_contact_quality,
    calculate_statcast_launch_quality_index,
    calculate_running_control
)

# Stats to track baselines for
HITTER_BASELINE_STATS = [
    'K%', 'BB%',  # Plate discipline
    'AVG', 'OBP', 'SLG', 'wOPS',  # Slash line + weighted OPS (park-adjusted)
    'GDP',  # Ground into double play
    'Positional_WAR', 'Enhanced_Baserunning', 'Enhanced_Defense'  # WAR components
]

PITCHER_BASE_STATS = [
    'K%', 'BB%', 'SwStr%', 'LOB%', 'Hard%',  # Skill/batted ball
    'ERA', 'GB%', 'HR/FB%',  # Results (park-adjusted)
    'WPA/LI'  # Leverage performance
]

PITCHER_COMPOSITE_STATS = [
    'damage_control_ratio',
    'Opportunity_Success',
    'strikeout_efficiency',
    'contact_management',
    'strikeout_contact_quality',
    'Statcast_Launch_Quality_Index',
    'Running_Control'
]

# Lower is better for these stats
LOWER_IS_BETTER_STATS = {
    'K%_hitter',  # Hitters want low K%
    'BB%_pitcher',  # Pitchers want low BB%
    'ERA',
    'Hard%'
}


def _is_lower_better(stat_name: str, player_type: str) -> bool:
    """Determine if lower values are better for this stat."""
    if player_type == 'hitter' and stat_name == 'K%':
        return True
    if player_type == 'pitcher' and stat_name in ['BB%', 'ERA', 'Hard%']:
        return True
    return False


def _calculate_pitcher_composites_for_row(row: pd.Series) -> Dict[str, float]:
    """
    Calculate all pitcher composite features for a single historical row.

    Args:
        row: DataFrame row with base stats (K%, BB%, LOB%, GB%, HR/FB%, Hard%, etc.)

    Returns:
        Dictionary with composite feature values
    """
    composites = {}

    # Need to get MLBAMID or create dummy
    pid = row.get('MLBAMID', 0)

    # damage_control_ratio
    if 'LOB%' in row and 'HR/FB%' in row and not pd.isna(row['LOB%']) and not pd.isna(row['HR/FB%']):
        result = calculate_damage_control_ratio(
            {pid: row['LOB%']},
            {pid: row['HR/FB%']}
        )
        composites['damage_control_ratio'] = result.get(pid, np.nan)
    else:
        composites['damage_control_ratio'] = np.nan

    # Opportunity_Success
    if all(k in row and not pd.isna(row[k]) for k in ['K%', 'BB%', 'LOB%']):
        result = calculate_opportunity_success(
            {pid: row['K%']},
            {pid: row['BB%']},
            {pid: row['LOB%']}
        )
        composites['Opportunity_Success'] = result.get(pid, np.nan)
    else:
        composites['Opportunity_Success'] = np.nan

    # strikeout_efficiency
    if all(k in row and not pd.isna(row[k]) for k in ['K%', 'BB%']):
        result = calculate_strikeout_efficiency(
            {pid: row['K%']},
            {pid: row['BB%']}
        )
        composites['strikeout_efficiency'] = result.get(pid, np.nan)
    else:
        composites['strikeout_efficiency'] = np.nan

    # contact_management
    if all(k in row and not pd.isna(row[k]) for k in ['GB%', 'BB%']):
        result = calculate_contact_management(
            {pid: row['GB%']},
            {pid: row['BB%']}
        )
        composites['contact_management'] = result.get(pid, np.nan)
    else:
        composites['contact_management'] = np.nan

    # strikeout_contact_quality
    if all(k in row and not pd.isna(row[k]) for k in ['K%', 'Hard%']):
        result = calculate_strikeout_contact_quality(
            {pid: row['K%']},
            {pid: row['Hard%']}
        )
        composites['strikeout_contact_quality'] = result.get(pid, np.nan)
    else:
        composites['strikeout_contact_quality'] = np.nan

    # Statcast_Launch_Quality_Index
    if all(k in row and not pd.isna(row[k]) for k in ['Avg_Hit_Angle', 'Angle_Sweet_Spot_Pct']):
        result = calculate_statcast_launch_quality_index({
            pid: {
                'avg_hit_angle': row['Avg_Hit_Angle'],
                'anglesweetspotpercent': row['Angle_Sweet_Spot_Pct']
            }
        })
        composites['Statcast_Launch_Quality_Index'] = result.get(pid, np.nan)
    else:
        composites['Statcast_Launch_Quality_Index'] = np.nan

    # Running_Control
    if all(k in row and not pd.isna(row[k]) for k in ['CS', 'PK', 'SB', 'BK']):
        composites['Running_Control'] = calculate_running_control(
            int(row.get('CS', 0)),
            int(row.get('PK', 0)),
            int(row.get('SB', 0)),
            int(row.get('BK', 0))
        )
    else:
        composites['Running_Control'] = np.nan

    return composites


def _calculate_wOPS(obp: float, slg: float) -> float:
    """
    Calculate weighted OPS (wOPS).

    wOPS properly weights OBP as 1.8x more important than SLG.
    Formula: (1.8 × OBP + SLG) / 2.8

    Args:
        obp: On-base percentage
        slg: Slugging percentage

    Returns:
        Weighted OPS value (or np.nan if inputs invalid)

    Example:
        >>> _calculate_wOPS(0.400, 0.600)  # Elite hitter
        0.500
        >>> _calculate_wOPS(0.320, 0.450)  # Average hitter
        0.367
    """
    if pd.isna(obp) or pd.isna(slg):
        return np.nan
    return (1.8 * obp + slg) / 2.8


def _add_wOPS_to_historical(
    player_historical: pd.DataFrame,
    player_type: str
) -> pd.DataFrame:
    """
    Add wOPS to historical hitter data.

    Args:
        player_historical: Historical data with OBP and SLG columns
        player_type: 'hitter' or 'pitcher'

    Returns:
        Historical data with wOPS column added
    """
    if player_type != 'hitter' or player_historical.empty:
        return player_historical

    result = player_historical.copy()

    # Calculate wOPS for each row if OBP and SLG exist
    if 'OBP' in result.columns and 'SLG' in result.columns:
        result['wOPS'] = result.apply(
            lambda row: _calculate_wOPS(row['OBP'], row['SLG']),
            axis=1
        )

    return result


def _add_composites_to_historical(
    player_historical: pd.DataFrame,
    player_type: str
) -> pd.DataFrame:
    """
    Add composite features to historical data.

    Args:
        player_historical: Historical data with base stats
        player_type: 'hitter' or 'pitcher'

    Returns:
        Historical data with composite columns added
    """
    if player_type != 'pitcher' or player_historical.empty:
        return player_historical

    result = player_historical.copy()

    # Calculate composites for each row
    for idx, row in result.iterrows():
        composites = _calculate_pitcher_composites_for_row(row)
        for comp_name, comp_value in composites.items():
            result.loc[idx, comp_name] = comp_value

    return result


def _is_qualifying_season(row: pd.Series, player_type: str) -> bool:
    """
    Check if season meets usage thresholds (matches IPFilter/PAFilter logic).

    Args:
        row: Historical season row
        player_type: 'hitter' or 'pitcher'

    Returns:
        True if season qualifies as meaningful sample

    Example:
        >>> # Starter with 15 GS
        >>> row = pd.Series({'GS': 15, 'G': 20, 'IP': 90.0})
        >>> _is_qualifying_season(row, 'pitcher')
        True

        >>> # Reliever with 50 G, 45 IP
        >>> row = pd.Series({'GS': 0, 'G': 50, 'IP': 45.0})
        >>> _is_qualifying_season(row, 'pitcher')
        True

        >>> # Hitter with 400 PA
        >>> row = pd.Series({'PA': 400})
        >>> _is_qualifying_season(row, 'hitter')
        True
    """
    if player_type == 'pitcher':
        ip = row.get('IP', 0)
        gs = row.get('GS', 0)
        g = row.get('G', 0)

        # Any starter qualifies (matches IPFilter line 149)
        if gs > 0:
            return True

        # Relievers need meaningful volume (matches IPFilter line 153)
        if g >= 10 and ip >= 10:
            return True

        # High-volume relievers (matches IPFilter line 157)
        if g >= 20:
            return True

        return False

    else:  # hitter
        pa = row.get('PA', 0)
        # Matches PAFilter default: 50 PA full season, adjusts for partial
        return pa >= 50


def _filter_qualifying_seasons(
    player_historical: pd.DataFrame,
    injury_data: Optional[pd.DataFrame],
    player_type: str
) -> pd.DataFrame:
    """
    Filter historical data to qualifying healthy seasons.

    Uses IPFilter/PAFilter-style usage thresholds + injury filter (when available).

    Args:
        player_historical: Historical data with usage columns
        injury_data: Injury data (optional, only available 2020+)
        player_type: 'hitter' or 'pitcher'

    Returns:
        Filtered DataFrame with only qualifying seasons

    Example:
        >>> history = pd.DataFrame({
        ...     'Year': [2018, 2019, 2020, 2021, 2022],
        ...     'GS': [32, 0, 28, 30, 25],
        ...     'G': [32, 55, 28, 30, 25],
        ...     'IP': [180, 60, 150, 170, 140],
        ...     'ERA': [3.2, 4.5, 2.9, 3.1, 3.8]
        ... })
        >>> # 2019 was reliever year with injury
        >>> injury_df = pd.DataFrame({'Year': [2019]})
        >>> filtered = _filter_qualifying_seasons(history, injury_df, 'pitcher')
        >>> len(filtered)
        4  # 2019 excluded due to injury
    """
    if player_historical.empty:
        return player_historical

    qualifying_rows = []

    for idx, row in player_historical.iterrows():
        # Usage filter (matches pipeline qualification)
        meets_usage = _is_qualifying_season(row, player_type)

        if not meets_usage:
            continue

        # Injury filter (only for 2020+ if data available)
        if injury_data is not None and not injury_data.empty:
            year = row.get('Year', None)
            if year is not None and year >= 2020:
                # Check if this year appears in injury data
                has_injury = (injury_data['Year'] == year).any()
                if has_injury:
                    continue  # Skip injured season

        qualifying_rows.append(row)

    if qualifying_rows:
        return pd.DataFrame(qualifying_rows)
    else:
        return pd.DataFrame()


def calculate_healthy_baseline(
    player_historical: pd.DataFrame,
    stat_name: str,
    injury_data: Optional[pd.DataFrame] = None,
    player_type: str = 'pitcher',
    min_seasons: int = 3,
    lower_is_better: bool = False
) -> float:
    """
    Calculate healthy baseline as best N-year average with adaptive fallback.

    Uses IPFilter/PAFilter-style qualification + injury filtering (when available).
    Implements Solutions A (fallback tiers), B (better healthy filter), C (non-consecutive years).

    Args:
        player_historical: Historical data with usage and stat columns
        stat_name: Stat to analyze
        injury_data: Injury data (optional, only available 2020+)
        player_type: 'hitter' or 'pitcher' (for usage thresholds)
        min_seasons: Minimum seasons for tier 1 (default: 3)
        lower_is_better: If True, use min instead of max (for ERA, K% for hitters, etc.)

    Returns:
        Healthy baseline value (or np.nan if no data)

    Example:
        >>> history = pd.DataFrame({
        ...     'Year': [2019, 2020, 2021, 2022, 2024],  # 2023 injured (skipped)
        ...     'GS': [32, 30, 0, 28, 30],
        ...     'G': [32, 30, 60, 28, 30],
        ...     'IP': [190, 180, 55, 165, 175],
        ...     'ERA': [3.2, 2.9, 4.8, 3.1, 3.4]
        ... })
        >>> injury_df = pd.DataFrame({'Year': [2021]})  # 2021 injured
        >>> # Uses 2019, 2020, 2024 (non-consecutive qualifying years)
        >>> calculate_healthy_baseline(history, 'ERA', injury_df, 'pitcher', lower_is_better=True)
        3.1  # Best 3-year average from qualifying years
    """
    if player_historical.empty or stat_name not in player_historical.columns:
        return np.nan

    # Solution B: Filter to qualifying healthy seasons
    qualifying_seasons = _filter_qualifying_seasons(player_historical, injury_data, player_type)

    if qualifying_seasons.empty:
        return np.nan

    # Sort by year
    qualifying_seasons = qualifying_seasons.sort_values('Year')

    # Solution C: Calculate best N-year average from ANY N qualifying years (non-consecutive OK)
    # Tier 1: Try 3-year average
    if len(qualifying_seasons) >= min_seasons:
        # Calculate all possible 3-year combinations
        stat_values = qualifying_seasons[stat_name].values
        best_avg = None

        # Use rolling window on sorted qualifying years (gaps OK)
        for i in range(len(stat_values) - min_seasons + 1):
            window_avg = stat_values[i:i+min_seasons].mean()
            if best_avg is None:
                best_avg = window_avg
            else:
                if lower_is_better:
                    best_avg = min(best_avg, window_avg)
                else:
                    best_avg = max(best_avg, window_avg)

        return best_avg if not pd.isna(best_avg) else np.nan

    # Solution A: Tier 2 fallback - use 2-year average
    elif len(qualifying_seasons) == 2:
        return qualifying_seasons[stat_name].mean()

    # Solution A: Tier 3 fallback - use single best qualifying season
    elif len(qualifying_seasons) == 1:
        return qualifying_seasons[stat_name].values[0]

    else:
        return np.nan


def calculate_peak_baseline(
    player_historical: pd.DataFrame,
    stat_name: str,
    lower_is_better: bool = False
) -> float:
    """
    Calculate peak performance (single best season).

    Args:
        player_historical: Historical data
        stat_name: Stat to analyze
        lower_is_better: If True, use min instead of max

    Returns:
        Peak value (or np.nan if no data)
    """
    if player_historical.empty or stat_name not in player_historical.columns:
        return np.nan

    if lower_is_better:
        return player_historical[stat_name].min()
    else:
        return player_historical[stat_name].max()


def calculate_recent_baseline(
    player_historical: pd.DataFrame,
    stat_name: str,
    lookback: int = 2
) -> float:
    """
    Calculate recent baseline (last N years average).

    Args:
        player_historical: Historical data
        stat_name: Stat to analyze
        lookback: Years to look back

    Returns:
        Recent average (or np.nan if insufficient data)
    """
    if len(player_historical) < lookback or stat_name not in player_historical.columns:
        return np.nan

    recent = player_historical.tail(lookback)
    return recent[stat_name].mean()


def calculate_career_baseline(
    player_historical: pd.DataFrame,
    stat_name: str
) -> float:
    """
    Calculate career baseline (simple average of ALL qualifying seasons).

    Args:
        player_historical: Historical data (should be pre-filtered to qualifying seasons)
        stat_name: Stat to analyze

    Returns:
        Career average (or np.nan if no data)

    Example:
        >>> history = pd.DataFrame({'ERA': [3.2, 2.8, 4.1, 3.5, 3.9]})
        >>> calculate_career_baseline(history, 'ERA')
        3.5
    """
    if player_historical.empty or stat_name not in player_historical.columns:
        return np.nan

    return player_historical[stat_name].mean()


def calculate_peak_baseline_direct(
    player_historical: pd.DataFrame,
    stat_name: str,
    lower_is_better: bool = False
) -> float:
    """
    Calculate peak performance value (not just for deviation, but as direct feature).

    Args:
        player_historical: Historical data
        stat_name: Stat to analyze
        lower_is_better: If True, use min instead of max

    Returns:
        Peak value (or np.nan if no data)

    Example:
        >>> history = pd.DataFrame({'ERA': [3.2, 2.8, 4.1, 3.5, 3.9]})
        >>> calculate_peak_baseline_direct(history, 'ERA', lower_is_better=True)
        2.8
    """
    if player_historical.empty or stat_name not in player_historical.columns:
        return np.nan

    if lower_is_better:
        return player_historical[stat_name].min()
    else:
        return player_historical[stat_name].max()


def calculate_recent_weighted(
    player_historical: pd.DataFrame,
    stat_name: str
) -> float:
    """
    Calculate weighted recent performance (3-year with fallback).

    Weights most recent years heavier:
    - 3 years: 50% (most recent), 30% (middle), 20% (oldest)
    - 2 years: 60% (most recent), 40% (older)
    - 1 year: 100%

    Args:
        player_historical: Historical data sorted by Year
        stat_name: Stat to analyze

    Returns:
        Weighted recent average (or np.nan if no data)

    Example:
        >>> history = pd.DataFrame({
        ...     'Year': [2020, 2021, 2022, 2023, 2024],
        ...     'ERA': [4.5, 3.2, 2.9, 3.1, 3.4]
        ... })
        >>> # Last 3 years: 2022 (2.9), 2023 (3.1), 2024 (3.4)
        >>> # Weighted: 0.2*2.9 + 0.3*3.1 + 0.5*3.4 = 3.21
        >>> calculate_recent_weighted(history, 'ERA')
        3.21
    """
    if player_historical.empty or stat_name not in player_historical.columns:
        return np.nan

    # Get most recent years (already sorted in extract_baseline_features)
    recent = player_historical.tail(3)

    if len(recent) == 3:
        # 3 years: 50%, 30%, 20% (newest to oldest)
        values = recent[stat_name].values
        weights = np.array([0.2, 0.3, 0.5])
        return np.average(values, weights=weights)

    elif len(recent) == 2:
        # 2 years: 60%, 40%
        values = recent[stat_name].values
        weights = np.array([0.4, 0.6])
        return np.average(values, weights=weights)

    elif len(recent) == 1:
        # 1 year: 100%
        return recent[stat_name].values[0]

    else:
        return np.nan


def extract_baseline_features(
    player_historical: pd.DataFrame,
    current_season: pd.Series,
    player_type: str = 'hitter',
    injury_data: Optional[pd.DataFrame] = None
) -> Dict[str, float]:
    """
    Extract all baseline comparison features for ROS model.

    Tracks baselines for ALL model features including composites.
    Generates 8 features per stat: career, healthy, peak, recent_3yr, vs_healthy, vs_recent, vs_peak, maintenance_ratio.

    Args:
        player_historical: Historical data with columns:
            - Year
            - All base stats (K%, BB%, AVG, OBP, SLG, ERA, etc.)
            - Usage columns (PA, IP, G, GS)
            - WAR_per_600 or WAR_per_162
        current_season: Current stats with same columns
        player_type: 'hitter' or 'pitcher'
        injury_data: Injury data (optional, only available 2020+)

    Returns:
        Dictionary with baseline features for all tracked stats.
        Format: career_{stat}, healthy_{stat}, peak_{stat}, recent_3yr_{stat},
                {stat}_vs_healthy, {stat}_vs_recent, {stat}_vs_peak, maintenance_ratio_{stat}

    Example:
        >>> history = pd.DataFrame({
        ...     'Year': [2019, 2020, 2021, 2022, 2023, 2024],
        ...     'K%': [17.0, 16.5, 17.2, 18.0, 17.5, 17.8],
        ...     'BB%': [18.5, 19.0, 18.8, 18.2, 17.5, 18.1],
        ...     'AVG': [0.292, 0.351, 0.313, 0.288, 0.275, 0.288],
        ...     'PA': [650, 580, 600, 620, 590, 610],
        ...     'WAR_per_600': [5.8, 4.1, 5.3, 5.5, 4.8, 5.2]
        ... })
        >>> current = pd.Series({'K%': 17.2, 'BB%': 18.9, 'AVG': 0.291, 'WAR_per_600': 7.0})
        >>> features = extract_baseline_features(history, current, 'hitter')
        >>> len(features)  # Multiple stats × 8 metrics each
        64+
    """
    features = {}

    # Add composites to historical data and current season if pitcher (BEFORE filtering)
    if player_type == 'pitcher':
        player_historical = _add_composites_to_historical(player_historical, player_type)

        # Also calculate composites for current season
        current_composites = _calculate_pitcher_composites_for_row(current_season)
        # Create a new series with both base stats and composites
        current_season_with_composites = current_season.copy()
        for comp_name, comp_value in current_composites.items():
            current_season_with_composites[comp_name] = comp_value
        current_season = current_season_with_composites

    # Add wOPS to historical data and current season if hitter (BEFORE filtering)
    if player_type == 'hitter':
        player_historical = _add_wOPS_to_historical(player_historical, player_type)

        # Also calculate wOPS for current season
        if 'OBP' in current_season.index and 'SLG' in current_season.index:
            current_season_with_wOPS = current_season.copy()
            current_season_with_wOPS['wOPS'] = _calculate_wOPS(
                current_season['OBP'],
                current_season['SLG']
            )
            current_season = current_season_with_wOPS

    # NOW filter historical data to qualifying seasons (after adding wOPS/composites)
    qualifying_historical = _filter_qualifying_seasons(player_historical, injury_data, player_type)

    # Handle rookies/players with no history: use current season as baseline
    is_rookie = qualifying_historical.empty

    # Determine which stats to track
    if player_type == 'pitcher':
        stat_list = PITCHER_BASE_STATS + PITCHER_COMPOSITE_STATS
    else:
        stat_list = HITTER_BASELINE_STATS

    # Track baselines for each stat
    for stat_name in stat_list:
        # Check if stat exists in current season
        current_value = current_season.get(stat_name)
        if pd.isna(current_value):
            # Skip stats not available
            continue

        # Determine if lower is better
        lower_is_better = _is_lower_better(stat_name, player_type)

        # Calculate baselines (using qualifying_historical for career/healthy/peak/recent)
        # For rookies with no history, use current season values as baselines
        if is_rookie:
            # Rookie: current season IS the baseline
            career_baseline = current_value
            healthy_baseline = current_value
            peak_baseline = current_value
            recent_weighted = current_value
        else:
            # Veteran: calculate from historical data
            career_baseline = calculate_career_baseline(qualifying_historical, stat_name)
            healthy_baseline = calculate_healthy_baseline(
                player_historical, stat_name, injury_data, player_type, lower_is_better=lower_is_better
            )
            peak_baseline = calculate_peak_baseline_direct(
                qualifying_historical, stat_name, lower_is_better=lower_is_better
            )
            recent_weighted = calculate_recent_weighted(qualifying_historical, stat_name)

        # Old peak baseline for vs_peak calculation (backward compat)
        peak_baseline_old = calculate_peak_baseline(
            player_historical, stat_name, lower_is_better=lower_is_better
        )

        # Calculate deviations
        if is_rookie:
            # Rookies: no history, so no deviation from baseline
            vs_healthy = 0.0
            vs_recent = 0.0
            vs_peak = 0.0
            maintenance = 1.0
        else:
            # Veterans: calculate deviations from historical baselines
            if not pd.isna(healthy_baseline):
                vs_healthy = current_value - healthy_baseline
            else:
                vs_healthy = 0.0
                healthy_baseline = current_value  # Default to current

            if not pd.isna(recent_weighted):
                vs_recent = current_value - recent_weighted
            else:
                vs_recent = 0.0

            if not pd.isna(peak_baseline):
                vs_peak = current_value - peak_baseline
            else:
                vs_peak = 0.0

            # Maintenance ratio (recent / peak)
            if not pd.isna(peak_baseline) and abs(peak_baseline) > 0.001:
                if not pd.isna(recent_weighted):
                    maintenance = recent_weighted / peak_baseline
                else:
                    maintenance = 1.0
            else:
                maintenance = 1.0

        # For lower-is-better stats, invert deviations so positive = good
        if lower_is_better:
            vs_healthy = -vs_healthy
            vs_recent = -vs_recent
            vs_peak = -vs_peak

        # Store features (8 per stat)
        # Use sanitized stat names (replace special chars)
        safe_stat_name = stat_name.replace('%', 'Pct').replace('/', '_')

        features[f'career_{safe_stat_name}'] = career_baseline
        features[f'healthy_{safe_stat_name}'] = healthy_baseline
        features[f'peak_{safe_stat_name}'] = peak_baseline
        features[f'recent_3yr_{safe_stat_name}'] = recent_weighted
        features[f'{safe_stat_name}_vs_healthy'] = vs_healthy
        features[f'{safe_stat_name}_vs_recent'] = vs_recent
        features[f'{safe_stat_name}_vs_peak'] = vs_peak
        features[f'maintenance_ratio_{safe_stat_name}'] = maintenance

    return features
