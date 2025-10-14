"""
Elite Player Detection Features for ROS Prediction

Extracts elite tier classification, injury compromise detection, and decline patterns.
Adapted from common_modules/elite_adjustment_base.py for use as model features.
"""

import pandas as pd
import numpy as np
from typing import Dict

# WAR tier thresholds (from elite_adjustment_base.py)
WAR_TIERS = {
    'mvp_level': 6.0,
    'superstar': 5.0,
    'all_star': 4.0,
    'good_player': 3.0,
    'solid_starter': 2.0,
    'role_player': 1.0,
    'replacement': 0.0
}


def _get_war_rate_column(df: pd.DataFrame) -> str:
    """
    Detect which WAR rate column is available.

    Hitters use WAR_per_600, pitchers use WAR_per_162.

    Args:
        df: DataFrame to check for WAR rate column

    Returns:
        Column name ('WAR_per_600' or 'WAR_per_162')

    Raises:
        ValueError: If neither column found

    Example:
        >>> hitter_df = pd.DataFrame({'WAR_per_600': [5.0, 6.0]})
        >>> _get_war_rate_column(hitter_df)
        'WAR_per_600'

        >>> pitcher_df = pd.DataFrame({'WAR_per_162': [3.0, 4.0]})
        >>> _get_war_rate_column(pitcher_df)
        'WAR_per_162'
    """
    if 'WAR_per_600' in df.columns:
        return 'WAR_per_600'
    elif 'WAR_per_162' in df.columns:
        return 'WAR_per_162'
    else:
        raise ValueError("No WAR rate column found (expected WAR_per_600 or WAR_per_162)")


def get_war_tier(war_value: float) -> str:
    """
    Get WAR tier classification for player.

    Args:
        war_value: Player's WAR value (normalized rate)

    Returns:
        Tier name (e.g., 'mvp_level', 'all_star', 'replacement')

    Example:
        >>> get_war_tier(8.1)
        'mvp_level'
        >>> get_war_tier(4.5)
        'all_star'
    """
    for tier, threshold in WAR_TIERS.items():
        if war_value >= threshold:
            return tier
    return 'replacement'


def encode_war_tier(tier_name: str) -> int:
    """
    Encode WAR tier name to integer for model input.

    Args:
        tier_name: Tier name from get_war_tier()

    Returns:
        Integer 0-6 (higher = better)
    """
    encoding = {
        'mvp_level': 6,
        'superstar': 5,
        'all_star': 4,
        'good_player': 3,
        'solid_starter': 2,
        'role_player': 1,
        'replacement': 0
    }
    return encoding.get(tier_name, 0)


def is_injury_compromised_legend(
    player_historical: pd.DataFrame,
    current_war: float
) -> bool:
    """
    Detect injury-compromised legend (Trout-type).

    Criteria:
    - Had elite peak (≥6.0 WAR)
    - Recent 2-year average <40% of peak
    - Current performance <2.0 WAR

    Args:
        player_historical: Historical WAR data
        current_war: Current season WAR rate

    Returns:
        True if injury-compromised legend

    Example:
        >>> # Trout: peak 10.8, recent avg 3.0, current 1.4
        >>> history = pd.DataFrame({'WAR_per_600': [10.8, 10.2, 8.3, 0.0, 4.6, 1.4]})
        >>> is_injury_compromised_legend(history, 1.4)
        True

        >>> # Judge: peak 11.4, recent avg 8.15, current 10.5
        >>> history = pd.DataFrame({'WAR_per_600': [8.1, 4.2, 6.2, 11.4, 5.5, 10.8]})
        >>> is_injury_compromised_legend(history, 10.5)
        False
    """
    if len(player_historical) < 4:
        return False

    # Detect WAR rate column (WAR_per_600 for hitters, WAR_per_162 for pitchers)
    war_col = _get_war_rate_column(player_historical)

    # Check for historical elite peak
    peak_war = player_historical[war_col].max()

    # Recent 2-year average
    recent_wars = player_historical.tail(2)[war_col].mean()

    # Criteria
    had_elite_peak = peak_war >= 6.0
    significant_recent_decline = recent_wars < peak_war * 0.4
    currently_struggling = current_war < 2.0

    return had_elite_peak and significant_recent_decline and currently_struggling


def is_declining_veteran(
    player_historical: pd.DataFrame,
    current_war: float,
    current_age: int = None
) -> bool:
    """
    Detect declining veteran using age gate + weighted recent performance.

    Age gate (32+) prevents flagging injured prime-age players (Acuña).
    Weighted average handles injury recovery spikes better than slope.

    Criteria:
    - Must be 32+ years old (age gate)
    - Weighted 3-year average < 60% of peak

    Weights: [0.2, 0.3, 0.5] for [oldest, middle, newest]

    Args:
        player_historical: Historical WAR data
        current_war: Current season WAR rate
        current_age: Player's current age (required)

    Returns:
        True if declining veteran

    Example:
        >>> # Trout (age 33): recent [2.9, 0.9, 1.8], peak 10.1
        >>> # weighted_avg = 1.75 < 6.06 -> True
        >>> # Acuña (age 27): age < 32 -> False (protected)
    """
    if len(player_historical) < 4:
        return False

    # Age gate: Must be past prime (32+) to be declining veteran
    # Protects Acuña (27) and other injured prime-age stars
    if current_age is None or current_age < 32:
        return False

    # Detect WAR rate column (WAR_per_600 for hitters, WAR_per_162 for pitchers)
    war_col = _get_war_rate_column(player_historical)

    # Get recent 3 years
    recent_3 = player_historical.tail(3)[war_col].values
    if len(recent_3) < 3:
        return False

    # Weighted average: recent years weighted heavier
    # Handles injury recovery spikes better than simple slope
    weights = np.array([0.2, 0.3, 0.5])  # [oldest, middle, newest]
    weighted_avg = np.average(recent_3, weights=weights)

    # Peak career WAR
    peak_war = player_historical[war_col].max()

    # Declining if weighted average significantly below peak
    return weighted_avg < peak_war * 0.6


def is_consistent_elite(
    player_historical: pd.DataFrame,
    current_war: float
) -> bool:
    """
    Detect consistent elite profile (Soto-type).

    Criteria:
    - ≥3 elite seasons (≥4.0 WAR)
    - ≥60% of seasons are elite
    - Currently performing well (≥3.0 WAR)

    Args:
        player_historical: Historical WAR data
        current_war: Current season WAR rate

    Returns:
        True if consistently elite
    """
    if len(player_historical) < 3:
        return False

    # Detect WAR rate column (WAR_per_600 for hitters, WAR_per_162 for pitchers)
    war_col = _get_war_rate_column(player_historical)

    # Count elite seasons
    elite_seasons = (player_historical[war_col] >= 4.0).sum()
    total_seasons = len(player_historical)
    elite_rate = elite_seasons / total_seasons

    # Criteria
    multiple_elite = elite_seasons >= 3
    consistent = elite_rate >= 0.6
    current_good = current_war >= 3.0

    return multiple_elite and consistent and current_good


def calculate_trajectory(player_historical: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate 5-year trajectory slope and consistency.

    Args:
        player_historical: Historical WAR data

    Returns:
        Dictionary with:
        - trajectory_slope: Linear trend (WAR change per year)
        - trajectory_r_squared: Trend consistency (0-1)

    Example:
        >>> # Judge: improving trend
        >>> history = pd.DataFrame({'WAR_per_600': [8.1, 4.2, 4.6, 6.2, 11.4, 5.5, 10.8]})
        >>> calculate_trajectory(history)
        {'trajectory_slope': 0.3, 'trajectory_r_squared': 0.65}
    """
    if len(player_historical) < 5:
        return {'trajectory_slope': 0.0, 'trajectory_r_squared': 0.0}

    # Detect WAR rate column (WAR_per_600 for hitters, WAR_per_162 for pitchers)
    war_col = _get_war_rate_column(player_historical)

    # Use last 5 years
    recent_wars = player_historical.tail(5)[war_col].values
    x = np.arange(len(recent_wars))

    # Fit linear trend
    coeffs = np.polyfit(x, recent_wars, 1)
    slope = coeffs[0]

    # Calculate R²
    predictions = np.polyval(coeffs, x)
    ss_res = np.sum((recent_wars - predictions) ** 2)
    ss_tot = np.sum((recent_wars - recent_wars.mean()) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    return {
        'trajectory_slope': slope,
        'trajectory_r_squared': max(0.0, r_squared)  # Clamp to [0, 1]
    }


def extract_elite_features(
    player_historical: pd.DataFrame,
    current_season: pd.Series
) -> Dict[str, float]:
    """
    Extract all elite detection features for ROS model.

    Args:
        player_historical: Historical data with columns:
            - Year
            - WAR_per_600 (normalized WAR rate)
            - Games (optional, for filtering)
            - injury_flag (optional, 0/1)
        current_season: Current stats with keys:
            - WAR_per_600
            - Age
            - Position

    Returns:
        Dictionary with 9 elite features:
        - elite_tier_level: int (0-6)
        - is_injury_compromised_legend: int (0/1)
        - is_declining_veteran: int (0/1)
        - is_consistent_elite: int (0/1)
        - peak_WAR: float
        - recent_2yr_avg: float
        - deviation_from_peak: float
        - trajectory_slope: float
        - trajectory_r_squared: float

    Example:
        >>> # Judge (healthy elite)
        >>> history = pd.DataFrame({
        ...     'Year': [2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024],
        ...     'WAR_per_600': [8.1, 4.2, 4.6, 1.9, 6.2, 11.4, 5.5, 10.8]
        ... })
        >>> current = pd.Series({'WAR_per_600': 10.5, 'Age': 33, 'Position': 'RF'})
        >>> features = extract_elite_features(history, current)
        >>> features['elite_tier_level']
        6  # mvp_level
        >>> features['is_injury_compromised_legend']
        0  # No
        >>> features['is_declining_veteran']
        0  # No
    """
    # Detect WAR rate column from historical data (or use hitter default)
    if len(player_historical) > 0:
        war_col = _get_war_rate_column(player_historical)
    else:
        war_col = 'WAR_per_600'  # Default to hitter scale

    # Get current WAR (try both columns for current_season)
    current_war = current_season.get('WAR_per_600', current_season.get('WAR_per_162', 0.0))

    # Tier classification
    tier_name = get_war_tier(current_war)
    elite_tier_level = encode_war_tier(tier_name)

    # Profile detection
    age_value = current_season.get('Age', 30)
    current_age = int(age_value) if age_value is not None and pd.notna(age_value) else 30
    injury_compromised = int(is_injury_compromised_legend(player_historical, current_war))
    declining = int(is_declining_veteran(player_historical, current_war, current_age))
    consistent = int(is_consistent_elite(player_historical, current_war))

    # Historical context
    peak_WAR = player_historical[war_col].max() if len(player_historical) > 0 else current_war
    recent_2yr_avg = player_historical.tail(2)[war_col].mean() if len(player_historical) >= 2 else current_war
    deviation_from_peak = (current_war - peak_WAR) / peak_WAR if peak_WAR > 0 else 0.0

    # Trajectory
    trajectory = calculate_trajectory(player_historical)

    return {
        'elite_tier_level': elite_tier_level,
        'is_injury_compromised_legend': injury_compromised,
        'is_declining_veteran': declining,
        'is_consistent_elite': consistent,
        'peak_WAR': peak_WAR,
        'recent_2yr_avg': recent_2yr_avg,
        'deviation_from_peak': deviation_from_peak,
        'trajectory_slope': trajectory['trajectory_slope'],
        'trajectory_r_squared': trajectory['trajectory_r_squared']
    }
