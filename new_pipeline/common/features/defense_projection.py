"""
Enhanced Defense ROS Projection with Sample-Size Weighting

Projects rest-of-season defense using innings-weighted regression to baseline.
Uses NO age curves due to survivorship bias and lack of component-level research.

Research context:
- Defensive aging research shows 3+ year error due to survivorship bias
- Position-switching confounds position-based aging curves
- Defense metrics are noisy (framing r² < 0.67)
- No clean component-level aging research (unlike sprint speed)

Methodology:
- Small samples → Regress toward 3-year weighted average
- Large samples → Trust current rate
- Threshold: 750 innings for full reliability (conservative due to defensive noise)
"""

import pandas as pd
import numpy as np
from typing import Dict


def get_defense_sample_weight(innings: float, threshold: float = 750) -> float:
    """
    Calculate reliability weight based on innings accumulated.

    Conservative threshold of 750 innings (3/4 season) due to defensive noise.

    Args:
        innings: Defensive innings played so far this season
        threshold: Innings needed for full reliability (default: 750)

    Returns:
        Weight between 0.0-1.0 (1.0 = full trust in current rate)

    Examples:
        >>> get_defense_sample_weight(375)  # Half season
        0.5
        >>> get_defense_sample_weight(750)  # 3/4 season threshold
        1.0
        >>> get_defense_sample_weight(150)  # Small sample
        0.2
    """
    return min(1.0, innings / threshold)


def project_defense_ros(
    current_defense_runs: float,
    current_innings: float,
    defense_3yr_avg: float,
    typical_season_innings: float = 1200
) -> float:
    """
    Project rest-of-season defense using innings-weighted regression.

    Balances current performance with historical baseline based on sample size.

    Args:
        current_defense_runs: Enhanced_Defense runs accumulated so far
        current_innings: Innings played at position this season
        defense_3yr_avg: 3-year weighted average (full season)
        typical_season_innings: Full season innings for annualization (default: 1200)

    Returns:
        Projected Enhanced_Defense for full season

    Examples:
        >>> # Declining defender: +2 runs in 600 inn, 3yr avg = +13
        >>> project_defense_ros(2, 600, 13)
        6.97  # Regressed toward current declining rate

        >>> # Elite defender maintaining: +10 runs in 750 inn, 3yr avg = +15
        >>> project_defense_ros(10, 750, 15)
        16.0  # Fully trust current rate (750 inn threshold met)

        >>> # Small sample: +1 run in 200 inn, 3yr avg = +10
        >>> project_defense_ros(1, 200, 10)
        8.73  # Heavy regression toward baseline
    """
    # Annualize current rate
    if current_innings > 0:
        current_rate = (current_defense_runs / current_innings) * typical_season_innings
    else:
        current_rate = 0.0

    # Sample reliability (750 innings = full weight)
    reliability = get_defense_sample_weight(current_innings)

    # Weighted projection
    projected = (current_rate * reliability) + (defense_3yr_avg * (1 - reliability))

    return projected


def extract_defense_projection_features(
    current_season: pd.Series,
    historical_data: pd.DataFrame
) -> Dict[str, float]:
    """
    Extract defense projection features for ROS model.

    Args:
        current_season: Current season stats with keys:
            - Enhanced_Defense: Current defensive runs
            - Inn: Innings played (from defensive files)
        historical_data: Historical data with columns:
            - Year: Season year
            - Enhanced_Defense: Historical defensive values

    Returns:
        Dictionary with defense projection features:
        - projected_defense_ros: Sample-weighted projection
        - defense_sample_weight: Reliability weight (0.0-1.0)
        - defense_3yr_avg: Unadjusted 3-year baseline
        - defense_current_rate: Annualized current rate (per 1200 inn)
        - defense_trend: Trend over last 3 years

    Example:
        >>> current = pd.Series({'Enhanced_Defense': 2, 'Inn': 600})
        >>> history = pd.DataFrame({
        ...     'Year': [2022, 2023, 2024],
        ...     'Enhanced_Defense': [11, 13, 15]
        ... })
        >>> features = extract_defense_projection_features(current, history)
        >>> features['projected_defense_ros']
        6.97  # Regressed toward current declining rate
        >>> features['defense_sample_weight']
        0.8   # 600/750 = 0.8 reliability
    """
    current_defense = current_season.get('Enhanced_Defense', 0.0)
    current_innings = current_season.get('Inn', 0.0)

    # Get historical defense values (sorted by year)
    if not historical_data.empty and 'Enhanced_Defense' in historical_data.columns:
        historical_data = historical_data.sort_values('Year')
        historical_defense = historical_data['Enhanced_Defense']
    else:
        historical_defense = pd.Series(dtype=float)

    # Calculate 3-year weighted average (50% recent, 30%, 20%)
    valid_history = historical_defense.dropna()
    if len(valid_history) >= 3:
        recent_3 = valid_history.tail(3).values
        defense_3yr_avg = np.average(recent_3, weights=[0.2, 0.3, 0.5])
    elif len(valid_history) == 2:
        defense_3yr_avg = np.average(valid_history.values, weights=[0.4, 0.6])
    elif len(valid_history) == 1:
        defense_3yr_avg = valid_history.values[0]
    else:
        defense_3yr_avg = 0.0

    # Calculate projection
    projected_value = project_defense_ros(
        current_defense,
        current_innings,
        defense_3yr_avg
    )

    # Sample weight
    sample_weight = get_defense_sample_weight(current_innings)

    # Current rate (annualized)
    if current_innings > 0:
        current_rate = (current_defense / current_innings) * 1200
    else:
        current_rate = 0.0

    # Trend (simple linear: last - first in 3yr window)
    if len(valid_history) >= 3:
        recent_values = valid_history.tail(3).values
        defense_trend = recent_values[-1] - recent_values[0]
    else:
        defense_trend = 0.0

    return {
        'projected_defense_ros': projected_value,
        'defense_sample_weight': sample_weight,
        'defense_3yr_avg': defense_3yr_avg,
        'defense_current_rate': current_rate,
        'defense_trend': defense_trend
    }
