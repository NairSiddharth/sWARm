"""
Age Curve Features for ROS Prediction

Implements Dynasty Guru age curve methodology with logarithmic growth,
peak range 26-29, and position-specific decline rates.

Adapted from future_season_modules/future_projections.py.
"""

import pandas as pd
import numpy as np
from typing import Dict

# Position-specific aging parameters (Dynasty Guru research)
POSITION_CURVES = {
    'C': {'peak': 26, 'decline_rate': 0.035, 'career_length': 8},
    'SS': {'peak': 27, 'decline_rate': 0.025, 'career_length': 10},
    '2B': {'peak': 27, 'decline_rate': 0.025, 'career_length': 9},
    '3B': {'peak': 28, 'decline_rate': 0.020, 'career_length': 10},
    '1B': {'peak': 29, 'decline_rate': 0.015, 'career_length': 11},
    'LF': {'peak': 28, 'decline_rate': 0.020, 'career_length': 10},
    'CF': {'peak': 27, 'decline_rate': 0.025, 'career_length': 9},
    'RF': {'peak': 28, 'decline_rate': 0.020, 'career_length': 10},
    'DH': {'peak': 30, 'decline_rate': 0.015, 'career_length': 8},
    'P': {'peak': 27, 'decline_rate': 0.030, 'career_length': 7}
}

# Age-performance tiers
AGE_PERFORMANCE_TIERS = {
    'young_elite': 0,       # <26, elite performer
    'young_average': 1,     # <26, average
    'prime_elite': 2,       # 26-29, elite
    'prime_average': 3,     # 26-29, average
    'veteran_elite': 4,     # 30-33, elite
    'veteran_average': 5,   # 30-33, average
    'late_career_elite': 6, # 34+, elite
    'late_career_average': 7  # 34+, average
}


def classify_age_phase(age: int) -> int:
    """
    Classify age into phase.

    Phases:
    0 = growth (20-23)
    1 = development (24-25)
    2 = prime_plateau (26-29)
    3 = decline (30-33)
    4 = late_career (34+)

    Args:
        age: Player's age

    Returns:
        Phase integer (0-4)
    """
    if age < 24:
        return 0  # growth
    elif age < 26:
        return 1  # development
    elif age <= 29:
        return 2  # prime_plateau
    elif age <= 33:
        return 3  # decline
    else:
        return 4  # late_career


def classify_age_performance_tier(age: int, war_value: float) -> str:
    """
    Classify player into age-performance matrix.

    Args:
        age: Player's age
        war_value: Current WAR rate

    Returns:
        Tier name from AGE_PERFORMANCE_TIERS

    Example:
        >>> classify_age_performance_tier(28, 8.0)
        'prime_elite'
        >>> classify_age_performance_tier(35, 4.5)
        'late_career_elite'
    """
    is_elite = war_value >= 4.0

    if age < 26:
        return 'young_elite' if is_elite else 'young_average'
    elif age <= 29:
        return 'prime_elite' if is_elite else 'prime_average'
    elif age <= 33:
        return 'veteran_elite' if is_elite else 'veteran_average'
    else:
        return 'late_career_elite' if is_elite else 'late_career_average'


def calculate_base_age_factor(age: int, position: str) -> float:
    """
    Calculate base age factor using Dynasty Guru methodology.

    Key features:
    - Logarithmic growth for ages 20-24
    - Peak plateau at 26-29
    - Position-specific decline after 30

    Args:
        age: Player's age
        position: Position code (C, 1B, 2B, etc.)

    Returns:
        Age factor (0.0-1.0, where 1.0 = peak)

    Example:
        >>> calculate_base_age_factor(28, 'RF')
        1.0  # Peak
        >>> calculate_base_age_factor(22, 'RF')
        0.82  # Growing
        >>> calculate_base_age_factor(34, 'RF')
        0.89  # Declining
    """
    # Get position curve
    pos_curve = POSITION_CURVES.get(position, POSITION_CURVES['CF'])

    # Ages < 20: Conservative baseline
    if age < 20:
        return 0.70

    # Ages 20-24: Logarithmic growth
    elif age < 24:
        age_progress = (age - 20) / 4.0  # 0-1 scale
        log_factor = np.log1p(age_progress) / np.log1p(1.0)
        return 0.70 + (0.25 * log_factor)  # 70% → 95%

    # Ages 24-25: Continued improvement
    elif age < 26:
        base_factor = 0.95
        improvement = (age - 24) * 0.025  # 2.5% per year
        return base_factor + improvement

    # Ages 26-29: Peak plateau with slight variation
    elif 26 <= age <= 29:
        range_position = (age - 26) / 3.0  # 0-1 scale
        peak_variation = 0.03 * (1 - 4 * (range_position - 0.5)**2)
        return 1.0 + peak_variation  # Slight parabola

    # Ages 30-31: Gentle decline
    elif age <= 31:
        years_past_peak = age - 29
        return 1.0 - (years_past_peak * 0.015)

    # Ages 32+: Position-based decline
    else:
        base_decline = 1.0 - 2 * 0.015  # From ages 30-31
        years_past_31 = age - 31
        return max(0.3, base_decline - (years_past_31 * pos_curve['decline_rate']))


def adjust_peak_for_late_bloomer(debut_age: int) -> float:
    """
    Adjust peak age for late bloomers.

    Late bloomers (debut age ≥25) have shifted peak ages.

    Args:
        debut_age: Age at MLB debut

    Returns:
        Peak age adjustment (years to add)

    Example:
        >>> adjust_peak_for_late_bloomer(25)
        1.5  # Judge: peak shifts from 28 to 29.5
        >>> adjust_peak_for_late_bloomer(22)
        0.0  # No adjustment
    """
    if debut_age >= 25:
        return (debut_age - 22) * 0.5
    return 0.0


def extract_age_features(
    current_season: pd.Series,
    player_historical: pd.DataFrame,
    position: str
) -> Dict[str, float]:
    """
    Extract all age curve features for ROS model.

    Args:
        current_season: Current stats with keys:
            - Age
            - WAR_per_600 or WAR_per_162
        player_historical: Historical data with Year, Age columns
        position: Position code

    Returns:
        Dictionary with 8 age features:
        - age: int
        - age_phase: int (0-4)
        - position_peak_age: float
        - years_from_peak: float
        - adjusted_peak_age: float
        - age_performance_tier_encoded: int (0-7)
        - base_age_factor: float (0.0-1.0)
        - position_decline_rate: float

    Example:
        >>> # Judge age 33 (late bloomer)
        >>> current = pd.Series({'Age': 33, 'WAR_per_600': 10.5})
        >>> history = pd.DataFrame({'Year': range(2016, 2025), 'Age': range(24, 33)})
        >>> features = extract_age_features(current, history, 'RF')
        >>> features['age']
        33
        >>> features['adjusted_peak_age']
        29.5  # +1.5 for late bloomer
        >>> features['age_performance_tier_encoded']
        4  # veteran_elite
    """
    age_value = current_season.get('Age', 28)
    age = int(age_value) if age_value is not None and pd.notna(age_value) else 28
    war_value = current_season.get('WAR_per_600', current_season.get('WAR_per_162', 0.0))

    # Get position curve
    pos_curve = POSITION_CURVES.get(position, POSITION_CURVES['CF'])
    position_peak_age = pos_curve['peak']
    position_decline_rate = pos_curve['decline_rate']

    # Determine debut age
    if not player_historical.empty:
        first_age = player_historical.sort_values('Year').iloc[0]['Age']
        debut_age = int(first_age) if first_age is not None and pd.notna(first_age) else max(age - 2, 22)
    else:
        debut_age = max(age - 2, 22)  # Estimate

    # Adjust peak for late bloomer
    peak_adjustment = adjust_peak_for_late_bloomer(debut_age)
    adjusted_peak_age = position_peak_age + peak_adjustment

    # Years from peak
    years_from_peak = age - adjusted_peak_age

    # Age phase
    age_phase = classify_age_phase(age)

    # Age-performance tier
    tier_name = classify_age_performance_tier(age, war_value)
    tier_encoded = AGE_PERFORMANCE_TIERS[tier_name]

    # Base age factor
    base_age_factor = calculate_base_age_factor(age, position)

    return {
        'age': age,
        'age_phase': age_phase,
        'position_peak_age': position_peak_age,
        'years_from_peak': years_from_peak,
        'adjusted_peak_age': adjusted_peak_age,
        'age_performance_tier_encoded': tier_encoded,
        'base_age_factor': base_age_factor,
        'position_decline_rate': position_decline_rate
    }
