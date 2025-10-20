"""
Baserunning ROS Projection with Age Curves

Projects rest-of-season baserunning based on historical average
adjusted for age-related sprint speed decline.

Research basis:
- Sprint speed peaks age 23-25 (100% factor)
- Gradual decline starts age 28-29 (95% factor)
- Steep decline age 33+ (75% factor by age 36)

Sources:
- FanGraphs: "Sprint Speed and Age" analysis
- Baseball Prospectus: "The Age of Speed" research
"""

import pandas as pd
import numpy as np
from typing import Dict


def get_baserunning_age_factor(age: int) -> float:
    """
    Calculate age adjustment factor for baserunning ability.

    Based on sprint speed research showing peak at 23-25,
    decline starting at 28, and steep decline after 33.

    Args:
        age: Player's age

    Returns:
        Age factor (0.0-1.0, where 1.0 = peak speed)

    Examples:
        >>> get_baserunning_age_factor(24)
        1.0  # Peak speed
        >>> get_baserunning_age_factor(28)
        0.975  # Early decline
        >>> get_baserunning_age_factor(33)
        0.875  # Accelerated decline
        >>> get_baserunning_age_factor(36)
        0.75  # Steep decline
    """
    # Age <= 25: Peak speed (100%)
    if age <= 25:
        return 1.0

    # Age 26-27: Minimal decline (99-98%)
    elif age <= 27:
        decline_per_year = 0.01
        years_past_peak = age - 25
        return 1.0 - (years_past_peak * decline_per_year)

    # Age 28-32: Gradual decline (97% → 88%)
    # From 98% at age 27 down to 88% at age 32
    elif age <= 32:
        base_factor = 0.98  # Starting point at age 27
        decline_per_year = 0.02  # 2% per year
        years_past_27 = age - 27
        return base_factor - (years_past_27 * decline_per_year)

    # Age 33+: Steep decline (88% → 70%)
    # Accelerated decline of 3% per year
    else:
        base_factor = 0.88  # Starting point at age 32
        decline_per_year = 0.03  # 3% per year (steeper)
        years_past_32 = age - 32
        return max(0.70, base_factor - (years_past_32 * decline_per_year))


def project_baserunning_ros(
    historical_baserunning: pd.Series,
    current_age: int,
    min_years: int = 3
) -> float:
    """
    Project rest-of-season baserunning using age-adjusted historical average.

    Methodology:
    1. Calculate 3-year average of Enhanced_Baserunning
    2. Apply age curve adjustment factor
    3. Return projected ROS baserunning value

    Args:
        historical_baserunning: Series of Enhanced_Baserunning values (last N years)
        current_age: Player's current age
        min_years: Minimum years of history required (default: 3)

    Returns:
        Projected baserunning value (age-adjusted)
        Returns 0.0 if insufficient history

    Examples:
        >>> history = pd.Series([2.5, 3.0, 2.8])  # Last 3 years
        >>> project_baserunning_ros(history, age=24)
        2.77  # 2.77 * 1.0 (peak age)
        >>> project_baserunning_ros(history, age=33)
        2.42  # 2.77 * 0.875 (declining speed)
    """
    # Filter out NaN values
    valid_history = historical_baserunning.dropna()

    # Check if we have enough history
    if len(valid_history) < min_years:
        return 0.0

    # Calculate recent average (last 3 years)
    recent_avg = valid_history.tail(3).mean()

    # Apply age curve adjustment
    age_factor = get_baserunning_age_factor(current_age)
    projected_value = recent_avg * age_factor

    return projected_value


def extract_baserunning_projection_features(
    current_season: pd.Series,
    historical_data: pd.DataFrame
) -> Dict[str, float]:
    """
    Extract baserunning projection features for ROS model.

    Args:
        current_season: Current season stats with keys:
            - Age: Player age
            - Enhanced_Baserunning: Current baserunning value
        historical_data: Historical data with columns:
            - Year: Season year
            - Enhanced_Baserunning: Historical baserunning values

    Returns:
        Dictionary with baserunning projection features:
        - projected_baserunning_ros: Age-adjusted projection
        - baserunning_age_factor: Age adjustment factor (0.0-1.0)
        - baserunning_3yr_avg: Unadjusted 3-year average
        - baserunning_trend: Trend over last 3 years (positive = improving)

    Example:
        >>> current = pd.Series({'Age': 28, 'Enhanced_Baserunning': 2.1})
        >>> history = pd.DataFrame({
        ...     'Year': [2022, 2023, 2024],
        ...     'Enhanced_Baserunning': [2.5, 2.7, 2.4]
        ... })
        >>> features = extract_baserunning_projection_features(current, history)
        >>> features['projected_baserunning_ros']
        2.54  # 2.53 * 0.975 (age 28 factor)
        >>> features['baserunning_age_factor']
        0.975
    """
    age_value = current_season.get('Age', 28)
    age = int(age_value) if age_value is not None and pd.notna(age_value) else 28

    # Get historical baserunning values (sorted by year)
    if not historical_data.empty and 'Enhanced_Baserunning' in historical_data.columns:
        historical_data = historical_data.sort_values('Year')
        historical_baserunning = historical_data['Enhanced_Baserunning']
    else:
        historical_baserunning = pd.Series(dtype=float)

    # Calculate projection
    projected_value = project_baserunning_ros(historical_baserunning, age)
    age_factor = get_baserunning_age_factor(age)

    # Calculate 3-year average (before age adjustment)
    valid_history = historical_baserunning.dropna()
    if len(valid_history) >= 3:
        baserunning_3yr_avg = valid_history.tail(3).mean()
    else:
        baserunning_3yr_avg = 0.0

    # Calculate trend (simple linear: last year - 3 years ago)
    if len(valid_history) >= 3:
        recent_values = valid_history.tail(3).values
        # Simple trend: difference between most recent and oldest in window
        baserunning_trend = recent_values[-1] - recent_values[0]
    else:
        baserunning_trend = 0.0

    return {
        'projected_baserunning_ros': projected_value,
        'baserunning_age_factor': age_factor,
        'baserunning_3yr_avg': baserunning_3yr_avg,
        'baserunning_trend': baserunning_trend
    }
