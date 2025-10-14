"""
Rookie Detection Features for ROS Prediction

Validates rookie status using MLB thresholds and classifies rookie tier.
NO CEILING APPLIED - let model learn from data.

Adapted from common_modules/rookie_elite_protection.py.
"""

import pandas as pd
from typing import Dict, Optional

# MLB Rookie Thresholds
PITCHER_IP_THRESHOLD = 50.0
HITTER_PA_THRESHOLD = 150.0  # MLB uses <130 AB; ~150 PA equivalent (AB ≈ 0.87*PA)
MIN_CURRENT_IP = 40.0
MIN_CURRENT_PA = 250.0  # Approximately 200 AB equivalent
MIN_WAR_THRESHOLD = 2.5


def validate_rookie_status(
    player_data: Dict,
    historical_data: Optional[pd.DataFrame] = None,
    player_type: str = 'hitter'
) -> Dict:
    """
    Validate rookie status using MLB official thresholds.

    MLB Rookie Definition:
    - Pitchers: <50 IP in all previous MLB seasons
    - Hitters: <130 AB (~150 PA) in all previous MLB seasons

    Args:
        player_data: Current season data with keys:
            - Name
            - Position
            - WAR (or WAR_per_600)
            - IP (for pitchers) or AB/PA (for hitters)
            - Year
        historical_data: Historical data for previous seasons validation
        player_type: 'pitcher' or 'hitter' (default: 'hitter')

    Returns:
        Dictionary with:
        - is_qualifying_rookie: bool
        - is_pitcher: bool
        - meets_minimum_volume: bool
        - meets_war_threshold: bool
        - has_previous_experience: bool
        - total_previous_volume: float

    Example:
        >>> # Judge 2017: Rookie with elite performance
        >>> judge_2017 = {
        ...     'Name': 'Aaron Judge',
        ...     'Position': 'RF',
        ...     'WAR_per_600': 8.1,
        ...     'PA': 542,
        ...     'Year': 2017
        ... }
        >>> history = pd.DataFrame({'Year': [2016], 'PA': [95]})
        >>> result = validate_rookie_status(judge_2017, history, player_type='hitter')
        >>> result['is_qualifying_rookie']
        True
    """
    player_name = player_data.get('Name', 'Unknown')
    current_war = player_data.get('WAR', player_data.get('WAR_per_600', 0.0))

    # Get current season stats
    current_ip = player_data.get('IP', 0.0)
    current_pa = player_data.get('PA', 0.0)

    # Use passed player_type instead of auto-detecting
    is_pitcher = (player_type == 'pitcher')

    # Check minimum thresholds
    if is_pitcher:
        meets_minimum = current_ip >= MIN_CURRENT_IP
    else:
        meets_minimum = current_pa >= MIN_CURRENT_PA

    # Check WAR threshold
    meets_war_threshold = current_war >= MIN_WAR_THRESHOLD

    # Historical validation
    has_previous_experience = False
    total_previous_volume = 0.0

    if historical_data is not None and not historical_data.empty:
        # Look for previous seasons
        previous_seasons = historical_data[
            (historical_data['Name'] == player_name) &
            (historical_data['Year'] < player_data.get('Year', 2025))
        ]

        if not previous_seasons.empty:
            if is_pitcher:
                # Safely access IP column (might not exist in edge cases)
                if 'IP' in previous_seasons.columns:
                    total_previous_volume = previous_seasons['IP'].sum()
                    has_previous_experience = total_previous_volume >= PITCHER_IP_THRESHOLD
                else:
                    total_previous_volume = 0.0
                    has_previous_experience = False
            else:
                # Safely access PA column (might not exist in edge cases)
                if 'PA' in previous_seasons.columns:
                    total_previous_volume = previous_seasons['PA'].sum()
                    has_previous_experience = total_previous_volume >= HITTER_PA_THRESHOLD
                else:
                    total_previous_volume = 0.0
                    has_previous_experience = False

    # Rookie qualification logic
    is_qualifying_rookie = (
        not has_previous_experience and
        meets_minimum and
        meets_war_threshold
    )

    return {
        'is_qualifying_rookie': is_qualifying_rookie,
        'is_pitcher': is_pitcher,
        'meets_minimum_volume': meets_minimum,
        'meets_war_threshold': meets_war_threshold,
        'has_previous_experience': has_previous_experience,
        'total_previous_volume': total_previous_volume
    }


def classify_rookie_tier(war_value: float) -> str:
    """
    Classify rookie into performance tier.

    Tiers:
    - elite_rookie: >=4.0 WAR (Judge 8.1, Trout 10.1)
    - good_rookie: >=3.0 WAR
    - average_rookie: <3.0 WAR

    Args:
        war_value: Current WAR value

    Returns:
        Tier name

    Example:
        >>> classify_rookie_tier(8.1)
        'elite_rookie'
        >>> classify_rookie_tier(3.5)
        'good_rookie'
    """
    if war_value >= 4.0:
        return 'elite_rookie'
    elif war_value >= 3.0:
        return 'good_rookie'
    else:
        return 'average_rookie'


def encode_rookie_tier(tier_name: str, is_rookie: bool) -> int:
    """
    Encode rookie tier to integer.

    Args:
        tier_name: Tier from classify_rookie_tier()
        is_rookie: Whether player is actually a rookie

    Returns:
        0 = not rookie
        1 = average rookie
        2 = good rookie
        3 = elite rookie
    """
    if not is_rookie:
        return 0

    encoding = {
        'elite_rookie': 3,
        'good_rookie': 2,
        'average_rookie': 1
    }
    return encoding.get(tier_name, 0)


def extract_rookie_features(
    player_data: Dict,
    historical_data: Optional[pd.DataFrame] = None,
    player_type: str = 'hitter'
) -> Dict[str, int]:
    """
    Extract all rookie features for ROS model.

    NOTE: NO CEILING APPLIED. Let model learn rookie secondhalf patterns.

    Args:
        player_data: Current season stats
        historical_data: Historical data for validation
        player_type: 'pitcher' or 'hitter' (default: 'hitter')

    Returns:
        Dictionary with 5 rookie features:
        - is_qualifying_rookie: int (0/1)
        - rookie_tier_level: int (0-3)
        - years_experience: int
        - debut_age: int or None
        - is_late_bloomer: int (0/1, debut_age >= 25)

    Example:
        >>> # Judge 2017
        >>> judge_2017 = {
        ...     'Name': 'Aaron Judge',
        ...     'Position': 'RF',
        ...     'WAR_per_600': 8.1,
        ...     'PA': 542,
        ...     'Age': 25,
        ...     'Year': 2017
        ... }
        >>> history = pd.DataFrame({'Year': [2016], 'PA': [95], 'Age': [24]})
        >>> features = extract_rookie_features(judge_2017, history, player_type='hitter')
        >>> features['is_qualifying_rookie']
        1  # True
        >>> features['rookie_tier_level']
        3  # elite_rookie
        >>> features['is_late_bloomer']
        1  # Yes (debut at 25)
    """
    # Validate rookie status
    validation = validate_rookie_status(player_data, historical_data, player_type)

    is_rookie = validation['is_qualifying_rookie']
    current_war = player_data.get('WAR', player_data.get('WAR_per_600', 0.0))

    # Classify tier
    tier_name = classify_rookie_tier(current_war)
    rookie_tier_level = encode_rookie_tier(tier_name, is_rookie)

    # Calculate years of experience
    if historical_data is not None and not historical_data.empty:
        player_history = historical_data[
            historical_data['Name'] == player_data.get('Name', '')
        ]
        years_experience = len(player_history['Year'].unique()) + 1  # +1 for current

        # Debut age (from first year in MLB)
        if len(player_history) > 0:
            first_season = player_history.sort_values('Year').iloc[0]
            age_value = first_season.get('Age', player_data.get('Age', 22))
            debut_age = int(age_value) if age_value is not None and pd.notna(age_value) else 22
        else:
            age_value = player_data.get('Age', 22)
            debut_age = int(age_value) if age_value is not None and pd.notna(age_value) else 22
    else:
        years_experience = 1
        age_value = player_data.get('Age', 22)
        debut_age = int(age_value) if age_value is not None and pd.notna(age_value) else 22

    # Late bloomer check
    is_late_bloomer = int(debut_age >= 25)

    return {
        'is_qualifying_rookie': int(is_rookie),
        'rookie_tier_level': rookie_tier_level,
        'years_experience': years_experience,
        'debut_age': debut_age,
        'is_late_bloomer': is_late_bloomer
    }
