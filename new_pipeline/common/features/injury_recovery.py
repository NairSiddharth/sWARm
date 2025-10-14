"""
Injury Recovery Features for ROS Prediction

Calculates injury-based features including recovery factors, severity, and timing.
Adapted from current_season_modules/injury_recovery_calculator.py.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
from datetime import datetime, timedelta

# Injury recovery coefficients (validated from future season modules)
INJURY_RECOVERY_COEFFICIENTS = {
    # Major surgeries
    'tommy_john': {
        'recovery_factor': 0.85,
        'recovery_duration_days': 365,
        'severity': 3
    },
    'shoulder_surgery': {
        'recovery_factor': 0.90,
        'recovery_duration_days': 180,
        'severity': 3
    },
    'hip_surgery': {
        'recovery_factor': 0.88,
        'recovery_duration_days': 120,
        'severity': 3
    },
    'elbow_internal_brace': {
        'recovery_factor': 0.92,
        'recovery_duration_days': 90,
        'severity': 2
    },
    'other_surgery': {
        'recovery_factor': 0.93,
        'recovery_duration_days': 90,
        'severity': 2
    },

    # Non-surgical injuries
    'oblique_strain': {
        'recovery_factor': 0.95,
        'recovery_duration_days': 60,
        'severity': 1
    },
    'hamstring_strain': {
        'recovery_factor': 0.95,
        'recovery_duration_days': 45,
        'severity': 1
    },
    'shoulder_strain': {
        'recovery_factor': 0.96,
        'recovery_duration_days': 30,
        'severity': 1
    },
    'back_strain': {
        'recovery_factor': 0.94,
        'recovery_duration_days': 45,
        'severity': 1
    },
    'groin_strain': {
        'recovery_factor': 0.96,
        'recovery_duration_days': 30,
        'severity': 1
    },

    # Default
    'unknown': {
        'recovery_factor': 0.98,
        'recovery_duration_days': 21,
        'severity': 0
    }
}


def classify_injury_type(injury_description: str) -> str:
    """
    Classify injury from description text.

    Args:
        injury_description: Description of injury (e.g., "Tommy John surgery")

    Returns:
        Injury type key from INJURY_RECOVERY_COEFFICIENTS

    Example:
        >>> classify_injury_type("Tommy John surgery")
        'tommy_john'
        >>> classify_injury_type("oblique strain")
        'oblique_strain'
    """
    if pd.isna(injury_description):
        return 'unknown'

    desc_lower = str(injury_description).lower()

    # Surgery keywords
    if 'tommy john' in desc_lower or 'ucl' in desc_lower:
        return 'tommy_john'
    elif 'shoulder surgery' in desc_lower or 'labrum' in desc_lower:
        return 'shoulder_surgery'
    elif 'hip surgery' in desc_lower:
        return 'hip_surgery'
    elif 'elbow' in desc_lower and ('brace' in desc_lower or 'internal' in desc_lower):
        return 'elbow_internal_brace'
    elif 'surgery' in desc_lower:
        return 'other_surgery'

    # Strain keywords
    elif 'oblique' in desc_lower:
        return 'oblique_strain'
    elif 'hamstring' in desc_lower:
        return 'hamstring_strain'
    elif 'shoulder' in desc_lower and 'strain' in desc_lower:
        return 'shoulder_strain'
    elif 'back' in desc_lower:
        return 'back_strain'
    elif 'groin' in desc_lower:
        return 'groin_strain'

    return 'unknown'


def calculate_recovery_progress(
    return_date: datetime,
    current_date: datetime,
    injury_type: str
) -> float:
    """
    Calculate recovery progress based on time since return.

    Recovery progress increases from base recovery_factor to 1.0 over the
    recovery duration period.

    Args:
        return_date: Date player returned from injury
        current_date: Current date (or prediction date)
        injury_type: Injury type key

    Returns:
        Recovery factor (0.0-1.0, where 1.0 = fully recovered)

    Example:
        >>> # Trout returned from surgery 180 days ago (shoulder surgery)
        >>> return_dt = datetime(2024, 12, 1)
        >>> current_dt = datetime(2025, 6, 1)
        >>> calculate_recovery_progress(return_dt, current_dt, 'shoulder_surgery')
        1.0  # Fully recovered (180 days = full recovery duration)
    """
    injury_info = INJURY_RECOVERY_COEFFICIENTS.get(
        injury_type,
        INJURY_RECOVERY_COEFFICIENTS['unknown']
    )

    base_factor = injury_info['recovery_factor']
    recovery_duration = injury_info['recovery_duration_days']

    # Days since return
    days_since_return = (current_date - return_date).days

    if days_since_return <= 0:
        # Not yet returned
        return 0.0
    elif days_since_return >= recovery_duration:
        # Fully recovered
        return 1.0
    else:
        # Progressive recovery
        progress = days_since_return / recovery_duration
        return base_factor + (1.0 - base_factor) * progress


def check_recurring_injury(
    player_injury_history: pd.DataFrame,
    current_injury_type: str,
    lookback_years: int = 3
) -> bool:
    """
    Check if player has recurring injury of similar type.

    Args:
        player_injury_history: Historical injury data for player
        current_injury_type: Current injury type
        lookback_years: Years to look back

    Returns:
        True if similar injury occurred within lookback period

    Example:
        >>> history = pd.DataFrame({
        ...     'Year': [2022, 2023, 2024],
        ...     'injury_type': ['oblique_strain', 'back_strain', 'oblique_strain']
        ... })
        >>> check_recurring_injury(history, 'oblique_strain', lookback_years=3)
        True  # Had oblique in 2022 and 2024
    """
    if player_injury_history.empty:
        return False

    # Ensure injury_type column exists
    history_copy = player_injury_history.copy()
    if 'injury_type' not in history_copy.columns:
        # Classify injury types from descriptions
        history_copy['injury_type'] = history_copy.get('injury_description', pd.Series()).apply(
            classify_injury_type
        )

    # Filter to lookback window
    current_year = history_copy['Year'].max()
    recent_injuries = history_copy[
        history_copy['Year'] >= (current_year - lookback_years)
    ]

    # Count occurrences of same injury type
    matching_injuries = recent_injuries[
        recent_injuries['injury_type'] == current_injury_type
    ]

    return len(matching_injuries) >= 2  # Recurring = 2+ occurrences


def extract_injury_features(
    player_injury_data: Optional[pd.DataFrame],
    current_date: datetime = None
) -> Dict[str, float]:
    """
    Extract all injury features for ROS model.

    Args:
        player_injury_data: Injury history for player with columns:
            - Year
            - injury_description
            - return_date (datetime or str)
            - injury_type (optional, will classify if missing)
        current_date: Date for calculating recovery (default: today)

    Returns:
        Dictionary with 5 injury features:
        - injury_flag: int (0/1, has active/recent injury)
        - injury_recovery_factor: float (0.0-1.0)
        - days_since_injury: int
        - injury_severity_encoded: int (0-3)
        - recurring_injury: int (0/1)

    Example:
        >>> # Player with recent shoulder surgery
        >>> injury_data = pd.DataFrame({
        ...     'Year': [2024],
        ...     'injury_description': ['shoulder surgery'],
        ...     'return_date': [datetime(2024, 12, 1)]
        ... })
        >>> features = extract_injury_features(injury_data, datetime(2025, 6, 1))
        >>> features['injury_flag']
        1
        >>> features['injury_recovery_factor']
        1.0  # Fully recovered after 6 months
        >>> features['injury_severity_encoded']
        3  # Surgery
    """
    if current_date is None:
        current_date = datetime.now()

    # Default values (no injury)
    default_features = {
        'injury_flag': 0,
        'injury_recovery_factor': 1.0,
        'days_since_injury': 0,
        'injury_severity_encoded': 0,
        'recurring_injury': 0
    }

    if player_injury_data is None or player_injury_data.empty:
        return default_features

    # Get most recent injury
    recent_injury = player_injury_data.sort_values('Year', ascending=False).iloc[0]

    # Classify injury type if not already classified
    if 'injury_type' not in recent_injury or pd.isna(recent_injury['injury_type']):
        injury_type = classify_injury_type(recent_injury.get('injury_description', ''))
    else:
        injury_type = recent_injury['injury_type']

    # Get injury info
    injury_info = INJURY_RECOVERY_COEFFICIENTS.get(
        injury_type,
        INJURY_RECOVERY_COEFFICIENTS['unknown']
    )

    # Parse return date
    return_date = recent_injury.get('return_date')
    if isinstance(return_date, str):
        return_date = pd.to_datetime(return_date)

    # Calculate days since injury
    if pd.notna(return_date):
        days_since = (current_date - return_date).days
    else:
        days_since = 365  # Assume old injury if no return date

    # Recovery factor
    if pd.notna(return_date):
        recovery_factor = calculate_recovery_progress(
            return_date, current_date, injury_type
        )
    else:
        recovery_factor = 1.0  # Default to fully recovered

    # Check for recurring injury
    is_recurring = check_recurring_injury(
        player_injury_data, injury_type, lookback_years=3
    )

    # Active injury flag (within last 180 days)
    injury_flag = int(days_since <= 180)

    return {
        'injury_flag': injury_flag,
        'injury_recovery_factor': recovery_factor,
        'days_since_injury': max(0, days_since),
        'injury_severity_encoded': injury_info['severity'],
        'recurring_injury': int(is_recurring)
    }
