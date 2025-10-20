"""
Real-time composite feature calculator for park-adjusted predictions.

Calculates composite features from park-adjusted raw components during prediction,
ensuring park factors properly affect the final features used by the model.
"""

from typing import Dict, Optional
import numpy as np
from .logging import get_logger

logger = get_logger(__name__)

# Normalization constants (matching training data)
Z_SCORE_CENTER = 50.0
Z_SCORE_SCALE = 15.0

# Population statistics from park-adjusted training data (2016-2024)
# These MUST match the values from the park-adjusted cache rebuild (with home/road split)
DAMAGE_CONTROL_MEAN = 33.742  # From park-adjusted cache
DAMAGE_CONTROL_STD = 61.656   # From park-adjusted cache

CONTACT_QUALITY_MEAN = 47.005  # From park-adjusted cache
CONTACT_QUALITY_STD = 0.124    # From park-adjusted cache

LAUNCH_QUALITY_MEAN = 50.0
LAUNCH_QUALITY_STD = 15.0


def calculate_damage_control_from_components(
    lob_pct: float,
    hr_pct: float,  # HR/FB%
    normalize: bool = False
) -> float:
    """
    Calculate RAW damage_control_ratio from park-adjusted components (NO normalization).

    PHASE 1 UPDATE: Removed normalization to fix double-normalization bug.
    Tree models (RandomForest, XGBoost) don't require normalization.
    StandardScaler will handle normalization for Keras if needed.

    Formula: LOB% / (HR% + 0.5)

    Args:
        lob_pct: Left on base percentage (already park-adjusted if applicable)
        hr_pct: Home run per fly ball percentage (MUST be park-adjusted with HR park factor)
        normalize: Deprecated parameter, kept for backwards compatibility (default: False)

    Returns:
        RAW damage control ratio (no normalization applied)
    """
    if hr_pct < 0 or lob_pct < 0:
        logger.warning(f"Invalid input: lob_pct={lob_pct}, hr_pct={hr_pct}")
        # Return league average raw ratio instead of normalized center
        return 72.0 / (10.0 + 0.5)  # League avg LOB% / (League avg HR% + 0.5)

    raw_ratio = lob_pct / (hr_pct + 0.5)
    return raw_ratio


def calculate_contact_quality_from_components(
    hard_pct: float,
    soft_pct: float,
    med_pct: float,
    normalize: bool = False
) -> float:
    """
    Calculate RAW Contact_Quality_Index from park-adjusted batted ball components (NO normalization).

    PHASE 1 UPDATE: Removed normalization to fix double-normalization bug.
    Tree models (RandomForest, XGBoost) don't require normalization.
    StandardScaler will handle normalization for Keras if needed.

    Formula: (100 - Hard%) * 0.6 + Soft% * 0.4

    Args:
        hard_pct: Hard contact percentage (MUST be park-adjusted with 3yr park factor)
        soft_pct: Soft contact percentage (MUST be park-adjusted with 3yr park factor)
        med_pct: Medium contact percentage (MUST be park-adjusted with 3yr park factor)
        normalize: Deprecated parameter, kept for backwards compatibility (default: False)

    Returns:
        RAW contact quality index (no normalization applied)
    """
    if hard_pct < 0 or soft_pct < 0 or med_pct < 0:
        logger.warning(f"Invalid input: hard={hard_pct}, soft={soft_pct}, med={med_pct}")
        # Return league average raw index
        return (100 - 35.0) * 0.6 + 18.0 * 0.4  # League avg calculation

    # Contact quality: lower hard% and higher soft% is better for pitcher
    # Inverse hard% (lower is better) and weight soft% positively
    raw_index = (100 - hard_pct) * 0.6 + soft_pct * 0.4
    return raw_index


def calculate_opportunity_success(
    k_pct: float,
    bb_pct: float,
    lob_pct: float
) -> float:
    """
    Calculate Opportunity_Success from rate stats.

    Formula: (K% - BB%) * (LOB% / 100) to capture strikeout ability with damage limitation

    Args:
        k_pct: Strikeout percentage (skill-based, NOT park-adjusted)
        bb_pct: Walk percentage (skill-based, NOT park-adjusted)
        lob_pct: Left on base percentage

    Returns:
        Opportunity success metric
    """
    if k_pct < 0 or bb_pct < 0 or lob_pct < 0:
        logger.warning(f"Invalid input: k={k_pct}, bb={bb_pct}, lob={lob_pct}")
        return 0.0

    return (k_pct - bb_pct) * (lob_pct / 100.0)


def get_composite_features_from_park_adjusted_stats(
    park_adjusted_stats: Dict[str, float],
    raw_stats: Dict[str, float]
) -> Dict[str, float]:
    """
    Calculate all composite features from park-adjusted statistics.

    Args:
        park_adjusted_stats: Dict with park-adjusted values for HR%, Hard%, Soft%, Med%, etc.
        raw_stats: Dict with raw (non-park-adjusted) values for skill-based stats like K%, BB%

    Returns:
        Dictionary with composite feature values:
        {
            'damage_control_ratio': float,
            'Contact_Quality_Index': float,
            'Opportunity_Success': float,
            'Statcast_Launch_Quality_Index': float  # Use cached value for now
        }
    """
    composites = {}

    # 1. Damage Control Ratio (uses park-adjusted HR%)
    lob_pct = park_adjusted_stats.get('LOB%', raw_stats.get('LOB%', 72.0))
    hr_pct = park_adjusted_stats.get('HR%', raw_stats.get('HR%', 10.0))

    composites['damage_control_ratio'] = calculate_damage_control_from_components(
        lob_pct=lob_pct,
        hr_pct=hr_pct,
        normalize=True
    )

    # 2. Contact Quality Index - use cached value (skill-based, not park-adjusted)
    # Hard%, Soft%, Med% are pitcher skill metrics, not park-dependent
    composites['Contact_Quality_Index'] = raw_stats.get('Contact_Quality_Index', 50.0)

    # 3. Opportunity Success (uses skill-based stats, NOT park-adjusted)
    k_pct = raw_stats.get('K%', 20.0)
    bb_pct = raw_stats.get('BB%', 8.0)

    composites['Opportunity_Success'] = calculate_opportunity_success(
        k_pct=k_pct,
        bb_pct=bb_pct,
        lob_pct=lob_pct
    )

    # 4. Statcast Launch Quality Index - keep cached value for now
    # This requires exit velocity + launch angle data, not simple calculation
    composites['Statcast_Launch_Quality_Index'] = raw_stats.get(
        'Statcast_Launch_Quality_Index', 50.0
    )

    logger.debug(f"Calculated composites from park-adjusted stats: {composites}")

    return composites
