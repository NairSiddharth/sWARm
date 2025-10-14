"""
ROS Projection Display Utilities

Formats ROS model predictions (cumulative WAR) for display with proper rate calculations.

Key Concept:
- ROS models predict CUMULATIVE remaining WAR directly (not rates)
- This module converts cumulative predictions to display-friendly formats
- Calculates implied rates for comparison: cumulative_war / (usage / normalization_basis)

Example:
    >>> predictions = {'mean': 1.5, 'q50': 1.4, 'q90': 2.1}
    >>> remaining_ip = 30
    >>> result = format_pitcher_ros_display(predictions, remaining_ip, role='reliever')
    >>> result
    {
        'ros_war': 1.5,  # Cumulative (use this for projections)
        'ros_rate': 2.41,  # Implied WAR_per_48.2 (for comparison only)
        'ros_q50': 1.4,
        'ros_q90': 2.1,
        ...
    }
"""

from typing import Dict, Union
import numpy as np
from ..constants import (
    WAR_NORMALIZATION_IP_STARTER,
    WAR_NORMALIZATION_IP_RELIEVER,
    WAR_NORMALIZATION_PA,
    PITCHER_STARTER_THRESHOLD
)


def format_pitcher_ros_display(
    predictions: Dict[str, np.ndarray],
    remaining_ip: Union[float, np.ndarray],
    role: str = 'starter',
    include_rates: bool = True
) -> Dict[str, np.ndarray]:
    """
    Format pitcher ROS predictions for display.

    The ROS model predicts cumulative remaining WAR directly. This function:
    1. Extracts cumulative WAR predictions (no conversion needed)
    2. Optionally calculates implied rates for comparison

    Args:
        predictions: Dict from predict_with_uncertainty() containing:
            - 'mean': Cumulative remaining WAR
            - 'q10', 'q25', 'q50', 'q75', 'q90': Quantile predictions
            - 'std', 'uncertainty_band': Optional uncertainty metrics
        remaining_ip: Projected remaining IP (scalar or array)
        role: 'starter' or 'reliever' (determines rate normalization basis)
        include_rates: If True, calculate implied WAR_per_X rates

    Returns:
        Dict with formatted predictions:
        - 'ros_war': Cumulative remaining WAR (use this for projections)
        - 'ros_q10', 'ros_q25', 'ros_q50', 'ros_q75', 'ros_q90': Quantiles
        - 'ros_rate': Implied rate (WAR_per_162 or WAR_per_48.2) if include_rates=True
        - 'ros_rate_q50', 'ros_rate_q90': Rate quantiles if include_rates=True

    Example:
        >>> # Reliever: 1.5 WAR over 30 remaining IP
        >>> preds = {'mean': np.array([1.5]), 'q50': np.array([1.4]), 'q90': np.array([2.1])}
        >>> result = format_pitcher_ros_display(preds, np.array([30]), role='reliever')
        >>> result['ros_war']
        array([1.5])  # Cumulative WAR (CORRECT - don't convert again!)
        >>> result['ros_rate']
        array([2.41])  # Implied WAR_per_48.2 = 1.5 / (30/48.2) = 2.41
    """
    # Determine normalization basis by role
    if role == 'starter':
        normalization_basis = WAR_NORMALIZATION_IP_STARTER  # 162
    elif role == 'reliever':
        normalization_basis = WAR_NORMALIZATION_IP_RELIEVER  # 48.2
    else:
        # Swing pitchers - use starter basis as default
        normalization_basis = WAR_NORMALIZATION_IP_STARTER

    result = {}

    # Extract cumulative WAR (NO CONVERSION - model already predicts cumulative)
    result['ros_war'] = predictions['mean']
    result['ros_q10'] = predictions.get('q10', predictions['mean'])
    result['ros_q25'] = predictions.get('q25', predictions['mean'])
    result['ros_q50'] = predictions.get('q50', predictions['mean'])
    result['ros_q75'] = predictions.get('q75', predictions['mean'])
    result['ros_q90'] = predictions.get('q90', predictions['mean'])

    # Copy uncertainty metrics if available
    if 'std' in predictions:
        result['ros_std'] = predictions['std']
    if 'uncertainty_band' in predictions:
        result['ros_uncertainty'] = predictions['uncertainty_band']

    # Calculate implied rates for comparison/display (optional)
    if include_rates:
        # Avoid division by zero
        remaining_ip_safe = np.where(remaining_ip > 0, remaining_ip, 1.0)

        # Implied rate: cumulative_war / (remaining_ip / normalization_basis)
        # = cumulative_war * (normalization_basis / remaining_ip)
        rate_multiplier = normalization_basis / remaining_ip_safe

        result['ros_rate'] = predictions['mean'] * rate_multiplier
        result['ros_rate_q50'] = predictions.get('q50', predictions['mean']) * rate_multiplier
        result['ros_rate_q90'] = predictions.get('q90', predictions['mean']) * rate_multiplier

        # Set rates to 0 where remaining_ip is 0
        zero_mask = remaining_ip <= 0
        if isinstance(zero_mask, (bool, np.bool_)):
            if zero_mask:
                result['ros_rate'] = 0.0
                result['ros_rate_q50'] = 0.0
                result['ros_rate_q90'] = 0.0
        else:
            result['ros_rate'] = np.where(zero_mask, 0.0, result['ros_rate'])
            result['ros_rate_q50'] = np.where(zero_mask, 0.0, result['ros_rate_q50'])
            result['ros_rate_q90'] = np.where(zero_mask, 0.0, result['ros_rate_q90'])

    return result


def format_hitter_ros_display(
    predictions: Dict[str, np.ndarray],
    remaining_pa: Union[float, np.ndarray],
    include_rates: bool = True
) -> Dict[str, np.ndarray]:
    """
    Format hitter ROS predictions for display.

    The ROS model predicts cumulative remaining WAR directly. This function:
    1. Extracts cumulative WAR predictions (no conversion needed)
    2. Optionally calculates implied WAR_per_600 rates for comparison

    Args:
        predictions: Dict from predict_with_uncertainty() containing:
            - 'mean': Cumulative remaining WAR
            - 'q10', 'q25', 'q50', 'q75', 'q90': Quantile predictions
            - 'std', 'uncertainty_band': Optional uncertainty metrics
        remaining_pa: Projected remaining PA (scalar or array)
        include_rates: If True, calculate implied WAR_per_600 rates

    Returns:
        Dict with formatted predictions:
        - 'ros_war': Cumulative remaining WAR (use this for projections)
        - 'ros_q10', 'ros_q25', 'ros_q50', 'ros_q75', 'ros_q90': Quantiles
        - 'ros_rate': Implied WAR_per_600 if include_rates=True
        - 'ros_rate_q50', 'ros_rate_q90': Rate quantiles if include_rates=True

    Example:
        >>> # Hitter: 2.5 WAR over 200 remaining PA
        >>> preds = {'mean': np.array([2.5]), 'q50': np.array([2.3]), 'q90': np.array([3.2])}
        >>> result = format_hitter_ros_display(preds, np.array([200]))
        >>> result['ros_war']
        array([2.5])  # Cumulative WAR (CORRECT - don't convert again!)
        >>> result['ros_rate']
        array([7.5])  # Implied WAR_per_600 = 2.5 / (200/600) = 7.5
    """
    result = {}

    # Extract cumulative WAR (NO CONVERSION - model already predicts cumulative)
    result['ros_war'] = predictions['mean']
    result['ros_q10'] = predictions.get('q10', predictions['mean'])
    result['ros_q25'] = predictions.get('q25', predictions['mean'])
    result['ros_q50'] = predictions.get('q50', predictions['mean'])
    result['ros_q75'] = predictions.get('q75', predictions['mean'])
    result['ros_q90'] = predictions.get('q90', predictions['mean'])

    # Copy uncertainty metrics if available
    if 'std' in predictions:
        result['ros_std'] = predictions['std']
    if 'uncertainty_band' in predictions:
        result['ros_uncertainty'] = predictions['uncertainty_band']

    # Calculate implied WAR_per_600 rates for comparison/display (optional)
    if include_rates:
        # Avoid division by zero
        remaining_pa_safe = np.where(remaining_pa > 0, remaining_pa, 1.0)

        # Implied rate: cumulative_war / (remaining_pa / 600)
        # = cumulative_war * (600 / remaining_pa)
        rate_multiplier = WAR_NORMALIZATION_PA / remaining_pa_safe

        result['ros_rate'] = predictions['mean'] * rate_multiplier
        result['ros_rate_q50'] = predictions.get('q50', predictions['mean']) * rate_multiplier
        result['ros_rate_q90'] = predictions.get('q90', predictions['mean']) * rate_multiplier

        # Set rates to 0 where remaining_pa is 0
        zero_mask = remaining_pa <= 0
        if isinstance(zero_mask, (bool, np.bool_)):
            if zero_mask:
                result['ros_rate'] = 0.0
                result['ros_rate_q50'] = 0.0
                result['ros_rate_q90'] = 0.0
        else:
            result['ros_rate'] = np.where(zero_mask, 0.0, result['ros_rate'])
            result['ros_rate_q50'] = np.where(zero_mask, 0.0, result['ros_rate_q50'])
            result['ros_rate_q90'] = np.where(zero_mask, 0.0, result['ros_rate_q90'])

    return result


def format_ros_predictions_for_display(
    predictions: Dict[str, np.ndarray],
    player_type: str,
    remaining_usage: Union[float, np.ndarray],
    role: str = 'starter',
    include_rates: bool = True
) -> Dict[str, np.ndarray]:
    """
    Format ROS predictions for display (unified interface).

    Args:
        predictions: Dict from predict_with_uncertainty()
        player_type: 'pitcher' or 'hitter'
        remaining_usage: Remaining IP (pitchers) or PA (hitters)
        role: 'starter', 'reliever', or 'swing' (pitchers only)
        include_rates: Whether to calculate implied rates

    Returns:
        Dict with formatted predictions

    Example:
        >>> # Pitcher
        >>> result = format_ros_predictions_for_display(
        ...     preds, 'pitcher', remaining_ip=30, role='reliever'
        ... )
        >>> # Hitter
        >>> result = format_ros_predictions_for_display(
        ...     preds, 'hitter', remaining_usage=200
        ... )
    """
    if player_type == 'pitcher':
        return format_pitcher_ros_display(
            predictions,
            remaining_usage,
            role=role,
            include_rates=include_rates
        )
    elif player_type == 'hitter':
        return format_hitter_ros_display(
            predictions,
            remaining_usage,
            include_rates=include_rates
        )
    else:
        raise ValueError(f"player_type must be 'pitcher' or 'hitter', got {player_type!r}")
