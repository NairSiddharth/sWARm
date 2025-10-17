"""
Tier threshold calculations and adjustment logic for ROS (Rest of Season) predictions.

Provides dynamic tier thresholds that scale based on remaining season percentage,
plus utilities for elite player identification and blend ratio adjustments.
"""

import numpy as np
import pandas as pd
from typing import Tuple

from new_pipeline.common.constants import (
    WAR_NORMALIZATION_IP_STARTER,
    WAR_NORMALIZATION_IP_RELIEVER,
    WAR_NORMALIZATION_IP_SWING,
    WAR_NORMALIZATION_PA
)


def subclassify_swing_pitcher(games_started: int, games_pitched: int, innings_pitched: float) -> str:
    """
    Subclassify swing pitchers into swing_starter or swing_reliever.

    Swing pitchers fall on a spectrum between starters and relievers. This function
    splits them based on their usage pattern to apply appropriate threshold scaling.

    Args:
        games_started: Number of games started (GS)
        games_pitched: Total games pitched (G)
        innings_pitched: Total innings pitched (IP)

    Returns:
        str: 'swing_starter' or 'swing_reliever'

    Classification Logic:
        SWING_STARTER: GS/G > 0.5 AND IP/G > 3.0
        - Failed starters still getting regular spot starts
        - Openers with multi-inning appearances
        - Example: 12 GS in 20 G (60%), 4.2 IP/G

        SWING_RELIEVER: GS/G ≤ 0.5 OR IP/G ≤ 3.0
        - Long relievers occasionally filling in
        - Former closers stretching out
        - Example: 5 GS in 45 G (11%), 2.3 IP/G
    """
    if games_pitched == 0:
        return 'swing_reliever'  # Default

    gs_ratio = games_started / games_pitched
    ip_per_g = innings_pitched / games_pitched

    # Swing-starters: More starter-like usage (frequent starts AND long outings)
    if gs_ratio > 0.5 and ip_per_g > 3.0:
        return 'swing_starter'
    else:
        return 'swing_reliever'


def get_threshold_scaling(role: str, season_pct: float) -> float:
    """
    Get scaling factor for thresholds based on remaining season percentage.

    Scaling approaches by role:
    - Starters: (1-s)^1.6 accelerated decay (looser as season progresses)
    - Swing_starters: (1-s)^1.3 (weighted toward starters)
    - Swing_relievers: (1-s)^0.9 (weighted toward relievers)
    - Relievers: (1-s)^0.7 inverted power (tighter control)
    - Hitters: (1-s)^1.0 linear (rate-preserving)

    Args:
        role: 'starter', 'swing_starter', 'swing_reliever', 'reliever', or 'hitter'
        season_pct: Fraction of season completed (0.0 to 1.0)

    Returns:
        Scaling factor (0.0 to 1.0)

    Example:
        >>> get_threshold_scaling('starter', 0.59)
        0.2435  # (1-0.59)^1.6 = 0.41^1.6
        >>> get_threshold_scaling('hitter', 0.59)
        0.41  # Linear: 1-0.59
    """
    remaining_pct = 1.0 - season_pct

    if role == 'starter':
        return min(remaining_pct ** 1.6, 1.0)
    elif role == 'swing_starter':
        return min(remaining_pct ** 1.3, 1.0)
    elif role == 'swing_reliever':
        return min(remaining_pct ** 0.9, 1.0)
    elif role == 'swing':
        return min(remaining_pct ** 1.1, 1.0)
    elif role == 'reliever':
        return min(remaining_pct ** 0.7, 1.0)
    elif role == 'hitter':
        return remaining_pct  # Linear
    else:
        raise ValueError(f"Unknown role: {role}")


def get_thresholds(role: str, season_pct: float) -> Tuple[float, float]:
    """
    Get dynamically scaled elite/good thresholds for a role at a season percentage.

    FOR ROS PREDICTIONS: Uses role-specific power scaling optimized for tier distributions.

    Scaling approaches by role:
    - Starters: (1-s)^1.6 accelerated decay (looser as season progresses)
    - Swing_starters: (1-s)^1.3 (weighted toward starters)
    - Swing_relievers: (1-s)^0.9 (weighted toward relievers)
    - Relievers: (1-s)^0.7 inverted power (tighter control)
    - Hitters: (1-s)^1.0 linear (rate-preserving)

    Args:
        role: 'starter', 'swing_starter', 'swing_reliever', 'reliever', or 'hitter'
        season_pct: Fraction of season completed (0.0 to 1.0)

    Returns:
        (good_threshold, elite_threshold) in cumulative remaining WAR

    Example:
        >>> # Starters at 59% completion
        >>> get_thresholds('starter', 0.59)
        (0.804, 1.217)  # Remaining WAR needed for good/elite
    """
    # Get base thresholds for role
    if role == 'starter':
        base_good = 3.3
        base_elite = 5.0
    elif role in ['swing_starter', 'swing_reliever', 'swing']:
        base_good = 2.0
        base_elite = 2.5
    elif role == 'reliever':
        base_good = 1.1
        base_elite = 1.6
    elif role == 'hitter':
        base_good = 3.8  # Raised from 3.3 to reduce "good" tier size
        base_elite = 5.0
    else:
        raise ValueError(f"Unknown role: {role}")

    # Get scaling factor
    scaling = get_threshold_scaling(role, season_pct)

    return base_good * scaling, base_elite * scaling


def classify_tier(
    predictions: np.ndarray,
    role: str,
    season_pct: float
) -> np.ndarray:
    """
    Classify ROS predictions into tier labels.

    Args:
        predictions: ROS WAR predictions (cumulative remaining)
        role: 'starter', 'swing_starter', 'swing_reliever', 'reliever', or 'hitter'
        season_pct: Fraction of season completed (0.0 to 1.0)

    Returns:
        Array of tier labels: 'average', 'good', or 'elite'

    Example:
        >>> predictions = np.array([0.5, 1.2, 2.8])
        >>> classify_tier(predictions, 'starter', 0.59)
        array(['average', 'good', 'elite'], dtype='<U7')
    """
    good_threshold, elite_threshold = get_thresholds(role, season_pct)

    return np.array([
        'average' if w < good_threshold
        else 'good' if w < elite_threshold
        else 'elite'
        for w in predictions
    ])


def get_tier_percentiles(role: str) -> Tuple[float, float]:
    """
    Get elite/good percentile cutoffs by role based on actual qualified player counts.

    These percentiles are calibrated to domain expectations:
    - Elite: Best or maybe 2nd at position (if neck-and-neck)
    - Good: 2nd or 3rd at position (not including elite)

    Based on 2025 first-half qualified counts:
    - Hitters: 418 (>= 75 PA)
    - Starters: 198
    - Swing: 66
    - Relievers: 333

    Args:
        role: 'starter', 'swing_starter', 'swing_reliever', 'swing', 'reliever', or 'hitter'

    Returns:
        (elite_pct, good_pct) where:
        - elite_pct: Top X% classified as elite
        - good_pct: Top Y% classified as good or better (includes elite)

    Example:
        >>> elite_pct, good_pct = get_tier_percentiles('hitter')
        >>> elite_pct
        0.03  # Top 3% = ~13 of 418 hitters
        >>> good_pct
        0.065  # Top 6.5% = ~27 of 418 hitters (27 - 13 = 14 in good tier)
    """
    if role in ['starter', 'swing_starter']:
        # Starters: 198 qualified
        # Elite: 12 players (~6%) - Cy Young conversation
        # Good+Elite: 28 players (~14%) - All-Star caliber
        return 0.06, 0.14

    elif role in ['swing', 'swing_reliever']:
        # Swing: 66 qualified
        # Elite: 4 players (~6%) - Elite versatility
        # Good+Elite: 8 players (~12%) - Valuable swing arms
        return 0.06, 0.12

    elif role == 'reliever':
        # Relievers: 333 qualified
        # Elite: 8 players (~2.5%) - Elite closers/setup men
        # Good+Elite: 17 players (~5%) - High-leverage arms
        return 0.025, 0.05

    elif role == 'hitter':
        # Hitters: 418 qualified
        # Elite: 13 players (~3%) - Best or 2nd at position
        # Good+Elite: 27 players (~6.5%) - Top 2-3 at position
        return 0.03, 0.065

    else:
        raise ValueError(f"Unknown role: {role}")


def classify_tier_percentile(
    predictions: np.ndarray,
    role: str
) -> np.ndarray:
    """
    Classify ROS predictions into tier labels using percentile-based thresholds.

    Unlike classify_tier() which uses fixed WAR thresholds scaled by season_pct,
    this function uses percentiles of the actual prediction distribution. This
    guarantees consistent tier sizes regardless of overall talent level.

    Use this for ROS projections where relative rankings matter more than
    absolute WAR values.

    Args:
        predictions: ROS WAR predictions (cumulative remaining)
        role: 'starter', 'swing_starter', 'swing_reliever', 'swing', 'reliever', or 'hitter'

    Returns:
        Array of tier labels: 'average', 'good', or 'elite'

    Example:
        >>> # 418 hitters with ROS predictions
        >>> predictions = np.array([2.5, 2.3, 1.8, 1.5, ...])  # 418 values
        >>> tiers = classify_tier_percentile(predictions, 'hitter')
        >>> np.sum(tiers == 'elite')
        13  # Top 3% of 418 = 13 players
        >>> np.sum(tiers == 'good')
        14  # Next 3.5% of 418 = 14 players (27 - 13)
    """
    elite_pct, good_pct = get_tier_percentiles(role)

    # Calculate percentile thresholds from actual distribution
    # np.percentile(predictions, 97) gives the 97th percentile (top 3%)
    elite_threshold = np.percentile(predictions, 100 * (1 - elite_pct))
    good_threshold = np.percentile(predictions, 100 * (1 - good_pct))

    return np.array([
        'average' if w < good_threshold
        else 'good' if w < elite_threshold
        else 'elite'
        for w in predictions
    ])


def calculate_war_rate(war: float, usage: float, role: str) -> float:
    """
    Calculate WAR rate (WAR_per_X) from cumulative WAR and usage.

    Args:
        war: Cumulative WAR
        usage: IP (pitchers) or PA (hitters)
        role: 'starter', 'swing_starter', 'swing_reliever', 'reliever', or 'hitter'

    Returns:
        WAR rate in role-specific normalization
    """
    if usage <= 0:
        return 0.0

    if role == 'starter':
        return war / usage * WAR_NORMALIZATION_IP_STARTER
    elif role in ['swing', 'swing_starter', 'swing_reliever']:
        return war / usage * WAR_NORMALIZATION_IP_SWING
    elif role == 'reliever':
        return war / usage * WAR_NORMALIZATION_IP_RELIEVER
    elif role == 'hitter':
        return war / usage * WAR_NORMALIZATION_PA
    else:
        raise ValueError(f"Unknown role: {role}")


def is_elite_candidate(
    player_history: pd.DataFrame,
    current_war_rate: float,
    role: str,
    elite_threshold: float,
    good_threshold: float
) -> Tuple[bool, str]:
    """
    Check if a player qualifies as an elite candidate for adjustment.

    Args:
        player_history: Historical seasons for this player (sorted by year)
        current_war_rate: Current season WAR rate
        role: Player role
        elite_threshold: Elite threshold for current season
        good_threshold: Good threshold for current season

    Returns:
        (is_candidate, reason) - reason is '1yr_elite', '2yr_elite', or 'none'
    """
    # Need at least 1 prior season
    if len(player_history) == 0:
        return False, 'none'

    # Get most recent 1-2 seasons
    recent_seasons = player_history.tail(2)

    # Calculate WAR rates for historical seasons
    war_rates = []
    for _, season in recent_seasons.iterrows():
        if role in ['starter', 'swing', 'swing_starter', 'swing_reliever', 'reliever']:
            war_rate = calculate_war_rate(season['full_WAR'], season['full_IP'], role)
        else:  # hitter
            war_rate = calculate_war_rate(season['full_WAR'], season['full_PA'], role)
        war_rates.append(war_rate)

    # 1-year elite: Single season at ≥elite, current season at ≥elite
    if len(war_rates) >= 1:
        if war_rates[-1] >= elite_threshold and current_war_rate >= elite_threshold:
            return True, '1yr_elite'

    # 2-year elite: Both seasons at ≥90% elite, current at ≥good AND within 10% of elite
    if len(war_rates) >= 2:
        elite_90pct = elite_threshold * 0.9
        elite_110pct = elite_threshold * 1.1

        if (war_rates[-1] >= elite_90pct and
            war_rates[-2] >= elite_90pct and
            current_war_rate >= good_threshold and
            current_war_rate >= elite_threshold * 0.9):
            return True, '2yr_elite'

    return False, 'none'


def calculate_blended_prediction(
    current_war_rate: float,
    remaining_usage: float,
    baseline_q50: float,
    blend_ratio: float,
    role: str
) -> float:
    """
    Calculate blended ROS WAR prediction.

    Args:
        current_war_rate: Current season WAR rate (WAR_per_X)
        remaining_usage: Remaining IP or PA
        baseline_q50: Baseline model's median prediction (cumulative WAR)
        blend_ratio: Weight for current rate (0.0 to 1.0)
        role: Player role

    Returns:
        Blended ROS WAR prediction (cumulative)
    """
    # Current rate projection
    if role in ['starter', 'swing', 'swing_starter', 'swing_reliever', 'reliever']:
        normalization = {
            'starter': WAR_NORMALIZATION_IP_STARTER,
            'swing': WAR_NORMALIZATION_IP_SWING,
            'swing_starter': WAR_NORMALIZATION_IP_SWING,
            'swing_reliever': WAR_NORMALIZATION_IP_SWING,
            'reliever': WAR_NORMALIZATION_IP_RELIEVER
        }[role]
    else:
        normalization = WAR_NORMALIZATION_PA

    current_projection = current_war_rate * (remaining_usage / normalization)

    # Blend
    return blend_ratio * current_projection + (1 - blend_ratio) * baseline_q50
