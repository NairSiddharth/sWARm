"""
Tier threshold calculations for current season accumulated WAR.

Provides dynamic tier thresholds that scale based on season completion percentage.
Used for classifying players into average/good/elite tiers based on their
accumulated WAR performance so far in the current season.
"""

from typing import Tuple


def get_thresholds(season_pct: float, role: str) -> Tuple[float, float]:
    """
    Calculate tier thresholds based on season progress for accumulated WAR.

    Returns thresholds on same scale as model predictions (WAR_per_[denominator]).
    Season scaling is applied after normalization to account for confidence level.

    Uses role-specific scaling formulas:
    - Starters: Quadratic (1.22*x² - 0.83*x + 0.61) fitted to starter WAR accumulation
    - Relievers: Square root (x^0.7) for more conservative early-season thresholds
    - Swing: Square root (x^0.7) similar to relievers
    - Hitters: Linear (x) for more consistent WAR accumulation

    Threshold scales by role:
    - Starters: WAR_per_162
    - Relievers: WAR_per_48.2
    - Swing: WAR_per_110
    - Hitters: WAR_per_600

    Args:
        season_pct: Fraction of season completed (0.0 to 1.0)
        role: Player role ('starter', 'reliever', 'swing', 'hitter')

    Returns:
        tuple: (good_threshold, elite_threshold) in WAR_per_[denominator]

    Example:
        >>> # Starter thresholds at 70% season (WAR_per_162):
        >>> thresholds = get_thresholds(0.7, 'starter')
        >>> # Returns: (2.24, 3.52) - scaled from base (3.3, 5.0)
        >>>
        >>> # Reliever thresholds at 58.6% season (WAR_per_48.2):
        >>> thresholds = get_thresholds(0.586, 'reliever')
        >>> # Returns: (0.68, 1.02) - scaled from base (1.03, 1.55)
        >>>
        >>> # Hitter thresholds at 70% season (WAR_per_600):
        >>> thresholds = get_thresholds(0.7, 'hitter')
        >>> # Returns: (1.40, 2.45) - scaled from base (2.0, 3.5)
    """
    if role == 'starter':
        # Quadratic scaling for starters: 1.22*x² - 0.83*x + 0.61
        # Fitted to starter WAR accumulation patterns
        scaling = min(
            1.22 * (season_pct ** 2) - 0.83 * season_pct + 0.61,
            1.0
        )
        # Base values in WAR_per_162 scale
        base_elite = 5.0   # Ace starters (top 13-14 per year)
        base_good = 3.3    # Top #1/#2 starters (All-Star caliber)

    elif role == 'reliever':
        # Square root scaling for relievers: x^0.7
        # More conservative early season to handle high variance in small IP samples
        scaling = min(season_pct ** 0.7, 1.0)

        # Convert base values from full-season actual WAR to WAR_per_48.2
        # Elite relievers: 2.25 WAR in ~70 IP typical
        # Good relievers: 1.5 WAR in ~70 IP typical
        base_elite_full = 2.25
        base_good_full = 1.5
        typical_ip = 70  # Typical full-season IP for quality reliever

        # Normalize to WAR_per_48.2: (WAR / typical_IP) * 48.2
        base_elite = base_elite_full / typical_ip * 48.2  # = 1.55
        base_good = base_good_full / typical_ip * 48.2    # = 1.03

    elif role == 'swing':
        # Square root scaling for swing (similar to relievers)
        scaling = min(season_pct ** 0.7, 1.0)

        # Base values in WAR_per_110 scale
        base_elite = 3.5  # Full season target
        base_good = 2.5   # Full season target

    elif role == 'hitter':
        # Linear scaling for hitters: x
        # Hitters accumulate WAR more linearly than pitchers
        scaling = min(season_pct, 1.0)

        # Base values in WAR_per_600 scale (full season thresholds)
        base_elite = 3.5  # Elite hitters: >3.5 WAR per 600 PA
        base_good = 2.0   # Good hitters: 2.0-3.5 WAR per 600 PA

    else:
        raise ValueError(f"Invalid role: {role}. Must be 'starter', 'reliever', 'swing', or 'hitter'")

    # Apply season scaling to normalized base values
    return base_good * scaling, base_elite * scaling
