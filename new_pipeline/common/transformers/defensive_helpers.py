"""
Defensive Metric Helper Functions.

Pure calculation functions for Enhanced_Defense composite.
Used by hitter_loaders.py to compute position-specific defensive value.

Key Concepts:
- Range Factor (RF): 9 × (A + PO) / Inn - measures plays per 9 innings
- Position-relative: Compare player RF to position average
- Position-specific bonuses: DP for IF, scoops for 1B, framing/throwing/blocking for C
- Position-specific caps: Elite SS/C can reach +30, elite 1B capped at +15

Design Philosophy:
- Elite defenders at easy positions (1B, corner OF) should reach ~0 total value (defense offsets penalty)
- Elite defenders at premium positions (SS, C) should reach +3-4 WAR from defense alone
- Asymmetric caps: Easier to be average than elite (wider negative range)
"""

from typing import Dict, Tuple


# ============================================================================
# Position-Specific Defensive Caps
# ============================================================================

ENHANCED_DEFENSE_CAPS = {
    'C':  (30, -30),   # Elite framers (Grandal, Hedges): +25-30, poor: -20-25
    'SS': (30, -25),   # Elite range (Simmons): +30 DRS, poor: -20-25
    'CF': (25, -20),   # Elite range (Kiermaier, Buxton): +20-25, poor: -15-20
    '2B': (20, -15),   # Elite: +15-20, less demanding than SS
    '3B': (15, -12),   # Elite (Arenado, Chapman): +12-15, moderate impact
    'LF': (18, -10),   # Elite corner OF (Tucker, Betts): +15-18, CF-caliber tools
    'RF': (18, -10),   # Elite corner OF (Judge, Acuña): +15-18, CF-caliber tools
    '1B': (15, -8),    # Elite (Goldschmidt, Rizzo, Olson): +13-15 (mostly scoops)
    'DH': (0, 0)       # No defense
}


# ============================================================================
# Range Factor Calculations
# ============================================================================

def calculate_range_factor(assists: int, putouts: int, innings: float) -> float:
    """
    Calculate Range Factor (RF) = 9 × (A + PO) / Inn.

    Measures defensive plays per 9 innings played.

    Args:
        assists: Assists (throws that record an out)
        putouts: Putouts (outs recorded directly)
        innings: Innings played at position

    Returns:
        float: Range Factor (plays per 9 innings)

    Example:
        SS with 250A, 400PO in 1200 innings:
        >>> calculate_range_factor(250, 400, 1200)
        4.875  # 4.88 plays per 9 innings
    """
    if innings <= 0:
        return 0.0

    return 9.0 * (assists + putouts) / innings


def calculate_position_relative_rf(
    player_rf: float,
    position_avg_rf: float,
    scaling_factor: float = 2.5
) -> float:
    """
    Calculate position-relative RF runs above/below average.

    Compares player's RF to position average and scales to run value.

    Args:
        player_rf: Player's Range Factor
        position_avg_rf: Average RF for the position
        scaling_factor: Runs per RF point above average (default 2.5)

    Returns:
        float: Defensive runs from range (can be negative)

    Example:
        SS with RF 4.88, position avg 4.50:
        >>> calculate_position_relative_rf(4.88, 4.50, 2.5)
        0.95  # +0.95 runs from range

    Notes:
        - scaling_factor = 2.5 is empirical (1 RF point ≈ 2.5 runs/season)
        - Negative values = below-average range
        - Positive values = above-average range
    """
    rf_diff = player_rf - position_avg_rf
    return rf_diff * scaling_factor


# ============================================================================
# Position-Specific Bonus Calculations
# ============================================================================

def calculate_infielder_dp_value(
    dps: int,
    dpt: int,
    dpf: int,
    position: str
) -> float:
    """
    Calculate double play value for infielders.

    Different DP roles have different difficulty/value:
    - DPS (started): Requires range + quick decision → 0.9 runs
    - DPT (turned): Hardest skill (pivot at 2B) → 1.0 runs
    - DPF (finished): Routine catch at 1B → 0.3 runs

    Args:
        dps: Double plays started
        dpt: Double plays turned (pivot)
        dpf: Double plays finished
        position: Player's position (2B, SS, 3B, 1B)

    Returns:
        float: Double play run value

    Examples:
        2B with 40 DPS, 60 DPT, 50 DPF:
        >>> calculate_infielder_dp_value(40, 60, 50, '2B')
        111.0  # (40×0.9) + (60×1.0) + (50×0.3)

        3B with 15 DPS, 0 DPT, 0 DPF (only starts):
        >>> calculate_infielder_dp_value(15, 0, 0, '3B')
        13.5  # 15 × 0.9
    """
    # Position-specific weighting
    if position in ['2B', 'SS']:
        # Middle infielders: All three components
        return (dps * 0.9) + (dpt * 1.0) + (dpf * 0.3)

    elif position == '3B':
        # Third basemen: Only start DPs (rarely turn/finish)
        return dps * 0.9

    elif position == '1B':
        # First basemen: Only finish DPs (rarely start/turn)
        return dpf * 0.3

    else:
        return 0.0


def calculate_first_base_scoop_value(scoops: int) -> float:
    """
    Calculate value from scooping bad throws at first base.

    Elite 1B defenders (Goldschmidt, Rizzo, Olson) make 12-20 scoops/season.
    Each scoop prevents an error (0.25-0.3 run value).

    Args:
        scoops: Number of scoops recorded

    Returns:
        float: Run value from scoops

    Example:
        Elite 1B with 18 scoops:
        >>> calculate_first_base_scoop_value(18)
        10.8  # 18 × 0.6

    Note:
        - 0.6 runs/scoop is slightly higher than error value (0.3)
        - Accounts for preventing cascading errors (runner advances, etc.)
        - Helps elite 1B defenders offset -1.25 WAR positional penalty
    """
    return scoops * 0.6


def calculate_catcher_metrics_value(
    framing: float,
    throwing: float,
    blocking: float
) -> float:
    """
    Calculate weighted catcher defensive value.

    Weighting based on run value variance (2024 data):
    - Framing: 33-run range, 5.0 std → 60% weight (dominant factor)
    - Throwing: 13-run range, 2.0 std → 25% weight
    - Blocking: 10-run range, 1.4 std → 15% weight

    Args:
        framing: Pitch framing run value (FanGraphs)
        throwing: CS/SB run value (FanGraphs)
        blocking: PB/WP run value (FanGraphs)

    Returns:
        float: Weighted catcher defensive value

    Example:
        Elite framer with +20 framing, +5 throwing, +2 blocking:
        >>> calculate_catcher_metrics_value(20, 5, 2)
        14.55  # (20×0.60) + (5×0.25) + (2×0.15)

    Notes:
        - FanGraphs 'Throwing' already includes CS/SB (no double counting)
        - FanGraphs 'Blocking' already includes PB (no double counting)
        - 'Arm' metric dropped (all NaN in recent data, overlaps with Throwing)
    """
    return (framing * 0.60) + (throwing * 0.25) + (blocking * 0.15)


# ============================================================================
# Cap Application
# ============================================================================

def apply_defensive_cap(defense_runs: float, position: str) -> float:
    """
    Apply position-specific caps to Enhanced_Defense value.

    Prevents noise/outliers from distorting model while respecting
    true talent differences across positions.

    Args:
        defense_runs: Raw defensive value (before capping)
        position: Player's position

    Returns:
        float: Capped defensive value

    Example:
        Elite SS with +35 runs (outlier):
        >>> apply_defensive_cap(35, 'SS')
        30.0  # Capped at +30

        Elite 1B with +20 runs (too high for easy position):
        >>> apply_defensive_cap(20, '1B')
        15.0  # Capped at +15

    Notes:
        - SS/C have widest caps (±30/±25): Hardest positions, widest skill range
        - 1B has tightest positive cap (+15): Easiest position, limited impact
        - Asymmetric caps: Wider negative range (easier to be bad than elite)
    """
    if position not in ENHANCED_DEFENSE_CAPS:
        # Default cap if position unknown
        return max(-20, min(20, defense_runs))

    pos_cap, neg_cap = ENHANCED_DEFENSE_CAPS[position]

    return max(neg_cap, min(pos_cap, defense_runs))


# ============================================================================
# Position Average RF (Empirical Baselines)
# ============================================================================

# These are approximate MLB averages from recent seasons
# Will be refined with actual data in loader
POSITION_AVG_RF = {
    'C': 7.5,   # High PO (strikeouts), low A
    '1B': 9.0,  # Very high PO, moderate A
    '2B': 4.8,  # Balanced A and PO
    '3B': 2.5,  # Moderate A and PO
    'SS': 4.5,  # High A and PO (busiest infield position)
    'LF': 1.9,  # Low opportunities
    'CF': 2.6,  # Highest OF opportunities
    'RF': 2.0,  # Low opportunities, but some A (throws to bases)
    'DH': 0.0   # No defense
}
