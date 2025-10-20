"""
Age Curve Adjustments for Future Projections

Position-specific aging patterns based on Dynasty Guru research.
Copied from future_season_modules/future_projections.py lines 55-135.

See FUTURE_PROJECTIONS_MIGRATION_GUIDE.md Section 3A.
"""

import numpy as np
from typing import Dict


class AgeCurveAdjuster:
    """
    Position-specific aging curve adjustments.

    Implements enhanced age curves with:
    - Peak performance range (26-29)
    - Logarithmic growth for young players
    - Position-specific decline rates
    """

    def __init__(self):
        """Initialize with position-specific aging parameters."""
        self.position_curves = self._initialize_position_curves()

    def _initialize_position_curves(self) -> Dict[str, Dict]:
        """
        Initialize position-specific aging parameters from research.

        Copied from future_projections.py lines 55-73.

        Returns:
            Dictionary of position -> aging parameters
        """
        return {
            'C': {'peak': 26, 'decline_rate': 0.035, 'career_length_median': 8},
            'SS': {'peak': 27, 'decline_rate': 0.025, 'career_length_median': 10},
            '2B': {'peak': 27, 'decline_rate': 0.025, 'career_length_median': 9},
            '3B': {'peak': 28, 'decline_rate': 0.020, 'career_length_median': 10},
            '1B': {'peak': 29, 'decline_rate': 0.015, 'career_length_median': 11},
            'LF': {'peak': 28, 'decline_rate': 0.020, 'career_length_median': 10},
            'CF': {'peak': 27, 'decline_rate': 0.025, 'career_length_median': 9},
            'RF': {'peak': 28, 'decline_rate': 0.020, 'career_length_median': 10},
            'DH': {'peak': 30, 'decline_rate': 0.015, 'career_length_median': 8},
            'P': {'peak': 27, 'decline_rate': 0.030, 'career_length_median': 7}
        }

    def get_age_factor(self, age: float, position: str = 'CF', use_log_transform: bool = True) -> float:
        """
        Calculate aging multiplier for given age and position.

        Copied from future_projections.py lines 75-135 (_calculate_enhanced_age_factor).

        Enhanced age curve calculation based on Dynasty Guru research:
        - Peak range 26-29 instead of single age
        - Logarithmic growth for young players (accelerating improvement)
        - Continued development 24-26
        - More realistic decline patterns

        Args:
            age: Player's age
            position: Player's primary position (default: 'CF' for general)
            use_log_transform: Whether to use logarithmic growth for young players

        Returns:
            Age factor multiplier (1.0 = peak performance)

        Examples:
            >>> adjuster = AgeCurveAdjuster()
            >>> adjuster.get_age_factor(27, 'SS')
            1.0  # At peak
            >>> adjuster.get_age_factor(32, 'SS')
            0.94  # 5 years past peak, moderate decline
        """
        # Ages < 20: Conservative baseline (limited data)
        if age < 20:
            return 0.70

        # Ages 20-24: Logarithmic growth implementation
        elif age < 24:
            if use_log_transform:
                # Dynasty Guru insight: accelerating improvement before age 24
                age_progress = (age - 20) / 4.0  # 0-1 scale for ages 20-24
                log_factor = np.log1p(age_progress) / np.log1p(1.0)  # Normalized log transform
                return 0.70 + (0.25 * log_factor)  # 70% -> 95% performance by age 24
            else:
                # Linear fallback
                return 0.70 + (0.25 * (age - 20) / 4.0)

        # Ages 24-25: Continued improvement phase
        elif age < 26:
            base_factor = 0.95  # From age 24 peak
            improvement = (age - 24) * 0.025  # 2.5% per year improvement
            return base_factor + improvement

        # Ages 26-29: Peak performance range (Dynasty Guru finding)
        elif 26 <= age <= 29:
            # Gentle variation within range using inverted parabola
            range_position = (age - 26) / 3.0  # 0-1 scale within peak range
            # Slight variation around 1.0, peaking in middle of range
            peak_variation = 0.03 * (1 - 4 * (range_position - 0.5)**2)
            return 1.0 + peak_variation

        # Ages 30+: Gradual decline with early-stage protection
        else:
            position_curve = self.position_curves.get(position, self.position_curves['CF'])

            if age <= 31:
                # Ages 30-31: Gentle decline (1.5% per year)
                years_past_peak = age - 29
                return 1.0 - (years_past_peak * 0.015)
            else:
                # Ages 32+: Standard position-based decline
                base_decline = 1.0 - 2 * 0.015  # From ages 30-31
                years_past_31 = age - 31
                position_decline_rate = position_curve['decline_rate']
                return max(0.3, base_decline - (years_past_31 * position_decline_rate))

    def get_decline_pattern(self, ages: np.ndarray, position: str = 'CF') -> np.ndarray:
        """
        Get age factors for an array of ages.

        Useful for visualizing aging curves.

        Args:
            ages: Array of ages
            position: Position for curve

        Returns:
            Array of age factors
        """
        return np.array([self.get_age_factor(age, position) for age in ages])
