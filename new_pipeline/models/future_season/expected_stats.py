"""
Expected Stats Calculator - Luck Correction for Projections

Implements regression to expected stats (xBA, xSLG) and 3-year weighted averages.
Adapted from future_season_modules/expected_stats.py with new pipeline features.

See FUTURE_PROJECTIONS_MIGRATION_GUIDE.md Section 2 for migration notes.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import warnings
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from new_pipeline.models.future_season.constants import FUTURE_HITTER_MODEL_FEATURES, FUTURE_PITCHER_MODEL_FEATURES


class ExpectedStatsCalculator:
    """
    Calculate expected statistics and regression adjustments.

    Key features:
    - 3-year weighted averages (20% t-3, 30% t-2, 50% t-1)
    - xBA/xSLG regression for luck correction
    - Consistency-adjusted blending

    Uses NEW pipeline features (not old wOBA/ISO):
    - Hitters: K%, BB%, AVG, OBP, SLG, GDP, etc.
    - Pitchers: K%, BB%, ERA, GB%, etc.
    """

    def __init__(
        self,
        player_type: str = 'hitter',
        real_weight: float = 0.75,
        expected_weight: float = 0.25,
        year_weights: Optional[List[float]] = None
    ):
        """
        Initialize expected stats calculator.

        Args:
            player_type: 'hitter' or 'pitcher'
            real_weight: Weight for real-world performance (default 0.75)
            expected_weight: Weight for expected performance (default 0.25)
            year_weights: Weights for years t-3, t-2, t-1 (default [0.2, 0.3, 0.5])
        """
        if player_type not in ['hitter', 'pitcher']:
            raise ValueError(f"player_type must be 'hitter' or 'pitcher', got {player_type}")

        # Use new pipeline features instead of old wOBA/ISO
        if player_type == 'hitter':
            # NEW: Use new pipeline features + WAR
            self.base_features = list(FUTURE_HITTER_MODEL_FEATURES) + ['WAR']
        else:
            self.base_features = list(FUTURE_PITCHER_MODEL_FEATURES) + ['WAR']

        # Consistency features (for variance calculation)
        # OLD: ['WAR', 'wOBA', 'OPS']
        # NEW: Use SLG+OBP as OPS equivalent
        self.consistency_features = ['WAR', 'SLG', 'OBP']

        if year_weights is None:
            year_weights = [0.2, 0.3, 0.5]  # t-3, t-2, t-1

        if abs(real_weight + expected_weight - 1.0) > 1e-6:
            raise ValueError("real_weight + expected_weight must equal 1.0")

        if abs(sum(year_weights) - 1.0) > 1e-6:
            raise ValueError("year_weights must sum to 1.0")

        self.player_type = player_type
        self.real_weight = real_weight
        self.expected_weight = expected_weight
        self.year_weights = year_weights

    def calculate_3yr_weighted_average(
        self,
        player_history: pd.DataFrame,
        target_year: int
    ) -> Dict[str, float]:
        """
        Calculate 3-year weighted averages for a player's statistics.

        Copied from old expected_stats.py lines 59-113, adapted for new features.

        Args:
            player_history: Historical data for the player (must have 'Year' column)
            target_year: Year for which to calculate the average

        Returns:
            Dictionary of feature_name -> weighted_average
        """
        # Get the 3 years prior to target year
        relevant_years = [target_year - 3, target_year - 2, target_year - 1]

        # Filter to relevant years
        player_data = player_history[
            player_history['Year'].isin(relevant_years)
        ].copy().sort_values('Year')

        if len(player_data) == 0:
            return {}

        weighted_averages = {}

        for feature in self.base_features:
            if feature not in player_data.columns:
                warnings.warn(f"Feature '{feature}' not found in player data")
                continue

            # Calculate weighted average based on available years
            feature_values = []
            weights = []

            for i, year in enumerate(relevant_years):
                year_data = player_data[player_data['Year'] == year]
                if len(year_data) > 0:
                    feature_value = year_data[feature].iloc[-1]
                    if pd.notna(feature_value):
                        feature_values.append(feature_value)
                        weights.append(self.year_weights[i])

            if feature_values:
                # Normalize weights for available years
                weights = np.array(weights)
                weights = weights / weights.sum()

                weighted_avg = np.average(feature_values, weights=weights)
                weighted_averages[feature] = weighted_avg

        return weighted_averages

    def apply_expected_stats_regression(
        self,
        real_averages: Dict[str, float],
        expected_metrics: Optional[Dict[str, float]] = None,
        consistency_factor: float = 1.0
    ) -> Dict[str, float]:
        """
        Apply regression to expected stats for luck correction.

        Uses xBA and xSLG if available to regress AVG and SLG toward expected values.

        Args:
            real_averages: 3-year weighted averages from real performance
            expected_metrics: Expected metrics (xBA, xSLG) if available
            consistency_factor: Adjustment for player consistency (0.5-1.5)

        Returns:
            Dictionary of regressed metric values
        """
        if expected_metrics is None:
            # No expected data, return real averages
            return real_averages.copy()

        regressed_metrics = real_averages.copy()

        # Expected metrics mapping
        # Only xBA and xSLG are commonly available
        expected_mappings = {
            'xBA': 'AVG',
            'xSLG': 'SLG'
        }

        for expected_col, actual_col in expected_mappings.items():
            if expected_col in expected_metrics and actual_col in real_averages:
                real_value = real_averages[actual_col]
                expected_value = expected_metrics[expected_col]

                if expected_value is not None and pd.notna(expected_value):
                    # Calculate consistency-adjusted weights
                    adjusted_real_weight = self._adjust_weight_for_consistency(
                        self.real_weight, consistency_factor
                    )
                    adjusted_expected_weight = 1.0 - adjusted_real_weight

                    # Blend the values (regression to expected)
                    regressed_value = (
                        adjusted_real_weight * real_value +
                        adjusted_expected_weight * expected_value
                    )
                    regressed_metrics[actual_col] = regressed_value

        return regressed_metrics

    def _adjust_weight_for_consistency(
        self,
        base_real_weight: float,
        consistency_factor: float
    ) -> float:
        """
        Adjust real vs expected weights based on player consistency.

        More consistent players get higher real weight.
        Less consistent players regress more toward expected.

        Args:
            base_real_weight: Base weight for real performance
            consistency_factor: Consistency multiplier (0.5-1.5)

        Returns:
            Adjusted real weight
        """
        # Adjust weight based on consistency
        # consistency_factor > 1.0 = more consistent = higher real weight
        # consistency_factor < 1.0 = less consistent = lower real weight (more regression)
        adjusted_weight = base_real_weight * consistency_factor

        # Keep within reasonable bounds
        adjusted_weight = max(0.5, min(0.95, adjusted_weight))

        return adjusted_weight

    def calculate_consistency_factor(
        self,
        player_history: pd.DataFrame,
        target_year: int
    ) -> float:
        """
        Calculate player consistency factor based on stat variance.

        Uses variance in consistency features (WAR, SLG, OBP) over past 3 years.
        Lower variance = more consistent = higher factor.

        Args:
            player_history: Historical data for the player
            target_year: Year for context

        Returns:
            Consistency factor (0.5-1.5, default 1.0)
        """
        # Get 3 years prior to target
        relevant_years = [target_year - 3, target_year - 2, target_year - 1]
        player_data = player_history[player_history['Year'].isin(relevant_years)].copy()

        if len(player_data) < 2:
            return 1.0  # Default consistency

        # Calculate coefficient of variation for consistency features
        cv_values = []
        for feature in self.consistency_features:
            if feature in player_data.columns:
                values = player_data[feature].dropna()
                if len(values) >= 2 and values.mean() != 0:
                    cv = values.std() / abs(values.mean())
                    cv_values.append(cv)

        if not cv_values:
            return 1.0

        # Average CV
        avg_cv = np.mean(cv_values)

        # Convert to consistency factor
        # Lower CV = more consistent = higher factor
        # Typical CV for MLB players: 0.3-0.5
        if avg_cv < 0.3:
            consistency_factor = 1.3  # Very consistent
        elif avg_cv < 0.4:
            consistency_factor = 1.1  # Above average
        elif avg_cv < 0.5:
            consistency_factor = 1.0  # Average
        elif avg_cv < 0.7:
            consistency_factor = 0.9  # Below average
        else:
            consistency_factor = 0.7  # Inconsistent

        return consistency_factor
