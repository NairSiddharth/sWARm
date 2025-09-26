"""
Expected Statistics Calculator Module
====================================

Handles 3-year weighted averages with expected metrics blending (75% real / 25% expected)
for regression-to-mean adjustments in future projections.

Classes:
    ExpectedStatsCalculator: Main class for calculating blended performance metrics
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import warnings


class ExpectedStatsCalculator:
    """
    Calculates 3-year weighted averages and blends real vs expected statistics.

    Implements ZIPS-style regression methodology with configurable weighting
    for luck-based over/under performance identification.
    """

    def __init__(self,
                 base_features: List[str] = None,
                 real_weight: float = 0.75,
                 expected_weight: float = 0.25,
                 year_weights: List[float] = None):
        """
        Initialize the expected statistics calculator.

        Args:
            base_features: List of statistical features to calculate
            real_weight: Weight for real-world performance (default 0.75)
            expected_weight: Weight for expected performance (default 0.25)
            year_weights: Weights for years t-3, t-2, t-1 (default [0.2, 0.3, 0.5])
        """
        if base_features is None:
            # Focus on key batting statistics that align with your one-season WAR calculation
            base_features = ['AVG', 'OBP', 'SLG', 'wOBA', 'ISO', 'BB%', 'K%', 'WAR']

        if year_weights is None:
            year_weights = [0.2, 0.3, 0.5]  # t-3, t-2, t-1

        if abs(real_weight + expected_weight - 1.0) > 1e-6:
            raise ValueError("real_weight + expected_weight must equal 1.0")

        if abs(sum(year_weights) - 1.0) > 1e-6:
            raise ValueError("year_weights must sum to 1.0")

        self.base_features = base_features
        self.real_weight = real_weight
        self.expected_weight = expected_weight
        self.year_weights = year_weights
        self.feature_cache = {}  # Cache calculated features

    def calculate_3yr_weighted_average(self,
                                     player_history: pd.DataFrame,
                                     target_season: int,
                                     player_id: str) -> Dict[str, float]:
        """
        Calculate 3-year weighted averages for a player's statistics.

        Args:
            player_history: Historical data for the player
            target_season: Season for which to calculate the average
            player_id: Player identifier for caching

        Returns:
            Dictionary of feature_name -> weighted_average
        """
        # Get the 3 years prior to target season
        relevant_years = [target_season - 3, target_season - 2, target_season - 1]

        # Filter to relevant seasons
        player_data = player_history[
            player_history['Season'].isin(relevant_years)
        ].copy().sort_values('Season')

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
                year_data = player_data[player_data['Season'] == year]
                if len(year_data) > 0:
                    # Use the most recent record if multiple (shouldn't happen with proper data)
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

    def get_expected_metrics(self,
                           current_season_data: pd.DataFrame,
                           player_id: str) -> Dict[str, Optional[float]]:
        """
        Extract expected metrics from Statcast data if available.

        Args:
            current_season_data: Current season data for the player
            player_id: Player identifier

        Returns:
            Dictionary of expected metric values (or None if unavailable)
        """
        expected_metrics = {}

        # Expected metrics mapping - only using xBA and xSLG as requested
        expected_features = {
            'xBA': 'AVG',
            'xSLG': 'SLG'
        }

        for expected_col, actual_col in expected_features.items():
            if expected_col in current_season_data.columns:
                expected_value = current_season_data[expected_col].iloc[-1]
                expected_metrics[actual_col] = expected_value if pd.notna(expected_value) else None
            else:
                expected_metrics[actual_col] = None

        # For other stats without expected values, return None (will use real values only)
        for feature in self.base_features:
            if feature not in expected_metrics:
                expected_metrics[feature] = None

        return expected_metrics

    def blend_real_vs_expected(self,
                              real_averages: Dict[str, float],
                              expected_metrics: Dict[str, Optional[float]],
                              consistency_factor: float = 1.0) -> Dict[str, float]:
        """
        Blend real-world averages with expected metrics for regression adjustment.

        Args:
            real_averages: 3-year weighted averages from real performance
            expected_metrics: Expected metrics from current season
            consistency_factor: Adjustment factor for player consistency (0.5-1.5)

        Returns:
            Dictionary of blended metric values
        """
        blended_metrics = {}

        for feature in real_averages.keys():
            real_value = real_averages[feature]
            expected_value = expected_metrics.get(feature)

            if expected_value is None or pd.isna(expected_value):
                # No expected data available, use real value
                blended_metrics[feature] = real_value
            else:
                # Calculate consistency-adjusted weights
                adjusted_real_weight = self._adjust_weight_for_consistency(
                    self.real_weight, consistency_factor
                )
                adjusted_expected_weight = 1.0 - adjusted_real_weight

                # Blend the values
                blended_value = (
                    adjusted_real_weight * real_value +
                    adjusted_expected_weight * expected_value
                )
                blended_metrics[feature] = blended_value

        return blended_metrics

    def _adjust_weight_for_consistency(self,
                                     base_real_weight: float,
                                     consistency_factor: float) -> float:
        """
        Adjust real vs expected weights based on player consistency.

        Higher consistency = more weight on real performance
        Lower consistency = more regression to expected

        Args:
            base_real_weight: Base weight for real performance
            consistency_factor: Player consistency metric (0.5-1.5)

        Returns:
            Adjusted weight for real performance
        """
        # Maximum 15% adjustment based on consistency
        max_adjustment = 0.15
        consistency_adjustment = (consistency_factor - 1.0) * max_adjustment

        adjusted_weight = base_real_weight + consistency_adjustment

        # Keep within reasonable bounds (0.6 - 0.9)
        adjusted_weight = max(0.6, min(0.9, adjusted_weight))

        return adjusted_weight

    def calculate_regression_factor(self,
                                  actual_vs_expected_gap: Dict[str, float]) -> float:
        """
        Calculate regression factor based on actual vs expected performance gaps.

        Args:
            actual_vs_expected_gap: Dictionary of feature -> (actual - expected) gaps

        Returns:
            Regression factor indicating likelihood of mean reversion
        """
        if not actual_vs_expected_gap:
            return 1.0  # No regression needed

        # Calculate standardized gaps (z-scores)
        gaps = list(actual_vs_expected_gap.values())
        if len(gaps) == 0:
            return 1.0

        # Convert gaps to absolute deviations
        abs_gaps = [abs(gap) for gap in gaps]
        mean_abs_gap = np.mean(abs_gaps)

        # Stronger regression for larger gaps
        # Scale: 0 gap = 1.0 factor, large gap = higher factor
        regression_factor = 1.0 + (mean_abs_gap * 2.0)  # Configurable multiplier

        # Cap the regression factor to prevent over-adjustment
        regression_factor = min(regression_factor, 2.0)

        return regression_factor

    def calculate_player_consistency_score(self,
                                         player_history: pd.DataFrame,
                                         lookback_years: int = 3) -> float:
        """
        Calculate a consistency score for a player based on performance variation.

        Args:
            player_history: Historical performance data
            lookback_years: Number of years to analyze

        Returns:
            Consistency score (higher = more consistent, typically 0.5-1.5)
        """
        if len(player_history) < 2:
            return 1.0  # Default consistency for players with limited history

        # Calculate coefficient of variation for key metrics
        consistency_features = ['WAR', 'wOBA', 'OPS']  # Key stability indicators
        available_features = [f for f in consistency_features if f in player_history.columns]

        if not available_features:
            return 1.0

        cv_scores = []

        for feature in available_features:
            feature_data = player_history[feature].dropna()
            if len(feature_data) >= 2:
                cv = feature_data.std() / abs(feature_data.mean()) if feature_data.mean() != 0 else 1.0
                cv_scores.append(cv)

        if not cv_scores:
            return 1.0

        # Convert coefficient of variation to consistency score
        mean_cv = np.mean(cv_scores)

        # Lower CV = higher consistency
        # Transform to 0.5-1.5 range with 1.0 as average
        consistency_score = max(0.5, min(1.5, 2.0 - mean_cv))

        return consistency_score

    def prepare_projection_features(self,
                                  player_data: pd.DataFrame,
                                  target_season: int,
                                  player_id: str) -> Dict[str, Union[float, Dict]]:
        """
        Main interface for preparing all projection features for a player.

        Args:
            player_data: Complete historical data for the player
            target_season: Season for which to prepare features
            player_id: Player identifier

        Returns:
            Dictionary with all calculated features for projections
        """
        # Calculate 3-year weighted averages
        real_averages = self.calculate_3yr_weighted_average(
            player_data, target_season, player_id
        )

        # Get expected metrics for most recent season
        current_data = player_data[player_data['Season'] == target_season - 1]
        expected_metrics = self.get_expected_metrics(current_data, player_id)

        # Calculate consistency score
        consistency_score = self.calculate_player_consistency_score(player_data)

        # Blend real and expected metrics
        blended_metrics = self.blend_real_vs_expected(
            real_averages, expected_metrics, consistency_score
        )

        # Calculate regression factors
        actual_vs_expected = {}
        for feature in blended_metrics.keys():
            if expected_metrics.get(feature) is not None:
                actual_vs_expected[feature] = (
                    real_averages[feature] - expected_metrics[feature]
                )

        regression_factor = self.calculate_regression_factor(actual_vs_expected)

        return {
            'real_averages': real_averages,
            'expected_metrics': expected_metrics,
            'blended_metrics': blended_metrics,
            'consistency_score': consistency_score,
            'regression_factor': regression_factor,
            'actual_vs_expected_gaps': actual_vs_expected
        }

    def batch_calculate_features(self,
                               data: pd.DataFrame,
                               target_season: int,
                               player_id_col: str = 'mlbid') -> pd.DataFrame:
        """
        Calculate projection features for all players in a dataset.

        Args:
            data: Complete dataset with all players
            target_season: Target season for projections
            player_id_col: Column name for player identifiers

        Returns:
            DataFrame with calculated features added
        """
        # Group by player and calculate features
        feature_results = []

        for player_id, player_data in data.groupby(player_id_col):
            features = self.prepare_projection_features(
                player_data, target_season, str(player_id)
            )

            # Flatten the features for DataFrame integration
            feature_row = {
                f'{player_id_col}': player_id,
                'target_season': target_season,
                'consistency_score': features['consistency_score'],
                'regression_factor': features['regression_factor']
            }

            # Add blended metrics
            for metric, value in features['blended_metrics'].items():
                feature_row[f'blended_{metric}'] = value

            # Add gaps
            for metric, gap in features['actual_vs_expected_gaps'].items():
                feature_row[f'gap_{metric}'] = gap

            feature_results.append(feature_row)

        features_df = pd.DataFrame(feature_results)

        # Merge back with original data if needed
        merge_cols = [player_id_col, 'target_season']
        if 'Season' in data.columns:
            # If merging with specific season data
            target_data = data[data['Season'] == target_season - 1].copy()
            result_df = target_data.merge(features_df, on=player_id_col, how='left')
        else:
            result_df = features_df

        return result_df