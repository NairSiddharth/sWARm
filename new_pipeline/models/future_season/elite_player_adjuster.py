"""
Elite Player Adjustment for New Pipeline

Adapted from common_modules/elite_adjustment_base.py for new pipeline.

Implements confidence-based regression reduction to protect elite players
from over-regression before mathematical constraints are applied.
"""

__version__ = '2.0.0'
__author__ = 'oWAR Development Team'

# Standard library imports
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Third-party imports
import numpy as np
import pandas as pd

# New pipeline imports
from new_pipeline.common.features.confidence_scorer import SimpleConfidenceScorer
from new_pipeline.common.logging_config import get_logger

logger = get_logger(__name__)


class ElitePlayerAdjuster:
    """
    Enhanced elite player adjustment system using WAR tier-based protection.

    Replaces the 3-tier system with standardized 7-tier WAR classifications
    and adds position-specific protection factors.

    Core Logic:
    - MVP Level (6+ WAR): 75-80% regression reduction
    - Superstar (5-6 WAR): 65-75% regression reduction
    - All-Star (4-5 WAR): 60-70% regression reduction
    - Position-specific factors for catchers and pitchers
    """

    def __init__(self,
                 use_enhanced_system: bool = True,
                 elite_threshold: float = 5.5,
                 very_good_threshold: float = 4.5,
                 good_threshold: float = 3.0,
                 elite_protection: float = 0.4,
                 very_good_protection: float = 0.6,
                 good_protection: float = 0.8):
        """
        Initialize elite adjustment parameters.

        Args:
                use_enhanced_system: Use enhanced WAR tier system (recommended)
                elite_threshold: Legacy confidence threshold for elite players
                very_good_threshold: Legacy confidence threshold for very good players
                good_threshold: Legacy confidence threshold for good players
                elite_protection: Legacy regression reduction factor for elite players
                very_good_protection: Legacy regression reduction factor for very good players
                good_protection: Legacy regression reduction factor for good players
        """
        self.use_enhanced_system = use_enhanced_system

        # Always initialize legacy attributes for backward compatibility
        self.elite_threshold = elite_threshold
        self.very_good_threshold = very_good_threshold
        self.good_threshold = good_threshold
        self.elite_protection = elite_protection
        self.very_good_protection = very_good_protection
        self.good_protection = good_protection

        if use_enhanced_system:
            # Enhanced WAR tier system
            self.war_tiers = {
                'mvp_level': 6.0,
                'superstar': 5.0,
                'all_star': 4.0,
                'good_player': 3.0,
                'solid_starter': 2.0,
                'role_player': 1.0,
                'scrub': 0.0
            }

            # Protection factors by tier and position type
            self.protection_matrix = {
                'mvp_level': {'hitters': 0.25, 'starting_pitchers': 0.20, 'relief_pitchers': 0.30, 'catchers': 0.20},
                'superstar': {'hitters': 0.30, 'starting_pitchers': 0.25, 'relief_pitchers': 0.35, 'catchers': 0.25},
                'all_star': {'hitters': 0.35, 'starting_pitchers': 0.30, 'relief_pitchers': 0.40, 'catchers': 0.30},
                'good_player': {'hitters': 0.50, 'starting_pitchers': 0.40, 'relief_pitchers': 0.50, 'catchers': 0.40},
                'solid_starter': {'hitters': 0.70, 'starting_pitchers': 0.60, 'relief_pitchers': 0.70, 'catchers': 0.60},
                'role_player': {'hitters': 0.85, 'starting_pitchers': 0.80, 'relief_pitchers': 0.85, 'catchers': 0.80},
                'scrub': {'hitters': 1.00, 'starting_pitchers': 1.00, 'relief_pitchers': 1.00, 'catchers': 1.00}
            }

        # Initialize confidence scorer
        self.confidence_scorer = SimpleConfidenceScorer()

        # Track adjustment statistics
        self.adjustment_stats = {
            'total_players': 0,
            'elite_adjusted': 0,
            'very_good_adjusted': 0,
            'good_adjusted': 0,
            'regular_adjusted': 0,
            'total_war_protection': 0.0,
            'avg_protection_factor': 0.0
        }

    def classify_position_type(self, position: str) -> str:
        """
        Classify position into broader categories for protection matrix.

        Args:
                position: Player's primary position

        Returns:
                Position type for protection matrix lookup
        """
        position = str(position).upper()

        if position in ['C']:
            return 'catchers'
        elif position in ['P', 'SP']:
            return 'starting_pitchers'
        elif position in ['RP', 'CL']:
            return 'relief_pitchers'
        else:
            return 'hitters'

    def get_war_tier(self, war_value: float) -> str:
        """
        Get WAR tier classification for player.

        Args:
                war_value: Player's WAR value

        Returns:
                WAR tier classification
        """
        for tier, threshold in self.war_tiers.items():
            if war_value >= threshold:
                return tier
        return 'scrub'

    def get_protection_factor(self, war_value: float, position: str = 'OF') -> float:
        """
        Get protection factor using enhanced or legacy system.

        Args:
                war_value: Player's WAR value
                position: Player's position (used in enhanced system)

        Returns:
                Protection factor (lower = more protection)
        """
        if self.use_enhanced_system:
            tier = self.get_war_tier(war_value)
            position_type = self.classify_position_type(position)
            return self.protection_matrix[tier][position_type]
        else:
            # Legacy system based on confidence scores
            confidence = self.confidence_scorer.calculate_confidence_score(war_value)

            if confidence >= self.elite_threshold:
                return self.elite_protection
            elif confidence >= self.very_good_threshold:
                return self.very_good_protection
            elif confidence >= self.good_threshold:
                return self.good_protection
            else:
                return 1.0  # No protection

    def apply_elite_adjustment(self,
                               predictions: pd.DataFrame,
                               confidence_scores: pd.Series,
                               base_projections: pd.Series) -> pd.DataFrame:
        """
        Apply elite player adjustment to predictions.

        Args:
                predictions: Base predictions DataFrame
                confidence_scores: Confidence scores for each player
                base_projections: Base projection values (usually previous year's WAR)

        Returns:
                DataFrame with adjusted predictions

        Raises:
                ValueError: If input dataframes have mismatched indices
        """
        logger.info(f"Applying elite adjustment to {len(predictions)} predictions")

        try:
            # Validate inputs
            if not predictions.index.equals(confidence_scores.index):
                raise ValueError("Predictions and confidence scores must have same index")

            if not predictions.index.equals(base_projections.index):
                raise ValueError("Predictions and base projections must have same index")

            # Create adjusted predictions copy
            adjusted = predictions.copy()

            # Reset statistics
            self.adjustment_stats = {
                'total_players': len(predictions),
                'elite_adjusted': 0,
                'very_good_adjusted': 0,
                'good_adjusted': 0,
                'regular_adjusted': 0,
                'total_war_protection': 0.0,
                'avg_protection_factor': 0.0
            }

            # Apply adjustments
            for idx in predictions.index:
                confidence = confidence_scores[idx]
                base_value = base_projections[idx]
                predicted_value = predictions.loc[idx, 'predicted_war']

                # Determine protection factor
                if self.use_enhanced_system:
                    # Use enhanced WAR tier system with position-specific factors
                    protection = self.get_protection_factor(confidence, position='OF')
                    tier = self.get_war_tier(confidence)
                    if tier in ('mvp_level', 'superstar'):
                        self.adjustment_stats['elite_adjusted'] += 1
                    elif tier in ('all_star', 'good_player'):
                        self.adjustment_stats['very_good_adjusted'] += 1
                    elif tier in ('solid_starter', 'role_player'):
                        self.adjustment_stats['good_adjusted'] += 1
                    else:
                        self.adjustment_stats['regular_adjusted'] += 1
                else:
                    # Legacy threshold system
                    if confidence >= self.elite_threshold:
                        protection = self.elite_protection
                        self.adjustment_stats['elite_adjusted'] += 1
                    elif confidence >= self.very_good_threshold:
                        protection = self.very_good_protection
                        self.adjustment_stats['very_good_adjusted'] += 1
                    elif confidence >= self.good_threshold:
                        protection = self.good_protection
                        self.adjustment_stats['good_adjusted'] += 1
                    else:
                        protection = 1.0  # No protection
                        self.adjustment_stats['regular_adjusted'] += 1

                # Apply protection (reduce regression)
                regression_amount = base_value - predicted_value
                protected_regression = regression_amount * protection
                adjusted_value = base_value - protected_regression

                adjusted.loc[idx, 'predicted_war'] = adjusted_value
                adjusted.loc[idx, 'protection_factor'] = protection
                adjusted.loc[idx, 'war_adjustment'] = adjusted_value - predicted_value

                self.adjustment_stats['total_war_protection'] += (adjusted_value - predicted_value)

            # Calculate average protection
            if self.adjustment_stats['total_players'] > 0:
                self.adjustment_stats['avg_protection_factor'] = (
                    self.adjustment_stats['total_war_protection'] /
                    self.adjustment_stats['total_players']
                )

            logger.info(
                f"Elite adjustment complete: {
                    self.adjustment_stats['elite_adjusted']} elite, " f"{
                    self.adjustment_stats['very_good_adjusted']} very good, " f"{
                    self.adjustment_stats['good_adjusted']} good players adjusted")

            return adjusted

        except Exception as e:
            logger.error(f"Error in elite adjustment: {str(e)}", exc_info=True)
            raise

    def apply_war_tier_adjustment(self,
                                  player_data: pd.DataFrame,
                                  war_column: str = 'WAR',
                                  position_column: str = 'Position') -> pd.DataFrame:
        """
        Apply WAR tier-based adjustments to player projections.

        Args:
                player_data: DataFrame containing player data
                war_column: Name of WAR column
                position_column: Name of position column

        Returns:
                DataFrame with tier-based adjustments applied

        Raises:
                ValueError: If required columns are missing
        """
        logger.info(f"Applying WAR tier adjustment to {len(player_data)} players")

        # Validate required columns
        required_columns = [war_column, position_column]
        missing_columns = set(required_columns) - set(player_data.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        try:
            result = player_data.copy()

            # Add tier classifications and protection factors
            result['war_tier'] = result[war_column].apply(self.get_war_tier)

            result['protection_factor'] = result.apply(
                lambda row: self.get_protection_factor(
                    row[war_column],
                    row[position_column]
                ),
                axis=1
            )

            # Calculate adjusted WAR with protection
            # This represents how much regression to mean is reduced
            result['regression_reduction'] = 1.0 - result['protection_factor']
            result['adjusted_war'] = result[war_column] * (1 + result['regression_reduction'] * 0.2)

            # Cap adjustments to reasonable bounds
            result['adjusted_war'] = result['adjusted_war'].clip(lower=0, upper=12)

            # Log summary statistics
            tier_counts = result['war_tier'].value_counts()
            logger.info(f"WAR tier distribution: {tier_counts.to_dict()}")

            avg_protection = result['protection_factor'].mean()
            logger.info(f"Average protection factor: {avg_protection:.3f}")

            return result

        except Exception as e:
            logger.error(f"Error in WAR tier adjustment: {str(e)}", exc_info=True)
            raise

    def calculate_dynamic_protection(self,
                                     current_war: float,
                                     historical_wars: pd.Series,
                                     position: str = 'OF') -> float:
        """
        Calculate dynamic protection based on current and historical performance.

        Args:
                current_war: Current season WAR
                historical_wars: Series of historical WAR values
                position: Player's position

        Returns:
                Dynamic protection factor
        """
        try:
            # Base protection from current WAR
            base_protection = self.get_protection_factor(current_war, position)

            # Adjust based on historical consistency
            if len(historical_wars) >= 3:
                # Calculate consistency bonus/penalty
                war_std = historical_wars.std()
                war_mean = historical_wars.mean()

                if war_mean > 0:
                    consistency_factor = 1.0 - (war_std / war_mean)
                    consistency_factor = np.clip(consistency_factor, 0.8, 1.2)
                else:
                    consistency_factor = 1.0

                # Check for trend (improving/declining)
                if len(historical_wars) >= 5:
                    recent_trend = np.polyfit(range(len(historical_wars)), historical_wars, 1)[0]
                    trend_factor = 1.0 + np.clip(recent_trend * 0.1, -0.2, 0.2)
                else:
                    trend_factor = 1.0

                # Apply adjustments
                dynamic_protection = base_protection * consistency_factor * trend_factor
            else:
                # Not enough history, use base protection with uncertainty penalty
                dynamic_protection = base_protection * 1.1

            # Ensure within reasonable bounds
            return np.clip(dynamic_protection, 0.15, 1.0)

        except Exception as e:
            logger.warning(f"Error calculating dynamic protection, using base: {str(e)}")
            return base_protection

    def generate_adjustment_report(self) -> Dict:
        """
        Generate report on elite adjustments applied.

        Returns:
                Dictionary containing adjustment statistics
        """
        return {
            'summary': self.adjustment_stats.copy(),
            'protection_tiers': {
                'elite': {
                    'threshold': self.elite_threshold,
                    'protection': self.elite_protection,
                    'count': self.adjustment_stats['elite_adjusted']
                },
                'very_good': {
                    'threshold': self.very_good_threshold,
                    'protection': self.very_good_protection,
                    'count': self.adjustment_stats['very_good_adjusted']
                },
                'good': {
                    'threshold': self.good_threshold,
                    'protection': self.good_protection,
                    'count': self.adjustment_stats['good_adjusted']
                },
                'regular': {
                    'threshold': 0.0,
                    'protection': 1.0,
                    'count': self.adjustment_stats['regular_adjusted']
                }
            },
            'war_impact': {
                'total_protection': self.adjustment_stats['total_war_protection'],
                'average_protection': self.adjustment_stats['avg_protection_factor'],
                'protection_percentage': (
                    self.adjustment_stats['elite_adjusted'] +
                    self.adjustment_stats['very_good_adjusted'] +
                    self.adjustment_stats['good_adjusted']
                ) / max(self.adjustment_stats['total_players'], 1) * 100
            }
        }

    def apply_historical_adjustment(self,
                                   current_performance: float,
                                   historical_performance: List[float],
                                   position: str = 'OF',
                                   player_name: str = None) -> Dict[str, float]:
        """
        Apply elite adjustment using historical context for current season projections.

        This method addresses the systematic undervaluation of elite players by using
        their historical performance as context, not just current undervalued predictions.

        Args:
            current_performance: Model's current WAR prediction (often undervalued)
            historical_performance: List of historical WAR values [most recent, ..., oldest]
            position: Player's position (for position-specific protection)
            player_name: Optional player name for logging

        Returns:
            Dictionary containing:
                - adjusted_war: The adjusted WAR value
                - adjustment_amount: How much adjustment was applied
                - protection_factor: The protection factor used
                - baseline_war: The calculated historical baseline
                - reasoning: Explanation of adjustment
        """
        player_str = f" for {player_name}" if player_name else ""
        logger.info(f"Applying historical adjustment{player_str}")

        # If no historical data, return current performance unchanged
        if not historical_performance or len(historical_performance) == 0:
            logger.warning(f"No historical data available{player_str}, using current performance")
            return {
                'adjusted_war': current_performance,
                'adjustment_amount': 0.0,
                'protection_factor': 1.0,
                'baseline_war': current_performance,
                'reasoning': 'No historical data available'
            }

        # Calculate weighted historical baseline
        # Use 60/40 blend instead of 70/30 as requested
        weights = [0.5, 0.3, 0.2][:len(historical_performance)]  # Normalize if fewer years
        if len(historical_performance) < 3:
            # Renormalize weights
            weight_sum = sum(weights[:len(historical_performance)])
            weights = [w / weight_sum for w in weights[:len(historical_performance)]]

        baseline_war = sum(w * war for w, war in zip(weights, historical_performance))

        # For elite players (baseline >= 4.0), blend historical with current
        # Using 60% historical, 40% current as requested
        if baseline_war >= 4.0:
            effective_war = (historical_performance[0] * 0.6) + (current_performance * 0.4)
        else:
            effective_war = baseline_war

        # Determine protection tier based on effective WAR
        if self.use_enhanced_system:
            tier = self.get_war_tier(effective_war)
            position_type = self.classify_position_type(position)
            protection_factor = self.protection_matrix[tier][position_type]
        else:
            # Legacy system - use thresholds
            if effective_war >= 6.0:
                protection_factor = 0.20  # MVP tier
            elif effective_war >= 5.0:
                protection_factor = 0.25  # Superstar
            elif effective_war >= 4.0:
                protection_factor = 0.30  # All-Star
            elif effective_war >= 3.0:
                protection_factor = 0.40  # Good player
            elif effective_war >= 2.0:
                protection_factor = 0.60  # Solid starter
            else:
                protection_factor = 1.0  # No protection

        # Calculate regression amount
        regression_amount = effective_war - current_performance

        # IMPORTANT: Only apply protection to downward regression
        if regression_amount > 0:  # Model predicts lower than baseline
            # Apply protection to reduce regression
            protected_regression = regression_amount * protection_factor
            adjusted_war = effective_war - protected_regression
            tier_str = tier if self.use_enhanced_system else 'tier'
            reasoning = f"Elite protection applied: {tier_str} ({protection_factor:.0%} regression)"
        else:  # Model predicts higher than baseline
            # Allow full upward movement (no penalty for improvement)
            adjusted_war = current_performance
            reasoning = "Model predicts improvement - no protection needed"

        # Cap at reasonable bounds
        adjusted_war = max(0.0, min(adjusted_war, 10.0))

        logger.info(
            f"Historical adjustment{player_str}: "
            f"baseline={baseline_war:.2f}, current={current_performance:.2f}, "
            f"adjusted={adjusted_war:.2f} (protection={protection_factor:.0%})"
        )

        return {
            'adjusted_war': adjusted_war,
            'adjustment_amount': adjusted_war - current_performance,
            'protection_factor': protection_factor,
            'baseline_war': baseline_war,
            'effective_war': effective_war,
            'reasoning': reasoning
        }

    def identify_breakout_candidates(self,
                                     player_data: pd.DataFrame,
                                     lookback_years: int = 3) -> pd.DataFrame:
        """
        Identify potential breakout candidates based on recent trends.

        Args:
                player_data: DataFrame with player historical data
                lookback_years: Number of years to analyze

        Returns:
                DataFrame of potential breakout candidates
        """
        logger.info(f"Identifying breakout candidates from {len(player_data)} players")

        breakout_candidates = []

        try:
            for player_name in player_data['Name'].unique():
                player_history = player_data[player_data['Name'] == player_name].copy()
                player_history = player_history.sort_values('Season')

                if len(player_history) < lookback_years:
                    continue

                recent_data = player_history.tail(lookback_years)
                wars = recent_data['WAR'].values

                # Check for upward trend
                if len(wars) >= 2:
                    trend = np.polyfit(range(len(wars)), wars, 1)[0]

                    # Criteria for breakout candidate
                    if (trend > 0.5 and  # Strong upward trend
                            wars[-1] > wars[0] * 1.3 and  # 30% improvement
                            wars[-1] >= 2.5 and  # Current performance is good
                            wars[-1] < 5.0):  # Not already elite

                        breakout_candidates.append({
                            'Name': player_name,
                            'Current_WAR': wars[-1],
                            'Trend': trend,
                            'Improvement': (wars[-1] / wars[0] - 1) * 100,
                            'Position': recent_data.iloc[-1]['Position']
                        })

            result = pd.DataFrame(breakout_candidates)

            if not result.empty:
                result = result.sort_values('Trend', ascending=False)
                logger.info(f"Identified {len(result)} breakout candidates")
            else:
                logger.info("No breakout candidates identified")

            return result

        except Exception as e:
            logger.error(f"Error identifying breakout candidates: {str(e)}", exc_info=True)
            return pd.DataFrame()

    def _is_elite_potential_player(self, player_history: pd.DataFrame, current_war: float) -> bool:
        """
        Check if player has elite potential based on history.

        Args:
                player_history: Historical performance data
                current_war: Current WAR value

        Returns:
                True if player shows elite potential
        """
        if len(player_history) < 2:
            return False

        # Check for past elite performance
        max_war = player_history['WAR'].max()
        avg_war = player_history['WAR'].mean()

        return (max_war >= 5.0 or  # Had elite season
                (avg_war >= 3.5 and current_war >= 4.0))  # Consistent very good player

    def _is_declining_veteran(self, player_history: pd.DataFrame, current_war: float) -> bool:
        """
        Check if player matches declining veteran profile.

        Args:
                player_history: Historical performance data
                current_war: Current WAR value

        Returns:
                True if player appears to be declining veteran
        """
        if len(player_history) < 4:
            return False

        # Check for declining trend in recent years
        recent_wars = player_history.tail(3)['WAR'].values
        if len(recent_wars) >= 3:
            trend = np.polyfit(range(len(recent_wars)), recent_wars, 1)[0]
            peak_war = player_history['WAR'].max()

            return (trend < -0.5 and  # Declining trend
                    current_war < peak_war * 0.6)  # Significant decline from peak

        return False

    def _is_consistent_elite(self, player_history: pd.DataFrame, current_war: float) -> bool:
        """
        Check if player matches consistent elite profile (Soto-type).

        Args:
                player_history: Historical performance data
                current_war: Current WAR value

        Returns:
                True if player is consistently elite
        """
        if len(player_history) < 3:
            return False

        # Count elite seasons (4+ WAR)
        elite_seasons = (player_history['WAR'] >= 4.0).sum()
        total_seasons = len(player_history)

        return (elite_seasons >= 3 and  # Multiple elite seasons
                elite_seasons / total_seasons >= 0.6 and  # Consistent
                current_war >= 3.0)  # Currently performing well

    def _is_injury_compromised_legend(
            self,
            player_history: pd.DataFrame,
            current_war: float) -> bool:
        """
        Check if player matches injury-compromised legend profile (Trout-type).

        Args:
                player_history: Historical performance data
                current_war: Current WAR value

        Returns:
                True if player appears to be injury-compromised legend
        """
        if len(player_history) < 4:
            return False

        # Check for historical elite peak and recent decline
        peak_war = player_history['WAR'].max()
        recent_wars = player_history.tail(2)['WAR'].mean()

        return (peak_war >= 6.0 and  # Had elite peak
                recent_wars < peak_war * 0.4 and  # Significant decline
                current_war < 2.0)  # Currently struggling
