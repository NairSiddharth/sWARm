"""
Elite Player Adjustment Module
=============================

Implements the two-stage pipeline approach for Option C:
Base Projections → Elite Adjustment → Constraint Optimization

This module applies confidence-based regression reduction to protect elite players
from over-regression before mathematical constraints are applied.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
import warnings
import sys
from pathlib import Path

# Add project root to path for confidence scorer import
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from common_modules.confidence_scorer import SimpleConfidenceScorer

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
            'no_adjustment': 0,
            'mean_elite_adjustment': 0.0,
            'max_adjustment': 0.0,
            'min_adjustment': 0.0
        }

    def calculate_regression_reduction(self, confidence: float = None, war_value: float = None, position: str = None) -> float:
        """
        Calculate regression reduction factor based on confidence score or WAR tier.

        Args:
            confidence: Player confidence score (0.5-8.0 range) - legacy system
            war_value: Current WAR value - enhanced system
            position: Player position - enhanced system

        Returns:
            Regression reduction factor (lower = more protection)
        """
        if self.use_enhanced_system and war_value is not None and position is not None:
            return self.get_enhanced_protection_factor(war_value, position)
        else:
            # Legacy confidence-based system
            confidence = confidence or 1.0
            if confidence >= self.elite_threshold:
                return self.elite_protection
            elif confidence >= self.very_good_threshold:
                return self.very_good_protection
            elif confidence >= self.good_threshold:
                return self.good_protection
            else:
                return 1.0  # No protection for average players

    def get_enhanced_protection_factor(self, war_value: float, position: str) -> float:
        """
        Get protection factor using enhanced WAR tier system.

        Args:
            war_value: Current WAR value
            position: Player position

        Returns:
            Protection factor (lower = more protection)
        """
        tier = self.classify_player_tier(war_value)
        position_type = self.classify_position_type(position)
        return self.protection_matrix[tier][position_type]

    def classify_player_tier(self, war_value: float) -> str:
        """Classify player into WAR tier."""
        if war_value >= self.war_tiers['mvp_level']:
            return 'mvp_level'
        elif war_value >= self.war_tiers['superstar']:
            return 'superstar'
        elif war_value >= self.war_tiers['all_star']:
            return 'all_star'
        elif war_value >= self.war_tiers['good_player']:
            return 'good_player'
        elif war_value >= self.war_tiers['solid_starter']:
            return 'solid_starter'
        elif war_value >= self.war_tiers['role_player']:
            return 'role_player'
        else:
            return 'scrub'

    def classify_position_type(self, position: str) -> str:
        """Classify position into protection category."""
        if position == 'C':
            return 'catchers'
        elif position in ['SP', 'P'] and position != 'C':
            return 'starting_pitchers'
        elif position in ['RP', 'CL']:
            return 'relief_pitchers'
        else:
            return 'hitters'

    def get_protection_factor(self, war_value: float, position: str) -> float:
        """
        Public interface for getting protection factor.

        Args:
            war_value: Current WAR value
            position: Player position

        Returns:
            Protection factor (lower = more protection)
        """
        if self.use_enhanced_system:
            return self.get_enhanced_protection_factor(war_value, position)
        else:
            # Legacy system fallback
            confidence = self.confidence_scorer.calculate_confidence({'WAR': war_value})
            return self.calculate_regression_reduction(confidence=confidence)

    def apply_positional_adjustments(self, position: str, base_adjustment: float) -> float:
        """
        Apply position-specific adjustments for elite players.

        Args:
            position: Player position
            base_adjustment: Base adjustment amount

        Returns:
            Position-adjusted amount
        """
        # Position scarcity factors - more valuable positions get slight boost
        position_factors = {
            'C': 1.05,    # Catcher scarcity
            'SS': 1.03,   # Shortstop scarcity
            'CF': 1.02,   # Center field premium
            '2B': 1.01,   # Second base slight premium
            '3B': 1.00,   # Third base baseline
            '1B': 0.99,   # First base slight penalty
            'LF': 0.98,   # Left field penalty
            'RF': 0.98,   # Right field penalty
            'DH': 0.95,   # Designated hitter penalty
            'P': 1.00,    # Pitcher baseline
            'OF': 0.99    # Generic outfield
        }

        factor = position_factors.get(position, 1.0)
        return base_adjustment * factor

    def apply_adjustment(self,
                        current_performance: float,
                        confidence_score: float,
                        player_type: str = 'hitter',
                        player_name: str = 'Unknown',
                        position: str = None) -> Dict:
        """
        Apply elite adjustment to a single player's performance.

        Args:
            current_performance: Current WAR/WARP value
            confidence_score: Confidence score (0-8 scale)
            player_type: 'hitter' or 'pitcher'
            player_name: Player name for logging
            position: Player position for positional adjustments

        Returns:
            Dictionary with adjustment results
        """
        # Calculate regression reduction factor
        regression_factor = self.calculate_regression_reduction(confidence_score)

        # For simplified implementation, assume target performance based on confidence
        # High confidence players should perform closer to elite levels
        if confidence_score >= self.elite_threshold:
            # Elite players: reduce regression by protection factor
            target_multiplier = 1.0 + (1.0 - regression_factor) * 0.7  # Boost toward elite
        elif confidence_score >= self.very_good_threshold:
            target_multiplier = 1.0 + (1.0 - regression_factor) * 0.5  # Moderate boost
        elif confidence_score >= self.good_threshold:
            target_multiplier = 1.0 + (1.0 - regression_factor) * 0.3  # Small boost
        else:
            target_multiplier = 1.0  # No adjustment

        # Calculate adjusted performance
        adjusted_performance = current_performance * target_multiplier

        # Apply positional adjustments if position provided
        if position:
            positional_factor = self.position_factors.get(position, 1.0)
            adjusted_performance *= positional_factor

        # Calculate adjustment amount
        adjustment_amount = adjusted_performance - current_performance

        # Determine protection level
        if confidence_score >= self.elite_threshold:
            protection_level = "elite"
        elif confidence_score >= self.very_good_threshold:
            protection_level = "very_good"
        elif confidence_score >= self.good_threshold:
            protection_level = "good"
        else:
            protection_level = "none"

        return {
            'original_performance': current_performance,
            'adjusted_performance': adjusted_performance,
            'adjustment_amount': adjustment_amount,
            'confidence_score': confidence_score,
            'protection_level': protection_level,
            'regression_factor': regression_factor
        }

    def adjust_elite_projections(self,
                                projections_df: pd.DataFrame,
                                confidence_scores: Dict[int, float],
                                training_data: pd.DataFrame = None) -> pd.DataFrame:
        """
        Apply confidence-based adjustments to reduce elite player over-regression.

        Args:
            projections_df: DataFrame with base projections
            confidence_scores: Dictionary mapping mlbid to confidence scores
            training_data: Original training data for fallback confidence calculation

        Returns:
            DataFrame with elite-adjusted projections
        """
        print("Applying elite player adjustments...")

        adjusted_df = projections_df.copy()
        adjustments_made = []

        # Reset adjustment statistics
        self.adjustment_stats = {key: 0 if 'adjustment' in key else 0.0
                               for key in self.adjustment_stats.keys()}
        self.adjustment_stats['total_players'] = len(adjusted_df)

        # Process each player
        for idx, row in adjusted_df.iterrows():
            player_id = row.get('mlbid')
            position = row.get('Position', 'OF')
            age = row.get('Age', 30)

            # Get confidence score (calculate if not provided)
            if confidence_scores is not None:
                confidence = confidence_scores.get(player_id, 1.0)
            else:
                # Calculate confidence using proper confidence scorer
                confidence = self._calculate_proper_confidence(row, training_data)

            # Calculate regression reduction factor
            regression_factor = self.calculate_regression_reduction(confidence)

            # Skip if no adjustment needed
            if regression_factor >= 1.0:
                self.adjustment_stats['no_adjustment'] += 1
                continue

            # Calculate adjustment for each projection year
            projection_cols = [col for col in adjusted_df.columns
                             if col.startswith('projected_') and 'year_' in col]

            player_adjustments = {}
            max_adjustment = 0.0

            for col in projection_cols:
                current_projection = row[col]

                if pd.isna(current_projection):
                    continue

                # Get corresponding current performance for baseline
                metric_type = col.split('_')[1]  # WAR or WARP
                current_col = f'Current_{metric_type}'
                current_performance = row.get(current_col, current_projection)

                if pd.isna(current_performance):
                    current_performance = current_projection

                # Calculate mean reversion amount (difference from current to projection)
                regression_amount = current_performance - current_projection

                # Reduce regression by protection factor
                reduced_regression = regression_amount * regression_factor

                # Calculate new projection (less regression toward mean)
                new_projection = current_performance - reduced_regression

                # Apply positional adjustment
                adjustment_amount = new_projection - current_projection
                adjusted_amount = self.apply_positional_adjustments(position, adjustment_amount)
                final_projection = current_projection + adjusted_amount

                # Update the projection
                adjusted_df.at[idx, col] = final_projection

                # Track adjustment
                actual_adjustment = final_projection - current_projection
                player_adjustments[col] = actual_adjustment
                max_adjustment = max(max_adjustment, abs(actual_adjustment))

            # Update statistics
            if confidence >= self.elite_threshold:
                self.adjustment_stats['elite_adjusted'] += 1
                category = 'elite'
            elif confidence >= self.very_good_threshold:
                self.adjustment_stats['very_good_adjusted'] += 1
                category = 'very_good'
            elif confidence >= self.good_threshold:
                self.adjustment_stats['good_adjusted'] += 1
                category = 'good'
            else:
                continue  # Should not reach here due to earlier check

            # Track for logging
            if player_adjustments:
                adjustments_made.append({
                    'player_id': player_id,
                    'name': row.get('Name', 'Unknown'),
                    'confidence': confidence,
                    'category': category,
                    'max_adjustment': max_adjustment,
                    'adjustments': player_adjustments
                })

        # Calculate summary statistics
        if adjustments_made:
            all_adjustments = []
            for adj in adjustments_made:
                all_adjustments.extend(adj['adjustments'].values())

            self.adjustment_stats['mean_elite_adjustment'] = np.mean(all_adjustments)
            self.adjustment_stats['max_adjustment'] = max(all_adjustments)
            self.adjustment_stats['min_adjustment'] = min(all_adjustments)

        # Log adjustment summary
        self._log_adjustment_summary(adjustments_made)

        return adjusted_df

    def _log_adjustment_summary(self, adjustments_made: list):
        """Log summary of adjustments made."""
        print(f"\nElite Player Adjustment Summary:")
        print(f"  Total players: {self.adjustment_stats['total_players']}")
        print(f"  Elite players adjusted: {self.adjustment_stats['elite_adjusted']}")
        print(f"  Very good players adjusted: {self.adjustment_stats['very_good_adjusted']}")
        print(f"  Good players adjusted: {self.adjustment_stats['good_adjusted']}")
        print(f"  No adjustment needed: {self.adjustment_stats['no_adjustment']}")

        if adjustments_made:
            print(f"  Mean adjustment: {self.adjustment_stats['mean_elite_adjustment']:.3f} WAR")
            print(f"  Max adjustment: {self.adjustment_stats['max_adjustment']:.3f} WAR")
            print(f"  Min adjustment: {self.adjustment_stats['min_adjustment']:.3f} WAR")

            # Show top adjustments
            top_adjustments = sorted(adjustments_made,
                                   key=lambda x: x['max_adjustment'],
                                   reverse=True)[:5]

            print(f"\n  Top 5 Elite Adjustments:")
            for adj in top_adjustments:
                print(f"    {adj['name']} ({adj['category']}): "
                      f"+{adj['max_adjustment']:.2f} WAR (confidence: {adj['confidence']:.1f})")

    def get_adjustment_statistics(self) -> Dict:
        """Return adjustment statistics for validation."""
        return self.adjustment_stats.copy()

    def validate_adjustments(self,
                           original_df: pd.DataFrame,
                           adjusted_df: pd.DataFrame) -> Dict[str, bool]:
        """
        Validate that adjustments are reasonable and don't break constraints.

        Args:
            original_df: Original projections
            adjusted_df: Elite-adjusted projections

        Returns:
            Validation results dictionary
        """
        validation_results = {
            'shape_preserved': False,
            'no_extreme_adjustments': False,
            'elite_players_improved': False,
            'total_war_reasonable': False
        }

        # Check shape preservation
        validation_results['shape_preserved'] = (
            original_df.shape == adjusted_df.shape and
            list(original_df.columns) == list(adjusted_df.columns)
        )

        # Check for extreme adjustments (> 5 WAR change)
        projection_cols = [col for col in original_df.columns
                         if col.startswith('projected_') and 'year_' in col]

        max_adjustment = 0.0
        for col in projection_cols:
            if col in original_df.columns and col in adjusted_df.columns:
                differences = (adjusted_df[col] - original_df[col]).dropna()
                if len(differences) > 0:
                    max_adjustment = max(max_adjustment, differences.abs().max())

        validation_results['no_extreme_adjustments'] = max_adjustment <= 5.0

        # Check if elite players improved
        if 'projected_WAR_year_1' in projection_cols:
            original_top = original_df.nlargest(10, 'projected_WAR_year_1')['projected_WAR_year_1'].mean()
            adjusted_top = adjusted_df.nlargest(10, 'projected_WAR_year_1')['projected_WAR_year_1'].mean()
            validation_results['elite_players_improved'] = adjusted_top > original_top

        # Check total WAR is still reasonable (within 20% of original)
        if 'projected_WAR_year_1' in projection_cols:
            original_total = original_df['projected_WAR_year_1'].sum()
            adjusted_total = adjusted_df['projected_WAR_year_1'].sum()
            if original_total > 0:
                change_pct = abs(adjusted_total - original_total) / original_total
                validation_results['total_war_reasonable'] = change_pct <= 0.20

        return validation_results

    def _calculate_proper_confidence(self, player_row, training_data) -> float:
        """
        Calculate confidence score using the proper SimpleConfidenceScorer system
        with profile-based enhancements.

        Args:
            player_row: Row from projections DataFrame with player info
            training_data: Training dataset for profile classification

        Returns:
            Enhanced confidence score on 1.5-8.0+ scale
        """
        # Extract current projected performance
        current_war = player_row.get('projected_WAR_year_1', player_row.get('Current_WAR', 0))
        age = player_row.get('Age', 28)
        position = player_row.get('Position', 'OF')

        # Base confidence from proper scorer
        base_confidence = self.confidence_scorer.calculate_confidence_score(
            war_value=current_war,
            age=age,
            position=position
        )

        # Apply profile-based enhancements
        enhanced_confidence = self._apply_profile_enhancements(
            base_confidence, player_row, training_data
        )

        return enhanced_confidence

    def _apply_profile_enhancements(self, base_confidence: float, player_row, training_data) -> float:
        """
        Apply profile-based confidence enhancements for elite players.

        Args:
            base_confidence: Base confidence score
            player_row: Player information
            training_data: Training dataset

        Returns:
            Enhanced confidence score
        """
        enhanced_confidence = base_confidence
        player_id = player_row.get('mlbid')

        if player_id is None or training_data is None:
            return enhanced_confidence

        try:
            # Get player's historical performance for profile classification
            player_history = training_data[training_data['mlbid'] == player_id]

            if len(player_history) == 0:
                return enhanced_confidence

            # Apply elite player profile enhancements
            age = player_row.get('Age', 28)
            current_war = player_row.get('projected_WAR_year_1', player_row.get('Current_WAR', 0))

            # Late bloomer enhancement (Judge-type) - Conservative
            if self._is_late_bloomer(player_history, age, current_war):
                enhanced_confidence *= 1.15  # 15% confidence boost
                print(f"    Late bloomer boost applied: {base_confidence:.2f} -> {enhanced_confidence:.2f}")

            # Consistent elite enhancement (Soto-type) - Conservative
            elif self._is_consistent_elite(player_history, current_war):
                enhanced_confidence *= 1.1  # 10% confidence boost
                print(f"    Consistent elite boost applied: {base_confidence:.2f} -> {enhanced_confidence:.2f}")

            # Injury-compromised legend floor (Trout-type) - Conservative
            elif self._is_injury_compromised_legend(player_history, current_war):
                # Apply modest minimum confidence floor for former elites
                min_confidence = 3.5  # Ensure at least modest protection
                enhanced_confidence = max(enhanced_confidence, min_confidence)
                if enhanced_confidence > base_confidence:
                    print(f"    Injury-compromised legend floor applied: {base_confidence:.2f} -> {enhanced_confidence:.2f}")

            # Cap at maximum
            enhanced_confidence = min(enhanced_confidence, 8.0)

        except Exception as e:
            print(f"    Profile enhancement error for player {player_id}: {e}")
            # Fall back to base confidence on error
            enhanced_confidence = base_confidence

        return enhanced_confidence

    def _is_late_bloomer(self, player_history, current_age: int, current_war: float) -> bool:
        """Check if player matches late bloomer profile (Judge-type)."""
        if len(player_history) < 2:
            return False

        # Check debut age and current elite performance
        first_season = player_history.iloc[0]
        debut_age = first_season.get('Age', 25)

        return (debut_age >= 25 and  # Late debut
                current_age <= 35 and  # Still in prime
                current_war >= 4.0)  # Currently elite

    def _is_consistent_elite(self, player_history, current_war: float) -> bool:
        """Check if player matches consistent elite profile (Soto-type)."""
        if len(player_history) < 3:
            return False

        # Count elite seasons (4+ WAR)
        elite_seasons = (player_history['WAR'] >= 4.0).sum()
        total_seasons = len(player_history)

        return (elite_seasons >= 3 and  # Multiple elite seasons
                elite_seasons / total_seasons >= 0.6 and  # Consistent
                current_war >= 3.0)  # Currently performing well

    def _is_injury_compromised_legend(self, player_history, current_war: float) -> bool:
        """Check if player matches injury-compromised legend profile (Trout-type)."""
        if len(player_history) < 4:
            return False

        # Check for historical elite peak and recent decline
        peak_war = player_history['WAR'].max()
        recent_wars = player_history.tail(2)['WAR'].mean()

        return (peak_war >= 6.0 and  # Had elite peak
                recent_wars < peak_war * 0.4 and  # Significant decline
                current_war < 2.0)  # Currently struggling


class TwoWayEliteProtection(ElitePlayerAdjuster):
    """
    Enhanced elite protection for two-way players using component-based approach.

    Provides separate protection for pitching and hitting components while
    considering cross-role interaction effects and injury-aware adjustments.
    """

    def __init__(self, use_enhanced_system: bool = True):
        """Initialize with base elite adjuster functionality."""
        super().__init__(use_enhanced_system=use_enhanced_system)

        # Component interaction factors
        self.interaction_factors = {
            'dual_role_bonus': 1.1,      # Slight bonus for maintaining both roles
            'injury_compensation': 1.15,  # Extra protection during role-specific injuries
            'focus_shift_penalty': 0.95   # Minor penalty when shifting focus
        }

    def get_two_way_protection(self,
                             pitcher_war: float,
                             hitter_war: float,
                             injury_status: dict = None) -> dict:
        """
        Calculate component-based protection factors.

        Args:
            pitcher_war: Pitcher WAR component
            hitter_war: Hitter WAR component
            injury_status: Current injury effects

        Returns:
            Dictionary with component protection factors
        """
        # Base component protections using the enhanced system
        pitcher_protection = self.get_protection_factor(pitcher_war, 'P')
        hitter_protection = self.get_protection_factor(hitter_war, 'DH')

        # Apply interaction effects
        if injury_status:
            if injury_status.get('pitcher_affected'):
                # Pitching injury: protect pitcher more, boost hitter slightly
                pitcher_protection *= self.interaction_factors['injury_compensation']
                hitter_protection *= 0.95  # Slight hitter boost from focus shift

            elif injury_status.get('hitter_affected'):
                # Hitting injury: protect hitter more, pitcher slightly less
                hitter_protection *= self.interaction_factors['injury_compensation']
                pitcher_protection *= self.interaction_factors['focus_shift_penalty']

        # Dual-role bonus (when both components active)
        if pitcher_war > 1.0 and hitter_war > 1.0:
            dual_bonus = self.interaction_factors['dual_role_bonus']
            pitcher_protection *= dual_bonus
            hitter_protection *= dual_bonus

        return {
            'pitcher_protection': pitcher_protection,
            'hitter_protection': hitter_protection,
            'combined_total': (pitcher_protection + hitter_protection) / 2,
            'interaction_effects': injury_status or {}
        }

    def apply_two_way_adjustment(self,
                               pitcher_war: float,
                               hitter_war: float,
                               injury_status: dict = None,
                               player_name: str = 'Two-Way Player') -> dict:
        """
        Apply two-way player specific adjustments.

        Args:
            pitcher_war: Current pitcher WAR
            hitter_war: Current hitter WAR
            injury_status: Current injury effects
            player_name: Player name for logging

        Returns:
            Dictionary with two-way adjustment results
        """
        # Get component protection factors
        protection_factors = self.get_two_way_protection(pitcher_war, hitter_war, injury_status)

        # Calculate adjusted components
        adjusted_pitcher_war = pitcher_war * (1 + (1 - protection_factors['pitcher_protection']) * 0.5)
        adjusted_hitter_war = hitter_war * (1 + (1 - protection_factors['hitter_protection']) * 0.5)

        # Total adjusted WAR
        total_adjusted = adjusted_pitcher_war + adjusted_hitter_war
        original_total = pitcher_war + hitter_war

        return {
            'player_name': player_name,
            'original_pitcher_war': pitcher_war,
            'original_hitter_war': hitter_war,
            'original_total_war': original_total,
            'adjusted_pitcher_war': adjusted_pitcher_war,
            'adjusted_hitter_war': adjusted_hitter_war,
            'adjusted_total_war': total_adjusted,
            'pitcher_adjustment': adjusted_pitcher_war - pitcher_war,
            'hitter_adjustment': adjusted_hitter_war - hitter_war,
            'total_adjustment': total_adjusted - original_total,
            'protection_factors': protection_factors
        }

    def detect_two_way_player(self, player_data: dict) -> bool:
        """
        Detect if a player qualifies as a two-way player.

        Args:
            player_data: Dictionary with player stats

        Returns:
            True if player qualifies as two-way
        """
        # Check for meaningful contribution in both roles
        pitcher_ip = player_data.get('IP', 0.0)
        hitter_pa = player_data.get('PA', 0.0)

        # Two-way qualification thresholds (based on Ohtani)
        min_pitcher_ip = 50.0    # Meaningful pitching contribution
        min_hitter_pa = 200.0    # Meaningful hitting contribution

        return pitcher_ip >= min_pitcher_ip and hitter_pa >= min_hitter_pa


class RookieEliteProtection(ElitePlayerAdjuster):
    """
    Enhanced elite protection for rookie players using MLB official thresholds.

    Provides special pathway for qualifying rookies (per MLB definition):
    - Pitchers: <50 IP in all previous MLB seasons
    - Hitters: <130 AB in all previous MLB seasons

    Prevents veteran "career year" false positives while capturing elite rookie talent.
    """

    def __init__(self, use_enhanced_system: bool = True):
        """Initialize with base elite adjuster functionality."""
        super().__init__(use_enhanced_system=use_enhanced_system)

        # MLB official rookie thresholds
        self.rookie_thresholds = {
            'pitcher_ip': 50.0,    # MLB rookie threshold for pitchers
            'hitter_ab': 130.0,    # MLB rookie threshold for hitters
            'minimum_current_ip': 40.0,   # Minimum IP to avoid tiny samples
            'minimum_current_ab': 200.0,  # Minimum AB to avoid tiny samples
            'minimum_war': 2.5     # Minimum WAR to trigger rookie protection
        }

        # Rookie-specific protection matrix (more conservative than veterans)
        self.rookie_protection_matrix = {
            'elite_rookie': {'hitters': 0.30, 'starting_pitchers': 0.25, 'relief_pitchers': 0.35},
            'good_rookie': {'hitters': 0.45, 'starting_pitchers': 0.40, 'relief_pitchers': 0.50},
            'average_rookie': {'hitters': 0.70, 'starting_pitchers': 0.65, 'relief_pitchers': 0.75}
        }

        # Uncertainty factors for small samples
        self.uncertainty_adjustments = {
            'sample_size_penalty': 0.15,  # Additional regression for small samples
            'rookie_ceiling': 6.0,        # Maximum projected WAR for rookies
            'confidence_discount': 0.9    # Slight confidence discount for inexperience
        }

    def validate_rookie_status(self, player_data: dict, historical_data: pd.DataFrame = None) -> dict:
        """
        Validate rookie status using MLB official thresholds.

        Args:
            player_data: Current season player data (must include Name, IP/AB, Position)
            historical_data: Historical player data for previous seasons validation

        Returns:
            Dict with rookie validation results
        """
        player_name = player_data.get('Name', 'Unknown')
        position = player_data.get('Position', 'OF')
        current_war = player_data.get('WAR', 0.0)

        # Get current season stats
        current_ip = player_data.get('IP', 0.0)
        current_ab = player_data.get('AB', 0.0)
        current_pa = player_data.get('PA', 0.0)

        # Determine if player is pitcher or hitter
        is_pitcher = position in ['P', 'SP', 'RP', 'CL'] or current_ip > 0

        # Check minimum thresholds to avoid tiny samples
        meets_minimum = False
        if is_pitcher:
            meets_minimum = current_ip >= self.rookie_thresholds['minimum_current_ip']
        else:
            meets_minimum = (current_ab >= self.rookie_thresholds['minimum_current_ab'] or
                           current_pa >= self.rookie_thresholds['minimum_current_ab'] * 1.5)

        # Check if performance merits rookie protection
        meets_war_threshold = current_war >= self.rookie_thresholds['minimum_war']

        # Historical validation (simplified for now - can be enhanced with actual data)
        # In practice, this would query historical_data for previous MLB experience
        has_previous_experience = False
        total_previous_volume = 0.0

        if historical_data is not None and not historical_data.empty:
            # Look for previous seasons of this player
            previous_seasons = historical_data[
                (historical_data['Name'] == player_name) &
                (historical_data['Season'] < player_data.get('Season', 2024))
            ]

            if not previous_seasons.empty:
                if is_pitcher:
                    total_previous_volume = previous_seasons['IP'].sum()
                    has_previous_experience = total_previous_volume >= self.rookie_thresholds['pitcher_ip']
                else:
                    total_previous_volume = previous_seasons['AB'].sum()
                    has_previous_experience = total_previous_volume >= self.rookie_thresholds['hitter_ab']

        # Rookie qualification logic
        is_qualifying_rookie = (
            not has_previous_experience and
            meets_minimum and
            meets_war_threshold
        )

        return {
            'is_qualifying_rookie': is_qualifying_rookie,
            'is_pitcher': is_pitcher,
            'meets_minimum_volume': meets_minimum,
            'meets_war_threshold': meets_war_threshold,
            'has_previous_experience': has_previous_experience,
            'total_previous_volume': total_previous_volume,
            'current_volume': current_ip if is_pitcher else current_ab,
            'validation_details': {
                'threshold_used': self.rookie_thresholds['pitcher_ip'] if is_pitcher else self.rookie_thresholds['hitter_ab'],
                'minimum_current': self.rookie_thresholds['minimum_current_ip'] if is_pitcher else self.rookie_thresholds['minimum_current_ab'],
                'minimum_war': self.rookie_thresholds['minimum_war']
            }
        }

    def classify_rookie_tier(self, war_value: float) -> str:
        """Classify rookie into performance tier for protection purposes."""
        if war_value >= 4.0:
            return 'elite_rookie'
        elif war_value >= 3.0:
            return 'good_rookie'
        else:
            return 'average_rookie'

    def get_rookie_protection_factor(self, war_value: float, position: str, rookie_validation: dict) -> float:
        """
        Get protection factor for qualifying rookies.

        Args:
            war_value: Current WAR value
            position: Player position
            rookie_validation: Results from validate_rookie_status()

        Returns:
            Protection factor (lower = more protection)
        """
        if not rookie_validation.get('is_qualifying_rookie', False):
            # Not a qualifying rookie, use standard protection
            return self.get_protection_factor(war_value, position)

        # Rookie-specific protection
        rookie_tier = self.classify_rookie_tier(war_value)
        position_type = self.classify_rookie_position_type(position, rookie_validation['is_pitcher'])

        base_protection = self.rookie_protection_matrix[rookie_tier][position_type]

        # Apply uncertainty adjustments for rookie inexperience
        sample_size_factor = 1.0 + self.uncertainty_adjustments['sample_size_penalty']
        confidence_factor = self.uncertainty_adjustments['confidence_discount']

        # Conservative adjustment: slightly higher regression for rookies vs veterans
        adjusted_protection = base_protection * sample_size_factor * confidence_factor

        # Cap at reasonable bounds
        return max(0.15, min(0.80, adjusted_protection))

    def classify_rookie_position_type(self, position: str, is_pitcher: bool) -> str:
        """Classify position type for rookie protection purposes."""
        if is_pitcher:
            if position in ['RP', 'CL']:
                return 'relief_pitchers'
            else:
                return 'starting_pitchers'
        else:
            return 'hitters'

    def apply_rookie_adjustments(self, player_data: dict, historical_data: pd.DataFrame = None) -> dict:
        """
        Apply rookie-specific projections with uncertainty factoring.

        Args:
            player_data: Current season player data
            historical_data: Historical data for rookie validation

        Returns:
            Dict with rookie adjustment results
        """
        # Validate rookie status
        rookie_validation = self.validate_rookie_status(player_data, historical_data)

        player_name = player_data.get('Name', 'Unknown')
        current_war = player_data.get('WAR', 0.0)
        position = player_data.get('Position', 'OF')

        # Get protection factor
        protection_factor = self.get_rookie_protection_factor(
            current_war, position, rookie_validation
        )

        # Apply ceiling for rookie projections (prevent over-optimism)
        ceiling_applied = False
        projected_war = current_war
        if rookie_validation.get('is_qualifying_rookie', False):
            if projected_war > self.uncertainty_adjustments['rookie_ceiling']:
                projected_war = self.uncertainty_adjustments['rookie_ceiling']
                ceiling_applied = True

        return {
            'player_name': player_name,
            'is_qualifying_rookie': rookie_validation.get('is_qualifying_rookie', False),
            'rookie_tier': self.classify_rookie_tier(current_war) if rookie_validation.get('is_qualifying_rookie', False) else None,
            'original_war': current_war,
            'projected_war': projected_war,
            'protection_factor': protection_factor,
            'ceiling_applied': ceiling_applied,
            'rookie_validation': rookie_validation,
            'adjustment_type': 'rookie_pathway' if rookie_validation.get('is_qualifying_rookie', False) else 'standard_pathway'
        }

    def get_rookie_players_from_data(self, current_data: pd.DataFrame, historical_data: pd.DataFrame = None) -> pd.DataFrame:
        """
        Identify all qualifying rookies from current season data.

        Args:
            current_data: Current season player data
            historical_data: Historical data for validation

        Returns:
            DataFrame of qualifying rookie players
        """
        rookie_players = []

        for _, player in current_data.iterrows():
            player_dict = player.to_dict()
            validation = self.validate_rookie_status(player_dict, historical_data)

            if validation.get('is_qualifying_rookie', False):
                adjustment_result = self.apply_rookie_adjustments(player_dict, historical_data)

                rookie_info = {
                    **player_dict,
                    'rookie_tier': adjustment_result['rookie_tier'],
                    'protection_factor': adjustment_result['protection_factor'],
                    'rookie_validation': validation
                }
                rookie_players.append(rookie_info)

        return pd.DataFrame(rookie_players) if rookie_players else pd.DataFrame()

    def generate_rookie_protection_report(self, current_data: pd.DataFrame, historical_data: pd.DataFrame = None) -> dict:
        """
        Generate comprehensive report on rookie protection system.

        Args:
            current_data: Current season player data
            historical_data: Historical data for validation

        Returns:
            Dict with rookie protection analysis
        """
        # Identify rookies
        rookies_df = self.get_rookie_players_from_data(current_data, historical_data)

        # Analyze high-WAR players for false positive check
        high_war_players = current_data[current_data['WAR'] > 3.0]

        false_positives = 0
        veteran_high_war = 0

        for _, player in high_war_players.iterrows():
            player_dict = player.to_dict()
            validation = self.validate_rookie_status(player_dict, historical_data)

            if validation.get('is_qualifying_rookie', False):
                # This is a rookie
                continue
            else:
                # This is a veteran with high WAR
                veteran_high_war += 1

        # Calculate summary statistics
        total_rookies = len(rookies_df)
        elite_rookies = len(rookies_df[rookies_df['WAR'] >= 4.0]) if total_rookies > 0 else 0
        good_rookies = len(rookies_df[(rookies_df['WAR'] >= 3.0) & (rookies_df['WAR'] < 4.0)]) if total_rookies > 0 else 0

        false_positive_rate = false_positives / len(high_war_players) if len(high_war_players) > 0 else 0.0

        return {
            'total_qualifying_rookies': total_rookies,
            'elite_rookies': elite_rookies,
            'good_rookies': good_rookies,
            'total_high_war_players': len(high_war_players),
            'veteran_high_war_players': veteran_high_war,
            'false_positive_count': false_positives,
            'false_positive_rate': false_positive_rate,
            'rookie_details': rookies_df.to_dict('records') if total_rookies > 0 else [],
            'thresholds_used': self.rookie_thresholds,
            'protection_matrix': self.rookie_protection_matrix
        }