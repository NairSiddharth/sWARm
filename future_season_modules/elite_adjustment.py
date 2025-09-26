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

class ElitePlayerAdjuster:
    """
    Applies confidence-based adjustments to reduce over-regression of elite players.

    Core Logic:
    - Elite players (confidence >= 4.0): 60% regression reduction
    - Very good players (confidence >= 2.5): 80% regression reduction
    - Good players (confidence >= 1.5): 90% regression reduction
    - Average players (confidence < 1.5): No regression reduction
    """

    def __init__(self,
                 elite_threshold: float = 4.0,
                 very_good_threshold: float = 2.5,
                 good_threshold: float = 1.5,
                 elite_protection: float = 0.6,
                 very_good_protection: float = 0.8,
                 good_protection: float = 0.9):
        """
        Initialize elite adjustment parameters.

        Args:
            elite_threshold: Confidence threshold for elite players
            very_good_threshold: Confidence threshold for very good players
            good_threshold: Confidence threshold for good players
            elite_protection: Regression reduction factor for elite players (0.6 = 40% less regression)
            very_good_protection: Regression reduction factor for very good players
            good_protection: Regression reduction factor for good players
        """
        self.elite_threshold = elite_threshold
        self.very_good_threshold = very_good_threshold
        self.good_threshold = good_threshold
        self.elite_protection = elite_protection
        self.very_good_protection = very_good_protection
        self.good_protection = good_protection

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

    def calculate_regression_reduction(self, confidence: float) -> float:
        """
        Calculate regression reduction factor based on confidence score.

        Args:
            confidence: Player confidence score (0.5-8.0 range)

        Returns:
            Regression reduction factor (lower = more protection)
        """
        if confidence >= self.elite_threshold:
            return self.elite_protection
        elif confidence >= self.very_good_threshold:
            return self.very_good_protection
        elif confidence >= self.good_threshold:
            return self.good_protection
        else:
            return 1.0  # No protection for average players

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

            # Get confidence score
            confidence = confidence_scores.get(player_id, 1.0)

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