"""
Constraint Optimization Module for Future Season Projections

This module handles zero-sum WAR constraint optimization and elite player adjustments,
extracted from integration.py for better modularity and maintainability.

Original functionality preserved with no modifications.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
from scipy.optimize import minimize


class ConstraintOptimizer:
    """
    Handles constraint optimization for WAR projections including:
    - Zero-sum WAR constraints
    - Elite player adjustments
    - Confidence-weighted optimization
    """

    def __init__(self, war_model=None, warp_model=None, elite_adjuster=None):
        """
        Initialize constraint optimizer.

        Args:
            war_model: WAR prediction model (for confidence scoring)
            warp_model: WARP prediction model (for confidence scoring)
            elite_adjuster: Elite player adjustment system
        """
        self.war_model = war_model
        self.warp_model = warp_model
        self.elite_adjuster = elite_adjuster

    def apply_zero_sum_war_constraint(self, projections_df: pd.DataFrame,
                                    target_total: float = 1000.0,
                                    hitter_pitcher_split: tuple = (570, 430),
                                    training_data: pd.DataFrame = None) -> pd.DataFrame:
        """
        Apply zero-sum WAR constraint using confidence-weighted optimization.

        Optimization Problem:
        Minimize: Σ(confidence_i × (original_i - adjusted_i)²)
        Subject to:
            - Σ(adjusted_i) = target_total
            - Σ(hitter_adjusted) = hitter_pitcher_split[0]
            - Σ(pitcher_adjusted) = hitter_pitcher_split[1]
            - adjusted_i >= 0 for all players
            - 0.5 * original_i <= adjusted_i <= 1.5 * original_i (reasonable bounds)

        Args:
            projections_df: DataFrame with individual projections
            target_total: Total league WAR (default 1000)
            hitter_pitcher_split: (hitter_war, pitcher_war) allocation
            training_data: Training data for confidence scoring

        Returns:
            DataFrame with constraint-adjusted projections
        """
        print("Applying zero-sum WAR constraint optimization...")

        # Separate hitters and pitchers
        hitters = projections_df[projections_df['Position'] != 'P'].copy()
        pitchers = projections_df[projections_df['Position'] == 'P'].copy()

        print(f"  Hitters: {len(hitters)} players")
        print(f"  Pitchers: {len(pitchers)} players")

        # Calculate confidence scores for all players
        hitter_confidences = []
        pitcher_confidences = []

        if self.war_model and training_data is not None:
            for _, row in hitters.iterrows():
                confidence = self.war_model.calculate_player_confidence_score(
                    row['mlbid'], training_data, row['Age'], row['Position']
                )
                hitter_confidences.append(confidence)

            for _, row in pitchers.iterrows():
                confidence = self.warp_model.calculate_player_confidence_score(
                    row['mlbid'], training_data, row['Age'], row['Position']
                ) if self.warp_model else 0.5
                pitcher_confidences.append(confidence)
        else:
            # Fallback to uniform confidence if models not available
            hitter_confidences = [0.5] * len(hitters)
            pitcher_confidences = [0.5] * len(pitchers)

        # Apply separate optimization for hitters and pitchers
        if len(hitters) > 0:
            adjusted_hitters = self._optimize_group_projections(
                hitters['projected_WAR_year_1'].values,
                hitter_confidences,
                target_total=hitter_pitcher_split[0]
            )
            hitters = hitters.copy()
            hitters['projected_WAR_year_1'] = adjusted_hitters

        if len(pitchers) > 0:
            adjusted_pitchers = self._optimize_group_projections(
                pitchers['projected_WAR_year_1'].values,
                pitcher_confidences,
                target_total=hitter_pitcher_split[1]
            )
            pitchers = pitchers.copy()
            pitchers['projected_WAR_year_1'] = adjusted_pitchers

        # Recombine and return
        result_df = pd.concat([hitters, pitchers], ignore_index=True) if len(hitters) > 0 and len(pitchers) > 0 else (hitters if len(hitters) > 0 else pitchers)

        # Log adjustment summary
        self._log_constraint_adjustments(projections_df, result_df)

        return result_df

    def _optimize_group_projections(self, original_projections: np.ndarray,
                                  confidence_scores: List[float],
                                  target_total: float) -> np.ndarray:
        """
        Optimize projections for a group (hitters or pitchers) with constraints.
        """
        current_total = original_projections.sum()

        # If already close to target, minimal adjustment needed
        if abs(current_total - target_total) < 25:
            print(f"    Group already close to target ({current_total:.1f} vs {target_total:.1f}), minimal adjustment")
            return original_projections

        print(f"    Optimizing group: {current_total:.1f} -> {target_total:.1f}")

        # Define optimization objective
        def objective_function(adjusted_projections):
            return sum(
                conf * (orig - adj)**2
                for conf, orig, adj in zip(confidence_scores, original_projections, adjusted_projections)
            )

        # Constraint: sum must equal target
        constraints = [
            {
                'type': 'eq',
                'fun': lambda x: x.sum() - target_total
            }
        ]

        # Bounds: reasonable adjustment limits (handle negative projections)
        bounds = []
        for orig in original_projections:
            if orig >= 0:
                # Positive projections: 50%-150% of original
                bounds.append((max(0.0, orig * 0.5), orig * 1.5))
            else:
                # Negative projections: allow wider range
                bounds.append((orig * 1.5, max(0.0, orig * 0.5)))

        # Validate bounds
        for i, (lower, upper) in enumerate(bounds):
            if lower > upper:
                # Fix invalid bounds by ensuring reasonable range
                orig = original_projections[i]
                bounds[i] = (min(0.0, orig * 1.5), max(0.5, abs(orig) * 2.0))
                print(f"    Warning: Fixed invalid bounds for projection {i}: orig={orig:.3f}, bounds=({bounds[i][0]:.3f}, {bounds[i][1]:.3f})")

        # Initial guess: proportional scaling
        scale_factor = target_total / current_total
        initial_guess = original_projections * scale_factor

        # Solve optimization
        result = minimize(
            objective_function,
            initial_guess,
            method='SLSQP',
            constraints=constraints,
            bounds=bounds,
            options={'ftol': 1e-6, 'disp': False}
        )

        if result.success:
            print(f"    Optimization successful: final total = {result.x.sum():.1f}")
            return result.x
        else:
            # Fallback to proportional scaling if optimization fails
            print(f"    Optimization failed, using proportional scaling: {result.message}")
            return original_projections * scale_factor

    def _log_constraint_adjustments(self, original_df: pd.DataFrame, adjusted_df: pd.DataFrame):
        """
        Log the adjustments made by the constraint optimization.
        """
        if len(original_df) != len(adjusted_df):
            print("Warning: Original and adjusted DataFrames have different lengths")
            return

        # Calculate adjustment statistics
        original_total = original_df['projected_WAR_year_1'].sum()
        adjusted_total = adjusted_df['projected_WAR_year_1'].sum()

        # Calculate per-player adjustments
        adjustments = adjusted_df['projected_WAR_year_1'].values - original_df['projected_WAR_year_1'].values

        print(f"  Constraint adjustment summary:")
        print(f"    Total WAR: {original_total:.1f} -> {adjusted_total:.1f}")
        print(f"    Mean adjustment: {adjustments.mean():.3f}")
        print(f"    Max increase: {adjustments.max():.3f}")
        print(f"    Max decrease: {adjustments.min():.3f}")
        print(f"    Players adjusted: {(adjustments != 0).sum()}/{len(adjustments)}")

    def apply_elite_adjustments(self, projections_df: pd.DataFrame,
                              confidence_scores: Optional[List[float]] = None,
                              training_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Apply elite player adjustments using the configured elite adjuster.

        Args:
            projections_df: DataFrame with player projections
            confidence_scores: Pre-calculated confidence scores (optional)
            training_data: Training data for confidence calculation

        Returns:
            DataFrame with elite adjustments applied
        """
        if self.elite_adjuster is None:
            print("No elite adjuster configured - skipping elite adjustments")
            return projections_df

        print("Applying elite player adjustments...")

        # Calculate confidence scores if not provided
        if confidence_scores is None:
            print("Calculating confidence scores for elite adjustments...")
            confidence_scores = self._calculate_batch_confidence_scores(projections_df, training_data)

        # Apply elite adjustments
        adjusted_df = self.elite_adjuster.adjust_elite_projections(
            projections_df,
            confidence_scores,
            training_data=training_data
        )

        return adjusted_df

    def _calculate_batch_confidence_scores(self, projections_df: pd.DataFrame,
                                         training_data: Optional[pd.DataFrame] = None) -> List[float]:
        """
        Calculate confidence scores for all players in batch.

        Args:
            projections_df: DataFrame with player projections
            training_data: Training data for confidence calculation

        Returns:
            List of confidence scores for each player
        """
        confidence_scores = []

        for _, row in projections_df.iterrows():
            player_id = row['mlbid']
            age = row['Age']
            position = row['Position']

            # Use appropriate model based on position
            if position == 'P' and self.warp_model:
                confidence = self.warp_model.calculate_player_confidence_score(
                    player_id, training_data, age, position
                )
            elif self.war_model:
                confidence = self.war_model.calculate_player_confidence_score(
                    player_id, training_data, age, position
                )
            else:
                # Fallback confidence
                confidence = 0.5

            confidence_scores.append(confidence)

        return confidence_scores