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
                                    training_data: pd.DataFrame = None,
                                    prediction_year: int = None) -> pd.DataFrame:
        """
        Apply dynamic budget allocation with performance-tiered protection factors.

        Instead of uniform cuts, applies graduated adjustments based on player
        performance tiers while maintaining exact WAR budget compliance.

        Protection Factors:
        - Superstar (6.0+ WAR): 1.15 (15% more WAR preserved vs average)
        - Elite (4.0+ WAR): 1.10 (10% more WAR preserved vs average)
        - Above Average (2.0+ WAR): 1.05 (5% more WAR preserved vs average)
        - Average (0.0+ WAR): 1.00 (baseline)
        - Below Average (<0.0 WAR): 0.95 (5% less WAR preserved vs average)

        Args:
            projections_df: DataFrame with individual projections
            target_total: Total league WAR (default 1000)
            hitter_pitcher_split: (hitter_war, pitcher_war) allocation
            training_data: Training data for performance classification

        Returns:
            DataFrame with dynamically allocated projections
        """
        print("Applying dynamic budget allocation with performance-tiered protection...")

        # Separate hitters and pitchers
        hitters = projections_df[projections_df['Position'] != 'P'].copy()
        pitchers = projections_df[projections_df['Position'] == 'P'].copy()

        print(f"  Hitters: {len(hitters)} players")
        print(f"  Pitchers: {len(pitchers)} players")

        # Calculate proportional budgets based on training data composition
        if training_data is not None and prediction_year is not None:
            hitter_budget, pitcher_budget = self._calculate_proportional_budgets(
                hitters, pitchers, training_data, prediction_year, target_total, hitter_pitcher_split
            )
        else:
            # Fallback to fixed split if no training data
            hitter_budget, pitcher_budget = hitter_pitcher_split

        # Apply dynamic allocation for each group
        if len(hitters) > 0:
            adjusted_hitters = self._apply_dynamic_allocation(
                hitters, target_total=hitter_budget, training_data=training_data, prediction_year=prediction_year
            )
            hitters = adjusted_hitters

        if len(pitchers) > 0:
            adjusted_pitchers = self._apply_dynamic_allocation(
                pitchers, target_total=pitcher_budget, training_data=training_data, prediction_year=prediction_year
            )
            pitchers = adjusted_pitchers

        # Recombine and return
        result_df = pd.concat([hitters, pitchers], ignore_index=True) if len(hitters) > 0 and len(pitchers) > 0 else (hitters if len(hitters) > 0 else pitchers)

        # Log adjustment summary
        self._log_constraint_adjustments(projections_df, result_df)

        return result_df

    def _calculate_proportional_budgets(self, hitters: pd.DataFrame, pitchers: pd.DataFrame,
                                      training_data: pd.DataFrame, prediction_year: int,
                                      target_total: float, default_split: tuple) -> tuple:
        """
        Calculate proportional budget allocation based on training fold composition.

        Args:
            hitters: Hitter projections for test fold
            pitchers: Pitcher projections for test fold
            training_data: Training data (historical only)
            prediction_year: Year being predicted
            target_total: Total WAR budget to allocate
            default_split: Default (hitter_war, pitcher_war) split

        Returns:
            (hitter_budget, pitcher_budget) based on training composition
        """
        # Get training data prior to prediction year (no leakage)
        historical_training = training_data[training_data['Season'] < prediction_year]

        if len(historical_training) == 0:
            print("    No historical training data - using default split")
            return default_split

        # Calculate training fold composition
        training_hitters = historical_training[historical_training['Position'] != 'P']
        training_pitchers = historical_training[historical_training['Position'] == 'P']

        if len(training_hitters) == 0 or len(training_pitchers) == 0:
            print("    Insufficient training data for both groups - using default split")
            return default_split

        # Get performance metric
        performance_metric = 'WAR' if 'WAR' in historical_training.columns else 'WARP'

        # Calculate training fold WAR totals
        training_hitter_war = training_hitters[performance_metric].sum()
        training_pitcher_war = training_pitchers[performance_metric].sum()
        training_total_war = training_hitter_war + training_pitcher_war

        if training_total_war <= 0:
            print("    Invalid training WAR totals - using default split")
            return default_split

        # Calculate proportional allocation based on training composition
        hitter_proportion = training_hitter_war / training_total_war
        pitcher_proportion = training_pitcher_war / training_total_war

        # Calculate test fold natural totals
        test_hitter_natural = hitters['projected_WAR_year_1'].sum() if len(hitters) > 0 else 0
        test_pitcher_natural = pitchers['projected_WAR_year_1'].sum() if len(pitchers) > 0 else 0
        test_total_natural = test_hitter_natural + test_pitcher_natural

        if test_total_natural <= 0:
            print("    Invalid test projections - using default split")
            return default_split

        # Allocate target total proportionally
        hitter_budget = test_total_natural * hitter_proportion
        pitcher_budget = test_total_natural * pitcher_proportion

        # Scale to match target total
        budget_scale = target_total / test_total_natural
        hitter_budget *= budget_scale
        pitcher_budget *= budget_scale

        print(f"    Proportional budget allocation:")
        print(f"      Training composition: {hitter_proportion:.1%} hitters, {pitcher_proportion:.1%} pitchers")
        print(f"      Test budgets: {hitter_budget:.0f} hitter WAR, {pitcher_budget:.0f} pitcher WAR")

        return hitter_budget, pitcher_budget

    def _apply_dynamic_allocation(self, player_group: pd.DataFrame,
                                target_total: float,
                                training_data: pd.DataFrame = None,
                                prediction_year: int = None) -> pd.DataFrame:
        """
        Apply dynamic budget allocation to a group of players (hitters or pitchers).

        Args:
            player_group: DataFrame of players (hitters or pitchers)
            target_total: Target WAR total for this group
            training_data: Training data for performance classification

        Returns:
            DataFrame with dynamically allocated projections
        """
        if len(player_group) == 0:
            return player_group

        group_df = player_group.copy()

        # Step 1: Classify players into performance tiers (no data leakage)
        performance_tiers = self._classify_performance_tiers(group_df, training_data, prediction_year)

        # Step 2: Calculate current WAR totals by tier
        tier_totals = {}
        tier_counts = {}
        protection_factors = {
            'superstar': 1.15,     # 15% more WAR preserved vs average
            'elite': 1.10,         # 10% more WAR preserved vs average
            'above_average': 1.05, # 5% more WAR preserved vs average
            'average': 1.00,       # Baseline
            'below_average': 0.95  # 5% less WAR preserved vs average
        }

        for tier in protection_factors.keys():
            tier_mask = performance_tiers == tier
            tier_totals[tier] = group_df.loc[tier_mask, 'projected_WAR_year_1'].sum()
            tier_counts[tier] = tier_mask.sum()

        current_total = sum(tier_totals.values())

        # Step 3: Calculate weighted total for proportional allocation
        weighted_total = sum(
            tier_totals[tier] * protection_factors[tier]
            for tier in tier_totals.keys()
            if tier_totals[tier] > 0
        )

        # Step 4: Calculate tier-specific reduction factors
        tier_reductions = {}
        for tier in tier_totals.keys():
            if tier_totals[tier] > 0:
                # Each tier gets WAR allocation proportional to: current_WAR * protection_factor
                tier_target = (tier_totals[tier] * protection_factors[tier] / weighted_total) * target_total
                tier_reductions[tier] = tier_target / tier_totals[tier]
            else:
                tier_reductions[tier] = 1.0

        # Step 5: Apply calculated reductions
        for tier, reduction_factor in tier_reductions.items():
            tier_mask = performance_tiers == tier
            if tier_mask.sum() > 0:
                current_values = group_df.loc[tier_mask, 'projected_WAR_year_1'].values
                adjusted_values = current_values * reduction_factor
                group_df.loc[tier_mask, 'projected_WAR_year_1'] = adjusted_values

        # Step 6: Verify total WAR = target
        final_total = group_df['projected_WAR_year_1'].sum()

        print(f"    Dynamic allocation applied:")
        print(f"      Original total: {current_total:.1f} WAR")
        print(f"      Target total: {target_total:.1f} WAR")
        print(f"      Final total: {final_total:.1f} WAR")
        print(f"      Error: {abs(final_total - target_total):.3f}")

        # Log tier-specific adjustments
        for tier in tier_counts.keys():
            if tier_counts[tier] > 0:
                reduction = tier_reductions[tier]
                factor = protection_factors[tier]
                print(f"      {tier}: {tier_counts[tier]} players, "
                      f"protection={factor:.2f}, reduction={reduction:.3f}")

        return group_df

    def _classify_performance_tiers(self, player_group: pd.DataFrame,
                                  training_data: pd.DataFrame = None,
                                  prediction_year: int = None) -> pd.Series:
        """
        Classify players into performance tiers based on recent performance.

        Args:
            player_group: DataFrame of players
            training_data: Training data for performance lookup
            prediction_year: Year being predicted (to avoid data leakage)

        Returns:
            Series with tier classification for each player
        """
        tiers = pd.Series(index=player_group.index, dtype=str)

        # If no training data, use current projection as proxy
        if training_data is None:
            for idx, row in player_group.iterrows():
                current_proj = row['projected_WAR_year_1']
                if current_proj >= 6.0:
                    tiers.loc[idx] = 'superstar'
                elif current_proj >= 4.0:
                    tiers.loc[idx] = 'elite'
                elif current_proj >= 2.0:
                    tiers.loc[idx] = 'above_average'
                elif current_proj >= 0.0:
                    tiers.loc[idx] = 'average'
                else:
                    tiers.loc[idx] = 'below_average'
            return tiers

        # Use training data to calculate recent performance
        performance_metric = 'WAR' if 'WAR' in training_data.columns else 'WARP'

        # Determine cutoff year to prevent data leakage
        if prediction_year is None:
            prediction_year = training_data['Season'].max() + 1

        for idx, row in player_group.iterrows():
            player_id = row['mlbid']

            if pd.isna(player_id):
                tiers.loc[idx] = 'average'  # Default for unknown players
                continue

            # Get player's historical performance (NO DATA LEAKAGE)
            player_data = training_data[training_data['mlbid'] == player_id]
            if len(player_data) == 0:
                tiers.loc[idx] = 'average'  # Default for players without history
                continue

            # CRITICAL FIX: Use only data PRIOR to prediction year
            historical_data = player_data[player_data['Season'] < prediction_year]

            if len(historical_data) == 0:
                tiers.loc[idx] = 'average'  # Default for players without sufficient history
                continue

            recent_seasons = historical_data.sort_values('Season').tail(3)
            recent_performance = recent_seasons[performance_metric].dropna()

            if len(recent_performance) == 0:
                tiers.loc[idx] = 'average'
                continue

            # Calculate weighted average (more recent years weighted higher)
            if len(recent_performance) == 1:
                avg_performance = recent_performance.iloc[0]
            elif len(recent_performance) == 2:
                avg_performance = (recent_performance.iloc[0] * 0.4 +
                                 recent_performance.iloc[1] * 0.6)
            else:
                avg_performance = (recent_performance.iloc[0] * 0.2 +
                                 recent_performance.iloc[1] * 0.3 +
                                 recent_performance.iloc[2] * 0.5)

            # Classify into tier
            if avg_performance >= 6.0:
                tiers.loc[idx] = 'superstar'
            elif avg_performance >= 4.0:
                tiers.loc[idx] = 'elite'
            elif avg_performance >= 2.0:
                tiers.loc[idx] = 'above_average'
            elif avg_performance >= 0.0:
                tiers.loc[idx] = 'average'
            else:
                tiers.loc[idx] = 'below_average'

        return tiers

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