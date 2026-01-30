"""
Constraint Optimizer - Zero-Sum WAR Enforcement

Ensures league-wide WAR projections sum to exactly 1000 (570 hitters, 430 pitchers).
Adapted from future_season_modules/constraint_optimizer.py.

See FUTURE_PROJECTIONS_MIGRATION_GUIDE.md Section 6.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Optional


class ConstraintOptimizer:
    """
    Zero-sum WAR constraint enforcement for future projections.

    Key principles:
    - League total: 1000 WAR
    - Hitter/Pitcher split: 570/430 (typical MLB distribution)
    - Proportional scaling preserves relative rankings
    - Performance-tiered protection (elite players protected more)
    """

    def __init__(
        self,
        target_total: float = 1000.0,
        hitter_target: float = 570.0,
        pitcher_target: float = 430.0
    ):
        """
        Initialize constraint optimizer.

        Args:
            target_total: Total league WAR (default: 1000)
            hitter_target: Target hitter WAR (default: 570)
            pitcher_target: Target pitcher WAR (default: 430)
        """
        self.target_total = target_total
        self.hitter_target = hitter_target
        self.pitcher_target = pitcher_target

        # Validate targets sum correctly
        if abs((hitter_target + pitcher_target) - target_total) > 0.1:
            raise ValueError(f"Hitter ({hitter_target}) + Pitcher ({pitcher_target}) must equal total ({target_total})")

    def apply_constraints(
        self,
        projections_df: pd.DataFrame,
        player_type_col: str = 'player_type',
        war_columns: List[str] = None,
        use_tiered_protection: bool = True
    ) -> pd.DataFrame:
        """
        Apply zero-sum constraints to projections.

        Args:
            projections_df: DataFrame with projections
                Must have player_type column ('hitter' or 'pitcher')
            player_type_col: Column name for player type
            war_columns: List of WAR columns to constrain
                Default: ['war_year_1', 'war_year_2', 'war_year_3']
            use_tiered_protection: Use performance-tiered scaling (recommended)

        Returns:
            DataFrame with constrained projections

        Example:
            >>> optimizer = ConstraintOptimizer()
            >>> constrained = optimizer.apply_constraints(projections_df)
            >>> print(constrained['war_year_1'].sum())  # Should be ~1000
        """
        if war_columns is None:
            war_columns = [col for col in ['war_year_1', 'war_year_2', 'war_year_3']
                          if col in projections_df.columns]

        if not war_columns:
            raise ValueError("No WAR projection columns found in projections_df")

        if player_type_col not in projections_df.columns:
            raise ValueError(f"Column '{player_type_col}' not found in projections_df")

        constrained_df = projections_df.copy()

        # Apply constraints to each projection year
        for war_col in war_columns:
            if war_col in constrained_df.columns:
                constrained_df[war_col] = self._apply_constraint_to_column(
                    constrained_df,
                    war_col,
                    player_type_col,
                    use_tiered_protection
                )

        # Report constraint results
        self._report_constraints(constrained_df, war_columns, player_type_col)

        return constrained_df

    def _apply_constraint_to_column(
        self,
        df: pd.DataFrame,
        war_col: str,
        player_type_col: str,
        use_tiered_protection: bool
    ) -> pd.Series:
        """
        Apply constraint to a single WAR column.

        Args:
            df: Projections dataframe
            war_col: Column to constrain
            player_type_col: Player type column
            use_tiered_protection: Use tiered scaling

        Returns:
            Constrained WAR values
        """
        # Separate hitters and pitchers
        hitters = df[df[player_type_col] == 'hitter'].copy()
        pitchers = df[df[player_type_col] == 'pitcher'].copy()

        # Calculate current totals
        hitter_total = hitters[war_col].sum() if len(hitters) > 0 else 0
        pitcher_total = pitchers[war_col].sum() if len(pitchers) > 0 else 0

        # Apply scaled adjustments
        if len(hitters) > 0:
            hitter_adjusted = self._scale_to_target(
                hitters[war_col],
                self.hitter_target,
                use_tiered_protection
            )
        else:
            hitter_adjusted = pd.Series(dtype=float)

        if len(pitchers) > 0:
            pitcher_adjusted = self._scale_to_target(
                pitchers[war_col],
                self.pitcher_target,
                use_tiered_protection
            )
        else:
            pitcher_adjusted = pd.Series(dtype=float)

        # Combine
        adjusted = pd.concat([hitter_adjusted, pitcher_adjusted])

        # Ensure proper ordering
        adjusted = adjusted.reindex(df.index)

        return adjusted

    def _scale_to_target(
        self,
        war_values: pd.Series,
        target_total: float,
        use_tiered_protection: bool
    ) -> pd.Series:
        """
        Scale WAR values to meet target total.

        Args:
            war_values: Original WAR values
            target_total: Target sum
            use_tiered_protection: Use performance-tiered scaling

        Returns:
            Scaled WAR values
        """
        current_total = war_values.sum()

        if current_total == 0:
            # All zeros, distribute target evenly
            return pd.Series(target_total / len(war_values), index=war_values.index)

        if not use_tiered_protection:
            # Simple proportional scaling
            scale_factor = target_total / current_total
            return war_values * scale_factor

        # Tiered protection scaling
        # Elite players get more protection, below-average get less

        # Define tiers based on WAR levels
        tier_thresholds = {
            'superstar': 6.0,    # 6+ WAR
            'elite': 4.0,        # 4-6 WAR
            'above_avg': 2.0,    # 2-4 WAR
            'average': 0.0,      # 0-2 WAR
            'below_avg': -10.0   # <0 WAR
        }

        tier_factors = {
            'superstar': 1.15,   # 15% more WAR preserved (6+ WAR players)
            'elite': 1.10,       # 10% more WAR preserved (4-6 WAR players)
            'above_avg': 1.05,   # 5% more WAR preserved (2-4 WAR players)
            'average': 1.00,     # Baseline (0-2 WAR players)
            'below_avg': 0.95    # 5% less WAR preserved (<0 WAR players)
        }
        # Preserve elite player rankings during zero-sum constraint enforcement

        # Assign tiers
        def get_tier(war):
            if war >= tier_thresholds['superstar']:
                return 'superstar'
            elif war >= tier_thresholds['elite']:
                return 'elite'
            elif war >= tier_thresholds['above_avg']:
                return 'above_avg'
            elif war >= tier_thresholds['average']:
                return 'average'
            else:
                return 'below_avg'

        tiers = war_values.apply(get_tier)

        # Apply tier factors
        tier_weights = tiers.map(tier_factors)

        # Calculate weighted scaling
        weighted_war = war_values * tier_weights
        weighted_total = weighted_war.sum()

        if weighted_total == 0:
            # Fallback to simple scaling
            scale_factor = target_total / current_total
            return war_values * scale_factor

        # Scale weighted values to target
        scale_factor = target_total / weighted_total
        scaled_war = weighted_war * scale_factor

        return scaled_war

    def _report_constraints(
        self,
        df: pd.DataFrame,
        war_columns: List[str],
        player_type_col: str
    ) -> None:
        """
        Report constraint application results.

        Args:
            df: Constrained dataframe
            war_columns: WAR columns
            player_type_col: Player type column
        """
        print("\nConstraint optimization results:")
        print("=" * 60)

        for war_col in war_columns:
            if war_col in df.columns:
                hitter_total = df[df[player_type_col] == 'hitter'][war_col].sum()
                pitcher_total = df[df[player_type_col] == 'pitcher'][war_col].sum()
                total = hitter_total + pitcher_total

                print(f"\n{war_col}:")
                print(f"  Hitters:  {hitter_total:>7.1f} WAR (target: {self.hitter_target})")
                print(f"  Pitchers: {pitcher_total:>7.1f} WAR (target: {self.pitcher_target})")
                print(f"  Total:    {total:>7.1f} WAR (target: {self.target_total})")

                # Check tolerance
                within_tolerance = abs(total - self.target_total) < 10
                status = "PASS" if within_tolerance else "FAIL"
                print(f"  Status: {status} (tolerance: ±10 WAR)")

    def validate_constraints(
        self,
        df: pd.DataFrame,
        war_columns: List[str],
        player_type_col: str = 'player_type',
        tolerance: float = 10.0
    ) -> Dict[str, bool]:
        """
        Validate that constraints are satisfied.

        Args:
            df: Projections dataframe
            war_columns: WAR columns to validate
            player_type_col: Player type column
            tolerance: Acceptable deviation from target (default: ±10)

        Returns:
            Dictionary of {war_column: is_valid}
        """
        results = {}

        for war_col in war_columns:
            if war_col in df.columns:
                total = df[war_col].sum()
                is_valid = abs(total - self.target_total) <= tolerance
                results[war_col] = is_valid

        return results
