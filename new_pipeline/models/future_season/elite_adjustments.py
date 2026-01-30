"""
Elite Player Adjustment for Future Projections

Wrapper around new_pipeline.models.future_season.elite_player_adjuster.ElitePlayerAdjuster
Protects elite players (6+ WAR) from over-regression in future projections.

See FUTURE_PROJECTIONS_MIGRATION_GUIDE.md Section 4.
"""

import pandas as pd
import numpy as np

from new_pipeline.models.future_season.elite_player_adjuster import ElitePlayerAdjuster


def apply_elite_adjustments(
    projections_df: pd.DataFrame,
    historical_war: pd.Series = None,
    war_columns: list = None,
    player_type: str = 'hitter'
) -> pd.DataFrame:
    """
    Apply elite player protection to future projections.

    Wrapper around new_pipeline.models.future_season.elite_player_adjuster.ElitePlayerAdjuster.

    Protects elite players from over-regression:
    - MVP Level (6+ WAR): 75-80% regression reduction
    - Superstar (5-6 WAR): 65-75% regression reduction
    - All-Star (4-5 WAR): 60-70% regression reduction

    Args:
        projections_df: DataFrame with projections
            Must have playerid column
        historical_war: Series of historical WAR values (optional)
            If None, uses war_year_1 from projections
        war_columns: List of WAR columns to adjust (optional)
            Default: ['war_year_1', 'war_year_2', 'war_year_3']
        player_type: 'hitter' or 'pitcher' (optional)
            Determines elite protection thresholds
            Default: 'hitter'

    Returns:
        DataFrame with elite-adjusted projections

    Example:
        >>> projections = joint_model.project_multiple_players(sequences)
        >>> adjusted_projections = apply_elite_adjustments(projections, player_type='pitcher')
    """
    if war_columns is None:
        # Default: adjust all projection years
        war_columns = [col for col in ['war_year_1', 'war_year_2', 'war_year_3']
                      if col in projections_df.columns]

    if not war_columns:
        raise ValueError("No WAR projection columns found in projections_df")

    # Initialize adjuster with player-type-specific thresholds
    if player_type == 'pitcher':
        # Pitchers: Lower WAR scale requires adjusted thresholds
        adjuster = ElitePlayerAdjuster(
            use_enhanced_system=True,
            elite_threshold=5.0,       # Elite: 5+ WAR (vs 5.5 for hitters)
            very_good_threshold=3.5,   # Very good: 3.5-5 WAR (vs 4.5 for hitters)
            good_threshold=2.5,        # Good: 2.5-3.5 WAR (vs 3.0 for hitters)
            elite_protection=0.25,     # 75% protection (vs 60% for hitters)
            very_good_protection=0.45, # 55% protection (vs 40% for hitters)
            good_protection=0.65       # 35% protection (vs 20% for hitters)
        )
    else:
        # Hitters: Use default thresholds
        adjuster = ElitePlayerAdjuster(use_enhanced_system=True)

    # Use historical WAR if provided, otherwise use war_year_1 as baseline
    if historical_war is None:
        if 'war_year_1' in projections_df.columns:
            historical_war = projections_df['war_year_1']
        else:
            # No baseline available, return projections unchanged
            print("Warning: No historical WAR provided and no war_year_1 column. Returning unadjusted projections.")
            return projections_df.copy()

    # Ensure indices match
    if 'playerid' in projections_df.columns:
        projections_df = projections_df.set_index('playerid')
        # Align historical_war index with projections_df
        if not historical_war.index.equals(projections_df.index):
            historical_war = historical_war.set_axis(projections_df.index)

    adjusted_df = projections_df.copy()

    # Generate confidence scores based on historical WAR
    # Higher WAR = higher confidence in elite status
    # Keep as WAR values (not scaled 0-1) to match elite_threshold defaults (5.5, 4.5, 3.0)
    confidence_scores = historical_war.clip(0, 10)

    # Apply adjustment to each projection year
    for war_col in war_columns:
        if war_col in adjusted_df.columns:
            # Create single-column DataFrame for adjustment
            # ElitePlayerAdjuster expects column named 'predicted_war'
            predictions = pd.DataFrame({'predicted_war': adjusted_df[war_col]})

            # Apply elite adjustment
            try:
                adjusted = adjuster.apply_elite_adjustment(
                    predictions=predictions,
                    confidence_scores=confidence_scores,
                    base_projections=historical_war
                )

                # Update the column
                adjusted_df[war_col] = adjusted['predicted_war']

            except Exception as e:
                print(f"Warning: Elite adjustment failed for {war_col}: {str(e)}")
                # Keep original values on failure

    # Get adjustment statistics
    stats = adjuster.adjustment_stats
    if stats:
        print(f"Elite adjustment applied:")
        print(f"  Elite adjusted: {stats.get('elite_adjusted', 0)} players")
        print(f"  Very good adjusted: {stats.get('very_good_adjusted', 0)} players")
        print(f"  Total WAR protection: {stats.get('total_war_protection', 0):.2f}")

    # Reset index if it was set
    if adjusted_df.index.name == 'playerid':
        adjusted_df = adjusted_df.reset_index()

    return adjusted_df


def apply_elite_adjustment_to_war(
    war_values: pd.Series,
    historical_war: pd.Series
) -> pd.Series:
    """
    Apply elite adjustment to a single WAR series.

    Convenience function for adjusting individual WAR projections.

    Args:
        war_values: Projected WAR values
        historical_war: Historical WAR for baseline

    Returns:
        Adjusted WAR values
    """
    # Ensure same index
    if not war_values.index.equals(historical_war.index):
        raise ValueError("war_values and historical_war must have same index")

    # Initialize adjuster
    adjuster = ElitePlayerAdjuster(use_enhanced_system=True)

    # Generate confidence scores (keep as WAR values to match thresholds)
    confidence_scores = historical_war.clip(0, 10)

    # Create predictions DataFrame
    # ElitePlayerAdjuster expects column named 'predicted_war'
    predictions = pd.DataFrame({'predicted_war': war_values})

    # Apply adjustment
    adjusted = adjuster.apply_elite_adjustment(
        predictions=predictions,
        confidence_scores=confidence_scores,
        base_projections=historical_war
    )

    return adjusted['predicted_war']


class FutureEliteProtector:
    """
    Elite player protection specifically for future projections.

    Provides additional context and utilities for multi-year adjustments.
    """

    def __init__(self, use_enhanced_system: bool = True):
        """
        Initialize future elite protector.

        Args:
            use_enhanced_system: Use enhanced WAR tier system (recommended)
        """
        self.adjuster = ElitePlayerAdjuster(use_enhanced_system=use_enhanced_system)

    def protect_multi_year_projections(
        self,
        projections_df: pd.DataFrame,
        current_war: pd.Series,
        years: int = 3
    ) -> pd.DataFrame:
        """
        Apply elite protection to multi-year projections.

        Protection strength decreases for later years due to increased uncertainty.

        Args:
            projections_df: DataFrame with war_year_1, war_year_2, war_year_3
            current_war: Current year WAR values
            years: Number of projection years (default: 3)

        Returns:
            DataFrame with protected projections
        """
        protected_df = projections_df.copy()

        # Protection factors by year (decrease with uncertainty)
        year_factors = {
            1: 1.0,   # Full protection for Year 1
            2: 0.8,   # 80% protection for Year 2
            3: 0.6    # 60% protection for Year 3
        }

        for year in range(1, min(years + 1, 4)):
            war_col = f'war_year_{year}'
            if war_col in protected_df.columns:
                # Adjust protection strength by year
                factor = year_factors.get(year, 0.5)

                # Scale confidence by year factor
                scaled_confidence = (current_war.clip(0, 10) / 10.0) * factor

                # ElitePlayerAdjuster expects column named 'predicted_war'
                predictions = pd.DataFrame({'predicted_war': protected_df[war_col]})

                try:
                    adjusted = self.adjuster.apply_elite_adjustment(
                        predictions=predictions,
                        confidence_scores=scaled_confidence,
                        base_projections=current_war
                    )
                    protected_df[war_col] = adjusted['predicted_war']
                except Exception as e:
                    print(f"Warning: Protection failed for {war_col}: {str(e)}")

        return protected_df
