"""
Complete Projection Generator - Combines current season + ROS models.

Architecture:
    Full Season Projection = Current WAR (actual) + ROS WAR (projected)

Process:
1. Current season model -> firsthalf WAR rate
2. Calculate actual firsthalf WAR (rate * usage_to_date)
3. Build ROS features (elite, age, injury, baselines, etc.)
4. ROS model -> secondhalf WAR rate (with quantiles)
5. Project remaining usage (IP/PA)
6. Calculate ROS WAR (rate * remaining_usage)
7. Combine: Total = firsthalf actual + ROS projected
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple

from ..features.ros_feature_builder import ROSFeatureBuilder
from ..constants import (
    HITTER_MODEL_FEATURES,
    PITCHER_MODEL_FEATURES,
    WAR_NORMALIZATION_PA,
    WAR_NORMALIZATION_IP,
    FULL_SEASON_GAMES,
    PITCHER_STARTER_THRESHOLD,
    PITCHER_RELIEVER_THRESHOLD
)
from .usage_projections import (
    get_team_games_from_data,
    calculate_remaining_usage
)

# Import filter_splits_to_nearest for matching historical splits to current timing
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from new_pipeline.models.ros.data_utils import filter_splits_to_nearest


def _get_pitcher_role(row: pd.Series) -> str:
    """
    Determine pitcher role from row data.

    Args:
        row: DataFrame row with GS and G columns

    Returns:
        str: 'starter', 'reliever', or 'swing'
    """
    if 'GS_per_G' in row.index:
        gs_per_g = row['GS_per_G']
    elif 'GS' in row.index and 'G' in row.index:
        gs_per_g = row['GS'] / max(row['G'], 1)
    else:
        return 'starter'  # Default if data missing

    if gs_per_g > PITCHER_STARTER_THRESHOLD:
        return 'starter'
    elif gs_per_g < PITCHER_RELIEVER_THRESHOLD:
        return 'reliever'
    else:
        return 'swing'


def _get_role_specific_denominator(role: str) -> float:
    """
    Get WAR normalization denominator for pitcher role.

    Args:
        role: 'starter', 'reliever', or 'swing'

    Returns:
        float: Normalization denominator (IP for full season equivalent)
    """
    role_denominators = {
        'starter': 162,    # WAR_NORMALIZATION_IP_STARTER
        'reliever': 48.2,  # WAR_NORMALIZATION_IP_RELIEVER
        'swing': 110       # WAR_NORMALIZATION_IP_SWING
    }
    return role_denominators.get(role, 162)


class CompleteProjectionGenerator:
    """
    Combines current season + ROS models for full season projections.

    Usage:
        generator = CompleteProjectionGenerator(
            current_season_model=hitter_model,
            ros_model=hitter_ros_model,
            player_type='hitter'
        )

        projections = generator.generate_projection(
            firsthalf_data=firsthalf_2025,
            historical_data=historical_2016_2024
        )
    """

    def __init__(
        self,
        current_season_model,  # HitterEnsemble or PitcherRoleEnsemble
        ros_model,  # HitterROSEnsemble or PitcherROSEnsemble
        player_type: str = 'hitter'
    ):
        """
        Initialize projection generator.

        Args:
            current_season_model: Trained current season model
            ros_model: Trained ROS model
            player_type: 'hitter' or 'pitcher'
        """
        self.current_season_model = current_season_model
        self.ros_model = ros_model
        self.player_type = player_type
        self.feature_builder = ROSFeatureBuilder(player_type=player_type)

        # Set player-specific constants
        if self.player_type == 'hitter':
            self.model_features = HITTER_MODEL_FEATURES
            self.usage_col = 'PA'
            self.usage_norm = WAR_NORMALIZATION_PA
        else:
            self.model_features = PITCHER_MODEL_FEATURES
            self.usage_col = 'IP'
            self.usage_norm = WAR_NORMALIZATION_IP

    def _get_pitcher_role(self, row: pd.Series) -> str:
        """
        Determine pitcher role from row data.

        Args:
            row: DataFrame row with GS and G columns

        Returns:
            str: 'starter', 'reliever', or 'swing'

        Thresholds:
            - Starter: GS/G > 0.7
            - Reliever: GS/G < 0.1
            - Swing: 0.1 <= GS/G <= 0.7
        """
        if 'GS_per_G' in row.index:
            gs_per_g = row['GS_per_G']
        elif 'GS' in row.index and 'G' in row.index:
            gs_per_g = row['GS'] / max(row['G'], 1)
        else:
            return 'starter'  # Default if data missing

        if gs_per_g > PITCHER_STARTER_THRESHOLD:
            return 'starter'
        elif gs_per_g < PITCHER_RELIEVER_THRESHOLD:
            return 'reliever'
        else:
            return 'swing'

    def generate_projection(
        self,
        firsthalf_data: pd.DataFrame,
        historical_data: pd.DataFrame,
        historical_splits: Optional[pd.DataFrame] = None,
        injury_data: Optional[pd.DataFrame] = None,
        season_length: int = FULL_SEASON_GAMES,
        team_games_source_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Generate complete season projection.

        Process:
        1. Current season model -> firsthalf WAR rate
        2. Calculate actual firsthalf WAR (rate * usage to date)
        3. Build ROS features (elite, age, injury, baselines)
        4. ROS model -> secondhalf WAR rate (with quantiles)
        5. Project remaining usage (IP/PA)
        6. Calculate ROS WAR (rate * remaining usage)
        7. Combine: Total = firsthalf actual + ROS projected

        Args:
            firsthalf_data: Current season stats (e.g., All-Star break data)
                Must include columns: Name, Team, playerid (optional), usage stats, model features
            historical_data: Full history 2016-2024 (for feature building)
                Must include columns: playerid or Name, Year, WAR_per_600/WAR_per_162, historical stats
            historical_splits: Historical split data with remaining_WAR (for ROS model lag features)
                Must include columns: playerid or Name, Year, remaining_WAR, split_point
                If None, will use historical_data (may fail if remaining_WAR missing)
            injury_data: Optional injury history
            season_length: Total season games (default: 162)
            team_games_source_df: DataFrame to use for calculating team games played.
                Should be hitter data for pitcher projections (pitchers don't play every game).
                If None, uses firsthalf_data (fine for hitters, problematic for pitchers)

        Returns:
            DataFrame with columns:
            - Name, Team, Position
            - Current_PA/IP, Current_WAR, Current_WAR_pace
            - Remaining_PA/IP, ROS_WAR, ROS_WAR_pace
            - Total_Projected_PA/IP, Total_Projected_WAR
            - Q10_WAR, Q25_WAR, Q50_WAR, Q75_WAR, Q90_WAR (uncertainty bands)
            - Uncertainty_Range (Q90 - Q10)

        Example:
            >>> gen = CompleteProjectionGenerator(hitter_model, hitter_ros, 'hitter')
            >>> proj = gen.generate_projection(current_2025, historical_2016_2024, hitter_splits)
            >>> proj.nlargest(10, 'Total_Projected_WAR')[['Name', 'Current_WAR', 'ROS_WAR', 'Total_Projected_WAR']]
        """
        print(f"Generating complete projections for {len(firsthalf_data)} {self.player_type}s...")

        # ===== Step 1: Current season predictions =====
        print("  Step 1: Running current season model...")
        current_features = firsthalf_data[self.model_features].values

        # Handle role-based prediction for pitcher ensembles
        if self.player_type == 'pitcher':
            # Calculate roles from GS_per_G for pitcher role ensemble
            roles = firsthalf_data.apply(lambda row: self._get_pitcher_role(row), axis=1).values
            current_war_rates = self.current_season_model.predict(current_features, roles)
        else:
            # Standard prediction for hitter ensembles
            current_war_rates = self.current_season_model.predict(current_features)

        # Add predictions to dataframe for ROS feature builder
        firsthalf_data_with_preds = firsthalf_data.copy()
        firsthalf_data_with_preds['Predicted_WAR_Rate'] = current_war_rates

        # ===== Step 2: Calculate actual WAR to date =====
        print("  Step 2: Calculating actual firsthalf WAR...")
        current_usage = firsthalf_data[self.usage_col].values

        # For pitchers, use role-specific denominators (matches pipeline_runner.py logic)
        if self.player_type == 'pitcher':
            role_denominators = firsthalf_data.apply(
                lambda row: _get_role_specific_denominator(_get_pitcher_role(row)),
                axis=1
            ).values
            current_WAR = current_war_rates * (current_usage / role_denominators)
        else:
            # Hitters use single denominator (600 PA)
            current_WAR = current_war_rates * (current_usage / self.usage_norm)

        # ===== Step 3: Build ROS features =====
        print("  Step 3: Building ROS features...")
        ros_features_df = self.feature_builder.build_features_batch(
            current_season_df=firsthalf_data_with_preds,
            historical_df=historical_data,
            injury_df=injury_data
        )

        print(f"    Built features for {len(ros_features_df)} players")

        # Diagnostic: Check for missing features
        print("    Checking for missing features...")
        nan_summary = ros_features_df.isna().sum()
        features_with_nans = nan_summary[nan_summary > 0]
        if len(features_with_nans) > 0:
            print(f"    WARNING: {len(features_with_nans)} features have NaN values:")
            for feat, count in features_with_nans.head(15).items():
                print(f"      {feat}: {count}/{len(ros_features_df)} players missing")

            # Sample a player with NaN
            nan_mask = ros_features_df.isna().any(axis=1)
            if nan_mask.any():
                sample_player = ros_features_df[nan_mask].iloc[0]
                print(f"    Example player with NaN: {sample_player.get('Name', 'Unknown')}")
                missing_feats = sample_player[sample_player.isna()].index.tolist()
                print(f"      Missing features ({len(missing_feats)} total): {missing_feats[:10]}")
        else:
            print("    All features complete (no NaN values)")

        # ===== Step 4: ROS predictions =====
        print("  Step 4: Running ROS model...")

        # Determine which historical data to use for ROS model
        # ROS model needs remaining_WAR for lag features - use splits if available
        if historical_splits is not None:
            # Detect current season completion percentage
            # Default to 0.5 if not in data
            if 'season_completion_pct' in ros_features_df.columns:
                current_completion = ros_features_df['season_completion_pct'].iloc[0]
            else:
                current_completion = 0.5
                print(f"    Warning: season_completion_pct not found, defaulting to {current_completion}")

            # Filter splits to nearest match
            print(f"    Filtering historical splits to nearest match (current: {current_completion:.2f})...")
            filtered_splits = filter_splits_to_nearest(historical_splits, current_completion)
            split_used = filtered_splits['split_point'].iloc[0]
            print(f"    Using split_point={split_used} ({len(filtered_splits)} historical observations)")

            # Use filtered splits for ROS model
            historical_for_model = filtered_splits
        else:
            # Fallback: use full historical data (may fail if remaining_WAR missing)
            print("    Warning: historical_splits not provided, using historical_data (may fail)")
            historical_for_model = historical_data

        ros_predictions = self.ros_model.predict_with_uncertainty(
            current_df=ros_features_df,
            historical_df=historical_for_model
        )

        # IMPORTANT: ROS model predicts CUMULATIVE remaining WAR, not rates!
        # These are already cumulative WAR values - do NOT multiply by usage ratios
        ros_cumulative_war = ros_predictions['mean']
        quantiles_cumulative = {
            'q10': ros_predictions['q10'],
            'q25': ros_predictions['q25'],
            'q50': ros_predictions['q50'],
            'q75': ros_predictions['q75'],
            'q90': ros_predictions['q90']
        }

        # ===== Step 5: Project remaining usage =====
        print("  Step 5: Projecting remaining usage...")
        # Use team_games_source_df if provided (critical for pitchers!)
        # Pitchers don't play every game, so max(pitcher_G) underestimates team games
        team_games_df = team_games_source_df if team_games_source_df is not None else firsthalf_data
        team_games_dict, league_median_games = get_team_games_from_data(team_games_df)

        remaining_usage = np.array([
            calculate_remaining_usage(
                player_row=row,
                player_type=self.player_type,
                team_games_dict=team_games_dict,
                league_median_games=league_median_games,
                season_length=season_length
            )
            for _, row in firsthalf_data.iterrows()
        ])

        # ===== Step 6: Calculate ROS WAR =====
        print("  Step 6: Using ROS WAR predictions...")

        # ROS predictions are already cumulative WAR - no conversion needed!
        # The model was trained on remaining_WAR (cumulative), not rates
        ros_WAR = ros_cumulative_war
        ros_WAR_quantiles = quantiles_cumulative

        # ===== Apply season-ending injury constraint =====
        # Zero out ROS projections for players with season-ending injuries
        if 'season_ending_injury' in ros_features_df.columns:
            season_ending_mask = ros_features_df['season_ending_injury'] == 1
            if season_ending_mask.any():
                n_injured = season_ending_mask.sum()
                print(f"  Applying season-ending injury constraint to {n_injured} player(s)...")

                # Zero out ROS WAR for season-ending injuries
                ros_WAR[season_ending_mask] = 0.0

                # Zero out all quantiles for season-ending injuries
                for q in ['q10', 'q25', 'q50', 'q75', 'q90']:
                    ros_WAR_quantiles[q][season_ending_mask] = 0.0

                # Zero out remaining usage as well
                remaining_usage[season_ending_mask] = 0.0

        # For display purposes, we can calculate implied rates (for debugging/comparison)
        # But we don't use these for the actual projections
        if self.player_type == 'pitcher':
            # Implied rate would be: cumulative_war / (remaining_usage / role_denominators)
            # But we already have the cumulative prediction, so we just use it directly
            pass
        else:
            # Same for hitters - model outputs cumulative WAR directly
            pass

        # ===== Step 7: Combine into final projections =====
        print("  Step 7: Combining projections...")
        # Extract Primary_Position with proper column handling
        if 'Primary_Position' in firsthalf_data.columns:
            primary_position = firsthalf_data['Primary_Position'].values
        elif 'Pos' in firsthalf_data.columns:
            primary_position = firsthalf_data['Pos'].values
        else:
            primary_position = ['P' if self.player_type == 'pitcher' else 'OF'] * len(firsthalf_data)

        # Build base results dictionary
        results_dict = {
            'playerid': firsthalf_data.get('playerid', pd.Series([None] * len(firsthalf_data))),
            'Name': firsthalf_data['Name'].values,
            'Team': firsthalf_data['Team'].values,
            'Primary_Position': primary_position,

            # Current (firsthalf actual)
            f'Current_{self.usage_col}': current_usage,
            'Current_WAR': current_WAR,
            'Current_WAR_pace': current_war_rates,

            # ROS (secondhalf projected)
            f'Remaining_{self.usage_col}': remaining_usage,
            'ROS_WAR': ros_WAR,
            'ROS_WAR_pace': ros_cumulative_war,  # Note: This is cumulative, not a rate (misleading column name)

            # Total projected
            f'Total_Projected_{self.usage_col}': current_usage + remaining_usage,
            'Total_Projected_WAR': current_WAR + ros_WAR,

            # Uncertainty bands (current actual + ROS quantiles)
            'Q10_WAR': current_WAR + ros_WAR_quantiles['q10'],
            'Q25_WAR': current_WAR + ros_WAR_quantiles['q25'],
            'Q50_WAR': current_WAR + ros_WAR_quantiles['q50'],
            'Q75_WAR': current_WAR + ros_WAR_quantiles['q75'],
            'Q90_WAR': current_WAR + ros_WAR_quantiles['q90'],

            # Uncertainty range (Q90 - Q10)
            'Uncertainty_Range': ros_WAR_quantiles['q90'] - ros_WAR_quantiles['q10']
        }

        # For pitchers, preserve GS, G, GS_per_G for type classification
        if self.player_type == 'pitcher':
            if 'GS' in firsthalf_data.columns:
                results_dict['GS'] = firsthalf_data['GS'].values
            if 'G' in firsthalf_data.columns:
                results_dict['G'] = firsthalf_data['G'].values
            if 'GS_per_G' in firsthalf_data.columns:
                results_dict['GS_per_G'] = firsthalf_data['GS_per_G'].values

        results = pd.DataFrame(results_dict)

        print(f"Complete! Generated projections for {len(results)} {self.player_type}s")
        return results

    def get_projection_summary(self, projections: pd.DataFrame, top_n: int = 10) -> str:
        """
        Generate a formatted summary of projections.

        Args:
            projections: Output from generate_projection()
            top_n: Number of top players to show (default: 10)

        Returns:
            Formatted string summary

        Example:
            >>> proj = gen.generate_projection(current_2025, historical_2016_2024)
            >>> print(gen.get_projection_summary(proj, top_n=10))
        """
        lines = []
        lines.append("=" * 90)
        lines.append(f"TOP {top_n} PROJECTED WAR ({self.player_type.upper()}S)")
        lines.append("=" * 90)

        top_players = projections.nlargest(top_n, 'Total_Projected_WAR')

        # Select columns for display
        display_cols = [
            'Name',
            'Team',
            'Current_WAR',
            'ROS_WAR',
            'Total_Projected_WAR',
            'Q10_WAR',
            'Q90_WAR',
            'Uncertainty_Range'
        ]

        # Format each column
        top_display = top_players[display_cols].copy()
        for col in ['Current_WAR', 'ROS_WAR', 'Total_Projected_WAR', 'Q10_WAR', 'Q90_WAR', 'Uncertainty_Range']:
            top_display[col] = top_display[col].round(1)

        lines.append(top_display.to_string(index=False))
        lines.append("=" * 90)

        return '\n'.join(lines)
