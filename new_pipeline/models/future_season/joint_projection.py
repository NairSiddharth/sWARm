"""
Joint Projection Model - Multi-Year WAR Forecasting

Combines longitudinal and survival models for 1-3 year projections.
Implements iterative forecasting with retirement risk discounting.

See FUTURE_PROJECTIONS_MODEL_ARCHITECTURE.md Module 3 for design specs.
See FUTURE_PROJECTIONS_MIGRATION_GUIDE.md Section 3D for migration notes.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from new_pipeline.common.constants import HITTER_MODEL_FEATURES, PITCHER_MODEL_FEATURES
from new_pipeline.models.future_season.longitudinal_model import LongitudinalModel
from new_pipeline.models.future_season.survival_model import SurvivalModel
from new_pipeline.models.future_season.age_curves import AgeCurveAdjuster
from new_pipeline.models.future_season.injury_recovery import apply_injury_recovery


class JointProjectionModel:
    """
    Multi-year WAR projections combining longitudinal and survival models.

    Implements iterative forecasting:
    - Year 1: Longitudinal model prediction
    - Year 2-3: Longitudinal + age curves + survival discounting

    Key features:
    - Retirement risk discounting (survival probabilities)
    - Age-adjusted performance trajectories
    - Compound survival probability over multiple years
    """

    def __init__(
        self,
        player_type: str,
        longitudinal_model: Optional[LongitudinalModel] = None,
        survival_model: Optional[SurvivalModel] = None,
        age_curves: Optional[AgeCurveAdjuster] = None
    ):
        """
        Initialize joint projection model.

        Args:
            player_type: 'hitter' or 'pitcher'
            longitudinal_model: Fitted LongitudinalModel (optional)
            survival_model: Fitted SurvivalModel (optional)
            age_curves: AgeCurveAdjuster instance (optional)
        """
        if player_type not in ['hitter', 'pitcher']:
            raise ValueError(f"player_type must be 'hitter' or 'pitcher', got {player_type}")

        self.player_type = player_type
        self.model_features = HITTER_MODEL_FEATURES if player_type == 'hitter' else PITCHER_MODEL_FEATURES

        # Initialize or use provided models
        self.longitudinal_model = longitudinal_model or LongitudinalModel(player_type)
        self.survival_model = survival_model or SurvivalModel()
        self.age_curves = age_curves or AgeCurveAdjuster()

    def project_player(
        self,
        current_features: pd.Series,
        years_ahead: int = 3,
        position: str = 'CF'
    ) -> Dict[str, float]:
        """
        Generate multi-year projections for a single player.

        Iterative approach:
        - Year 1: Direct longitudinal prediction
        - Year 2+: Age-adjusted + survival-discounted

        Args:
            current_features: Player's current year features (from sequences)
                Must include: age_n, war_n, [feature]_n for all model features
            years_ahead: Number of years to project (max 3)
            position: Player position for age curves

        Returns:
            Dictionary with projections: {'war_year_1': float, 'war_year_2': float, ...}
        """
        if not self.longitudinal_model.is_fitted:
            raise RuntimeError("Longitudinal model not fitted. Train models first.")

        # Extract current state
        current_age = current_features['age_n']
        current_war = current_features['war_n']

        # Initialize projections and tracking
        projections = {}
        cumulative_survival = 1.0
        age_factor_year_1 = None  # Track Year 1 age factor for relative scaling

        # For iterative projection, we need to simulate future years
        for year in range(1, years_ahead + 1):
            # Age for this projection year
            proj_age = current_age + year

            # Get age factor
            age_factor = self.age_curves.get_age_factor(proj_age, position)

            if year == 1:
                # Year 1: Direct prediction from longitudinal model
                # Check if using EnsembleLongitudinalModel (requires different format)
                from new_pipeline.models.future_season.ensemble_model import EnsembleLongitudinalModel

                if isinstance(self.longitudinal_model, EnsembleLongitudinalModel):
                    # Convert sequence format to historical DataFrame format
                    # Ensemble expects: Year, Age, WAR, [features] (no _n suffix)
                    historical_row = {
                        'Year': current_features.get('year_n', 2024),
                        'Age': current_features['age_n'],
                        'WAR': current_features['war_n']
                    }

                    # Add features without _n suffix
                    for col in current_features.index:
                        if col.endswith('_n') and col not in ['year_n', 'age_n', 'war_n']:
                            feature_name = col[:-2]  # Remove _n suffix
                            historical_row[feature_name] = current_features[col]

                    historical_df = pd.DataFrame([historical_row])
                    war_raw = self.longitudinal_model.predict_from_dataframe(historical_df)
                else:
                    # Old LongitudinalModel - use sequence format directly
                    features_df = pd.DataFrame([current_features])
                    war_raw = self.longitudinal_model.predict(features_df)[0]

                # Apply age curve adjustment
                war_adjusted = war_raw * age_factor

                # Store Year 1 age factor for relative scaling in future years
                age_factor_year_1 = age_factor

            else:
                # Year 2+: Apply RELATIVE age change from Year 1
                # Calculate how much the age factor changed relative to Year 1
                relative_age_change = age_factor / age_factor_year_1
                war_adjusted = projections['war_year_1'] * relative_age_change

            # Calculate survival probability for this year
            # Create survival features for this projection
            survival_features = pd.DataFrame([{
                'age_at_end': proj_age,
                'final_war': current_war,
                'career_war': current_war * 5,  # Rough estimate (5 year career avg)
                'peak_war': current_war * 1.2,  # Assume peak ~20% above current
                'war_decline': max(0, current_war * 0.1)  # Small decline
            }])

            try:
                # Get survival probability for this year
                if self.survival_model.is_fitted:
                    survival_prob = self.survival_model.predict_survival_probability(
                        survival_features,
                        years_ahead=year
                    )[0]
                else:
                    # Default survival probability if model not fitted
                    base_prob = 0.9
                    age_penalty = max(0, (proj_age - 30) * 0.02)
                    survival_prob = max(0.5, base_prob - age_penalty)

                # Cumulative survival (compound probability)
                if year > 1:
                    cumulative_survival *= survival_prob
                else:
                    cumulative_survival = survival_prob

            except Exception as e:
                # Fallback survival probability
                print(f"Warning: Survival prediction failed ({str(e)}). Using default.")
                cumulative_survival *= 0.9

            # Final projection: performance * survival probability
            war_final = war_adjusted * cumulative_survival

            # Store projection
            projections[f'war_year_{year}'] = max(0, war_final)  # Floor at 0

        return projections

    def project_multiple_players(
        self,
        sequences_df: pd.DataFrame,
        years_ahead: int = 3,
        position_col: str = None,
        injury_records: Optional[pd.DataFrame] = None,
        apply_injury_adjustments: bool = True
    ) -> pd.DataFrame:
        """
        Generate multi-year projections for multiple players.

        Args:
            sequences_df: DataFrame with player features (from build_longitudinal_sequences)
            years_ahead: Number of years to project
            position_col: Column name for player positions (optional)
            injury_records: DataFrame with injury information (optional)
                Must have columns: playerid, injury_type, surgery_year
            apply_injury_adjustments: Apply injury recovery adjustments if injury_records provided

        Returns:
            DataFrame with columns: playerid, war_year_1, war_year_2, war_year_3
        """
        if not self.longitudinal_model.is_fitted:
            raise RuntimeError("Longitudinal model not fitted. Train models first.")

        print(f"Generating {years_ahead}-year projections for {len(sequences_df)} players...")

        projections_list = []

        for idx, row in sequences_df.iterrows():
            # Get position (default to CF for hitters, P for pitchers)
            position = 'P' if self.player_type == 'pitcher' else 'CF'
            if position_col and position_col in row:
                position = row[position_col]

            # Project this player
            try:
                player_proj = self.project_player(row, years_ahead, position)
                player_proj['playerid'] = row['playerid']
                projections_list.append(player_proj)
            except Exception as e:
                print(f"Warning: Projection failed for player {row.get('playerid', 'unknown')}: {str(e)}")
                # Add default zero projections
                default_proj = {'playerid': row.get('playerid')}
                for year in range(1, years_ahead + 1):
                    default_proj[f'war_year_{year}'] = 0.0
                projections_list.append(default_proj)

        projections_df = pd.DataFrame(projections_list)

        print(f"  Completed: {len(projections_df)} player projections")
        print(f"  Year 1 avg: {projections_df['war_year_1'].mean():.2f} WAR")
        if years_ahead >= 2:
            print(f"  Year 2 avg: {projections_df['war_year_2'].mean():.2f} WAR")
        if years_ahead >= 3:
            print(f"  Year 3 avg: {projections_df['war_year_3'].mean():.2f} WAR")

        # Apply injury recovery adjustments if requested
        if apply_injury_adjustments and injury_records is not None and len(injury_records) > 0:
            # Add Age and Position from sequences_df for injury adjustments
            age_position_data = sequences_df[['playerid', 'age_n']].copy()
            age_position_data.rename(columns={'age_n': 'Age'}, inplace=True)

            # Add position if available
            if position_col and position_col in sequences_df.columns:
                age_position_data['Position'] = sequences_df[position_col]
            else:
                # Default position
                age_position_data['Position'] = 'P' if self.player_type == 'pitcher' else 'CF'

            # Merge age/position into projections
            projections_df = projections_df.merge(age_position_data, on='playerid', how='left')

            # Apply injury recovery
            war_cols = [f'war_year_{i}' for i in range(1, years_ahead + 1)]
            projections_df = apply_injury_recovery(
                projections_df,
                injury_records,
                war_columns=war_cols
            )

            print(f"\nPost-injury adjustment averages:")
            print(f"  Year 1 avg: {projections_df['war_year_1'].mean():.2f} WAR")
            if years_ahead >= 2:
                print(f"  Year 2 avg: {projections_df['war_year_2'].mean():.2f} WAR")
            if years_ahead >= 3:
                print(f"  Year 3 avg: {projections_df['war_year_3'].mean():.2f} WAR")

        return projections_df

    def get_performance_trajectory(
        self,
        current_features: pd.Series,
        years_ahead: int = 3,
        position: str = 'CF'
    ) -> Tuple[List[float], List[float], List[float]]:
        """
        Get detailed performance trajectory showing components.

        Returns raw predictions, age-adjusted, and survival-adjusted values.

        Args:
            current_features: Current year features
            years_ahead: Number of years
            position: Player position

        Returns:
            (raw_wars, age_adjusted_wars, final_wars)
        """
        projections = self.project_player(current_features, years_ahead, position)

        # For detailed breakdown, re-run with tracking
        current_age = current_features['age_n']
        raw_wars = []
        age_adjusted_wars = []
        final_wars = []

        for year in range(1, years_ahead + 1):
            proj_age = current_age + year
            age_factor = self.age_curves.get_age_factor(proj_age, position)

            if year == 1:
                # Check if using EnsembleLongitudinalModel (requires different format)
                from new_pipeline.models.future_season.ensemble_model import EnsembleLongitudinalModel

                if isinstance(self.longitudinal_model, EnsembleLongitudinalModel):
                    # Convert sequence format to historical DataFrame format
                    historical_row = {
                        'Year': current_features.get('year_n', 2024),
                        'Age': current_features['age_n'],
                        'WAR': current_features['war_n']
                    }
                    for col in current_features.index:
                        if col.endswith('_n') and col not in ['year_n', 'age_n', 'war_n']:
                            feature_name = col[:-2]
                            historical_row[feature_name] = current_features[col]

                    historical_df = pd.DataFrame([historical_row])
                    war_raw = self.longitudinal_model.predict_from_dataframe(historical_df)
                else:
                    # Old LongitudinalModel
                    features_df = pd.DataFrame([current_features])
                    war_raw = self.longitudinal_model.predict(features_df)[0]

                war_age_adj = war_raw * age_factor
            else:
                war_raw = projections['war_year_1']
                war_age_adj = war_raw * age_factor

            war_final = projections[f'war_year_{year}']

            raw_wars.append(war_raw)
            age_adjusted_wars.append(war_age_adj)
            final_wars.append(war_final)

        return raw_wars, age_adjusted_wars, final_wars
