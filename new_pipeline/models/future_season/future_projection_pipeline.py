"""
Future Projection Pipeline - End-to-End Orchestration

Complete orchestration for generating 1-3 year WAR projections.
Integrates all Phase 1-4 modules into a single cohesive workflow.

See FUTURE_PROJECTIONS_MASTER_PLAN.md Phase 5.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import sys
from datetime import datetime

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

# Darts for TimeSeries conversion
from darts import TimeSeries

# Phase 1: Data preparation
from new_pipeline.models.future_season.data_preparation import (
    load_historical_player_data,
    build_longitudinal_sequences
)

# Phase 2: Core models
from new_pipeline.models.future_season.ensemble_model import EnsembleLongitudinalModel
from new_pipeline.models.future_season.survival_model import SurvivalModel
from new_pipeline.models.future_season.age_curves import AgeCurveAdjuster

# Phase 3: Joint projection
from new_pipeline.models.future_season.joint_projection import JointProjectionModel
from new_pipeline.models.future_season.expected_stats import ExpectedStatsCalculator

# Phase 4: Adjustments
from new_pipeline.models.future_season.elite_adjustments import apply_elite_adjustments
from new_pipeline.models.future_season.injury_recovery import apply_injury_recovery
from new_pipeline.models.future_season.constraint_optimizer import ConstraintOptimizer


class FutureProjectionPipeline:
    """
    Complete orchestration pipeline for future WAR projections.

    Handles entire workflow:
    1. Data loading and sequence building
    2. Model training (Longitudinal + Survival)
    3. Multi-year projection generation
    4. Adjustment application (Elite, Injury, Expected Stats)
    5. Constraint enforcement (Zero-sum WAR)
    6. Output generation and saving
    """

    def __init__(
        self,
        player_type: str,
        base_year: int,
        years_ahead: int = 3,
        historical_years: Optional[List[int]] = None
    ):
        """
        Initialize future projection pipeline.

        Args:
            player_type: 'hitter' or 'pitcher'
            base_year: Year to project from (e.g., 2024 to project 2025-2027)
            years_ahead: Number of years to project (default: 3)
            historical_years: Years of historical data to use (default: base_year-8 to base_year)
        """
        if player_type not in ['hitter', 'pitcher']:
            raise ValueError(f"player_type must be 'hitter' or 'pitcher', got {player_type}")

        self.player_type = player_type
        self.base_year = base_year
        self.years_ahead = years_ahead

        # Default: Use 9 years of historical data (base_year - 8 through base_year)
        if historical_years is None:
            self.historical_years = list(range(base_year - 8, base_year + 1))
        else:
            self.historical_years = historical_years

        # Initialize models
        self.longitudinal_model = EnsembleLongitudinalModel(player_type)
        self.survival_model = SurvivalModel()
        self.age_curves = AgeCurveAdjuster()
        self.expected_stats_calculator = ExpectedStatsCalculator(player_type)

        # Data storage
        self.historical_data = None
        self.sequences_df = None
        self.projections_df = None

        print(f"\nInitialized {player_type} projection pipeline:")
        print(f"  Base year: {base_year}")
        print(f"  Projection years: {base_year + 1} - {base_year + years_ahead}")
        print(f"  Historical data: {min(self.historical_years)} - {max(self.historical_years)}")

    def _convert_to_timeseries(
        self,
        historical_df: pd.DataFrame
    ) -> Tuple[List[TimeSeries], List[TimeSeries], Dict]:
        """
        Convert historical DataFrame to Darts TimeSeries format for ensemble training.

        Args:
            historical_df: Player-season data from load_historical_player_data()

        Returns:
            Tuple of (target_series_list, covariate_series_list, player_info_dict)
        """
        from new_pipeline.models.future_season.constants import FUTURE_HITTER_MODEL_FEATURES, FUTURE_PITCHER_MODEL_FEATURES

        feature_cols = FUTURE_HITTER_MODEL_FEATURES if self.player_type == 'hitter' else FUTURE_PITCHER_MODEL_FEATURES

        target_series_list = []
        covariate_series_list = []
        player_info = {}

        for playerid in historical_df['playerid'].unique():
            player_data = historical_df[
                historical_df['playerid'] == playerid
            ].sort_values('Year').copy()

            if len(player_data) < 3:
                continue  # Need at least 3 years for longitudinal modeling

            # Check for consecutive years
            years = player_data['Year'].values
            if not np.all(np.diff(years) == 1):
                continue  # Skip players with gaps in history

            # Handle missing values
            player_data['years_since_tommy_john'].fillna(-1, inplace=True)

            # Create target series (WAR)
            target_series = TimeSeries.from_dataframe(
                df=player_data,
                time_col='Year',
                value_cols=['WAR'],
                fill_missing_dates=False
            )

            # Create covariate series (features + Age)
            covariate_cols = feature_cols + ['Age']
            available_cols = [col for col in covariate_cols if col in player_data.columns]

            covariate_series = TimeSeries.from_dataframe(
                df=player_data,
                time_col='Year',
                value_cols=available_cols,
                fill_missing_dates=False
            )

            target_series_list.append(target_series)
            covariate_series_list.append(covariate_series)
            player_info[playerid] = {
                'history_length': len(target_series),
                'years': years
            }

        return target_series_list, covariate_series_list, player_info

    def load_and_prepare_data(self) -> pd.DataFrame:
        """
        Load historical data and build longitudinal sequences.

        Returns:
            Sequences DataFrame ready for model training
        """
        print(f"\n--- Step 1: Loading & Preparing Data ---")

        # Load historical player data
        print(f"Loading {self.player_type} data for years {min(self.historical_years)}-{max(self.historical_years)}...")
        self.historical_data = load_historical_player_data(
            player_type=self.player_type,
            years=self.historical_years
        )

        print(f"  Loaded {len(self.historical_data)} player-season records")

        # Build longitudinal sequences
        print("Building longitudinal sequences...")
        self.sequences_df = build_longitudinal_sequences(
            self.historical_data,
            player_type=self.player_type
        )

        print(f"  Created {len(self.sequences_df)} sequences")

        return self.sequences_df

    def train_models(self) -> Dict[str, Dict]:
        """
        Train longitudinal and survival models.

        Returns:
            Dictionary with training metrics
        """
        print(f"\n--- Step 2: Training Models ---")

        if self.sequences_df is None or self.historical_data is None:
            raise RuntimeError("No data loaded. Call load_and_prepare_data() first.")

        training_metrics = {}

        # Train longitudinal model (ensemble requires TimeSeries format)
        print("Training longitudinal model...")
        print("  Converting historical data to TimeSeries format...")
        target_series, covariate_series, player_info = self._convert_to_timeseries(self.historical_data)
        print(f"  Converted {len(target_series)} players with consecutive seasons")

        # Train ensemble model
        long_metrics = self.longitudinal_model.train(target_series, covariate_series)
        training_metrics['longitudinal'] = long_metrics

        # Display training results
        print(f"\n  Ensemble Training Results:")
        print(f"    XGBoost trained: {long_metrics['xgboost']['trained']}")
        print(f"    RNN trained: {long_metrics['rnn']['trained']}")
        print(f"    ExtraTrees (Darts) trained: {long_metrics['extratrees']['trained']}")
        print(f"    Fallback (ExtraTrees) trained: {long_metrics['fallback']['trained']}")
        print(f"    Fallback R²: {long_metrics['fallback']['r2']:.3f}")
        print(f"    Fallback RMSE: {long_metrics['fallback']['rmse']:.3f}")
        print(f"    Players with <4 seasons (fallback only): {long_metrics['short_history_players']}")

        # Prepare and train survival model
        print("\nTraining survival model...")
        survival_data = self.survival_model.prepare_survival_data(
            self.historical_data,
            current_year=self.base_year
        )

        survival_metrics = self.survival_model.train(survival_data)
        training_metrics['survival'] = survival_metrics

        print(f"  Concordance Index: {survival_metrics['concordance_index']:.3f}")
        event_rate = survival_metrics['n_events'] / survival_metrics['n_observations']
        print(f"  Event rate: {event_rate:.2%}")
        print(f"  Training samples: {survival_metrics['n_observations']}")

        return training_metrics

    def generate_projections(
        self,
        position_col: str = 'primary_position'
    ) -> pd.DataFrame:
        """
        Generate multi-year projections using joint model.

        Args:
            position_col: Column name for player positions

        Returns:
            DataFrame with projections (war_year_1, war_year_2, war_year_3)
        """
        print(f"\n--- Step 3: Generating Projections ---")

        if not self.longitudinal_model.is_fitted:
            raise RuntimeError("Models not trained. Call train_models() first.")

        # Create joint projection model
        joint_model = JointProjectionModel(
            player_type=self.player_type,
            longitudinal_model=self.longitudinal_model,
            survival_model=self.survival_model,
            age_curves=self.age_curves
        )

        # Get current year players from historical_data (most recent year = base_year)
        # Sequences are for training; for projection we use the latest year's actual data
        current_year_data = self.historical_data[
            self.historical_data['Year'] == self.base_year
        ].copy()

        if len(current_year_data) == 0:
            # If base_year not in data, use most recent year
            most_recent_year = self.historical_data['Year'].max()
            print(f"Warning: No data for base_year {self.base_year}, using {most_recent_year} instead")
            current_year_data = self.historical_data[
                self.historical_data['Year'] == most_recent_year
            ].copy()

        # Convert current year data to sequence format (without target)
        from new_pipeline.models.future_season.data_preparation import add_age_context_features
        from new_pipeline.models.future_season.constants import FUTURE_HITTER_MODEL_FEATURES, FUTURE_PITCHER_MODEL_FEATURES

        model_features = FUTURE_HITTER_MODEL_FEATURES if self.player_type == 'hitter' else FUTURE_PITCHER_MODEL_FEATURES

        current_sequences = []
        for _, row in current_year_data.iterrows():
            sequence = {
                'playerid': row['playerid'],
                'year_n': int(row['Year']),
                'age_n': float(row['Age']),
                'war_n': float(row['WAR'])
            }

            # Add model features
            for feature in model_features:
                if feature in row:
                    sequence[f'{feature}_n'] = float(row[feature])

            current_sequences.append(sequence)

        current_sequences = pd.DataFrame(current_sequences)

        # Add age context features
        current_sequences = add_age_context_features(current_sequences, self.player_type)

        print(f"Projecting {len(current_sequences)} {self.player_type}s for {self.years_ahead} years...")

        # Generate projections
        self.projections_df = joint_model.project_multiple_players(
            current_sequences,
            years_ahead=self.years_ahead,
            position_col=position_col
        )

        return self.projections_df

    def apply_adjustments(
        self,
        injury_records: Optional[pd.DataFrame] = None,
        apply_expected_stats: bool = True,
        apply_elite: bool = True,
        apply_injury: bool = True
    ) -> pd.DataFrame:
        """
        Apply all projection adjustments.

        Args:
            injury_records: Optional DataFrame with injury information
            apply_expected_stats: Apply expected stats regression
            apply_elite: Apply elite player protection
            apply_injury: Apply injury recovery adjustments

        Returns:
            Adjusted projections DataFrame
        """
        print(f"\n--- Step 4: Applying Adjustments ---")

        if self.projections_df is None:
            raise RuntimeError("No projections generated. Call generate_projections() first.")

        adjusted_df = self.projections_df.copy()

        # Expected stats regression (luck correction)
        if apply_expected_stats:
            print("Applying expected stats regression...")
            # This would typically use xBA, xSLG data if available
            # For now, projections are used as-is
            print("  (Expected stats data not available, skipping)")

        # Elite player protection
        if apply_elite:
            print("Applying elite player protection...")
            # Get actual historical WAR from base year as baseline for elite detection
            # (NOT the projection - we need to compare against actual performance)
            historical_war_series = self.historical_data[
                self.historical_data['Year'] == self.base_year
            ].set_index('playerid')['WAR']

            # Align with projections DataFrame
            historical_war_aligned = adjusted_df['playerid'].map(historical_war_series)

            adjusted_df = apply_elite_adjustments(
                adjusted_df,
                historical_war=historical_war_aligned,  # Use actual base year WAR
                war_columns=[f'war_year_{i}' for i in range(1, self.years_ahead + 1)],
                player_type=self.player_type  # Use pitcher-specific thresholds for pitchers
            )

        # Injury recovery adjustments
        if apply_injury and injury_records is not None:
            print("Applying injury recovery adjustments...")
            adjusted_df = apply_injury_recovery(
                adjusted_df,
                injury_records,
                war_columns=[f'war_year_{i}' for i in range(1, self.years_ahead + 1)]
            )
        elif apply_injury:
            print("No injury records provided, skipping injury adjustments")

        self.projections_df = adjusted_df
        return adjusted_df

    def apply_constraints(
        self,
        hitter_projections: Optional[pd.DataFrame] = None,
        pitcher_projections: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Apply zero-sum WAR constraints.

        Args:
            hitter_projections: Hitter projections (if combining with pitcher projections)
            pitcher_projections: Pitcher projections (if combining with hitter projections)

        Returns:
            Constrained projections DataFrame
        """
        print(f"\n--- Step 5: Applying Zero-Sum Constraints ---")

        if hitter_projections is not None and pitcher_projections is not None:
            # Combining hitter and pitcher projections
            hitter_projections['player_type'] = 'hitter'
            pitcher_projections['player_type'] = 'pitcher'

            combined_df = pd.concat([hitter_projections, pitcher_projections], ignore_index=True)

            optimizer = ConstraintOptimizer(
                target_total=1000.0,
                hitter_target=570.0,
                pitcher_target=430.0
            )

            constrained_df = optimizer.apply_constraints(
                combined_df,
                player_type_col='player_type',
                war_columns=[f'war_year_{i}' for i in range(1, self.years_ahead + 1)],
                use_tiered_protection=True
            )

            # Split back into hitters and pitchers
            if self.player_type == 'hitter':
                self.projections_df = constrained_df[constrained_df['player_type'] == 'hitter'].copy()
            else:
                self.projections_df = constrained_df[constrained_df['player_type'] == 'pitcher'].copy()

        else:
            # Single player type - can't enforce full constraint without both
            print("Warning: Constraint enforcement requires both hitter and pitcher projections")
            print("         Run pipeline for both types and pass to apply_constraints()")

        return self.projections_df

    def _categorize_war_tier(self, war_value: float) -> str:
        """Categorize WAR into tiers."""
        if war_value >= 5.0:
            return 'Elite'
        elif war_value >= 3.0:
            return 'Good'
        elif war_value >= 0.0:
            return 'Average'
        else:
            return 'Below Replacement Level'

    def _prepare_output_dataframe(self, include_metadata: bool = True) -> pd.DataFrame:
        """
        Prepare output DataFrame with metadata and formatting.

        Args:
            include_metadata: Include metadata columns (age, position, etc.)

        Returns:
            Enriched DataFrame with player metadata and formatted projections
        """
        if self.projections_df is None:
            raise RuntimeError("No projections to prepare. Run pipeline first.")

        # Start with projections
        output_df = self.projections_df.copy()

        # Merge player info from current year data
        if include_metadata and self.historical_data is not None:
            current_year_data = self.historical_data[
                self.historical_data['Year'] == self.base_year
            ].copy()

            # Select identification columns
            id_cols = ['playerid', 'Name', 'Team', 'Age']

            # Add position/role column
            if self.player_type == 'hitter':
                if 'primary_position' in current_year_data.columns:
                    id_cols.append('primary_position')
                usage_col = 'PA'
            else:
                if 'Role' in current_year_data.columns:
                    id_cols.append('Role')
                elif 'primary_position' in current_year_data.columns:
                    id_cols.append('primary_position')
                usage_col = 'IP'

            # Add usage column
            if usage_col in current_year_data.columns:
                id_cols.append(usage_col)

            # Add baseline WAR
            id_cols.append('WAR')

            # Filter to available columns
            available_cols = [col for col in id_cols if col in current_year_data.columns]
            player_info = current_year_data[available_cols].copy()

            # Rename columns for clarity
            rename_map = {}
            if 'WAR' in player_info.columns:
                rename_map['WAR'] = f'war_{self.base_year}'
            if 'primary_position' in player_info.columns:
                rename_map['primary_position'] = 'Position' if self.player_type == 'hitter' else 'Role'

            if rename_map:
                player_info = player_info.rename(columns=rename_map)

            # Merge with projections
            output_df = output_df.merge(player_info, on='playerid', how='left')

        # Rename projection columns to actual years
        year_rename = {}
        for i in range(1, self.years_ahead + 1):
            old_col = f'war_year_{i}'
            new_col = f'war_{self.base_year + i}'
            if old_col in output_df.columns:
                year_rename[old_col] = new_col

        if year_rename:
            output_df = output_df.rename(columns=year_rename)

        # Round WAR columns to 2 decimal places
        war_cols = [col for col in output_df.columns if col.startswith('war_')]
        for col in war_cols:
            output_df[col] = output_df[col].round(2)

        # Add tier and percentile for Year 1 projection
        year_1_col = f'war_{self.base_year + 1}'
        if year_1_col in output_df.columns:
            # Add tier
            output_df[f'{year_1_col}_tier'] = output_df[year_1_col].apply(self._categorize_war_tier)

            # Add percentile (0-100)
            output_df[f'{year_1_col}_percentile'] = (
                output_df[year_1_col].rank(pct=True) * 100
            ).round(1)

        # Add standard metadata
        if include_metadata:
            output_df['projection_date'] = datetime.now().strftime('%Y-%m-%d')
            output_df['base_year'] = self.base_year
            output_df['model_version'] = 'v1.0'

        # Reorder columns for usability
        priority_cols = ['playerid', 'Name', 'Team', 'Age']

        # Add position/role
        if 'Position' in output_df.columns:
            priority_cols.append('Position')
        elif 'Role' in output_df.columns:
            priority_cols.append('Role')

        # Add baseline stats
        baseline_war = f'war_{self.base_year}'
        if baseline_war in output_df.columns:
            priority_cols.append(baseline_war)

        usage_col = 'PA' if self.player_type == 'hitter' else 'IP'
        if usage_col in output_df.columns:
            priority_cols.append(usage_col)

        # Add projection columns
        for i in range(1, self.years_ahead + 1):
            proj_col = f'war_{self.base_year + i}'
            if proj_col in output_df.columns:
                priority_cols.append(proj_col)

        # Add tier and percentile
        if f'{year_1_col}_tier' in output_df.columns:
            priority_cols.extend([f'{year_1_col}_tier', f'{year_1_col}_percentile'])

        # Add metadata columns
        metadata_cols = ['projection_date', 'base_year', 'model_version', 'player_type']
        for col in metadata_cols:
            if col in output_df.columns:
                priority_cols.append(col)

        # Add any remaining columns
        remaining_cols = [col for col in output_df.columns if col not in priority_cols]
        final_cols = priority_cols + remaining_cols

        # Filter to columns that exist
        final_cols = [col for col in final_cols if col in output_df.columns]
        output_df = output_df[final_cols]

        return output_df

    def save_projections(
        self,
        output_path: Optional[str] = None,
        include_metadata: bool = True
    ) -> Tuple[str, pd.DataFrame]:
        """
        Save projections to CSV file with enhanced readability.

        Args:
            output_path: Path to save file (default: predictions/future_projections_{player_type}_{year}.csv)
            include_metadata: Include metadata columns (age, position, etc.)

        Returns:
            Tuple of (path where file was saved, enriched DataFrame with metadata)
        """
        print(f"\n--- Step 6: Saving Projections ---")

        # Default output path
        if output_path is None:
            predictions_dir = project_root / "predictions"
            predictions_dir.mkdir(exist_ok=True)

            filename = f"future_projections_{self.player_type}_{self.base_year + 1}.csv"
            output_path = predictions_dir / filename

        # Prepare output DataFrame with metadata
        output_df = self._prepare_output_dataframe(include_metadata=include_metadata)

        # Save to CSV
        output_df.to_csv(output_path, index=False)

        print(f"Projections saved to: {output_path}")
        print(f"  Total players: {len(output_df)}")
        print(f"  Columns: {len(output_df.columns)}")
        print(f"  Projection years: {self.base_year + 1}-{self.base_year + self.years_ahead}")

        return str(output_path), output_df

    def run_full_pipeline(
        self,
        injury_records: Optional[pd.DataFrame] = None,
        save_output: bool = True
    ) -> pd.DataFrame:
        """
        Run complete projection pipeline end-to-end.

        Args:
            injury_records: Optional injury records for recovery adjustments
            save_output: Save projections to CSV

        Returns:
            Final projections DataFrame

        Example:
            >>> pipeline = FutureProjectionPipeline('hitter', base_year=2024, years_ahead=3)
            >>> projections = pipeline.run_full_pipeline()
            >>> print(projections[['playerid', 'war_year_1', 'war_year_2', 'war_year_3']].head())
        """
        print("="*70)
        print(f"FUTURE PROJECTION PIPELINE - {self.player_type.upper()}S")
        print("="*70)

        # Step 1: Load data
        self.load_and_prepare_data()

        # Step 2: Train models
        training_metrics = self.train_models()

        # Step 3: Generate projections
        self.generate_projections()

        # Step 4: Apply adjustments
        self.apply_adjustments(injury_records=injury_records)

        # Step 5: Apply constraints (requires both hitters and pitchers)
        # Skip for single player type pipeline
        print(f"\n--- Step 5: Constraint Enforcement ---")
        print("Note: Zero-sum constraints require both hitter and pitcher projections")
        print("      Run separate hitter/pitcher pipelines and combine with apply_constraints()")

        # Step 6: Save output
        if save_output:
            self.save_projections()

        print("\n" + "="*70)
        print("PIPELINE COMPLETE")
        print("="*70)

        return self.projections_df


def generate_league_projections(
    base_year: int,
    years_ahead: int = 3,
    injury_records: Optional[pd.DataFrame] = None,
    save_output: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate projections for entire league (hitters + pitchers) with constraints.

    Args:
        base_year: Year to project from
        years_ahead: Number of years to project
        injury_records: Optional injury records
        save_output: Save projections to CSV

    Returns:
        Tuple of (hitter_projections, pitcher_projections)

    Example:
        >>> hitters, pitchers = generate_league_projections(2024, years_ahead=3)
        >>> print(f"Hitters: {len(hitters)}, Pitchers: {len(pitchers)}")
    """
    print("="*70)
    print("LEAGUE-WIDE PROJECTION GENERATION")
    print("="*70)

    # Run hitter pipeline
    print("\n### HITTERS ###")
    hitter_pipeline = FutureProjectionPipeline('hitter', base_year, years_ahead)
    hitter_pipeline.run_full_pipeline(injury_records=injury_records, save_output=False)

    # Run pitcher pipeline
    print("\n### PITCHERS ###")
    pitcher_pipeline = FutureProjectionPipeline('pitcher', base_year, years_ahead)
    pitcher_pipeline.run_full_pipeline(injury_records=injury_records, save_output=False)

    # Apply league-wide constraints
    print("\n### LEAGUE-WIDE CONSTRAINT ENFORCEMENT ###")
    hitter_pipeline.apply_constraints(
        hitter_projections=hitter_pipeline.projections_df,
        pitcher_projections=pitcher_pipeline.projections_df
    )

    pitcher_pipeline.apply_constraints(
        hitter_projections=hitter_pipeline.projections_df,
        pitcher_projections=pitcher_pipeline.projections_df
    )

    # Prepare enriched DataFrames with metadata
    hitter_output = hitter_pipeline._prepare_output_dataframe(include_metadata=True)
    pitcher_output = pitcher_pipeline._prepare_output_dataframe(include_metadata=True)

    # Save outputs if requested
    if save_output:
        hitter_pipeline.save_projections()
        pitcher_pipeline.save_projections()

    print("\n" + "="*70)
    print("LEAGUE-WIDE PROJECTIONS COMPLETE")
    print("="*70)

    return hitter_output, pitcher_output
