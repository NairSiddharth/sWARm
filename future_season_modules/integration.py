"""
SYSTEM 2 Integration Pipeline
============================

Complete orchestration pipeline for SYSTEM 2: Future Performance Projections
Integrates all components for end-to-end joint longitudinal-survival modeling.

Classes:
    System2Pipeline: Main orchestration class for complete SYSTEM 2 workflow
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import warnings
import os
from scipy.optimize import minimize

from .expected_stats import ExpectedStatsCalculator
from .future_projections import FutureProjectionAgeCurve
from .validation import AgeCurveValidator
from .elite_adjustment import ElitePlayerAdjuster
# from .positional_adjustments import PositionalAdjustmentCalculator  # Not needed - adjustments already in WAR/WARP
DATA_DIR = r"C:\Users\nairs\Documents\GithubProjects\oWAR\MLB Player Data"

class System2Pipeline:
    """
    Complete pipeline for SYSTEM 2 future performance projections.

    Orchestrates data loading, feature engineering, model training, validation,
    and projection generation for 1-3 year player performance forecasting.
    """

    def __init__(self,
                 bp_data_path: str = r"C:\Users\nairs\Documents\GithubProjects\oWAR\MLB Player Data\BP_Data",
                 fg_data_path: str = r"C:\Users\nairs\Documents\GithubProjects\oWAR\MLB Player Data\FanGraphs_Data",
                 max_projection_years: int = 3,
                 use_dynasty_guru: bool = False,
                 use_zero_sum_constraint: bool = False,
                 use_injury_modeling: bool = True):
        """
        Initialize the SYSTEM 2 pipeline.

        Args:
            bp_data_path: Path to Baseball Prospectus data
            fg_data_path: Path to FanGraphs data
            max_projection_years: Maximum years to project (default 3)
            use_dynasty_guru: Whether to use Dynasty Guru enhanced aging curves
            use_zero_sum_constraint: Whether to apply zero-sum WAR constraints
            use_injury_modeling: Whether to apply injury recovery modeling
        """
        self.bp_data_path = bp_data_path
        self.fg_data_path = fg_data_path
        self.max_projection_years = max_projection_years
        self.use_dynasty_guru = use_dynasty_guru
        self.use_zero_sum_constraint = use_zero_sum_constraint
        self.use_injury_modeling = use_injury_modeling

        # Initialize components
        self.expected_calculator = ExpectedStatsCalculator()
        self.war_model = None
        self.warp_model = None
        self.projection_model = None  # Primary model for backward compatibility
        self.validator = AgeCurveValidator()
        self.elite_adjuster = ElitePlayerAdjuster()  # OPTION C: Elite player adjustment
        # self.position_calculator = PositionalAdjustmentCalculator()  # Not needed - adjustments already in WAR/WARP

        # Data storage
        self.raw_data = None
        self.processed_data = None
        self.training_data = None
        self.injury_data = None  # Injury recovery modeling data
        self.model_performance = None

    def load_complete_dataset(self,
                            years: Optional[List[int]] = None,
                            player_types: List[str] = ['hitters', 'pitchers']) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load complete dataset from all sources using enhanced_data_loading patterns.

        Args:
            years: Years to load (default 2016-2024)
            player_types: Types of players to load

        Returns:
            Combined dataset with all player data
        """
        if years is None:
            years = list(range(2016, 2024))

        print("SYSTEM 2: Loading complete dataset...")
        print("=" * 50)

        all_data = []

        # Load data using enhanced_data_loading patterns for each player type
        for player_type in player_types:
            # Load FanGraphs data (WAR)
            fg_data = self._load_fangraphs_comprehensive(player_type, years)
            if not fg_data.empty:
                fg_data['DataSource'] = 'WAR'
                # Standardize column names
                fg_data = self._standardize_fg_columns(fg_data)
                # Remove any duplicate columns that might have been created
                fg_data = fg_data.loc[:, ~fg_data.columns.duplicated()]
                all_data.append(fg_data)

            # Load Baseball Prospectus data (WARP)
            bp_data = self._load_bp_comprehensive(player_type, years)
            if not bp_data.empty:
                bp_data['DataSource'] = 'WARP'
                # Standardize column names (keep WARP separate from WAR)
                bp_data = self._standardize_bp_columns(bp_data)
                # Remove any duplicate columns that might have been created
                bp_data = bp_data.loc[:, ~bp_data.columns.duplicated()]
                all_data.append(bp_data)

        if not all_data:
            raise ValueError("No data could be loaded from any source")

        # Combine all data (keeping separate WAR and WARP records)
        combined_data = pd.concat(all_data, ignore_index=True)

        # CRITICAL FIX: Merge Age information from BP data to FG data
        combined_data = self._merge_age_information(combined_data)

        # Add position information from FanGraphs defensive data to both WAR and WARP records
        combined_data = self._merge_position_information(combined_data, years)

        # Load and integrate expected stats
        combined_data = self._load_and_merge_expected_stats(combined_data, years)

        print(f"Complete dataset loaded: {len(combined_data)} records")
        print(f"  Years: {combined_data['Season'].min()}-{combined_data['Season'].max()}")
        print(f"  Players: {combined_data['mlbid'].nunique()} unique")
        print(f"  Data sources: {combined_data['DataSource'].value_counts().to_dict()}")

        self.raw_data = combined_data
        return combined_data

    def _load_fangraphs_comprehensive(self, player_type: str, years: List[int]) -> pd.DataFrame:
        """
        Load comprehensive FanGraphs data following enhanced_data_loading patterns.

        Args:
            player_type: 'hitters' or 'pitchers'
            years: List of years to load

        Returns:
            Combined DataFrame with all FanGraphs metrics and season information
        """
        print(f"  Loading FanGraphs {player_type} data...")


        all_data = []

        for year in years:
            # Load the 3 main file types for this year - defensive data handled separately
            main_file = os.path.join(self.fg_data_path, player_type, f"fangraphs_{player_type}_{year}.csv")
            advanced_file = os.path.join(self.fg_data_path, player_type, f"fangraphs_{player_type}_{year}_advanced.csv")
            standard_file = os.path.join(self.fg_data_path, player_type, f"fangraphs_{player_type}_{year}_standard.csv")

            # Load main file (has WAR, wOBA, wRC+, etc.)
            if os.path.exists(main_file):
                try:
                    df_main = pd.read_csv(main_file, encoding='utf-8-sig')
                    df_main['Season'] = year  # Use Season instead of Year for consistency
                    df_main['PlayerType'] = player_type.rstrip('s').title()  # 'hitters' -> 'Hitter'

                    # Load advanced file (has UBR, wSB, wRAA, etc.)
                    if os.path.exists(advanced_file):
                        try:
                            df_advanced = pd.read_csv(advanced_file, encoding='utf-8-sig')
                            # Merge with main on common identifiers
                            merge_cols = ['Name', 'Team', 'PlayerId', 'MLBAMID']
                            available_merge_cols = [col for col in merge_cols if col in df_main.columns and col in df_advanced.columns]
                            if available_merge_cols:
                                # Select only unique columns from advanced (avoid duplicates)
                                advanced_unique_cols = available_merge_cols + [col for col in df_advanced.columns if col not in df_main.columns]
                                df_advanced_unique = df_advanced[advanced_unique_cols]
                                df_main = df_main.merge(df_advanced_unique, on=available_merge_cols, how='left')
                        except Exception as e:
                            warnings.warn(f"Could not merge advanced file {year}: {str(e)}")

                    # Load standard file (has traditional counting stats)
                    if os.path.exists(standard_file):
                        try:
                            df_standard = pd.read_csv(standard_file, encoding='utf-8-sig')
                            # Merge with main on common identifiers
                            merge_cols = ['Name', 'Team', 'PlayerId', 'MLBAMID']
                            available_merge_cols = [col for col in merge_cols if col in df_main.columns and col in df_standard.columns]
                            if available_merge_cols:
                                # Select only unique columns from standard
                                standard_unique_cols = available_merge_cols + [col for col in df_standard.columns if col not in df_main.columns]
                                df_standard_unique = df_standard[standard_unique_cols]
                                df_main = df_main.merge(df_standard_unique, on=available_merge_cols, how='left')
                        except Exception as e:
                            warnings.warn(f"Could not merge standard file {year}: {str(e)}")

                    all_data.append(df_main)

                except Exception as e:
                    warnings.warn(f"Could not load FanGraphs {player_type} {year}: {str(e)}")
                    continue

        if not all_data:
            warnings.warn(f"No FanGraphs {player_type} data found!")
            return pd.DataFrame()

        # Combine all years
        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"    Loaded {len(combined_df)} {player_type} player-seasons from {len(years)} years")
        return combined_df

    def _load_bp_comprehensive(self, player_type: str, years: List[int]) -> pd.DataFrame:
        """
        Load comprehensive Baseball Prospectus data following enhanced_data_loading patterns.

        Args:
            player_type: 'hitters' or 'pitchers'
            years: List of years to load

        Returns:
            Combined DataFrame with all BP metrics and season information
        """
        print(f"  Loading BP {player_type} data...")

        all_data = []

        for year in years:
            # Load the 2 file types for this year
            main_file = os.path.join(self.bp_data_path,player_type, f"bp_{player_type}_{year}.csv")
            standard_file = os.path.join(self.bp_data_path,player_type, f"bp_{player_type}_{year}_standard.csv")

            # Load main file (has WARP, DRC+, DRA, etc.)
            if os.path.exists(main_file):
                try:
                    df_main = pd.read_csv(main_file, encoding='utf-8-sig')
                    df_main['Season'] = year  # Use Season instead of Year for consistency
                    df_main['PlayerType'] = player_type.rstrip('s').title()  # 'hitters' -> 'Hitter'

                    # Load standard file if available and merge
                    if os.path.exists(standard_file):
                        try:
                            df_standard = pd.read_csv(standard_file, encoding='utf-8-sig')
                            # Merge on common identifiers
                            merge_cols = ['bpid', 'mlbid', 'Name', 'Team']
                            available_merge_cols = [col for col in merge_cols if col in df_main.columns and col in df_standard.columns]
                            if available_merge_cols:
                                # Select only unique columns from standard
                                standard_unique_cols = available_merge_cols + [col for col in df_standard.columns if col not in df_main.columns]
                                df_standard_unique = df_standard[standard_unique_cols]
                                df_main = df_main.merge(df_standard_unique, on=available_merge_cols, how='left')
                        except Exception as e:
                            warnings.warn(f"Could not merge standard file {year}: {str(e)}")

                    all_data.append(df_main)

                except Exception as e:
                    warnings.warn(f"Could not load BP {player_type} {year}: {str(e)}")
                    continue

        if not all_data:
            warnings.warn(f"No BP {player_type} data found!")
            return pd.DataFrame()

        # Combine all years
        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"    Loaded {len(combined_df)} {player_type} player-seasons from {len(years)} years")
        return combined_df

    def _standardize_bp_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize Baseball Prospectus column names."""
        column_mapping = {
            'MLBAMID': 'mlbid',
            'mlbamid': 'mlbid',
            'player_name': 'Name',
            'Player': 'Name',
            'WARP': 'WARP',  # Keep WARP as separate metric from WAR
            'warp': 'WARP',
            'pos': 'Position',
            'POS': 'Position',
            'position': 'Position',
            'Age': 'Age',  # Keep age data from BP files
            'age': 'Age'
        }
        return df.rename(columns=column_mapping)

    def _standardize_fg_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize FanGraphs column names."""
        column_mapping = {
            'MLBAMID': 'mlbid',
            'mlbamid': 'mlbid',
            'playerid': 'mlbid',
            # Note: PlayerId is different from MLBAMID - using MLBAMID as primary ID
            'Name': 'Name',
            'player_name': 'Name',
            'Pos': 'Position',
            'position': 'Position',
            'WAR': 'WAR',  # Preserve WAR column - critical for WAR projections
            'war': 'WAR'   # Handle lowercase variation
        }
        return df.rename(columns=column_mapping)

    def _merge_war_warp_datasets(self, all_data: List[pd.DataFrame]) -> pd.DataFrame:
        """
        Merge WAR and WARP datasets into unified player records.

        Args:
            all_data: List of DataFrames from different sources

        Returns:
            Unified DataFrame with both WAR and WARP columns for each player
        """
        print("  Merging WAR and WARP data into unified player records...")

        war_data = []
        warp_data = []

        # Separate WAR and WARP data
        for df in all_data:
            if 'DataSource' in df.columns:
                if (df['DataSource'] == 'WAR').any():
                    war_data.append(df[df['DataSource'] == 'WAR'].copy())
                if (df['DataSource'] == 'WARP').any():
                    warp_data.append(df[df['DataSource'] == 'WARP'].copy())

        # Combine each type
        if war_data:
            war_combined = pd.concat(war_data, ignore_index=True)
        else:
            war_combined = pd.DataFrame()

        if warp_data:
            warp_combined = pd.concat(warp_data, ignore_index=True)
        else:
            warp_combined = pd.DataFrame()

        print(f"    WAR records: {len(war_combined)}")
        print(f"    WARP records: {len(warp_combined)}")

        # Merge on Name + Season (since mlbids are different)
        if not war_combined.empty and not warp_combined.empty:
            # Use WARP as base (has Age data)
            base_df = warp_combined.copy()
            base_df = base_df.rename(columns={'WAR': 'WARP', 'WARP': 'WARP'})  # Ensure WARP column

            # Merge WAR data
            war_merge = war_combined[['Name', 'Season', 'WAR']].copy()
            unified_df = base_df.merge(
                war_merge,
                on=['Name', 'Season'],
                how='outer',
                suffixes=('', '_war')
            )

            # Fill missing values and cleanup
            if 'WAR_war' in unified_df.columns:
                unified_df['WAR'] = unified_df['WAR'].fillna(unified_df['WAR_war'])
                unified_df = unified_df.drop(columns=['WAR_war'])

            # Add position data to unified dataset
            unified_df = self._merge_position_information(unified_df, years)

            # Keep DataSource as WARP for consistency but now has both metrics
            unified_df['DataSource'] = 'UNIFIED'

            print(f"    Unified records: {len(unified_df)}")
            print(f"    Players with both WAR and WARP: {(unified_df['WAR'].notna() & unified_df['WARP'].notna()).sum()}")

            return unified_df

        elif not war_combined.empty:
            print("    Only WAR data available")
            war_with_positions = self._merge_position_information(war_combined, years)
            return war_with_positions
        elif not warp_combined.empty:
            print("    Only WARP data available")
            warp_with_positions = self._merge_position_information(warp_combined, years)
            return warp_with_positions
        else:
            print("    No data available")
            return pd.DataFrame()

    def _merge_age_information(self, combined_data: pd.DataFrame) -> pd.DataFrame:
        """
        Merge Age information from BP data to FG data for matching players.

        Args:
            combined_data: Combined dataset with both WAR and WARP data

        Returns:
            Dataset with Age filled for WAR records from WARP records
        """
        print("  Filling missing Age data from BP records...")

        # Get Age information from WARP records with both mlbid and Name for matching
        warp_age_data = combined_data[
            (combined_data['DataSource'] == 'WARP') &
            combined_data['Age'].notna()
        ][['mlbid', 'Name', 'Season', 'Age']].drop_duplicates()

        # Find WAR records missing Age
        war_missing_age = combined_data[
            (combined_data['DataSource'] == 'WAR') &
            combined_data['Age'].isna()
        ].copy()

        initial_missing = len(war_missing_age)

        # Merge Age information using primary mlbid matching, fallback to Name matching
        age_mapping = combined_data.copy()
        matched_by_id = 0
        matched_by_name = 0

        for idx, war_row in war_missing_age.iterrows():
            # Primary match: mlbid + Season
            # NOTE: Both datasets use 'mlbid' - standardized from BP's 'MLBAMID' and FG's 'MLBAMID' in _standardize_*_columns()
            matching_age = warp_age_data[
                (warp_age_data['mlbid'] == war_row['mlbid']) &
                (warp_age_data['Season'] == war_row['Season'])
            ]

            if not matching_age.empty:
                matched_by_id += 1
            else:
                # Fallback match: Name + Season (for cases where IDs don't match)
                matching_age = warp_age_data[
                    (warp_age_data['Name'] == war_row['Name']) &
                    (warp_age_data['Season'] == war_row['Season'])
                ]
                if not matching_age.empty:
                    matched_by_name += 1

            if not matching_age.empty:
                age_mapping.at[idx, 'Age'] = matching_age.iloc[0]['Age']

        # Count how many were filled
        remaining_missing = age_mapping[
            (age_mapping['DataSource'] == 'WAR') &
            age_mapping['Age'].isna()
        ].shape[0]

        filled_count = initial_missing - remaining_missing
        print(f"    Filled Age for {filled_count}/{initial_missing} WAR records from WARP data")
        print(f"      Primary matches (mlbid): {matched_by_id}")
        print(f"      Fallback matches (name): {matched_by_name}")

        if remaining_missing > 0:
            print(f"    {remaining_missing} WAR records still missing Age (no matching WARP record)")

        return age_mapping

    def _merge_position_information(self, combined_data: pd.DataFrame, years: List[int]) -> pd.DataFrame:
        """
        Merge position information from FanGraphs defensive data to both WAR and WARP records.

        Args:
            combined_data: Combined dataset with both WAR and WARP data
            years: List of years to load defensive data for

        Returns:
            Dataset with Position filled for both WAR and WARP records from defensive data
        """
        print("  Adding position data from FanGraphs defensive files...")

        position_data_frames = []

        # Load defensive data for all years
        for year in years:
            defensive_file = os.path.join(self.fg_data_path, "defensive", f"fangraphs_defensive_standard_{year}.csv")
            if os.path.exists(defensive_file):
                try:
                    df_defensive = pd.read_csv(defensive_file, encoding='utf-8-sig')
                    df_defensive['Season'] = year
                    # Only keep mlbid, season, and position data
                    df_defensive = self._standardize_fg_columns(df_defensive)
                    if 'mlbid' in df_defensive.columns and 'Position' in df_defensive.columns:
                        position_data_frames.append(df_defensive[['mlbid', 'Season', 'Position', 'Name']])
                except Exception as e:
                    warnings.warn(f"Could not load defensive file {year}: {str(e)}")

        if not position_data_frames:
            print("    No defensive data found")
            return combined_data

        # Combine all position data
        all_positions = pd.concat(position_data_frames, ignore_index=True)
        all_positions = all_positions.drop_duplicates(subset=['mlbid', 'Season', 'Position'])

        # Initialize Position column if it doesn't exist
        if 'Position' not in combined_data.columns:
            combined_data['Position'] = None

        initial_missing = len(combined_data[combined_data['Position'].isna()])

        # Merge position data for both WAR and WARP records
        combined_with_positions = combined_data.merge(
            all_positions[['mlbid', 'Season', 'Position']],
            on=['mlbid', 'Season'],
            how='left',
            suffixes=('', '_defensive')
        )

        # Fill missing positions from defensive data
        combined_with_positions['Position'] = combined_with_positions['Position'].fillna(
            combined_with_positions['Position_defensive']
        )

        # Drop the temporary column
        if 'Position_defensive' in combined_with_positions.columns:
            combined_with_positions = combined_with_positions.drop(['Position_defensive'], axis=1)

        final_missing = len(combined_with_positions[combined_with_positions['Position'].isna()])
        filled_count = initial_missing - final_missing

        print(f"    Filled Position for {filled_count}/{initial_missing} records from defensive data")

        if final_missing > 0:
            print(f"    {final_missing} records still missing Position (no matching defensive record)")

        return combined_with_positions

    def _load_and_merge_expected_stats(self, combined_data: pd.DataFrame, years: List[int]) -> pd.DataFrame:
        """
        Load expected stats from Statcast data and merge with main dataset.

        Args:
            combined_data: Main dataset
            years: Years to load expected stats for

        Returns:
            Dataset with expected stats merged
        """
        print("  Loading expected stats from Statcast data...")

        expected_stats_data = []

        for player_type in ['hitters', 'pitchers']:
            for year in years:
                expected_file = os.path.join(
                    r"C:\Users\nairs\Documents\GithubProjects\oWAR\MLB Player Data\Statcast_Data\expected_stats",
                    player_type,
                    f"expectedstats_statcast_{player_type}_{year}.csv"
                )

                if os.path.exists(expected_file):
                    try:
                        df_expected = pd.read_csv(expected_file, encoding='utf-8-sig')
                        df_expected['Season'] = year
                        df_expected['PlayerType'] = player_type.rstrip('s').title()

                        # Standardize ID column name
                        if 'player_id' in df_expected.columns:
                            df_expected = df_expected.rename(columns={'player_id': 'mlbid'})
                        elif 'id' in df_expected.columns:
                            df_expected = df_expected.rename(columns={'id': 'mlbid'})

                        # Standardize expected stats column names
                        column_mapping = {
                            'est_ba': 'xBA',
                            'est_slg': 'xSLG'
                        }
                        df_expected = df_expected.rename(columns=column_mapping)

                        # Select only xBA and xSLG as requested
                        expected_cols = ['mlbid', 'Season', 'PlayerType']
                        if 'xBA' in df_expected.columns:
                            expected_cols.append('xBA')
                        if 'xSLG' in df_expected.columns:
                            expected_cols.append('xSLG')

                        if len(expected_cols) > 3:  # More than just the base columns
                            df_expected_filtered = df_expected[expected_cols].copy()
                            expected_stats_data.append(df_expected_filtered)

                    except Exception as e:
                        warnings.warn(f"Could not load expected stats {player_type} {year}: {str(e)}")
                        continue

        if expected_stats_data:
            expected_df = pd.concat(expected_stats_data, ignore_index=True)
            print(f"    Loaded expected stats: {len(expected_df)} player-seasons")

            # Merge with main dataset
            merge_cols = ['mlbid', 'Season', 'PlayerType']
            combined_data = combined_data.merge(expected_df, on=merge_cols, how='left')

            # Report coverage
            xBA_coverage = combined_data['xBA'].notna().sum()
            xSLG_coverage = combined_data['xSLG'].notna().sum()
            total_records = len(combined_data)

            print(f"    Expected stats coverage:")
            print(f"      xBA: {xBA_coverage}/{total_records} ({xBA_coverage/total_records*100:.1f}%)")
            print(f"      xSLG: {xSLG_coverage}/{total_records} ({xSLG_coverage/total_records*100:.1f}%)")

        else:
            warnings.warn("No expected stats data found")

        return combined_data

    def prepare_projection_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare all features needed for future projections.

        Args:
            data: Raw dataset

        Returns:
            Dataset with projection features added
        """
        print("\nPreparing projection features...")

        processed_data = data.copy()

        # Note: Positional adjustments already included in WAR/WARP values, no need to add separately

        # Add regression factor for expected stats blending
        if 'regression_factor' not in processed_data.columns:
            print("  Adding default regression factor...")
            processed_data['regression_factor'] = 1.0

        print(f"Projection features prepared for {len(processed_data)} records")

        self.processed_data = processed_data
        return processed_data

    def prepare_training_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare final training dataset with all required features.

        Args:
            data: Processed dataset

        Returns:
            Training-ready dataset
        """
        print("\nPreparing training data...")

        # Base required columns for all training data
        base_required_columns = ['mlbid', 'Season', 'Age', 'DataSource']

        # Create TARGET_METRIC column from appropriate source
        data_with_target = data.copy()
        data_with_target['TARGET_METRIC'] = data_with_target.apply(
            lambda row: row.get('WAR') if row['DataSource'] == 'WAR' else row.get('WARP'),
            axis=1
        )

        # Validate and filter records with complete data
        all_required_columns = base_required_columns + ['TARGET_METRIC']
        training_data = self._validate_and_filter_training_data(data_with_target, all_required_columns)

        # Add default position if missing
        if 'Position' not in training_data.columns:
            print("  No position data found, using default positions based on player type")
            training_data['Position'] = training_data['PlayerType'].apply(
                lambda x: 'P' if x == 'Pitcher' else 'OF'  # Default to P for pitchers, OF for hitters
            )
        else:
            # Fill any missing position values
            training_data['Position'] = training_data['Position'].fillna(
                training_data['PlayerType'].apply(lambda x: 'P' if x == 'Pitcher' else 'OF')
            )

        # Filter age range
        training_data = training_data[
            (training_data['Age'] >= 18) & (training_data['Age'] <= 45)
        ].copy()

        # Remove outlier target metric values (WAR/WARP) - but preserve elite performance
        if len(training_data) > 0:
            # Use more conservative bounds to avoid filtering elite players
            # Lower bound: Remove only clearly erroneous negative values
            metric_lower = max(-3.0, training_data['TARGET_METRIC'].quantile(0.005))  # 0.5th percentile, floor at -3

            # Upper bound: Use 99.5th percentile with minimum of 12 WAR to preserve truly elite seasons
            metric_upper = max(12.0, training_data['TARGET_METRIC'].quantile(0.995))  # 99.5th percentile, floor at 12 WAR

            training_data = training_data[
                (training_data['TARGET_METRIC'] >= metric_lower) & (training_data['TARGET_METRIC'] <= metric_upper)
            ].copy()

        print(f"Training data prepared: {len(training_data)} records")
        if len(training_data) > 0:
            print(f"  Age range: {training_data['Age'].min():.1f}-{training_data['Age'].max():.1f}")
            print(f"  Target metric range: {training_data['TARGET_METRIC'].min():.1f}-{training_data['TARGET_METRIC'].max():.1f}")
            print(f"  Seasons: {training_data['Season'].min()}-{training_data['Season'].max()}")
            print(f"  Data sources: WAR={len(training_data[training_data['DataSource']=='WAR'])}, WARP={len(training_data[training_data['DataSource']=='WARP'])}")
        else:
            print("  No valid training records found")

        self.training_data = training_data
        return training_data

    def _validate_and_filter_training_data(self, data: pd.DataFrame, required_columns: List[str]) -> pd.DataFrame:
        """
        Validate training data, drop incomplete records, and log dropped players to file.

        Args:
            data: Input data with TARGET_METRIC column
            required_columns: List of columns that must have valid values

        Returns:
            Filtered dataset with complete records only
        """
        import os
        from datetime import datetime

        initial_count = len(data)

        # Identify incomplete records
        incomplete_mask = data[required_columns].isnull().any(axis=1)


        incomplete_data = data[incomplete_mask].copy()
        complete_data = data[~incomplete_mask].copy()

        # Log dropped players if any exist
        if len(incomplete_data) > 0:
            log_file = "dropped_players_log.txt"

            with open(log_file, 'w', encoding='utf-8') as f:
                f.write(f"\n{'='*60}\n")
                f.write(f"Data Validation Run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"{'='*60}\n")
                f.write(f"Total records processed: {initial_count}\n")
                f.write(f"Complete records kept: {len(complete_data)}\n")
                f.write(f"Incomplete records dropped: {len(incomplete_data)}\n\n")

                # Group dropped players by reason
                for _, row in incomplete_data.iterrows():
                    missing_cols = [col for col in required_columns if pd.isnull(row[col])]
                    player_info = f"Player: {row.get('Name', 'Unknown')} (ID: {row.get('mlbid', 'Unknown')})"
                    season_info = f"Season: {row.get('Season', 'Unknown')}, DataSource: {row.get('DataSource', 'Unknown')}"
                    missing_info = f"Missing: {', '.join(missing_cols)}"

                    f.write(f"{player_info}\n")
                    f.write(f"  {season_info}\n")
                    f.write(f"  {missing_info}\n\n")

            print(f"  Dropped {len(incomplete_data)} incomplete records (see dropped_players_log.txt)")

        return complete_data

    def train_projection_model(self, data: pd.DataFrame) -> Dict[str, Union[float, Dict]]:
        """
        Train the complete confidence-aware joint longitudinal-survival model.

        Args:
            data: Training dataset (used for both training and confidence calculation)

        Returns:
            Model performance metrics
        """
        print("\nTraining separate confidence-aware WAR and WARP projection models...")

        # Store training data reference for confidence calculations (OPTION B)
        self.training_data = data

        # Separate data by source
        war_data = data[data['DataSource'] == 'WAR'].copy()
        warp_data = data[data['DataSource'] == 'WARP'].copy()

        print(f"  WAR training data: {len(war_data)} records")
        print(f"  WARP training data: {len(warp_data)} records")

        # Train WAR model with confidence features
        if len(war_data) > 100:
            war_data['TARGET_METRIC'] = war_data['WAR']
            self.war_model = FutureProjectionAgeCurve(
                max_projection_years=self.max_projection_years,
                use_dynasty_guru=self.use_dynasty_guru
            )
            war_metrics = self.war_model.fit_joint_model(war_data, training_data=data)
        else:
            self.war_model = None
            war_metrics = {'error': 'insufficient_data'}

        # Train WARP model with confidence features
        if len(warp_data) > 100:
            warp_data['TARGET_METRIC'] = warp_data['WARP']
            self.warp_model = FutureProjectionAgeCurve(
                max_projection_years=self.max_projection_years,
                use_dynasty_guru=self.use_dynasty_guru
            )
            warp_metrics = self.warp_model.fit_joint_model(warp_data, training_data=data)
        else:
            self.warp_model = None
            warp_metrics = {'error': 'insufficient_data'}

        # Set primary model for backward compatibility
        if len(war_data) >= len(warp_data) and self.war_model:
            self.projection_model = self.war_model
        elif self.warp_model:
            self.projection_model = self.warp_model
        else:
            raise ValueError("Unable to train either model")

        combined_metrics = {
            'war_model': war_metrics,
            'warp_model': warp_metrics
        }

        self.model_performance = combined_metrics

        print("Confidence-aware separate model training complete!")
        return combined_metrics

    def validate_model(self, data: pd.DataFrame, n_splits: int = 5) -> Dict[str, Union[float, Dict]]:
        """
        Perform temporal cross-validation of the joint model.

        Args:
            data: Complete dataset for validation
            n_splits: Number of validation folds

        Returns:
            Comprehensive validation results
        """
        print(f"\nValidating both models with {n_splits}-fold temporal cross-validation...")

        validation_results = {}

        # Separate data by source for validation
        war_data = data[data['DataSource'] == 'WAR'].copy()
        warp_data = data[data['DataSource'] == 'WARP'].copy()

        # Validate WAR model if it exists
        if hasattr(self, 'war_model') and self.war_model is not None and len(war_data) > 100:
            print("Validating WAR model...")
            war_data['TARGET_METRIC'] = war_data['WAR']
            war_validation = self.validator.validate_joint_model(self.war_model, war_data, n_splits)
            validation_results['war_model_validation'] = war_validation

        # Validate WARP model if it exists
        if hasattr(self, 'warp_model') and self.warp_model is not None and len(warp_data) > 100:
            print("Validating WARP model...")
            warp_data['TARGET_METRIC'] = warp_data['WARP']
            warp_validation = self.validator.validate_joint_model(self.warp_model, warp_data, n_splits)
            validation_results['warp_model_validation'] = warp_validation

        # For backward compatibility, also include primary model validation
        if hasattr(self, 'projection_model') and self.projection_model is not None:
            validation_results['n_folds'] = n_splits

            # Set primary validation results based on which model exists
            if 'war_model_validation' in validation_results:
                validation_results.update(validation_results['war_model_validation'])
            elif 'warp_model_validation' in validation_results:
                validation_results.update(validation_results['warp_model_validation'])

        print("Model validation complete!")
        return validation_results

    def generate_player_projections(self,
                                  player_id: str,
                                  current_season: int,
                                  years_ahead: int = 3) -> Dict[str, float]:
        """
        Generate future projections for a specific player.

        Args:
            player_id: Player MLBID
            current_season: Current season for projection baseline
            years_ahead: Number of years to project

        Returns:
            Dictionary with projected WAR for each future year
        """
        if not (self.war_model or self.warp_model):
            raise ValueError("Models must be trained before generating projections")

        # Get player data from training_data to ensure data quality
        player_data = self.training_data[self.training_data['mlbid'] == player_id].copy()

        if player_data.empty:
            raise ValueError(f"No data found for player {player_id}")

        # Get current state
        current_data = player_data[player_data['Season'] == current_season]
        if current_data.empty:
            current_data = player_data.iloc[-1:]  # Use most recent season

        # Extract current state with proper NaN handling
        current_row = current_data.iloc[0]
        current_state = {
            'age': current_row.get('Age', 27) if pd.notna(current_row.get('Age')) else 27,
            'position': current_row.get('Position', 'OF') if pd.notna(current_row.get('Position')) else 'OF',
            'war': current_row.get('WAR', 0) if pd.notna(current_row.get('WAR')) else 0,
            'pa': current_row.get('PA', 600) if pd.notna(current_row.get('PA')) else 600,
            'regression_factor': current_row.get('regression_factor', 1.0) if pd.notna(current_row.get('regression_factor')) else 1.0
        }

        # Get player history (seasons before current)
        player_history = player_data[player_data['Season'] < current_season].copy()

        # Clean player history of NaN values that could cause feature extraction issues
        critical_columns = ['Age', 'WAR', 'Position']
        for col in critical_columns:
            if col in player_history.columns:
                player_history[col] = player_history[col].fillna(
                    {'Age': 27, 'WAR': 0, 'Position': 'OF'}.get(col, 0)
                )

        # Generate projections
        projections = self.projection_model.generate_future_projections(
            current_state, player_history, years_ahead
        )

        return projections

    def generate_player_projections_by_source(self,
                                             player_id: str,
                                             current_season: int,
                                             years_ahead: int = 3,
                                             source: str = 'WAR') -> Dict[str, float]:
        """
        Generate future projections for a specific player using specific data source.

        Args:
            player_id: Player MLBID
            current_season: Current season for projection baseline
            years_ahead: Number of years to project
            source: Data source to use ('WAR' or 'WARP')

        Returns:
            Dictionary with projected values for each future year
        """
        if not (self.war_model or self.warp_model):
            raise ValueError("Models must be trained before generating projections")

        # Get player data filtered by source
        player_data = self.training_data[
            (self.training_data['mlbid'] == player_id) &
            (self.training_data['DataSource'] == source)
        ].copy()

        if player_data.empty:
            return {f'year_{i+1}': np.nan for i in range(years_ahead)}

        # Get current state
        current_data = player_data[player_data['Season'] == current_season]
        if current_data.empty:
            current_data = player_data.iloc[-1:]

        current_row = current_data.iloc[0]
        player_history = player_data[player_data['Season'] <= current_season].sort_values('Season')

        # Use appropriate model
        model = self.war_model if source == 'WAR' else self.warp_model
        if not model:
            return {f'year_{i+1}': np.nan for i in range(years_ahead)}

        # Generate projections using confidence-aware source-specific model
        projected_values = model.predict_performance_path(
            age=current_row['Age'],
            position=current_row.get('Position', 'OF'),
            current_war=current_row['TARGET_METRIC'],
            player_history=player_history,
            years_ahead=years_ahead,
            training_data=getattr(self, 'training_data', None)  # Pass training data for confidence
        )

        return {f'year_{i+1}': projection for i, projection in enumerate(projected_values[:years_ahead])}

    def batch_generate_projections(self,
                                 target_season: int,
                                 years_ahead: int = 3,
                                 min_career_length: int = 2) -> pd.DataFrame:
        """
        Generate projections for all eligible players.

        Args:
            target_season: Season to project from
            years_ahead: Number of years to project
            min_career_length: Minimum career length for projections

        Returns:
            DataFrame with projections for all eligible players
        """
        if not (self.war_model or self.warp_model):
            raise ValueError("Models must be trained before generating projections")

        print(f"\nGenerating {years_ahead}-year projections from {target_season}...")

        projection_results = []

        # Use training_data instead of processed_data to ensure data quality
        # This ensures we only project for players with complete, validated data
        target_players = self.training_data[
            self.training_data['Season'] == target_season
        ]['mlbid'].unique()

        for player_id in target_players:
            try:
                player_data = self.training_data[
                    self.training_data['mlbid'] == player_id
                ].copy()

                # Check career length requirement
                if len(player_data) < min_career_length:
                    continue

                # Generate projections
                projections = self.generate_player_projections(
                    player_id, target_season, years_ahead
                )

                # Get player info
                player_info = player_data[player_data['Season'] == target_season].iloc[0]


                # Get current values from both sources if available
                war_data = player_data[(player_data['Season'] == target_season) & (player_data['DataSource'] == 'WAR')]
                warp_data = player_data[(player_data['Season'] == target_season) & (player_data['DataSource'] == 'WARP')]

                current_war = war_data['WAR'].iloc[0] if len(war_data) > 0 else np.nan
                current_warp = warp_data['WARP'].iloc[0] if len(warp_data) > 0 else np.nan

                # Generate both WAR and WARP projections
                war_projections = self.generate_player_projections_by_source(player_id, target_season, years_ahead, 'WAR')
                warp_projections = self.generate_player_projections_by_source(player_id, target_season, years_ahead, 'WARP')

                result = {
                    'mlbid': player_id,
                    'Name': player_info.get('Name', 'Unknown'),
                    'Age': player_info.get('Age', 0),
                    'Position': player_info.get('Position', 'Unknown'),
                    'Current_WAR': current_war,
                    'Current_WARP': current_warp,
                    'projection_season': target_season
                }

                # Add projected WAR values
                for year, projected_value in war_projections.items():
                    result[f'projected_WAR_{year}'] = projected_value

                # Add projected WARP values
                for year, projected_value in warp_projections.items():
                    result[f'projected_WARP_{year}'] = projected_value

                projection_results.append(result)

            except Exception as e:
                warnings.warn(f"Could not generate projections for player {player_id}: {str(e)}")
                continue

        projections_df = pd.DataFrame(projection_results)

        print(f"Projections generated for {len(projections_df)} players")

        # OPTION C: Apply elite player adjustments BEFORE injury recovery
        print("\nApplying elite player adjustments...")

        # Calculate confidence scores for all players
        confidence_scores = self._calculate_batch_confidence_scores(projections_df)

        # Apply elite adjustments
        projections_df = self.elite_adjuster.adjust_elite_projections(
            projections_df,
            confidence_scores,
            training_data=self.training_data
        )

        # Apply injury recovery modeling if enabled
        if self.use_injury_modeling and hasattr(self, 'injury_data') and self.injury_data is not None and not self.injury_data.empty:
            print("\nApplying injury recovery adjustments...")
            projections_df = self._apply_injury_recovery_adjustments(projections_df, target_season)
        elif self.use_injury_modeling:
            print("\nInjury modeling enabled but no injury data loaded - skipping injury adjustments")

        # Apply zero-sum constraint if enabled (AFTER elite adjustments and injury recovery)
        if self.use_zero_sum_constraint:
            print("\nApplying zero-sum WAR constraint optimization...")

            # Calculate original totals for comparison
            original_war_total = projections_df['projected_WAR_year_1'].sum()
            original_warp_total = projections_df['projected_WARP_year_1'].sum()

            # Apply constraint to both WAR and WARP projections
            projections_df = self.apply_zero_sum_war_constraint(projections_df)

            # Recalculate totals
            adjusted_war_total = projections_df['projected_WAR_year_1'].sum()
            adjusted_warp_total = projections_df['projected_WARP_year_1'].sum()

            print(f"WAR total adjustment: {original_war_total:.1f} -> {adjusted_war_total:.1f}")
            print(f"WARP total adjustment: {original_warp_total:.1f} -> {adjusted_warp_total:.1f}")

        return projections_df

    def _apply_injury_recovery_adjustments(self, projections_df: pd.DataFrame, target_season: int) -> pd.DataFrame:
        """
        Apply comprehensive injury recovery modeling adjustments to player projections.

        Covers Tommy John surgery plus 5 additional high-priority injury types:
        - Shoulder Surgery, Elbow Surgery (non-TJ), Hip Surgery, Back Surgery, Oblique Strain

        Args:
            projections_df: DataFrame with player projections
            target_season: Season being projected from

        Returns:
            DataFrame with comprehensive injury-adjusted projections
        """
        if not hasattr(self.projector, 'apply_tommy_john_recovery'):
            print("    Tommy John recovery model not available - skipping injury adjustments")
            return projections_df

        # Apply Tommy John recovery (existing functionality)
        adjusted_df = self._apply_tommy_john_adjustments(projections_df, target_season)

        # Apply comprehensive injury recovery (surgical injuries)
        if hasattr(self.projector, 'apply_comprehensive_injury_recovery') and self.injury_data is not None:
            print("    Applying comprehensive surgical injury recovery modeling...")

            # Filter injury data for relevant timeframe (within 3 years of target season)
            recent_injuries = self.injury_data[
                pd.to_datetime(self.injury_data['injury_date']).dt.year >= (target_season - 3)
            ].copy()

            if len(recent_injuries) > 0:
                adjusted_df = self.projector.apply_comprehensive_injury_recovery(
                    adjusted_df, recent_injuries
                )
            else:
                print("    No recent injuries found for comprehensive surgical modeling")
        else:
            print("    Comprehensive surgical injury recovery model not available or no injury data")

        # Apply general injury recovery (non-surgical injuries)
        if hasattr(self.projector, 'apply_general_injury_recovery') and self.injury_data is not None:
            print("    Applying general injury recovery modeling...")

            # Filter injury data for relevant timeframe (within 2 years for non-surgical)
            recent_general_injuries = self.injury_data[
                pd.to_datetime(self.injury_data['injury_date']).dt.year >= (target_season - 2)
            ].copy()

            if len(recent_general_injuries) > 0:
                adjusted_df = self.projector.apply_general_injury_recovery(
                    adjusted_df, recent_general_injuries
                )
            else:
                print("    No recent injuries found for general injury modeling")
        else:
            print("    General injury recovery model not available or no injury data")

        return adjusted_df

    def _apply_tommy_john_adjustments(self, projections_df: pd.DataFrame, target_season: int) -> pd.DataFrame:
        """
        Apply Tommy John surgery recovery adjustments (legacy method preserved for compatibility).

        Args:
            projections_df: DataFrame with player projections
            target_season: Season being projected from

        Returns:
            DataFrame with Tommy John adjustments applied
        """
        adjusted_df = projections_df.copy()
        adjustment_count = 0

        # Skip if no injury data available
        if self.injury_data is None:
            print("    No injury data available for Tommy John adjustments")
            return adjusted_df

        for idx, player_row in projections_df.iterrows():
            player_id = player_row['mlbid']
            player_name = player_row['Name']

            # Check if player has Tommy John surgery in injury data
            player_injuries = self.injury_data[
                self.injury_data['mlbamid'] == player_id
            ]

            tommy_john_injuries = player_injuries[
                player_injuries['injury_type'] == 'Tommy John Surgery'
            ]

            if len(tommy_john_injuries) > 0:
                # Get most recent Tommy John surgery
                recent_tj = tommy_john_injuries.sort_values('injury_date').iloc[-1]
                injury_year = pd.to_datetime(recent_tj['injury_date']).year

                # Only apply if injury is recent enough to affect projections
                years_since_injury = target_season - injury_year

                if 0 <= years_since_injury <= 3:  # Apply for 3 years post-surgery
                    try:
                        # Get player age and position for recovery calculation
                        player_age = player_row['Age']
                        player_position = player_row['Position']

                        # Apply Tommy John recovery adjustments to all projection years
                        for year_num in range(1, 4):  # Adjust years 1-3
                            war_col = f'projected_WAR_year_{year_num}'
                            warp_col = f'projected_WARP_year_{year_num}'

                            if war_col in adjusted_df.columns:
                                original_war = adjusted_df.loc[idx, war_col]
                                if not pd.isna(original_war):
                                    recovery_factor = self.projector.apply_tommy_john_recovery(
                                        player_age, player_position, years_since_injury + year_num - 1
                                    )
                                    adjusted_df.loc[idx, war_col] = original_war * recovery_factor

                            if warp_col in adjusted_df.columns:
                                original_warp = adjusted_df.loc[idx, warp_col]
                                if not pd.isna(original_warp):
                                    recovery_factor = self.projector.apply_tommy_john_recovery(
                                        player_age, player_position, years_since_injury + year_num - 1
                                    )
                                    adjusted_df.loc[idx, warp_col] = original_warp * recovery_factor

                        adjustment_count += 1

                    except Exception as e:
                        print(f"    Warning: Could not apply Tommy John adjustment for {player_name}: {e}")

        if adjustment_count > 0:
            print(f"    Applied Tommy John recovery adjustments to {adjustment_count} players")

        return adjusted_df

    def run_injury_recovery_backtest(self, training_years: List[int],
                                   test_years: List[int]) -> Dict[str, Union[float, Dict]]:
        """
        Run historical validation backtest for injury recovery modeling.

        Tests injury recovery projections against actual outcomes to validate model accuracy.

        Args:
            training_years: Years to use for training/calibration
            test_years: Years to test predictions against

        Returns:
            Dictionary with backtest results and performance metrics
        """
        print(f"\nRunning injury recovery backtest...")
        print(f"Training years: {training_years}")
        print(f"Test years: {test_years}")

        # Load injury data for backtest period
        all_years = sorted(set(training_years + test_years))
        if not hasattr(self, 'injury_data') or self.injury_data.empty:
            print("Loading injury data for backtest...")
            self.injury_data = self.load_injury_data(all_years)

        # Initialize backtest results
        backtest_results = {
            'test_summary': {
                'training_years': training_years,
                'test_years': test_years,
                'total_cases': 0,
                'successful_predictions': 0,
                'accuracy_metrics': {}
            },
            'injury_type_performance': {},
            'position_performance': {},
            'detailed_results': []
        }

        total_cases = 0
        successful_predictions = 0
        prediction_errors = []

        # Process each test year
        for test_year in test_years:
            print(f"\n  Testing predictions for {test_year}...")

            # Get injured players in training years (before test year)
            injured_players = self.injury_data[
                (pd.to_datetime(self.injury_data['injury_date']).dt.year >= test_year - 3) &
                (pd.to_datetime(self.injury_data['injury_date']).dt.year < test_year)
            ].copy()

            if len(injured_players) == 0:
                print(f"    No injured players found for {test_year} test")
                continue

            # Get actual performance data for test year
            actual_data = self.training_data[
                self.training_data['Season'] == test_year
            ].copy()

            if len(actual_data) == 0:
                print(f"    No actual performance data found for {test_year}")
                continue

            # Test each injured player's recovery
            for _, injury_case in injured_players.iterrows():
                player_id = injury_case.get('mlbamid')
                injury_type = injury_case.get('injury_type')
                injury_date = injury_case.get('injury_date')

                # Find player's actual performance in test year
                player_actual = actual_data[actual_data['mlbid'] == player_id]
                if len(player_actual) == 0:
                    continue

                # Get player's pre-injury baseline (average of 2 years before injury)
                injury_year = pd.to_datetime(injury_date).year
                baseline_data = self.training_data[
                    (self.training_data['mlbid'] == player_id) &
                    (self.training_data['Season'] >= injury_year - 2) &
                    (self.training_data['Season'] < injury_year)
                ]

                if len(baseline_data) == 0:
                    continue

                # Calculate baseline performance
                baseline_war = baseline_data['WAR'].mean()
                actual_war = player_actual['WAR'].iloc[0]

                # Get predicted recovery factor
                player_age = player_actual['Age'].iloc[0]
                player_position = player_actual['Position'].iloc[0]

                predicted_recovery = self._get_predicted_recovery_factor(
                    injury_type, player_age, player_position,
                    injury_year, test_year
                )

                if predicted_recovery is None:
                    continue

                # Calculate predicted vs actual performance
                predicted_war = baseline_war * predicted_recovery
                actual_recovery = actual_war / baseline_war if baseline_war != 0 else 1.0

                # Calculate prediction error
                prediction_error = abs(predicted_war - actual_war)
                prediction_errors.append(prediction_error)

                # Track detailed results
                case_result = {
                    'player_id': player_id,
                    'injury_type': injury_type,
                    'injury_year': injury_year,
                    'test_year': test_year,
                    'position': player_position,
                    'age': player_age,
                    'baseline_war': round(baseline_war, 2),
                    'predicted_war': round(predicted_war, 2),
                    'actual_war': round(actual_war, 2),
                    'predicted_recovery': round(predicted_recovery, 3),
                    'actual_recovery': round(actual_recovery, 3),
                    'prediction_error': round(prediction_error, 2),
                    'accurate_prediction': prediction_error <= 1.0  # Within 1.0 WAR
                }

                backtest_results['detailed_results'].append(case_result)

                # Update counters
                total_cases += 1
                if prediction_error <= 1.0:
                    successful_predictions += 1

                # Track by injury type
                if injury_type not in backtest_results['injury_type_performance']:
                    backtest_results['injury_type_performance'][injury_type] = {
                        'cases': 0, 'accurate': 0, 'avg_error': 0, 'errors': []
                    }

                injury_perf = backtest_results['injury_type_performance'][injury_type]
                injury_perf['cases'] += 1
                injury_perf['errors'].append(prediction_error)
                if prediction_error <= 1.0:
                    injury_perf['accurate'] += 1

                # Track by position
                pos_key = self._map_position_to_coefficient_key(player_position)
                if pos_key not in backtest_results['position_performance']:
                    backtest_results['position_performance'][pos_key] = {
                        'cases': 0, 'accurate': 0, 'avg_error': 0, 'errors': []
                    }

                pos_perf = backtest_results['position_performance'][pos_key]
                pos_perf['cases'] += 1
                pos_perf['errors'].append(prediction_error)
                if prediction_error <= 1.0:
                    pos_perf['accurate'] += 1

        # Calculate final metrics
        if total_cases > 0:
            overall_accuracy = successful_predictions / total_cases
            mean_error = sum(prediction_errors) / len(prediction_errors)
            median_error = sorted(prediction_errors)[len(prediction_errors) // 2]

            backtest_results['test_summary'].update({
                'total_cases': total_cases,
                'successful_predictions': successful_predictions,
                'overall_accuracy': round(overall_accuracy, 3),
                'mean_absolute_error': round(mean_error, 3),
                'median_absolute_error': round(median_error, 3)
            })

            # Calculate per-injury-type metrics
            for injury_type, perf in backtest_results['injury_type_performance'].items():
                if perf['cases'] > 0:
                    perf['accuracy'] = round(perf['accurate'] / perf['cases'], 3)
                    perf['avg_error'] = round(sum(perf['errors']) / len(perf['errors']), 3)
                    del perf['errors']  # Remove raw errors to save space

            # Calculate per-position metrics
            for position, perf in backtest_results['position_performance'].items():
                if perf['cases'] > 0:
                    perf['accuracy'] = round(perf['accurate'] / perf['cases'], 3)
                    perf['avg_error'] = round(sum(perf['errors']) / len(perf['errors']), 3)
                    del perf['errors']  # Remove raw errors to save space

            print(f"\nBacktest completed:")
            print(f"  Total cases tested: {total_cases}")
            print(f"  Overall accuracy: {overall_accuracy:.1%}")
            print(f"  Mean absolute error: {mean_error:.2f} WAR")

        else:
            print("No suitable test cases found for backtest")

        return backtest_results

    def _get_predicted_recovery_factor(self, injury_type: str, age: float, position: str,
                                     injury_year: int, prediction_year: int) -> float:
        """
        Get predicted recovery factor for a specific injury case.

        Args:
            injury_type: Type of injury
            age: Player age
            position: Player position
            injury_year: Year injury occurred
            prediction_year: Year being predicted

        Returns:
            Predicted recovery factor (0.0-1.0)
        """
        years_since_injury = prediction_year - injury_year

        if years_since_injury < 0 or years_since_injury > 3:
            return None

        # Map to recovery model method names
        recovery_methods = {
            'Tommy John Surgery': 'tommy_john',
            'Shoulder Surgery': 'shoulder_surgery',
            'Elbow Surgery': 'elbow_surgery',
            'Hip Surgery': 'hip_surgery',
            'Back Surgery': 'back_surgery',
            'Oblique Strain': 'oblique_strain',
            'Hamstring Strain': 'hamstring_strain',
            'Wrist Injury': 'wrist_injury',
            'Groin Strain': 'groin_strain',
            'Shoulder Strain': 'shoulder_strain',
            'Knee Injury': 'knee_injury',
            'Ankle Injury': 'ankle_injury',
            'Hand/Finger Injury': 'hand_finger'
        }

        if injury_type not in recovery_methods:
            return None

        # Use existing recovery coefficient logic
        coeff_key = self._map_position_to_coefficient_key(position)
        age_adjustment = age - 28

        # Get recovery factors based on injury type
        if injury_type == 'Tommy John Surgery':
            coeffs = {
                'SP': {'year1_base': 0.833, 'year1_age': -0.008, 'year2_improvement': 0.15, 'year3_improvement': 0.08},
                'RP': {'year1_base': 0.854, 'year1_age': -0.006, 'year2_improvement': 0.15, 'year3_improvement': 0.08},
                'INF': {'year1_base': 0.891, 'year1_age': -0.009, 'year2_improvement': 0.15, 'year3_improvement': 0.08},
                'C': {'year1_base': 0.785, 'year1_age': -0.012, 'year2_improvement': 0.15, 'year3_improvement': 0.08},
                'OF': {'year1_base': 0.904, 'year1_age': -0.005, 'year2_improvement': 0.15, 'year3_improvement': 0.08}
            }.get(coeff_key, {'year1_base': 0.85, 'year1_age': -0.007, 'year2_improvement': 0.15, 'year3_improvement': 0.08})

            year1_factor = coeffs['year1_base'] + (coeffs['year1_age'] * age_adjustment)
            year1_factor = max(0.5, min(1.0, year1_factor))

            if years_since_injury == 1:
                return year1_factor
            elif years_since_injury == 2:
                return min(1.0, year1_factor + coeffs['year2_improvement'])
            else:  # years_since_injury == 3
                return min(1.0, year1_factor + coeffs['year2_improvement'] + coeffs['year3_improvement'])

        # For other injury types, use simplified recovery patterns
        else:
            # Use general recovery pattern based on injury severity
            if injury_type in ['Shoulder Surgery', 'Elbow Surgery', 'Hip Surgery', 'Back Surgery']:
                # Surgical injuries
                year1_base = {'Shoulder Surgery': 0.75, 'Elbow Surgery': 0.82,
                             'Hip Surgery': 0.87, 'Back Surgery': 0.81}.get(injury_type, 0.80)
                year1_factor = year1_base + (age_adjustment * -0.005)
                year1_factor = max(0.4, min(1.0, year1_factor))

                if years_since_injury == 1:
                    return year1_factor
                elif years_since_injury == 2:
                    return min(1.0, year1_factor + 0.12)
                else:
                    return min(1.0, year1_factor + 0.20)

            else:
                # General injuries
                year1_base = 0.90  # Most general injuries have less severe impact
                year1_factor = year1_base + (age_adjustment * -0.003)
                year1_factor = max(0.6, min(1.0, year1_factor))

                if years_since_injury == 1:
                    return year1_factor
                else:
                    return min(1.0, year1_factor + 0.08)

        return 1.0  # Default to no impact if unable to calculate

    def run_comprehensive_injury_validation(self) -> Dict[str, Union[float, Dict]]:
        """
        Run comprehensive historical validation of injury recovery models.

        Performs full validation across multiple test periods and provides
        detailed performance metrics and model improvement recommendations.

        Returns:
            Dictionary with comprehensive validation results
        """
        print("\n=== COMPREHENSIVE INJURY RECOVERY VALIDATION ===")

        # Define validation periods
        validation_periods = [
            {'training': [2020, 2021], 'test': [2022]},
            {'training': [2020, 2021, 2022], 'test': [2023]},
            {'training': [2021, 2022], 'test': [2023, 2024]},
        ]

        comprehensive_results = {
            'validation_summary': {
                'total_periods': len(validation_periods),
                'overall_accuracy': 0.0,
                'overall_mean_error': 0.0,
                'model_performance_grade': 'Pending'
            },
            'period_results': [],
            'injury_type_analysis': {},
            'position_analysis': {},
            'model_recommendations': []
        }

        all_period_accuracies = []
        all_period_errors = []

        # Run validation for each period
        for i, period in enumerate(validation_periods, 1):
            print(f"\n--- Validation Period {i}/{len(validation_periods)} ---")

            try:
                # Run backtest for this period
                period_results = self.run_injury_recovery_backtest(
                    training_years=period['training'],
                    test_years=period['test']
                )

                if period_results['test_summary']['total_cases'] > 0:
                    period_accuracy = period_results['test_summary']['overall_accuracy']
                    period_error = period_results['test_summary']['mean_absolute_error']

                    all_period_accuracies.append(period_accuracy)
                    all_period_errors.append(period_error)

                    period_summary = {
                        'period': f"{period['training']} → {period['test']}",
                        'cases_tested': period_results['test_summary']['total_cases'],
                        'accuracy': period_accuracy,
                        'mean_error': period_error,
                        'grade': self._grade_model_performance(period_accuracy, period_error)
                    }

                    comprehensive_results['period_results'].append(period_summary)

                    # Aggregate injury type performance
                    for injury_type, performance in period_results['injury_type_performance'].items():
                        if injury_type not in comprehensive_results['injury_type_analysis']:
                            comprehensive_results['injury_type_analysis'][injury_type] = {
                                'total_cases': 0, 'total_accurate': 0, 'total_errors': []
                            }

                        injury_analysis = comprehensive_results['injury_type_analysis'][injury_type]
                        injury_analysis['total_cases'] += performance['cases']
                        injury_analysis['total_accurate'] += performance['accurate']
                        injury_analysis['total_errors'].append(performance['avg_error'])

                    # Aggregate position performance
                    for position, performance in period_results['position_performance'].items():
                        if position not in comprehensive_results['position_analysis']:
                            comprehensive_results['position_analysis'][position] = {
                                'total_cases': 0, 'total_accurate': 0, 'total_errors': []
                            }

                        position_analysis = comprehensive_results['position_analysis'][position]
                        position_analysis['total_cases'] += performance['cases']
                        position_analysis['total_accurate'] += performance['accurate']
                        position_analysis['total_errors'].append(performance['avg_error'])

                else:
                    print(f"    No test cases available for period {i}")

            except Exception as e:
                print(f"    Error in validation period {i}: {e}")

        # Calculate overall metrics
        if all_period_accuracies:
            overall_accuracy = sum(all_period_accuracies) / len(all_period_accuracies)
            overall_error = sum(all_period_errors) / len(all_period_errors)

            comprehensive_results['validation_summary'].update({
                'overall_accuracy': round(overall_accuracy, 3),
                'overall_mean_error': round(overall_error, 3),
                'model_performance_grade': self._grade_model_performance(overall_accuracy, overall_error)
            })

            # Finalize injury type analysis
            for injury_type, analysis in comprehensive_results['injury_type_analysis'].items():
                if analysis['total_cases'] > 0:
                    analysis['overall_accuracy'] = round(analysis['total_accurate'] / analysis['total_cases'], 3)
                    analysis['average_error'] = round(sum(analysis['total_errors']) / len(analysis['total_errors']), 3)
                    del analysis['total_errors']

            # Finalize position analysis
            for position, analysis in comprehensive_results['position_analysis'].items():
                if analysis['total_cases'] > 0:
                    analysis['overall_accuracy'] = round(analysis['total_accurate'] / analysis['total_cases'], 3)
                    analysis['average_error'] = round(sum(analysis['total_errors']) / len(analysis['total_errors']), 3)
                    del analysis['total_errors']

            # Generate model recommendations
            comprehensive_results['model_recommendations'] = self._generate_model_recommendations(
                comprehensive_results
            )

            print(f"\n=== COMPREHENSIVE VALIDATION SUMMARY ===")
            print(f"Overall Accuracy: {overall_accuracy:.1%}")
            print(f"Overall Mean Error: {overall_error:.2f} WAR")
            print(f"Model Grade: {comprehensive_results['validation_summary']['model_performance_grade']}")

        else:
            print("No validation results available - insufficient data")

        return comprehensive_results

    def _grade_model_performance(self, accuracy: float, mean_error: float) -> str:
        """
        Grade model performance based on accuracy and error metrics.

        Args:
            accuracy: Model accuracy (0.0-1.0)
            mean_error: Mean absolute error in WAR

        Returns:
            Performance grade (A+ to F)
        """
        # Grading criteria based on baseball analytics standards
        if accuracy >= 0.85 and mean_error <= 0.5:
            return 'A+'
        elif accuracy >= 0.80 and mean_error <= 0.7:
            return 'A'
        elif accuracy >= 0.75 and mean_error <= 0.9:
            return 'B+'
        elif accuracy >= 0.70 and mean_error <= 1.1:
            return 'B'
        elif accuracy >= 0.65 and mean_error <= 1.3:
            return 'C+'
        elif accuracy >= 0.60 and mean_error <= 1.5:
            return 'C'
        elif accuracy >= 0.50 and mean_error <= 2.0:
            return 'D'
        else:
            return 'F'

    def _generate_model_recommendations(self, validation_results: Dict) -> List[str]:
        """
        Generate model improvement recommendations based on validation results.

        Args:
            validation_results: Comprehensive validation results

        Returns:
            List of actionable recommendations
        """
        recommendations = []

        overall_accuracy = validation_results['validation_summary']['overall_accuracy']
        overall_error = validation_results['validation_summary']['overall_mean_error']

        # Overall performance recommendations
        if overall_accuracy < 0.70:
            recommendations.append(
                "CRITICAL: Overall accuracy below 70%. Consider recalibrating recovery coefficients "
                "or adding more sophisticated injury severity modeling."
            )
        elif overall_accuracy < 0.80:
            recommendations.append(
                "MODERATE: Overall accuracy below 80%. Fine-tune age and position-specific coefficients."
            )

        if overall_error > 1.2:
            recommendations.append(
                "HIGH ERROR: Mean error exceeds 1.2 WAR. Consider adding player-specific factors "
                "such as injury history, conditioning, or pre-injury performance trends."
            )

        # Injury-specific recommendations
        injury_analysis = validation_results.get('injury_type_analysis', {})
        for injury_type, performance in injury_analysis.items():
            if performance.get('total_cases', 0) >= 3:  # Only analyze injuries with sufficient data
                accuracy = performance.get('overall_accuracy', 0)
                error = performance.get('average_error', 0)

                if accuracy < 0.60:
                    recommendations.append(
                        f"INJURY FOCUS: {injury_type} model accuracy ({accuracy:.1%}) needs improvement. "
                        f"Consider position-specific or severity-based sub-models."
                    )
                elif error > 1.5:
                    recommendations.append(
                        f"ERROR FOCUS: {injury_type} predictions have high error ({error:.2f} WAR). "
                        f"Review recovery timeline assumptions."
                    )

        # Position-specific recommendations
        position_analysis = validation_results.get('position_analysis', {})
        for position, performance in position_analysis.items():
            if performance.get('total_cases', 0) >= 5:  # Only analyze positions with sufficient data
                accuracy = performance.get('overall_accuracy', 0)

                if accuracy < 0.65:
                    recommendations.append(
                        f"POSITION FOCUS: {position} recovery predictions ({accuracy:.1%}) underperforming. "
                        f"Consider position-specific age curves or injury susceptibility factors."
                    )

        # General recommendations based on model maturity
        total_cases = sum(p.get('total_cases', 0) for p in injury_analysis.values())
        if total_cases < 20:
            recommendations.append(
                "DATA LIMITATION: Limited validation cases. Model performance may improve with more "
                "historical data or expanded injury coverage."
            )

        if overall_accuracy >= 0.80 and overall_error <= 0.8:
            recommendations.append(
                "EXCELLENT PERFORMANCE: Model performing well. Consider implementing in production "
                "with regular monitoring and recalibration."
            )

        return recommendations if recommendations else ["No specific recommendations - model performance adequate."]

    def run_complete_pipeline(self,
                            years: Optional[List[int]] = None,
                            validation_splits: int = 5,
                            save_model: bool = True,
                            model_path: str = "system2_projection_model.pkl") -> Dict[str, Union[float, Dict]]:
        """
        Run the complete SYSTEM 2 pipeline from start to finish.

        Args:
            years: Years to include in analysis
            validation_splits: Number of cross-validation folds
            save_model: Whether to save the trained model
            model_path: Path to save the model

        Returns:
            Complete pipeline results
        """
        print("RUNNING COMPLETE SYSTEM 2 PIPELINE")
        print("=" * 50)

        results = {}

        # Step 1: Load data
        raw_data = self.load_complete_dataset(years)
        results['data_loading'] = {
            'total_records': len(raw_data),
            'unique_players': raw_data['mlbid'].nunique(),
            'season_range': f"{raw_data['Season'].min()}-{raw_data['Season'].max()}"
        }

        # Step 2: Prepare features
        processed_data = self.prepare_projection_features(raw_data)
        training_data = self.prepare_training_data(processed_data)
        results['data_preparation'] = {
            'processed_records': len(processed_data),
            'training_records': len(training_data)
        }

        # Step 3: Train model
        training_metrics = self.train_projection_model(training_data)
        results['model_training'] = training_metrics

        # Step 4: Validate model
        validation_metrics = self.validate_model(training_data, validation_splits)
        results['model_validation'] = validation_metrics

        # Step 5: Save model if requested
        if save_model:
            self.projection_model.save_model(model_path)
            results['model_saved'] = model_path

        print("\n" + "=" * 50)
        print("SYSTEM 2 PIPELINE COMPLETE")
        print("=" * 50)

        return results

    def apply_zero_sum_war_constraint(self, projections_df: pd.DataFrame,
                                    target_total: float = 1000.0,
                                    hitter_pitcher_split: tuple = (570, 430)) -> pd.DataFrame:
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

        for _, row in hitters.iterrows():
            confidence = self.war_model.calculate_player_confidence_score(
                row['mlbid'], self.training_data, row['Age'], row['Position']
            )
            hitter_confidences.append(confidence)

        for _, row in pitchers.iterrows():
            confidence = self.warp_model.calculate_player_confidence_score(
                row['mlbid'], self.training_data, row['Age'], row['Position']
            )
            pitcher_confidences.append(confidence)

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
            print("    Warning: DataFrame length mismatch in adjustment logging")
            return

        # Calculate adjustment statistics
        original_total = original_df['projected_WAR_year_1'].sum()
        adjusted_total = adjusted_df['projected_WAR_year_1'].sum()

        adjustments = adjusted_df['projected_WAR_year_1'].values - original_df['projected_WAR_year_1'].values

        print(f"  Constraint adjustment summary:")
        print(f"    Original total WAR: {original_total:.1f}")
        print(f"    Adjusted total WAR: {adjusted_total:.1f}")
        print(f"    Mean adjustment: {np.mean(adjustments):.3f}")
        print(f"    Adjustment std: {np.std(adjustments):.3f}")
        print(f"    Max positive adjustment: {np.max(adjustments):.3f}")
        print(f"    Max negative adjustment: {np.min(adjustments):.3f}")

        # Show elite player adjustments
        hitters = original_df[original_df['Position'] != 'P'].copy()
        if len(hitters) > 0:
            hitter_original = hitters['projected_WAR_year_1']
            hitter_indices = hitters.index
            hitter_adjusted = adjusted_df.loc[hitter_indices, 'projected_WAR_year_1']
            hitter_adjustments = hitter_adjusted.values - hitter_original.values

            # Find biggest positive and negative adjustments
            max_positive_idx = np.argmax(hitter_adjustments)
            max_negative_idx = np.argmin(hitter_adjustments)

            if abs(hitter_adjustments[max_positive_idx]) > 0.1:
                player_name = hitters.iloc[max_positive_idx]['Name']
                adjustment = hitter_adjustments[max_positive_idx]
                original_val = hitter_original.iloc[max_positive_idx]
                print(f"    Largest positive adjustment: {player_name} ({original_val:.1f} -> {original_val + adjustment:.1f}, +{adjustment:.1f})")

            if abs(hitter_adjustments[max_negative_idx]) > 0.1:
                player_name = hitters.iloc[max_negative_idx]['Name']
                adjustment = hitter_adjustments[max_negative_idx]
                original_val = hitter_original.iloc[max_negative_idx]
                print(f"    Largest negative adjustment: {player_name} ({original_val:.1f} -> {original_val + adjustment:.1f}, {adjustment:.1f})")

    def _calculate_batch_confidence_scores(self, projections_df: pd.DataFrame) -> Dict[int, float]:
        """
        Calculate confidence scores for all players in batch projections.

        Args:
            projections_df: DataFrame with player projections

        Returns:
            Dictionary mapping mlbid to confidence score
        """
        confidence_scores = {}

        for _, row in projections_df.iterrows():
            player_id = row.get('mlbid')
            age = row.get('Age', 30)
            position = row.get('Position', 'OF')

            # Use the enhanced confidence calculation from our models
            if self.war_model and hasattr(self.war_model, 'calculate_player_confidence_score'):
                try:
                    confidence = self.war_model.calculate_player_confidence_score(
                        player_id, self.training_data, age, position
                    )
                    confidence_scores[player_id] = confidence
                except Exception:
                    # Fallback confidence based on current performance
                    current_war = row.get('Current_WAR', 0)
                    current_warp = row.get('Current_WARP', 0)
                    best_current = max(current_war if pd.notna(current_war) else 0,
                                     current_warp if pd.notna(current_warp) else 0)

                    # Simple confidence based on performance and age
                    base_confidence = min(best_current * 0.5 + 1.0, 8.0)
                    age_factor = max(0.5, (35 - age) / 10) if age < 35 else 0.5
                    confidence_scores[player_id] = base_confidence * age_factor
            else:
                # Fallback for missing model
                confidence_scores[player_id] = 1.0

        return confidence_scores

    def get_pipeline_summary(self) -> Dict[str, Union[int, float, str]]:
        """
        Get a summary of the current pipeline state.

        Returns:
            Dictionary with pipeline summary statistics
        """
        summary = {
            'pipeline_status': 'initialized',
            'data_loaded': self.raw_data is not None,
            'features_prepared': self.processed_data is not None,
            'training_ready': self.training_data is not None,
            'model_fitted': self.projection_model.is_fitted,
            'max_projection_years': self.max_projection_years
        }

        if self.raw_data is not None:
            summary.update({
                'total_records': len(self.raw_data),
                'unique_players': self.raw_data['mlbid'].nunique(),
                'season_range': f"{self.raw_data['Season'].min()}-{self.raw_data['Season'].max()}"
            })

        if self.model_performance:
            summary['model_performance'] = self.model_performance

        return summary

    def load_injury_data(self, years: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Load and process FanGraphs injury database for recovery modeling.

        Args:
            years: Years to load injury data for (default: 2020-2024)

        Returns:
            DataFrame with columns: ['mlbid', 'Name', 'Position', 'injury_date',
                                   'injury_type', 'return_date', 'recovery_days',
                                   'data_year', 'MLBAMID']
        """
        if years is None:
            years = [2020, 2021, 2022, 2023, 2024]

        print(f"Loading injury data for years: {years}")

        injury_files = []
        injury_path = os.path.join(self.fg_data_path, "injuries")

        for year in years:
            file_path = os.path.join(injury_path, f"fangraphs_injuryreport_{year}.xlsx")
            if os.path.exists(file_path):
                try:
                    df = pd.read_excel(file_path)
                    df['data_year'] = year
                    injury_files.append(df)
                    print(f"  Loaded {len(df)} injury records from {year}")
                except Exception as e:
                    print(f"  Warning: Could not load {year} injury data: {e}")
            else:
                print(f"  Warning: Injury file not found for {year}: {file_path}")

        if not injury_files:
            print("  No injury data files found")
            return pd.DataFrame()

        # Combine all years
        combined_injuries = pd.concat(injury_files, ignore_index=True)

        # Standardize column names and data types
        processed_injuries = self._process_injury_data(combined_injuries)

        print(f"Total injury records loaded: {len(processed_injuries)}")
        print(f"Unique players with injuries: {processed_injuries['MLBAMID'].nunique()}")
        print(f"Tommy John cases: {len(processed_injuries[processed_injuries['injury_type'] == 'tommy_john'])}")

        # Store for use in projections
        self.injury_data = processed_injuries

        return processed_injuries

    def _process_injury_data(self, raw_injury_data: pd.DataFrame) -> pd.DataFrame:
        """
        Process and standardize raw injury data for modeling.

        Args:
            raw_injury_data: Raw injury data from FanGraphs Excel files

        Returns:
            Processed injury data with standardized classifications
        """
        processed = raw_injury_data.copy()

        # Standardize injury classifications
        processed['injury_type'] = processed['Injury / Surgery'].apply(self._classify_injury_type)

        # Convert dates
        processed['injury_date'] = pd.to_datetime(processed['Injury / Surgery Date'], errors='coerce')
        processed['return_date'] = pd.to_datetime(processed['Return Date'], errors='coerce')

        # Calculate recovery time
        recovery_mask = processed['injury_date'].notna() & processed['return_date'].notna()
        processed.loc[recovery_mask, 'recovery_days'] = (
            processed.loc[recovery_mask, 'return_date'] - processed.loc[recovery_mask, 'injury_date']
        ).dt.days

        # Clean up columns
        processed = processed.rename(columns={
            'Name': 'Name',
            'Pos': 'Position',
            'MLBAMID': 'MLBAMID'
        })

        # Add mlbid mapping (will implement player linking later)
        processed['mlbid'] = processed['MLBAMID']  # Direct mapping for now

        # Keep only relevant columns
        keep_columns = ['mlbid', 'Name', 'Position', 'injury_date', 'injury_type',
                       'return_date', 'recovery_days', 'data_year', 'MLBAMID']

        processed = processed[keep_columns].copy()

        return processed

    def _classify_injury_type(self, injury_description: str) -> str:
        """
        Classify injury descriptions into standardized categories.

        Args:
            injury_description: Raw injury description from FanGraphs

        Returns:
            Standardized injury type classification
        """
        if pd.isna(injury_description):
            return 'unknown'

        injury_lower = str(injury_description).lower()

        # Tommy John surgery (highest priority)
        if 'tommy john' in injury_lower:
            return 'tommy_john'

        # Other major surgeries
        elif 'shoulder surgery' in injury_lower:
            return 'shoulder_surgery'
        elif 'hip surgery' in injury_lower:
            return 'hip_surgery'
        elif 'thoracic outlet' in injury_lower:
            return 'thoracic_outlet'
        elif 'knee surgery' in injury_lower:
            return 'knee_surgery'
        elif 'wrist surgery' in injury_lower:
            return 'wrist_surgery'
        elif 'elbow surgery' in injury_lower and 'internal brace' in injury_lower:
            return 'elbow_internal_brace'
        elif 'surgery' in injury_lower:
            return 'other_surgery'

        # Non-surgical injuries by body part
        elif 'hamstring' in injury_lower:
            return 'hamstring_strain'
        elif 'oblique' in injury_lower:
            return 'oblique_strain'
        elif 'shoulder' in injury_lower and ('strain' in injury_lower or 'inflammation' in injury_lower):
            return 'shoulder_strain'
        elif 'elbow' in injury_lower and 'inflammation' in injury_lower:
            return 'elbow_inflammation'
        elif 'groin' in injury_lower:
            return 'groin_strain'
        elif 'back' in injury_lower:
            return 'back_strain'
        elif 'calf' in injury_lower:
            return 'calf_strain'
        elif 'concussion' in injury_lower:
            return 'concussion'

        # Default classification
        else:
            return 'other_injury'