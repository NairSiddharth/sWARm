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
import glob
from pathlib import Path

from .expected_stats import ExpectedStatsCalculator
from .future_projections import FutureProjectionAgeCurve
from .validation import AgeCurveValidator
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
                 max_projection_years: int = 3):
        """
        Initialize the SYSTEM 2 pipeline.

        Args:
            bp_data_path: Path to Baseball Prospectus data
            fg_data_path: Path to FanGraphs data
            max_projection_years: Maximum years to project (default 3)
        """
        self.bp_data_path = bp_data_path
        self.fg_data_path = fg_data_path
        self.max_projection_years = max_projection_years

        # Initialize components
        self.expected_calculator = ExpectedStatsCalculator()
        self.war_model = None
        self.warp_model = None
        self.projection_model = None  # Primary model for backward compatibility
        self.validator = AgeCurveValidator()
        # self.position_calculator = PositionalAdjustmentCalculator()  # Not needed - adjustments already in WAR/WARP

        # Data storage
        self.raw_data = None
        self.processed_data = None
        self.training_data = None
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
                        if 'id' in df_expected.columns:
                            df_expected = df_expected.rename(columns={'id': 'mlbid'})

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

        # Remove outlier target metric values (WAR/WARP)
        if len(training_data) > 0:
            metric_lower = training_data['TARGET_METRIC'].quantile(0.01)
            metric_upper = training_data['TARGET_METRIC'].quantile(0.99)
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
        Train the complete joint longitudinal-survival model.

        Args:
            data: Training dataset

        Returns:
            Model performance metrics
        """
        print("\nTraining separate WAR and WARP projection models...")

        # Separate data by source
        war_data = data[data['DataSource'] == 'WAR'].copy()
        warp_data = data[data['DataSource'] == 'WARP'].copy()

        print(f"  WAR training data: {len(war_data)} records")
        print(f"  WARP training data: {len(warp_data)} records")

        # Train WAR model
        if len(war_data) > 100:
            war_data['TARGET_METRIC'] = war_data['WAR']
            self.war_model = FutureProjectionAgeCurve(max_projection_years=self.max_projection_years)
            war_metrics = self.war_model.fit_joint_model(war_data)
        else:
            self.war_model = None
            war_metrics = {'error': 'insufficient_data'}

        # Train WARP model
        if len(warp_data) > 100:
            warp_data['TARGET_METRIC'] = warp_data['WARP']
            self.warp_model = FutureProjectionAgeCurve(max_projection_years=self.max_projection_years)
            warp_metrics = self.warp_model.fit_joint_model(warp_data)
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

        print("Separate model training complete!")
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

        # Generate projections using source-specific model
        projected_values = model.predict_performance_path(
            age=current_row['Age'],
            position=current_row.get('Position', 'OF'),
            current_war=current_row['TARGET_METRIC'],
            player_history=player_history,
            years_ahead=years_ahead
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

        return projections_df

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