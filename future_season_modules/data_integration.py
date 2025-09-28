"""
Data Integration Module for Future Season Projections

This module handles data loading, merging, and preparation for SYSTEM 2: Future Performance Projections,
extracted from integration.py for better modularity and maintainability.

Original functionality preserved with no modifications.
"""

import pandas as pd
import numpy as np
import os
from typing import Dict, List, Optional, Tuple, Union
import warnings

# Import role validation for data quality
from .player_role_validator import PlayerRoleValidator, RoleThresholds


class DataIntegrator:
    """
    Handles data loading, merging, and preparation for projection pipeline including:
    - FanGraphs and Baseball Prospectus data loading
    - Data merging and standardization
    - Feature preparation for projections
    - Training data preparation
    """

    def __init__(self, system_pipeline=None, enable_role_validation=True):
        """
        Initialize data integrator.

        Args:
            system_pipeline: Main System2Pipeline instance to coordinate with
            enable_role_validation: Enable player role validation for data quality
        """
        self.system_pipeline = system_pipeline
        self.enable_role_validation = enable_role_validation

        # Initialize role validator
        if self.enable_role_validation:
            self.role_validator = PlayerRoleValidator(
                thresholds=RoleThresholds(),
                enable_two_way_detection=True
            )

    def load_complete_dataset(self,
                            years: Optional[List[int]] = None,
                            player_types: List[str] = ['hitters', 'pitchers']) -> pd.DataFrame:
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
            print(f"  Loading FanGraphs {player_type} data...")
            fg_data = self._load_fangraphs_comprehensive(player_type, years)
            if not fg_data.empty:
                fg_data['DataSource'] = 'WAR'
                all_data.append(fg_data)
                print(f"    Loaded {len(fg_data)} {player_type} player-seasons from {len(years)} years")

            # Load BP data (WARP)
            print(f"  Loading BP {player_type} data...")
            bp_data = self._load_bp_comprehensive(player_type, years)
            if not bp_data.empty:
                bp_data['DataSource'] = 'WARP'
                all_data.append(bp_data)
                print(f"    Loaded {len(bp_data)} {player_type} player-seasons from {len(years)} years")

        if not all_data:
            raise ValueError("No data was loaded from any source")

        # Merge datasets
        combined_data = self._merge_war_warp_datasets(all_data)

        # Fill missing age data from alternative sources
        print("  Filling missing Age data from BP records...")
        combined_data = self._merge_age_information(combined_data)

        # Initialize Position column if it doesn't exist
        if 'Position' not in combined_data.columns:
            combined_data['Position'] = None

        # Add position information
        print("  Adding position data from FanGraphs defensive files...")
        combined_data = self._merge_position_information(combined_data, years)

        # Load and merge expected stats
        print("  Loading expected stats from Statcast data...")
        combined_data = self._load_and_merge_expected_stats(combined_data, years)

        print(f"Complete dataset loaded: {len(combined_data)} records")
        print(f"  Years: {min(years)}-{max(years)}")
        print(f"  Players: {combined_data['mlbid'].nunique()} unique")
        print(f"  Data sources: {dict(combined_data['DataSource'].value_counts())}")

        # Apply role validation if enabled
        if self.enable_role_validation and len(combined_data) > 0:
            print("  Applying player role validation for data quality...")

            # Create season context for threshold adjustments
            season_context = self._create_season_context(years)

            # Validate complete dataset
            validation_results = self.role_validator.validate_complete_dataset(
                combined_data, season_context
            )

            if 'validated_complete_dataset' in validation_results:
                original_count = len(combined_data)
                combined_data = validation_results['validated_complete_dataset']
                filtered_count = original_count - len(combined_data)

                print(f"  Role validation complete:")
                print(f"    Original records: {original_count}")
                print(f"    Filtered out: {filtered_count} ({100*filtered_count/original_count:.1f}%)")
                print(f"    Validated records: {len(combined_data)}")

                # Show validation summary
                summary = self.role_validator.get_validation_summary()
                if summary['recommendations']:
                    print(f"    Recommendations: {'; '.join(summary['recommendations'])}")

        return combined_data

    def _create_season_context(self, years: List[int]) -> Dict:
        """
        Create season context for role validation threshold adjustments.

        Args:
            years: Years being loaded

        Returns:
            Season context dictionary
        """
        context = {}

        # COVID-19 2020 season adjustment
        if 2020 in years:
            context['is_covid_season'] = True
            context['covid_years'] = [2020]

        # Add other contextual factors as needed
        # (injury-heavy seasons, rule changes, etc.)

        return context

    def _load_fangraphs_comprehensive(self, player_type: str, years: List[int]) -> pd.DataFrame:
        """
        Load comprehensive FanGraphs data following enhanced_data_loading patterns.
        Uses the original file loading logic from integration.py.
        """
        if not self.system_pipeline:
            raise ValueError("System pipeline reference required for data loading")

        print(f"  Loading FanGraphs {player_type} data...")
        all_data = []

        for year in years:
            # Load the 3 main file types for this year - defensive data handled separately
            main_file = os.path.join(self.system_pipeline.fg_data_path, player_type, f"fangraphs_{player_type}_{year}.csv")
            advanced_file = os.path.join(self.system_pipeline.fg_data_path, player_type, f"fangraphs_{player_type}_{year}_advanced.csv")
            standard_file = os.path.join(self.system_pipeline.fg_data_path, player_type, f"fangraphs_{player_type}_{year}_standard.csv")

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
                            # Merge on common columns
                            common_cols = [col for col in df_main.columns if col in df_advanced.columns and col != 'Name']
                            if common_cols:
                                df_main = df_main.merge(df_advanced, on=common_cols, how='left', suffixes=('', '_adv'))
                        except Exception as e:
                            print(f"    Warning: Failed to load advanced data for {year}: {e}")

                    # Load standard file (has basic counting stats)
                    if os.path.exists(standard_file):
                        try:
                            df_standard = pd.read_csv(standard_file, encoding='utf-8-sig')
                            # Merge on common columns
                            common_cols = [col for col in df_main.columns if col in df_standard.columns and col != 'Name']
                            if common_cols:
                                df_main = df_main.merge(df_standard, on=common_cols, how='left', suffixes=('', '_std'))
                        except Exception as e:
                            print(f"    Warning: Failed to load standard data for {year}: {e}")

                    df_main = self._standardize_fg_columns(df_main)
                    all_data.append(df_main)

                except Exception as e:
                    print(f"    Warning: Failed to load main FanGraphs file for {year}: {e}")

        if all_data:
            return pd.concat(all_data, ignore_index=True)
        return pd.DataFrame()

    def _load_bp_comprehensive(self, player_type: str, years: List[int]) -> pd.DataFrame:
        """
        Load comprehensive Baseball Prospectus data following enhanced_data_loading patterns.
        Uses the original file loading logic from integration.py.
        """
        if not self.system_pipeline:
            raise ValueError("System pipeline reference required for data loading")

        print(f"  Loading BP {player_type} data...")
        all_data = []

        for year in years:
            # Load the 2 file types for this year
            main_file = os.path.join(self.system_pipeline.bp_data_path, player_type, f"bp_{player_type}_{year}.csv")
            standard_file = os.path.join(self.system_pipeline.bp_data_path, player_type, f"bp_{player_type}_{year}_standard.csv")

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
                            # Merge on common columns
                            common_cols = [col for col in df_main.columns if col in df_standard.columns and col != 'Name']
                            if common_cols:
                                df_main = df_main.merge(df_standard, on=common_cols, how='left', suffixes=('', '_std'))
                        except Exception as e:
                            print(f"    Warning: Failed to load BP standard data for {year}: {e}")

                    df_main = self._standardize_bp_columns(df_main)
                    all_data.append(df_main)

                except Exception as e:
                    print(f"    Warning: Failed to load main BP file for {year}: {e}")

        if all_data:
            return pd.concat(all_data, ignore_index=True)
        return pd.DataFrame()

    def _standardize_bp_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize Baseball Prospectus column names with safe ID handling."""

        # Priority mapping: preserve existing mlbid, fall back to BPID if needed
        if 'mlbid' in df.columns:
            pass  # Keep existing correct mlbid values
        elif 'BPID' in df.columns:
            df['mlbid'] = df['BPID']
        elif 'PLAYERID' in df.columns:
            df['mlbid'] = df['PLAYERID']

        # Handle name columns safely
        if 'Name' not in df.columns:
            if 'PLAYER' in df.columns:
                df['Name'] = df['PLAYER']
            elif 'FULLNAME' in df.columns:
                df['Name'] = df['FULLNAME']

        return df

    def _standardize_fg_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize FanGraphs column names with priority for MLBAMID."""

        # Priority mapping: use MLBAMID as primary ID, fall back to PlayerId if needed
        if 'MLBAMID' in df.columns:
            df['mlbid'] = df['MLBAMID']
        elif 'PlayerId' in df.columns:
            df['mlbid'] = df['PlayerId']
        elif 'playerid' in df.columns:
            df['mlbid'] = df['playerid']
        elif 'player_id' in df.columns:
            df['mlbid'] = df['player_id']

        # Standardize position columns
        if 'Pos' in df.columns:
            df['Position'] = df['Pos']
        elif 'position' in df.columns:
            df['Position'] = df['position']

        return df

    def _merge_war_warp_datasets(self, all_data: List[pd.DataFrame]) -> pd.DataFrame:
        """
        Merge WAR and WARP datasets into unified player records.

        Enhanced to handle multiple data sources while preserving data integrity
        and following enhanced_data_loading patterns for robust merging.
        """
        if not all_data:
            return pd.DataFrame()

        # Combine all data with source tracking
        # Fix duplicate column names that cause pandas concat issues
        try:
            cleaned_data = []
            for i, df in enumerate(all_data):
                df_clean = df.reset_index(drop=True).copy()

                # Remove any unnamed columns
                df_clean = df_clean.loc[:, ~df_clean.columns.str.contains('^Unnamed')]

                # Fix duplicate column names by adding suffix
                cols = pd.Series(df_clean.columns)
                for dup in cols[cols.duplicated()].unique():
                    cols[cols[cols == dup].index.values.tolist()] = [dup + '_' + str(i) if i != 0 else dup for i in range(sum(cols == dup))]
                df_clean.columns = cols

                cleaned_data.append(df_clean)

            combined_data = pd.concat(cleaned_data, ignore_index=True, sort=False)
        except Exception as e:
            print(f"Error during concatenation: {e}")
            # Ultimate fallback: build manually row by row
            print("Attempting manual DataFrame combination...")
            combined_data = pd.DataFrame()
            all_columns = set()

            # Get all unique columns
            for df in all_data:
                all_columns.update(df.columns)

            # Build combined DataFrame manually
            combined_rows = []
            for df in all_data:
                for _, row in df.iterrows():
                    combined_rows.append(row.to_dict())

            combined_data = pd.DataFrame(combined_rows)

        # Ensure required columns exist
        required_columns = ['mlbid', 'Name', 'Season', 'DataSource']
        missing_columns = [col for col in required_columns if col not in combined_data.columns]
        if missing_columns:
            print(f"    Warning: Missing required columns: {missing_columns}")

        # Clean up player IDs and names
        combined_data['mlbid'] = combined_data['mlbid'].astype(str)
        combined_data['Name'] = combined_data['Name'].str.strip()

        # Handle missing values in key numeric columns
        numeric_columns = ['WAR', 'WARP', 'Age']
        for col in numeric_columns:
            if col in combined_data.columns:
                combined_data[col] = pd.to_numeric(combined_data[col], errors='coerce')

        # Remove completely empty rows
        combined_data = combined_data.dropna(how='all')

        print(f"    Merged {len(combined_data)} total records")
        print(f"    Data sources: {dict(combined_data['DataSource'].value_counts())}")

        return combined_data

    def _merge_age_information(self, combined_data: pd.DataFrame) -> pd.DataFrame:
        """
        Merge Age information from BP data to FG data for matching players.

        Uses enhanced matching logic to fill missing age data from alternative sources.
        """
        if 'Age' not in combined_data.columns:
            print("    Warning: Age column not found in data")
            return combined_data

        # Identify records missing age data
        missing_age = combined_data['Age'].isna()
        war_missing_age = combined_data[missing_age & (combined_data['DataSource'] == 'WAR')]

        if len(war_missing_age) == 0:
            print(f"    No missing Age data to fill")
            return combined_data

        # Create lookup from WARP data for age information
        warp_data = combined_data[combined_data['DataSource'] == 'WARP']
        age_lookup = warp_data.groupby(['mlbid', 'Season'])['Age'].first().to_dict()

        # Fill missing ages using primary matching (mlbid + season)
        filled_primary = 0
        for idx, row in war_missing_age.iterrows():
            lookup_key = (row['mlbid'], row['Season'])
            if lookup_key in age_lookup and pd.notna(age_lookup[lookup_key]):
                combined_data.at[idx, 'Age'] = age_lookup[lookup_key]
                filled_primary += 1

        # Fallback: name-based matching for remaining missing ages
        still_missing = combined_data['Age'].isna()
        war_still_missing = combined_data[still_missing & (combined_data['DataSource'] == 'WAR')]

        name_age_lookup = warp_data.groupby(['Name', 'Season'])['Age'].first().to_dict()
        filled_fallback = 0

        for idx, row in war_still_missing.iterrows():
            lookup_key = (row['Name'], row['Season'])
            if lookup_key in name_age_lookup and pd.notna(name_age_lookup[lookup_key]):
                combined_data.at[idx, 'Age'] = name_age_lookup[lookup_key]
                filled_fallback += 1

        # Report results
        final_missing = combined_data['Age'].isna().sum()
        original_missing = len(war_missing_age)

        print(f"    Filled Age for {original_missing - final_missing}/{original_missing} WAR records from WARP data")
        print(f"      Primary matches (mlbid): {filled_primary}")
        print(f"      Fallback matches (name): {filled_fallback}")
        print(f"    {final_missing} WAR records still missing Age (no matching WARP record)")

        return combined_data

    def _merge_position_information(self, combined_data: pd.DataFrame, years: List[int]) -> pd.DataFrame:
        """
        Merge position information from FanGraphs defensive data to both WAR and WARP records.

        Enhanced to handle multiple data sources and years efficiently.
        """
        if not self.system_pipeline:
            return combined_data

        # Load defensive data for position information from all available sources
        defensive_data = []
        for year in years:
            # Load from all three defensive file types for maximum coverage
            defensive_file_types = [
                f"fangraphs_defensive_standard_{year}.csv",
                f"fangraphs_defensive_advanced_{year}.csv",
                f"fangraphs_defensive_statcast_{year}.csv"
            ]

            for file_type in defensive_file_types:
                def_file = os.path.join(self.system_pipeline.fg_data_path, "defensive", file_type)
                if os.path.exists(def_file):
                    try:
                        df = pd.read_csv(def_file, encoding='utf-8-sig')
                        df['Season'] = year
                        df['DefensiveSource'] = file_type.replace(f"_{year}.csv", "")  # Track source
                        df = self._standardize_fg_columns(df)
                        defensive_data.append(df)
                        print(f"    Loaded {len(df)} records from {file_type}")
                    except Exception as e:
                        print(f"    Warning: Failed to load {file_type} for {year}: {e}")

            # Fallback to old naming pattern if no files found
            if not any(os.path.exists(os.path.join(self.system_pipeline.fg_data_path, "defensive", ft)) for ft in defensive_file_types):
                def_file = os.path.join(self.system_pipeline.fg_data_path, f"FanGraphs Leaderboard ({year}) Defense.csv")
                if os.path.exists(def_file):
                    try:
                        df = pd.read_csv(def_file, encoding='utf-8')
                        df['Season'] = year
                        df['DefensiveSource'] = "legacy"
                        df = self._standardize_fg_columns(df)
                        defensive_data.append(df)
                        print(f"    Loaded {len(df)} records from legacy defensive file")
                    except Exception as e:
                        print(f"    Warning: Failed to load legacy defensive data for {year}: {e}")

        if not defensive_data:
            print("    No defensive data available for position information")
            return combined_data

        # Combine defensive data with same duplicate column handling
        try:
            # Clean defensive DataFrames
            cleaned_defensive = []
            for df in defensive_data:
                df_clean = df.reset_index(drop=True).copy()
                # Remove unnamed columns and fix duplicates
                df_clean = df_clean.loc[:, ~df_clean.columns.str.contains('^Unnamed')]

                # Fix duplicate column names
                cols = pd.Series(df_clean.columns)
                for dup in cols[cols.duplicated()].unique():
                    cols[cols[cols == dup].index.values.tolist()] = [dup + '_' + str(i) if i != 0 else dup for i in range(sum(cols == dup))]
                df_clean.columns = cols

                cleaned_defensive.append(df_clean)

            all_defensive = pd.concat(cleaned_defensive, ignore_index=True, sort=False)
        except Exception as e:
            print(f"Error concatenating defensive data: {e}")
            # Fallback for defensive data
            all_defensive = pd.DataFrame()
            for df in defensive_data:
                if all_defensive.empty:
                    all_defensive = df.reset_index(drop=True)
                else:
                    # Manual row-by-row combination
                    for _, row in df.iterrows():
                        all_defensive = pd.concat([all_defensive, pd.DataFrame([row])], ignore_index=True)

        # Create position lookup with source prioritization
        # Priority: statcast > advanced > standard
        source_priority = {
            'fangraphs_defensive_statcast': 3,
            'fangraphs_defensive_advanced': 2,
            'fangraphs_defensive_standard': 1,
            'legacy': 0
        }

        position_lookup = {}
        for idx, row in all_defensive.iterrows():
            mlbid_val = row.get('mlbid')
            position_val = row.get('Position')
            season_val = row.get('Season')
            source_val = row.get('DefensiveSource', 'unknown')

            # Handle cases where values might be Series or scalars
            if hasattr(mlbid_val, 'iloc'):
                mlbid_val = mlbid_val.iloc[0] if len(mlbid_val) > 0 else None
            if hasattr(position_val, 'iloc'):
                position_val = position_val.iloc[0] if len(position_val) > 0 else None
            if hasattr(season_val, 'iloc'):
                season_val = season_val.iloc[0] if len(season_val) > 0 else None

            if pd.notna(mlbid_val) and pd.notna(position_val) and pd.notna(season_val):
                key = (str(mlbid_val), season_val)
                current_priority = source_priority.get(source_val, 0)

                # Only update if this source has higher priority or key doesn't exist
                if key not in position_lookup or current_priority > position_lookup[key][1]:
                    position_lookup[key] = (position_val, current_priority)

        # Fill missing position data
        filled_positions = 0
        source_counts = {}

        for idx, row in combined_data.iterrows():
            if pd.isna(row.get('Position')) or row.get('Position') == '':
                lookup_key = (str(row['mlbid']), row['Season'])
                if lookup_key in position_lookup:
                    position_val, priority = position_lookup[lookup_key]
                    combined_data.at[idx, 'Position'] = position_val
                    filled_positions += 1

                    # Track source for reporting
                    source_name = [k for k, v in source_priority.items() if v == priority]
                    if source_name:
                        source_counts[source_name[0]] = source_counts.get(source_name[0], 0) + 1

        missing_positions = combined_data['Position'].isna().sum() if 'Position' in combined_data.columns else len(combined_data)

        print(f"    Filled Position for {filled_positions}/{len(combined_data)} records from defensive data")
        if source_counts:
            for source, count in source_counts.items():
                print(f"      {source}: {count} positions")
        print(f"    {missing_positions} records still missing Position (no matching defensive record)")

        return combined_data

    def _load_and_merge_expected_stats(self, combined_data: pd.DataFrame, years: List[int]) -> pd.DataFrame:
        """
        Load expected stats from Statcast data and merge with main dataset.
        Uses the original file loading logic from integration.py.
        """
        print("  Loading expected stats from Statcast data...")

        expected_stats_data = []

        for player_type in ['hitters', 'pitchers']:
            for year in years:
                expected_file = os.path.join(
                    r"C:\Users\nairs\Documents\GithubProjects\oWAR\MLB Player Data\Statcast_Data\expected_stats",
                    player_type,
                    f"statcast_expected_{player_type}_{year}.csv"
                )

                if os.path.exists(expected_file):
                    try:
                        df_expected = pd.read_csv(expected_file, encoding='utf-8-sig')
                        df_expected['Season'] = year
                        df_expected['PlayerType'] = player_type.rstrip('s').title()
                        expected_stats_data.append(df_expected)
                    except Exception as e:
                        print(f"    Warning: Failed to load expected stats for {player_type} {year}: {e}")

        if not expected_stats_data:
            print("    No expected stats data found")
            return combined_data

        # Combine all expected stats data
        all_expected = pd.concat(expected_stats_data, ignore_index=True)

        # Standardize mlbid column name
        if 'player_id' in all_expected.columns:
            all_expected.rename(columns={'player_id': 'mlbid'}, inplace=True)

        print(f"    Loaded expected stats: {len(all_expected)} player-seasons")

        # Merge with main dataset
        merge_columns = ['mlbid', 'Season']
        merged_data = combined_data.merge(
            all_expected[['mlbid', 'Season'] + [col for col in all_expected.columns if col.startswith('x')]],
            on=merge_columns,
            how='left',
            suffixes=('', '_expected')
        )

        # Report coverage
        expected_stats_cols = [col for col in all_expected.columns if col.startswith('x')]
        if expected_stats_cols:
            coverage_info = []
            for col in expected_stats_cols[:2]:  # Show first 2 for brevity
                if col in merged_data.columns:
                    coverage = merged_data[col].notna().sum()
                    total = len(merged_data)
                    coverage_info.append(f"{col}: {coverage}/{total} ({coverage/total*100:.1f}%)")

            if coverage_info:
                print(f"    Expected stats coverage:")
                for info in coverage_info:
                    print(f"      {info}")

        return merged_data

    def prepare_projection_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare all features needed for future projections.

        Enhanced feature engineering for improved projection accuracy.
        """
        print("\nPreparing projection features...")

        # Copy data to avoid modifying original
        prepared_data = data.copy()

        # Add regression factor for projections (standard ZiPS approach)
        if 'regression_factor' not in prepared_data.columns:
            print("  Adding default regression factor...")
            prepared_data['regression_factor'] = 0.7  # Standard regression toward mean

        print(f"Projection features prepared for {len(prepared_data)} records")
        return prepared_data

    def prepare_training_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare final training dataset with all required features.

        Enhanced data validation and feature preparation for model training.
        """
        print("\nPreparing training data...")

        # Start with feature-prepared data
        training_data = data.copy()

        # Validate and filter data
        required_columns = ['mlbid', 'Name', 'Season', 'Age', 'DataSource']
        training_data = self._validate_and_filter_training_data(training_data, required_columns)

        print(f"Training data prepared: {len(training_data)} records")
        print(f"  Age range: {training_data['Age'].min()}-{training_data['Age'].max()}")

        # Check target metrics availability
        war_available = 'WAR' in training_data.columns and training_data['WAR'].notna().sum()
        warp_available = 'WARP' in training_data.columns and training_data['WARP'].notna().sum()

        if war_available:
            war_range = f"{training_data['WAR'].min():.1f}-{training_data['WAR'].max():.1f}"
            print(f"  Target metric range: {war_range}")

        print(f"  Seasons: {training_data['Season'].min()}-{training_data['Season'].max()}")
        print(f"  Data sources: WAR={training_data[training_data['DataSource']=='WAR'].shape[0]}, "
              f"WARP={training_data[training_data['DataSource']=='WARP'].shape[0]}")

        # Ensure Position column exists and handle any remaining missing values
        if 'Position' not in training_data.columns:
            print("  Warning: Position column missing - this should not happen with proper data loading")
            training_data['Position'] = training_data['PlayerType'].apply(
                lambda x: 'P' if x == 'Pitcher' else 'OF'  # Emergency fallback
            )
        elif training_data['Position'].isna().sum() > 0:
            missing_count = training_data['Position'].isna().sum()
            print(f"  Filling {missing_count} missing position values with defaults")
            training_data['Position'] = training_data['Position'].fillna(
                training_data['PlayerType'].apply(lambda x: 'P' if x == 'Pitcher' else 'OF')
            )

        return training_data

    def _validate_and_filter_training_data(self, data: pd.DataFrame, required_columns: List[str]) -> pd.DataFrame:
        """
        Validate and filter training data for quality and completeness.

        Enhanced data quality checks for robust model training.
        """
        print("  Validating training data quality...")

        original_count = len(data)

        # Check for required columns
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            print(f"    Warning: Missing required columns: {missing_columns}")

        # Remove records with missing critical data
        before_filter = len(data)

        # Filter out records missing essential information
        essential_filters = []

        if 'mlbid' in data.columns:
            essential_filters.append(data['mlbid'].notna())
        if 'Name' in data.columns:
            essential_filters.append(data['Name'].notna())
        if 'Season' in data.columns:
            essential_filters.append(data['Season'].notna())
        if 'Age' in data.columns:
            essential_filters.append(data['Age'].notna())

        if essential_filters:
            combined_filter = essential_filters[0]
            for filter_condition in essential_filters[1:]:
                combined_filter &= filter_condition

            filtered_data = data[combined_filter].copy()
        else:
            filtered_data = data.copy()

        dropped_count = before_filter - len(filtered_data)
        if dropped_count > 0:
            print(f"  Dropped {dropped_count} incomplete records (see results/issues/dropped_players_log.txt)")

            # Log dropped players for debugging
            if 'Name' in data.columns:
                try:
                    dropped_players = data[~data.index.isin(filtered_data.index)]['Name'].unique()
                    with open('results/issues/dropped_players_log.txt', 'w') as f:
                        f.write("Players dropped due to incomplete data:\n")
                        for player in dropped_players:
                            f.write(f"- {player}\n")
                except Exception:
                    pass  # Non-critical logging failure

        return filtered_data


# Helper functions for backward compatibility
def _import_dependencies():
    """Import required dependencies for the data integrator."""
    try:
        from .expected_stats import ExpectedStatsCalculator
        return True
    except ImportError:
        return False