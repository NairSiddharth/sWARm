"""
Age Data Loading and Integration Module
======================================

Handles extraction of age data from Baseball Prospectus files and integration
with existing player data via MLBID matching.

Classes:
    AgeDataLoader: Main class for age data extraction and integration
"""

import pandas as pd
import numpy as np
import glob
import os
from typing import Dict, List, Optional, Tuple
import warnings


class AgeDataLoader:
    """
    Extracts age data from Baseball Prospectus files and integrates with player data.

    Supports multiple years (2016-2024) and handles missing age data through
    estimation and position-based median fallbacks.
    """

    def __init__(self, bp_data_path: str = "MLB Player Data/BaseballProspectus_Data"):
        """
        Initialize the age data loader.

        Args:
            bp_data_path: Path to Baseball Prospectus data files
        """
        self.bp_data_path = bp_data_path
        self.age_cache = {}  # Cache loaded age data
        self.mlbid_cache = {}  # Cache MLBID mappings

    def load_ages_from_bp(self, years: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Extract age data from Baseball Prospectus files.

        Args:
            years: List of years to load. If None, loads all available years.

        Returns:
            DataFrame with columns: ['mlbid', 'Name', 'Age', 'Season', 'Primary_Position']
        """
        if years is None:
            years = list(range(2016, 2025))  # Default to 2016-2024

        all_age_data = []

        for year in years:
            if year in self.age_cache:
                all_age_data.append(self.age_cache[year])
                continue

            year_data = self._load_single_year_bp_data(year)
            if year_data is not None:
                self.age_cache[year] = year_data
                all_age_data.append(year_data)
            else:
                warnings.warn(f"No age data found for year {year}")

        if not all_age_data:
            raise ValueError("No age data could be loaded from any specified years")

        combined_data = pd.concat(all_age_data, ignore_index=True)
        return self._validate_age_data(combined_data)

    def _load_single_year_bp_data(self, year: int) -> Optional[pd.DataFrame]:
        """
        Load age data for a single year from BP files.

        Args:
            year: Year to load data for

        Returns:
            DataFrame with age data for the specified year, or None if not found
        """
        # Try multiple possible file patterns for BP data
        possible_patterns = [
            f"{self.bp_data_path}/bp_hitters_{year}.csv",
            f"{self.bp_data_path}/bp_pitchers_{year}.csv",
            f"{self.bp_data_path}/hitters_{year}.csv",
            f"{self.bp_data_path}/pitchers_{year}.csv",
            f"{self.bp_data_path}/{year}_hitters.csv",
            f"{self.bp_data_path}/{year}_pitchers.csv"
        ]

        year_data_frames = []

        for pattern in possible_patterns:
            try:
                if os.path.exists(pattern):
                    df = pd.read_csv(pattern)

                    # Standardize column names
                    df = self._standardize_bp_columns(df)

                    # Add season information
                    df['Season'] = year

                    # Validate required columns exist
                    required_cols = ['mlbid', 'Name', 'Age']
                    if all(col in df.columns for col in required_cols):
                        # Filter to valid age data
                        df = df[df['Age'].notna() & (df['Age'] > 0)].copy()
                        year_data_frames.append(df)
                    else:
                        warnings.warn(f"Missing required columns in {pattern}")

            except Exception as e:
                warnings.warn(f"Could not load {pattern}: {str(e)}")
                continue

        if year_data_frames:
            # Combine hitters and pitchers for the year
            combined = pd.concat(year_data_frames, ignore_index=True)
            # Remove duplicates (in case player appears in both files)
            combined = combined.drop_duplicates(subset=['mlbid', 'Season'], keep='first')
            return combined

        return None

    def _standardize_bp_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize column names across different BP file formats.

        Args:
            df: Raw dataframe from BP file

        Returns:
            DataFrame with standardized column names
        """
        # Common column name mappings
        column_mapping = {
            'MLBAMID': 'mlbid',
            'mlbamid': 'mlbid',
            'MLBID': 'mlbid',
            'player_name': 'Name',
            'Player': 'Name',
            'player': 'Name',
            'age': 'Age',
            'AGE': 'Age',
            'pos': 'Primary_Position',
            'position': 'Primary_Position',
            'Position': 'Primary_Position',
            'POS': 'Primary_Position'
        }

        # Apply mappings
        df_renamed = df.rename(columns=column_mapping)

        # Ensure critical columns exist
        if 'Name' not in df_renamed.columns and 'name' in df_renamed.columns:
            df_renamed['Name'] = df_renamed['name']

        return df_renamed

    def _validate_age_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate age data for reasonable ranges and consistency.

        Args:
            df: Combined age dataframe

        Returns:
            Validated dataframe with warnings for issues
        """
        original_count = len(df)

        # Filter to reasonable age ranges (18-45)
        df = df[(df['Age'] >= 18) & (df['Age'] <= 45)].copy()

        if len(df) < original_count:
            warnings.warn(f"Filtered {original_count - len(df)} records with unreasonable ages")

        # Check for age consistency across seasons for same player
        if 'Season' in df.columns:
            age_consistency = df.groupby('mlbid').apply(self._check_age_progression)
            inconsistent_players = age_consistency[~age_consistency].index.tolist()

            if inconsistent_players:
                warnings.warn(f"Found {len(inconsistent_players)} players with inconsistent age progression")

        return df

    def _check_age_progression(self, player_data: pd.DataFrame) -> bool:
        """
        Check if a player's age progression across seasons is reasonable.

        Args:
            player_data: Data for a single player across multiple seasons

        Returns:
            True if age progression is reasonable, False otherwise
        """
        if len(player_data) <= 1:
            return True

        player_data = player_data.sort_values('Season')
        age_diffs = player_data['Age'].diff().dropna()

        # Age should increase by 0.8-1.2 years per season (allowing for birthday timing)
        reasonable_progression = age_diffs.between(0.8, 1.2).all()

        return reasonable_progression

    def merge_ages_with_pipeline(self, main_df: pd.DataFrame, age_data: pd.DataFrame) -> pd.DataFrame:
        """
        Join age data with main pipeline data using MLBID.

        Args:
            main_df: Main dataframe from existing pipeline
            age_data: Age data loaded from BP files

        Returns:
            Merged dataframe with age information added
        """
        # Ensure both dataframes have mlbid column
        if 'mlbid' not in main_df.columns:
            raise ValueError("Main dataframe must have 'mlbid' column for merging")

        # Prepare age data for merging
        merge_columns = ['mlbid', 'Age', 'Season']
        if 'Primary_Position' in age_data.columns:
            merge_columns.append('Primary_Position')

        age_merge_data = age_data[merge_columns].copy()

        # Perform left join to preserve all main data
        merged_df = main_df.merge(
            age_merge_data,
            on=['mlbid', 'Season'],
            how='left',
            suffixes=('', '_bp')
        )

        # Handle position conflicts (prefer main_df position if available)
        if 'Primary_Position_bp' in merged_df.columns and 'Primary_Position' in merged_df.columns:
            merged_df['Primary_Position'] = merged_df['Primary_Position'].fillna(merged_df['Primary_Position_bp'])
            merged_df = merged_df.drop('Primary_Position_bp', axis=1)
        elif 'Primary_Position_bp' in merged_df.columns:
            merged_df = merged_df.rename(columns={'Primary_Position_bp': 'Primary_Position'})

        # Report merge statistics
        age_coverage = merged_df['Age'].notna().sum()
        total_records = len(merged_df)
        coverage_pct = (age_coverage / total_records) * 100 if total_records > 0 else 0

        print(f"Age data merge complete:")
        print(f"  Total records: {total_records}")
        print(f"  Age coverage: {age_coverage}/{total_records} ({coverage_pct:.1f}%)")

        return merged_df

    def handle_missing_ages(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing age data through estimation and fallbacks.

        Args:
            df: DataFrame with potential missing age data

        Returns:
            DataFrame with missing ages filled using estimation methods
        """
        original_missing = df['Age'].isna().sum()

        if original_missing == 0:
            return df

        df = df.copy()

        # Method 1: Estimate from debut year (if available)
        if 'debut_year' in df.columns or 'first_season' in df.columns:
            debut_col = 'debut_year' if 'debut_year' in df.columns else 'first_season'
            df = self._estimate_age_from_debut(df, debut_col)

        # Method 2: Use position-based median ages
        df = self._fill_age_by_position_median(df)

        # Method 3: Use overall median as final fallback
        overall_median = df['Age'].median()
        df['Age'] = df['Age'].fillna(overall_median)

        final_missing = df['Age'].isna().sum()
        filled_count = original_missing - final_missing

        if filled_count > 0:
            print(f"Age estimation complete:")
            print(f"  Originally missing: {original_missing}")
            print(f"  Successfully filled: {filled_count}")
            print(f"  Still missing: {final_missing}")

        return df

    def _estimate_age_from_debut(self, df: pd.DataFrame, debut_col: str) -> pd.DataFrame:
        """
        Estimate age based on debut year, assuming typical debut age by position.
        """
        # Typical debut ages by position (based on research)
        debut_ages = {
            'C': 24.5, 'SS': 22.5, '2B': 23.0, '3B': 23.5, '1B': 24.0,
            'LF': 23.5, 'CF': 22.5, 'RF': 23.5, 'DH': 25.0, 'P': 24.0
        }

        default_debut_age = 23.5  # Overall average

        mask = df['Age'].isna() & df[debut_col].notna() & df['Season'].notna()

        for idx in df[mask].index:
            position = df.loc[idx, 'Primary_Position']
            debut_year = df.loc[idx, debut_col]
            current_season = df.loc[idx, 'Season']

            expected_debut_age = debut_ages.get(position, default_debut_age)
            estimated_age = expected_debut_age + (current_season - debut_year)

            # Only use if reasonable (18-45 range)
            if 18 <= estimated_age <= 45:
                df.loc[idx, 'Age'] = estimated_age

        return df

    def _fill_age_by_position_median(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fill missing ages using position-specific median ages.
        """
        if 'Primary_Position' not in df.columns:
            return df

        position_medians = df.groupby('Primary_Position')['Age'].median()

        for position in position_medians.index:
            mask = (df['Age'].isna()) & (df['Primary_Position'] == position)
            df.loc[mask, 'Age'] = position_medians[position]

        return df

    def get_age_summary_stats(self, df: pd.DataFrame) -> Dict:
        """
        Generate summary statistics for age data quality and coverage.

        Args:
            df: DataFrame with age data

        Returns:
            Dictionary with summary statistics
        """
        if 'Age' not in df.columns:
            return {'error': 'No Age column found'}

        total_records = len(df)
        age_available = df['Age'].notna().sum()
        coverage_pct = (age_available / total_records) * 100 if total_records > 0 else 0

        stats = {
            'total_records': total_records,
            'age_coverage': age_available,
            'coverage_percentage': coverage_pct,
            'age_range': {
                'min': df['Age'].min(),
                'max': df['Age'].max(),
                'median': df['Age'].median(),
                'mean': df['Age'].mean()
            }
        }

        # Position breakdown if available
        if 'Primary_Position' in df.columns:
            position_stats = df.groupby('Primary_Position')['Age'].agg([
                'count', 'mean', 'median', 'min', 'max'
            ]).round(1)
            stats['by_position'] = position_stats.to_dict('index')

        # Season breakdown if available
        if 'Season' in df.columns:
            season_coverage = df.groupby('Season')['Age'].agg([
                'count', lambda x: x.notna().sum()
            ]).rename(columns={'<lambda_0>': 'age_available'})
            season_coverage['coverage_pct'] = (
                season_coverage['age_available'] / season_coverage['count'] * 100
            ).round(1)
            stats['by_season'] = season_coverage.to_dict('index')

        return stats