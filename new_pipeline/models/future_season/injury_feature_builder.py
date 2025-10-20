"""
Injury Feature Engineering - Load and extract injury features from FanGraphs data.

This module handles:
- Loading FanGraphs injury reports (2020-2025)
- Temporal alignment (no future leakage)
- Feature extraction for 5 Tier 1 features
- Join with historical player data on MLBAMID

Author: Claude Code (Phase 1 Implementation)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict


class InjuryFeatureBuilder:
    """
    Load FanGraphs injury data and extract features for model training.

    Handles temporal alignment, missing data, and join with historical data.
    """

    def __init__(self, injury_data_dir: Path):
        """
        Initialize injury feature builder.

        Args:
            injury_data_dir: Path to FanGraphs injury data directory
                Example: Path("MLB Player Data/FanGraphs_Data/injuries")
        """
        self.injury_data_dir = Path(injury_data_dir)
        self.injury_data = None
        self.major_surgery_keywords = [
            'Tommy John',
            'torn ACL',
            'torn labrum',
            'elbow surgery',
            'shoulder surgery'
        ]

    def load_injury_data(self, years: range = range(2020, 2026)) -> pd.DataFrame:
        """
        Load and consolidate injury reports from multiple years.

        Args:
            years: Range of years to load (default: 2020-2025)

        Returns:
            Consolidated DataFrame with all injury records

        Columns:
            - MLBAMID (int): Player ID for joining
            - injury_date (datetime): Parsed injury date
            - return_date (datetime): Parsed return date
            - injury_type (str): Injury description
            - il_days (int): Days on IL
            - pos (str): Player position
        """
        dfs = []

        for year in years:
            filepath = self.injury_data_dir / f"fangraphs_injuryreport_{year}.xlsx"

            if not filepath.exists():
                print(f"Warning: {filepath} not found, skipping")
                continue

            df_year = pd.read_excel(filepath)

            # Parse dates
            df_year['injury_date'] = pd.to_datetime(
                df_year['Injury / Surgery Date'],
                format='%m/%d/%y',
                errors='coerce'
            )
            df_year['return_date'] = pd.to_datetime(
                df_year['Return Date'],
                format='%m/%d/%y',
                errors='coerce'
            )

            # Calculate IL days
            df_year['il_days'] = (
                df_year['return_date'] - df_year['injury_date']
            ).dt.days

            # Select relevant columns
            df_year = df_year[[
                'MLBAMID',
                'injury_date',
                'return_date',
                'Injury / Surgery',
                'il_days',
                'Pos'
            ]].copy()

            df_year.rename(columns={
                'Injury / Surgery': 'injury_type',
                'Pos': 'pos'
            }, inplace=True)

            dfs.append(df_year)

        if len(dfs) == 0:
            raise FileNotFoundError(f"No injury data files found in {self.injury_data_dir}")

        # Consolidate all years
        self.injury_data = pd.concat(dfs, ignore_index=True)

        # Drop rows with missing MLBAMID or injury_date
        self.injury_data = self.injury_data.dropna(subset=['MLBAMID', 'injury_date'])

        print(f"Loaded {len(self.injury_data)} injury records from {len(dfs)} years")
        print(f"  Covering {len(self.injury_data['MLBAMID'].unique())} unique players")

        return self.injury_data

    def extract_features_for_player_year(
        self,
        playerid: int,
        year: int
    ) -> Dict[str, any]:
        """
        Extract all Tier 1 injury features for a single player-year.

        Args:
            playerid: MLBAMID
            year: Season year (e.g., 2024)

        Returns:
            Dict with 5 features:
                - has_injury_data: Binary flag (1 if year >= 2020)
                - had_tommy_john_ever: Binary flag for TJ history
                - years_since_tommy_john: Float (years since TJ) or None
                - total_il_days_past_year: Int (IL days in past 12 months)
                - had_major_injury_past_year: Binary flag for major surgery
        """
        # Feature 1: Data availability
        has_data = 1 if year >= 2020 else 0

        # Initialize features
        features = {
            'has_injury_data': has_data,
            'had_tommy_john_ever': 0,
            'years_since_tommy_john': None,
            'total_il_days_past_year': 0,
            'had_major_injury_past_year': 0
        }

        if not has_data:
            return features  # No injury data for pre-2020

        # Get player's injuries
        player_injuries = self.injury_data[
            self.injury_data['MLBAMID'] == playerid
        ].copy()

        if len(player_injuries) == 0:
            return features  # Player has no injuries on record

        # Define time boundaries
        season_end = pd.to_datetime(f"{year}-10-31")
        year_ago = season_end - pd.DateOffset(years=1)

        # Filter to injuries before season end (avoid temporal leakage)
        prior_injuries = player_injuries[
            player_injuries['injury_date'] <= season_end
        ]
        past_year_injuries = player_injuries[
            (player_injuries['injury_date'] >= year_ago) &
            (player_injuries['injury_date'] <= season_end)
        ]

        # Feature 2 & 3: Tommy John
        tj_surgeries = prior_injuries[
            prior_injuries['injury_type'].str.contains(
                'Tommy John',
                case=False,
                na=False
            )
        ]

        if len(tj_surgeries) > 0:
            features['had_tommy_john_ever'] = 1
            most_recent_tj = tj_surgeries['injury_date'].max()
            years_elapsed = (season_end - most_recent_tj).days / 365.25
            features['years_since_tommy_john'] = round(years_elapsed, 2)

        # Feature 4: Total IL days past year
        total_il = past_year_injuries['il_days'].sum()
        features['total_il_days_past_year'] = int(total_il) if pd.notna(total_il) else 0

        # Feature 5: Major injury past year
        for keyword in self.major_surgery_keywords:
            if past_year_injuries['injury_type'].str.contains(
                keyword,
                case=False,
                na=False
            ).any():
                features['had_major_injury_past_year'] = 1
                break

        return features

    def add_injury_features_to_historical_data(
        self,
        historical_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Add injury features to historical player data.

        Args:
            historical_df: Historical data with columns:
                - MLBAMID (or playerid mapped to MLBAMID)
                - Year
                - (other features)

        Returns:
            historical_df with 5 new injury feature columns added
        """
        if self.injury_data is None:
            raise RuntimeError("Injury data not loaded. Call load_injury_data() first.")

        # Ensure MLBAMID column exists
        if 'MLBAMID' not in historical_df.columns and 'playerid' in historical_df.columns:
            # Assuming playerid IS MLBAMID in historical data
            historical_df['MLBAMID'] = historical_df['playerid']

        print(f"Adding injury features to {len(historical_df)} player-seasons...")

        # Initialize feature columns
        injury_features = []

        for idx, row in historical_df.iterrows():
            playerid = row['MLBAMID']
            year = row['Year']

            features = self.extract_features_for_player_year(playerid, year)
            injury_features.append(features)

        # Add features to DataFrame
        injury_df = pd.DataFrame(injury_features)
        historical_df = pd.concat([historical_df, injury_df], axis=1)

        print(f"  Injury features added:")
        print(f"    Players with Tommy John history: {historical_df['had_tommy_john_ever'].sum()}")
        print(f"    Player-years with major injury: {historical_df['had_major_injury_past_year'].sum()}")
        total_il_mean = historical_df[historical_df['total_il_days_past_year'] > 0]['total_il_days_past_year'].mean()
        if pd.notna(total_il_mean):
            print(f"    Average IL days (when >0): {total_il_mean:.1f}")

        return historical_df
