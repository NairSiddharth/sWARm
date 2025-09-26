"""
Positional Adjustment Module for Future Projections
=================================================

Handles position-specific WAR adjustments and defensive data integration
for the SYSTEM 2 future projection pipeline.

Based on modules/positional_adjustments.py but streamlined for projection use.
"""

import pandas as pd
import numpy as np
import glob
import os
from typing import Dict, List, Optional, Tuple

# Standard positional WAR adjustments per 600 PA (FanGraphs scale)
POSITION_WAR_ADJUSTMENTS = {
    'C': +1.25,   # Catcher: highest positive adjustment
    'SS': +0.75,  # Shortstop: high positive adjustment
    'CF': +0.25,  # Center field: small positive adjustment
    '3B': +0.25,  # Third base: small positive adjustment
    '2B': +0.3,   # Second base: small positive adjustment
    'LF': -0.7,   # Left field: negative adjustment
    'RF': -0.75,  # Right field: negative adjustment
    '1B': -1.25,  # First base: large negative adjustment
    'DH': -1.75,  # Designated hitter: largest negative adjustment
    'P': 0.0      # Pitcher: neutral (they rarely hit)
}


class PositionalAdjustmentCalculator:
    """Calculates position-specific WAR adjustments for future projections."""

    def __init__(self, data_dir: Optional[str] = None):
        """
        Initialize the positional adjustment calculator.

        Args:
            data_dir: Base directory for MLB data (defaults to standard path)
        """
        if data_dir is None:
            data_dir = r"C:\Users\nairs\Documents\GithubProjects\oWAR\MLB Player Data"

        self.data_dir = data_dir
        self.bp_positions = None
        self.fg_positions = None
        self._positions_loaded = False

    def load_defensive_data(self) -> None:
        """Load defensive/fielding data from both BP and FanGraphs sources."""
        print("Loading positional data for projections...")

        # Load FanGraphs defensive data (comprehensive coverage)
        self.fg_positions = self._load_fangraphs_defensive_data()

        # Load BP fielding data (fallback)
        self.bp_positions = self._load_bp_fielding_data()

        self._positions_loaded = True

        print(f"Positional data loaded:")
        print(f"  FG positions: {len(self.fg_positions) if self.fg_positions is not None else 0}")
        print(f"  BP positions: {len(self.bp_positions) if self.bp_positions is not None else 0}")

    def _load_fangraphs_defensive_data(self) -> Optional[pd.DataFrame]:
        """Load FanGraphs defensive data."""
        defensive_files = glob.glob(
            os.path.join(self.data_dir, "FanGraphs_Data", "defensive", "fangraphs_defensive_standard_*.csv")
        )

        if not defensive_files:
            print("  Warning: No FanGraphs defensive files found")
            return None

        all_defensive_data = []

        for file in sorted(defensive_files):
            try:
                year = int(os.path.basename(file).split('_')[-1].replace('.csv', ''))
                df = pd.read_csv(file, encoding='utf-8-sig')

                df['Season'] = year
                defensive_cols = ['MLBAMID', 'Name', 'Season', 'Pos', 'Inn']
                df_filtered = df[defensive_cols].copy()
                df_filtered = df_filtered.rename(columns={'Pos': 'Primary_Position', 'Inn': 'Total_Innings'})
                df_filtered = df_filtered.dropna(subset=['MLBAMID', 'Primary_Position'])

                all_defensive_data.append(df_filtered)

            except Exception as e:
                print(f"    Error loading {file}: {e}")

        if all_defensive_data:
            combined_defensive = pd.concat(all_defensive_data, ignore_index=True)
            print(f"  FG defensive data: {len(combined_defensive)} records")
            return combined_defensive

        return None

    def _load_bp_fielding_data(self) -> Optional[pd.DataFrame]:
        """Load BP fielding data."""
        fielding_files = glob.glob(
            os.path.join(self.data_dir, "BP_Data", "fielding", "bp_fielding_*.csv")
        )

        if not fielding_files:
            print("  Warning: No BP fielding files found")
            return None

        all_fielding_data = []

        for file in sorted(fielding_files):
            try:
                year = int(os.path.basename(file).split('_')[-1].replace('.csv', ''))
                df = pd.read_csv(file, encoding='utf-8-sig')

                df['Season'] = year
                fielding_cols = ['mlbid', 'Name', 'Season', 'Position', 'Games', 'Innings']
                df_filtered = df[fielding_cols].copy()
                df_filtered = df_filtered.dropna(subset=['mlbid', 'Position'])

                all_fielding_data.append(df_filtered)

            except Exception as e:
                print(f"    Error loading {file}: {e}")

        if all_fielding_data:
            combined_fielding = pd.concat(all_fielding_data, ignore_index=True)
            # Determine primary positions
            primary_positions = self._determine_primary_positions(combined_fielding)
            print(f"  BP fielding data: {len(primary_positions)} player-seasons")
            return primary_positions

        return None

    def _determine_primary_positions(self, fielding_df: pd.DataFrame) -> pd.DataFrame:
        """Determine primary position for each player-year based on innings played."""
        player_year_positions = []

        for (mlbid, season), group in fielding_df.groupby(['mlbid', 'Season']):
            group_clean = group.copy()
            group_clean['Innings'] = group_clean['Innings'].fillna(0)

            if group_clean['Innings'].sum() > 0:
                primary_pos_row = group_clean.loc[group_clean['Innings'].idxmax()]
                total_innings = group_clean['Innings'].sum()
            else:
                group_clean['Games'] = group_clean['Games'].fillna(0)
                primary_pos_row = group_clean.loc[group_clean['Games'].idxmax()]
                total_innings = group_clean['Games'].sum()

            player_year_positions.append({
                'mlbid': mlbid,
                'Name': primary_pos_row['Name'],
                'Season': season,
                'Primary_Position': primary_pos_row['Position'],
                'Total_Innings': total_innings
            })

        return pd.DataFrame(player_year_positions)

    def calculate_positional_adjustment(self,
                                     primary_position: str,
                                     playing_time_pa: float,
                                     full_season_pa: float = 600) -> float:
        """
        Calculate positional WAR adjustment based on position and playing time.

        Args:
            primary_position: Player's primary position (e.g., 'SS', '1B')
            playing_time_pa: Player's plate appearances
            full_season_pa: Plate appearances for full season (default 600)

        Returns:
            Positional WAR adjustment value
        """
        if pd.isna(primary_position) or primary_position not in POSITION_WAR_ADJUSTMENTS:
            return 0.0

        base_adjustment = POSITION_WAR_ADJUSTMENTS[primary_position]
        playing_time_ratio = playing_time_pa / full_season_pa
        playing_time_ratio = min(playing_time_ratio, 1.5)  # Cap at reasonable maximum

        return base_adjustment * playing_time_ratio

    def add_positional_adjustments(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add positional adjustments to a dataset.

        Args:
            data: DataFrame with offensive stats (must have player ID, Season, PA)

        Returns:
            DataFrame with added Positional_WAR column
        """
        if not self._positions_loaded:
            self.load_defensive_data()

        print("Adding positional adjustments to dataset...")

        df_enhanced = data.copy()

        # Determine column names
        year_col = 'Season' if 'Season' in df_enhanced.columns else 'Year'
        id_col = 'mlbid' if 'mlbid' in df_enhanced.columns else 'MLBAMID'

        # Initialize position columns
        df_enhanced['Primary_Position'] = None
        df_enhanced['Total_Innings'] = None

        # Try FanGraphs data first (comprehensive coverage)
        if self.fg_positions is not None and len(self.fg_positions) > 0:
            if id_col == 'MLBAMID':
                merged = df_enhanced.merge(
                    self.fg_positions[['MLBAMID', 'Season', 'Primary_Position', 'Total_Innings']],
                    left_on=[id_col, year_col],
                    right_on=['MLBAMID', 'Season'],
                    how='left',
                    suffixes=('', '_fg')
                )
                df_enhanced['Primary_Position'] = merged['Primary_Position_fg']
                df_enhanced['Total_Innings'] = merged['Total_Innings_fg']
            elif id_col == 'mlbid':
                # MLBAMID = mlbid, just rename for merge
                fg_renamed = self.fg_positions.copy()
                fg_renamed = fg_renamed.rename(columns={'MLBAMID': 'mlbid'})

                merged = df_enhanced.merge(
                    fg_renamed[['mlbid', 'Season', 'Primary_Position', 'Total_Innings']],
                    left_on=[id_col, year_col],
                    right_on=['mlbid', 'Season'],
                    how='left',
                    suffixes=('', '_fg')
                )
                df_enhanced['Primary_Position'] = merged['Primary_Position_fg']
                df_enhanced['Total_Innings'] = merged['Total_Innings_fg']

        # Fill missing positions with BP data
        if self.bp_positions is not None and len(self.bp_positions) > 0 and id_col == 'mlbid':
            missing_positions = df_enhanced['Primary_Position'].isna()

            if missing_positions.sum() > 0:
                bp_merge = df_enhanced[missing_positions].merge(
                    self.bp_positions[['mlbid', 'Season', 'Primary_Position', 'Total_Innings']],
                    left_on=[id_col, year_col],
                    right_on=['mlbid', 'Season'],
                    how='left',
                    suffixes=('', '_bp')
                )

                df_enhanced.loc[missing_positions, 'Primary_Position'] = bp_merge['Primary_Position_bp'].values
                df_enhanced.loc[missing_positions, 'Total_Innings'] = bp_merge['Total_Innings_bp'].values

        # Calculate positional WAR adjustments
        df_enhanced['Positional_WAR'] = df_enhanced.apply(
            lambda row: self.calculate_positional_adjustment(
                row['Primary_Position'],
                row.get('PA', 600)  # Default to 600 if PA missing
            ), axis=1
        )

        # Report success rate
        total_with_positions = df_enhanced['Primary_Position'].notna().sum()
        print(f"  Position assignment: {total_with_positions}/{len(df_enhanced)} ({total_with_positions/len(df_enhanced)*100:.1f}%)")

        if total_with_positions > 0:
            adj_stats = df_enhanced['Positional_WAR'].describe()
            print(f"  Positional WAR range: {adj_stats['min']:.3f} to {adj_stats['max']:.3f} (mean: {adj_stats['mean']:.3f})")

        return df_enhanced