"""
Real-Time Data Loader Module - Live Data Integration
Primary CSV-based with pybaseball API enhancement for current season analysis
"""

import pandas as pd
import numpy as np
import os
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, date
import sys

# Import game progress calculator
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common_modules.game_progress_calculator import GameProgressCalculator

# Try to import pybaseball
try:
    import pybaseball as pyb
    HAS_PYBASEBALL = True
    print("pybaseball available for live data integration")
except ImportError:
    HAS_PYBASEBALL = False
    print("pybaseball not available - using CSV-only mode")


class CurrentSeasonDataLoader:
    """
    Loads current season data with CSV-first approach and pybaseball enhancement

    Data Flow:
    1. Check for current season CSV files (2025)
    2. Load and validate data completeness
    3. Calculate games played and remaining games
    4. Optionally enhance with pybaseball live data
    5. Prepare feature matrix for WARP calculation
    """

    def __init__(self, season_year: int = 2025, data_path: str = None):
        """
        Initialize data loader

        Args:
            season_year: Current season year (default 2025)
            data_path: Path to MLB Player Data directory
        """
        self.season_year = season_year

        if data_path is None:
            # Default path based on project structure
            self.data_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "MLB Player Data"
            )
        else:
            self.data_path = data_path

        self.game_calculator = GameProgressCalculator(season_year=season_year)
        self.cached_data = {}
        self.data_freshness = {}

    def load_current_season_hitters(self, use_pybaseball: bool = True) -> Optional[pd.DataFrame]:
        """
        Load current season hitter data

        Args:
            use_pybaseball: Whether to try pybaseball API first

        Returns:
            DataFrame with current season hitter statistics
        """
        print(f"Loading {self.season_year} hitter data...")

        # Try pybaseball first if requested and available
        if use_pybaseball and HAS_PYBASEBALL:
            try:
                live_data = self._load_live_hitters_pybaseball()
                if live_data is not None and not live_data.empty:
                    print(f"  Loaded {len(live_data)} hitters from pybaseball API")
                    self.data_freshness['hitters'] = datetime.now()
                    return live_data
                else:
                    print("  pybaseball returned no hitter data, falling back to CSV")
            except Exception as e:
                print(f"  pybaseball failed: {e}, falling back to CSV")

        # Fallback to CSV
        csv_data = self._load_csv_hitters()
        if csv_data is not None:
            print(f"  Loaded {len(csv_data)} hitters from CSV files")
            self.data_freshness['hitters'] = datetime.now()

        return csv_data

    def load_current_season_pitchers(self, use_pybaseball: bool = True) -> Optional[pd.DataFrame]:
        """
        Load current season pitcher data

        Args:
            use_pybaseball: Whether to try pybaseball API first

        Returns:
            DataFrame with current season pitcher statistics
        """
        print(f"Loading {self.season_year} pitcher data...")

        # Try pybaseball first if requested and available
        if use_pybaseball and HAS_PYBASEBALL:
            try:
                live_data = self._load_live_pitchers_pybaseball()
                if live_data is not None and not live_data.empty:
                    print(f"  Loaded {len(live_data)} pitchers from pybaseball API")
                    self.data_freshness['pitchers'] = datetime.now()
                    return live_data
                else:
                    print("  pybaseball returned no pitcher data, falling back to CSV")
            except Exception as e:
                print(f"  pybaseball failed: {e}, falling back to CSV")

        # Fallback to CSV
        csv_data = self._load_csv_pitchers()
        if csv_data is not None:
            print(f"  Loaded {len(csv_data)} pitchers from CSV files")
            self.data_freshness['pitchers'] = datetime.now()

        return csv_data

    def _load_live_hitters_pybaseball(self) -> Optional[pd.DataFrame]:
        """Load live hitter data using pybaseball"""
        if not HAS_PYBASEBALL:
            return None

        try:
            # Get current season batting stats
            # Note: This is the basic approach - may need refinement based on pybaseball API
            batting_data = pyb.batting_stats(
                start_season=self.season_year,
                end_season=self.season_year,
                qual=1  # Minimum 1 PA to get all active players
            )

            if batting_data.empty:
                return None

            # Standardize column names to match your existing schema
            batting_data = self._standardize_hitter_columns(batting_data)

            # Add games played calculation
            batting_data = self._add_games_played_info(batting_data, 'hitter')

            return batting_data

        except Exception as e:
            print(f"Error loading pybaseball hitter data: {e}")
            return None

    def _load_live_pitchers_pybaseball(self) -> Optional[pd.DataFrame]:
        """Load live pitcher data using pybaseball"""
        if not HAS_PYBASEBALL:
            return None

        try:
            # Get current season pitching stats
            pitching_data = pyb.pitching_stats(
                start_season=self.season_year,
                end_season=self.season_year,
                qual=1  # Minimum 1 IP to get all active players
            )

            if pitching_data.empty:
                return None

            # Standardize column names
            pitching_data = self._standardize_pitcher_columns(pitching_data)

            # Add games played calculation
            pitching_data = self._add_games_played_info(pitching_data, 'pitcher')

            return pitching_data

        except Exception as e:
            print(f"Error loading pybaseball pitcher data: {e}")
            return None

    def _load_csv_hitters(self) -> Optional[pd.DataFrame]:
        """Load hitter data from CSV files"""
        possible_paths = [
            os.path.join(self.data_path, "FanGraphs_Data", "hitters", f"fangraphs_hitters_{self.season_year}.csv"),
            os.path.join(self.data_path, "Statcast_Data", "hitters", f"statcast_hitters_{self.season_year}.csv"),
            os.path.join(self.data_path, "BP_Data", "hitters", f"bp_hitters_{self.season_year}.csv"),
            os.path.join(self.data_path, f"current_season_hitters_{self.season_year}.csv")
        ]

        for path in possible_paths:
            if os.path.exists(path):
                try:
                    data = pd.read_csv(path)
                    print(f"  Found CSV file: {os.path.basename(path)}")

                    # Standardize and add games played info
                    data = self._standardize_hitter_columns(data)
                    data = self._add_games_played_info(data, 'hitter')

                    return data
                except Exception as e:
                    print(f"  Error reading {path}: {e}")
                    continue

        print(f"  No {self.season_year} hitter CSV files found in expected locations")
        return None

    def _load_csv_pitchers(self) -> Optional[pd.DataFrame]:
        """Load pitcher data from CSV files"""
        possible_paths = [
            os.path.join(self.data_path, "FanGraphs_Data", "pitchers", f"fangraphs_pitchers_{self.season_year}.csv"),
            os.path.join(self.data_path, "Statcast_Data", "pitchers", f"statcast_pitchers_{self.season_year}.csv"),
            os.path.join(self.data_path, "BP_Data", "pitchers", f"bp_pitchers_{self.season_year}.csv"),
            os.path.join(self.data_path, f"current_season_pitchers_{self.season_year}.csv")
        ]

        for path in possible_paths:
            if os.path.exists(path):
                try:
                    data = pd.read_csv(path)
                    print(f"  Found CSV file: {os.path.basename(path)}")

                    # Standardize and add games played info
                    data = self._standardize_pitcher_columns(data)
                    data = self._add_games_played_info(data, 'pitcher')

                    return data
                except Exception as e:
                    print(f"  Error reading {path}: {e}")
                    continue

        print(f"  No {self.season_year} pitcher CSV files found in expected locations")
        return None

    def _standardize_hitter_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize hitter column names to match historical data schema"""

        # Common column mappings
        column_mappings = {
            # Basic info
            'Name': 'player_name',
            'PlayerName': 'player_name',
            'name': 'player_name',
            'Team': 'team',
            'Tm': 'team',
            'Season': 'season',
            'Year': 'season',

            # Games and PA
            'G': 'games_played',
            'Games': 'games_played',
            'PA': 'PA',
            'AB': 'AB',
            'H': 'H',
            'HR': 'HR',
            'RBI': 'RBI',
            'R': 'R',
            'SB': 'SB',
            'BB': 'BB',
            'SO': 'SO',
            'K': 'SO',

            # Advanced stats
            'AVG': 'AVG',
            'OBP': 'OBP',
            'SLG': 'SLG',
            'OPS': 'OPS',
            'wOBA': 'wOBA',
            'wRC+': 'wRC_plus',
            'WAR': 'WAR',
            'WARP': 'WARP'
        }

        # Apply mappings
        df = df.rename(columns=column_mappings)

        # Ensure required columns exist with defaults
        required_columns = {
            'player_name': 'Unknown',
            'team': 'UNK',
            'season': self.season_year,
            'games_played': 0,
            'PA': 0,
            'AB': 0,
            'H': 0,
            'HR': 0,
            'RBI': 0,
            'AVG': 0.250,
            'OBP': 0.320,
            'SLG': 0.400,
            'OPS': 0.720
        }

        for col, default_val in required_columns.items():
            if col not in df.columns:
                df[col] = default_val

        return df

    def _standardize_pitcher_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize pitcher column names to match historical data schema"""

        column_mappings = {
            # Basic info
            'Name': 'player_name',
            'PlayerName': 'player_name',
            'name': 'player_name',
            'Team': 'team',
            'Tm': 'team',
            'Season': 'season',
            'Year': 'season',

            # Games and innings
            'G': 'games_played',
            'Games': 'games_played',
            'GS': 'GS',
            'IP': 'IP',
            'W': 'W',
            'L': 'L',
            'SV': 'SV',
            'SO': 'SO',
            'K': 'SO',
            'BB': 'BB',
            'H': 'H',
            'ER': 'ER',
            'HR': 'HR',

            # Advanced stats
            'ERA': 'ERA',
            'WHIP': 'WHIP',
            'K/9': 'K_9',
            'BB/9': 'BB_9',
            'HR/9': 'HR_9',
            'FIP': 'FIP',
            'xFIP': 'xFIP',
            'WAR': 'WAR',
            'WARP': 'WARP'
        }

        # Apply mappings
        df = df.rename(columns=column_mappings)

        # Ensure required columns exist
        required_columns = {
            'player_name': 'Unknown',
            'team': 'UNK',
            'season': self.season_year,
            'games_played': 0,
            'IP': 0.0,
            'W': 0,
            'L': 0,
            'SO': 0,
            'BB': 0,
            'ERA': 4.00,
            'WHIP': 1.30,
            'FIP': 4.00
        }

        for col, default_val in required_columns.items():
            if col not in df.columns:
                df[col] = default_val

        return df

    def _add_games_played_info(self, df: pd.DataFrame, player_type: str) -> pd.DataFrame:
        """Add games played and remaining games information"""

        # Calculate games remaining and season progress
        df['games_remaining'] = df['games_played'].apply(
            lambda x: self.game_calculator.calculate_games_remaining(x)
        )

        df['season_progress'] = df['games_played'].apply(
            lambda x: self.game_calculator.calculate_season_progress(x)
        )

        df['season_phase'] = df['games_played'].apply(
            lambda x: self.game_calculator.get_season_phase(x)
        )

        df['projection_confidence'] = df['games_played'].apply(
            lambda x: self.game_calculator.calculate_projection_confidence(x)
        )

        return df

    def get_player_data(self, player_name: str, player_type: str = 'hitter') -> Optional[Dict]:
        """
        Get current season data for a specific player

        Args:
            player_name: Name of the player
            player_type: 'hitter' or 'pitcher'

        Returns:
            Dictionary with player's current season data
        """
        if player_type == 'hitter':
            data = self.load_current_season_hitters()
        else:
            data = self.load_current_season_pitchers()

        if data is None or data.empty:
            return None

        # Find player (case-insensitive partial match)
        player_matches = data[
            data['player_name'].str.contains(player_name, case=False, na=False)
        ]

        if player_matches.empty:
            print(f"Player '{player_name}' not found in {player_type} data")
            return None

        if len(player_matches) > 1:
            print(f"Multiple matches found for '{player_name}':")
            print(player_matches[['player_name', 'team']].to_string())
            # Return first match
            player_row = player_matches.iloc[0]
        else:
            player_row = player_matches.iloc[0]

        return player_row.to_dict()

    def validate_data_quality(self, data: pd.DataFrame, player_type: str) -> Dict:
        """
        Validate data quality and completeness

        Args:
            data: DataFrame to validate
            player_type: 'hitter' or 'pitcher'

        Returns:
            Dictionary with validation results
        """
        if data is None or data.empty:
            return {'valid': False, 'error': 'No data provided'}

        validation_results = {
            'valid': True,
            'total_players': len(data),
            'missing_games': len(data[data['games_played'] == 0]),
            'warnings': []
        }

        # Check for missing critical columns
        if player_type == 'hitter':
            critical_cols = ['player_name', 'games_played', 'PA', 'AVG', 'OBP', 'SLG']
        else:
            critical_cols = ['player_name', 'games_played', 'IP', 'ERA', 'WHIP']

        missing_cols = [col for col in critical_cols if col not in data.columns]
        if missing_cols:
            validation_results['warnings'].append(f"Missing columns: {missing_cols}")

        # Check for unreasonable values
        if 'games_played' in data.columns:
            if data['games_played'].max() > 162:
                validation_results['warnings'].append("Some players have >162 games played")

        if player_type == 'hitter' and 'PA' in data.columns:
            avg_pa_per_game = data[data['games_played'] > 0]['PA'] / data[data['games_played'] > 0]['games_played']
            if avg_pa_per_game.mean() > 8:
                validation_results['warnings'].append("High PA per game average")

        validation_results['data_freshness'] = self.data_freshness.get(f'{player_type}s')

        return validation_results

    def get_available_seasons(self) -> List[int]:
        """Get list of seasons with available data"""
        seasons = []

        # Check different data sources
        data_dirs = [
            os.path.join(self.data_path, "FanGraphs_Data", "hitters"),
            os.path.join(self.data_path, "BP_Data", "hitters"),
            os.path.join(self.data_path, "Statcast_Data", "hitters")
        ]

        for data_dir in data_dirs:
            if os.path.exists(data_dir):
                for filename in os.listdir(data_dir):
                    if filename.endswith('.csv'):
                        # Extract year from filename
                        for year in range(2015, 2030):
                            if str(year) in filename and year not in seasons:
                                seasons.append(year)

        return sorted(seasons)


def load_current_season_data(season_year: int = 2025,
                           use_pybaseball: bool = True,
                           data_path: str = None) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Convenience function to load current season data for both hitters and pitchers

    Args:
        season_year: Year to load data for
        use_pybaseball: Whether to try pybaseball API
        data_path: Path to data directory

    Returns:
        Tuple of (hitters_df, pitchers_df)
    """
    loader = CurrentSeasonDataLoader(season_year=season_year, data_path=data_path)

    hitters = loader.load_current_season_hitters(use_pybaseball=use_pybaseball)
    pitchers = loader.load_current_season_pitchers(use_pybaseball=use_pybaseball)

    return hitters, pitchers


def test_data_loading(season_year: int = 2025) -> None:
    """
    Test data loading functionality

    Args:
        season_year: Year to test
    """
    print(f"Testing data loading for {season_year} season...")

    loader = CurrentSeasonDataLoader(season_year=season_year)

    # Test hitters
    hitters = loader.load_current_season_hitters()
    if hitters is not None:
        print(f"Hitters loaded: {len(hitters)} players")
        validation = loader.validate_data_quality(hitters, 'hitter')
        print(f"Hitter data validation: {validation}")
    else:
        print("No hitter data available")

    # Test pitchers
    pitchers = loader.load_current_season_pitchers()
    if pitchers is not None:
        print(f"Pitchers loaded: {len(pitchers)} players")
        validation = loader.validate_data_quality(pitchers, 'pitcher')
        print(f"Pitcher data validation: {validation}")
    else:
        print("No pitcher data available")

    print(f"Available seasons: {loader.get_available_seasons()}")


if __name__ == "__main__":
    # Test the data loader
    test_data_loading()