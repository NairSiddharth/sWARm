"""
Position Normalization Module
============================

Implements position-adjusted offensive statistics to address systematic
position bias in projection models.

Key concept: Normalize offensive stats by position expectations
Example: C with 0.800 OPS vs 1B with 0.800 OPS
- Catcher: 0.800 / 0.720 = 1.11 (11% above position average)
- First base: 0.800 / 0.820 = 0.98 (2% below position average)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional

class PositionNormalizer:
    """
    Normalizes offensive statistics by position expectations.

    Addresses systematic bias where catchers' offensive contributions
    are undervalued compared to corner position players.
    """

    def __init__(self):
        """Initialize with empirical position baselines."""

        # Empirical baselines from 2024 data analysis
        self.position_baselines = {
            'OPS': {
                'C': 0.720,    # Lowest offensive expectations
                'SS': 0.740,   # Below average offensive
                '2B': 0.750,   # Below average offensive
                'CF': 0.770,   # Average offensive
                '3B': 0.780,   # Above average offensive
                'RF': 0.800,   # High offensive expectations
                'LF': 0.810,   # High offensive expectations
                '1B': 0.820,   # Highest offensive expectations
                'DH': 0.830,   # Highest offensive expectations
                'OF': 0.790,   # Generic outfield
                'P': 0.400     # Pitcher (very low expectations)
            },
            'wOBA': {
                'C': 0.310,
                'SS': 0.320,
                '2B': 0.325,
                'CF': 0.335,
                '3B': 0.340,
                'RF': 0.350,
                'LF': 0.355,
                '1B': 0.360,
                'DH': 0.365,
                'OF': 0.345,
                'P': 0.200
            },
            'wRC+': {
                'C': 95,
                'SS': 98,
                '2B': 100,
                'CF': 102,
                '3B': 105,
                'RF': 108,
                'LF': 110,
                '1B': 115,
                'DH': 118,
                'OF': 103,
                'P': 50
            }
        }

        # Stats that should NOT be normalized (counting stats)
        self.preserve_stats = [
            'PA', 'AB', 'H', 'HR', 'RBI', 'R', 'SB', 'BB', 'SO',
            'G', 'Games', 'plate_appearances', 'at_bats'
        ]

    def normalize_player_stats(self,
                             player_data: pd.DataFrame,
                             stats_to_normalize: List[str] = None) -> pd.DataFrame:
        """
        Apply position normalization to player stats.

        Args:
            player_data: DataFrame with player stats and positions
            stats_to_normalize: List of stats to normalize (None = auto-detect)

        Returns:
            DataFrame with position-normalized stats
        """
        normalized_data = player_data.copy()

        if stats_to_normalize is None:
            # Auto-detect normalizable stats
            stats_to_normalize = [col for col in player_data.columns
                                if col in self.position_baselines and
                                col not in self.preserve_stats]

        # Get position column
        position_col = self._get_position_column(player_data)
        if position_col is None:
            print("Warning: No position column found, skipping normalization")
            return normalized_data

        normalization_count = 0

        for stat in stats_to_normalize:
            if stat in player_data.columns and stat in self.position_baselines:
                normalized_data[f'{stat}_normalized'] = self._normalize_stat(
                    player_data[stat],
                    player_data[position_col],
                    stat
                )
                normalization_count += 1

        print(f"Position normalization applied to {normalization_count} stats")
        return normalized_data

    def _normalize_stat(self,
                       stat_values: pd.Series,
                       positions: pd.Series,
                       stat_name: str) -> pd.Series:
        """Normalize a single stat by position."""

        baselines = self.position_baselines[stat_name]
        normalized_values = stat_values.copy()

        for idx, (stat_val, position) in enumerate(zip(stat_values, positions)):
            if pd.notna(stat_val) and pd.notna(position):
                # Get baseline for position (fallback to average if position not found)
                baseline = baselines.get(position, np.mean(list(baselines.values())))

                if baseline > 0:  # Avoid division by zero
                    normalized_values.iloc[idx] = stat_val / baseline
                else:
                    normalized_values.iloc[idx] = stat_val

        return normalized_values

    def _get_position_column(self, data: pd.DataFrame) -> Optional[str]:
        """Find the position column in the data."""
        position_candidates = ['Position', 'Pos', 'position', 'pos']

        for candidate in position_candidates:
            if candidate in data.columns:
                return candidate

        return None

    def get_position_impact_analysis(self,
                                   original_data: pd.DataFrame,
                                   normalized_data: pd.DataFrame) -> Dict:
        """
        Analyze the impact of position normalization.

        Returns:
            Dictionary with impact analysis by position
        """
        position_col = self._get_position_column(original_data)
        if position_col is None:
            return {}

        impact_analysis = {}

        # Find normalized columns
        normalized_cols = [col for col in normalized_data.columns
                          if col.endswith('_normalized')]

        for col in normalized_cols:
            base_col = col.replace('_normalized', '')

            if base_col in original_data.columns:
                impact_by_position = {}

                for position in original_data[position_col].unique():
                    if pd.notna(position):
                        pos_mask = original_data[position_col] == position

                        original_mean = original_data[pos_mask][base_col].mean()
                        normalized_mean = normalized_data[pos_mask][col].mean()

                        if not pd.isna(original_mean) and not pd.isna(normalized_mean):
                            impact_by_position[position] = {
                                'original_mean': original_mean,
                                'normalized_mean': normalized_mean,
                                'impact_factor': normalized_mean / original_mean if original_mean != 0 else 1.0
                            }

                impact_analysis[base_col] = impact_by_position

        return impact_analysis

    def get_position_adjustment_factors(self, position: str) -> Dict[str, float]:
        """
        Get adjustment factors for a specific position.

        Args:
            position: Position code (e.g., 'C', '1B', 'SS')

        Returns:
            Dictionary of adjustment factors by stat
        """
        adjustment_factors = {}

        for stat, baselines in self.position_baselines.items():
            # Calculate adjustment factor relative to league average
            league_average = np.mean(list(baselines.values()))
            position_baseline = baselines.get(position, league_average)

            if position_baseline > 0:
                adjustment_factors[stat] = league_average / position_baseline
            else:
                adjustment_factors[stat] = 1.0

        return adjustment_factors