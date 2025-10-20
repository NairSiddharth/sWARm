"""
ROS Feature Builder - Orchestrator

Coordinates all feature extraction modules to build complete feature vectors
for ROS prediction models.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime

from .elite_detection import extract_elite_features
from .rookie_detection import extract_rookie_features
from .injury_recovery import extract_injury_features
from .baseline_comparison import extract_baseline_features
from .age_curves import extract_age_features
from .baserunning_projection import extract_baserunning_projection_features
from .defense_projection import extract_defense_projection_features


class ROSFeatureBuilder:
    """
    Orchestrates feature extraction for ROS predictions.

    Combines all feature modules into single feature vector.
    """

    def __init__(self, player_type: str = 'hitter'):
        """
        Initialize feature builder.

        Args:
            player_type: 'hitter' or 'pitcher'
        """
        self.player_type = player_type

    def build_features(
        self,
        player_name: str,
        current_season: pd.Series,
        historical_data: pd.DataFrame,
        injury_data: Optional[pd.DataFrame] = None,
        current_date: datetime = None
    ) -> Dict[str, float]:
        """
        Build complete feature vector for single player.

        Args:
            player_name: Player name
            current_season: Current season stats (Series)
            historical_data: Historical data for player (DataFrame)
            injury_data: Injury history (optional)
            current_date: Date for injury calculations (default: today)

        Returns:
            Dictionary with all features (86 for hitters, 128 for pitchers)

        Example:
            >>> builder = ROSFeatureBuilder('hitter')
            >>> current = pd.Series({
            ...     'Name': 'Juan Soto',
            ...     'Age': 26,
            ...     'Position': 'RF',
            ...     'wOBA': 0.411,
            ...     'K%': 17.2,
            ...     'BB%': 18.9,
            ...     'WAR_per_600': 7.0,
            ...     'PA': 350
            ... })
            >>> history = pd.DataFrame({...})  # Historical data
            >>> features = builder.build_features('Juan Soto', current, history)
            >>> len(features)
            86  # All hitter features
        """
        # Initialize feature dict with current stats
        features = {}

        # 1. Current performance stats (copy from current_season)
        if self.player_type == 'hitter':
            stat_cols = ['K%', 'BB%', 'AVG', 'OBP', 'SLG', 'GDP',
                        'Positional_WAR', 'Enhanced_Baserunning', 'Enhanced_Defense']
        else:
            # Pitcher stats (use column names from processed pipeline data)
            stat_cols = ['K%', 'BB%', 'GB%', 'SwStr%', 'WPA/LI', 'ERA',
                        'damage_control_ratio', 'Opportunity_Success', 'strikeout_efficiency',
                        'contact_management', 'Running_Control']

        for col in stat_cols:
            features[col] = current_season.get(col, 0.0)

        # 2. Current season model output (if available)
        features['current_predicted_war_rate'] = current_season.get(
            'Predicted_WAR_Rate',
            current_season.get('WAR_per_600', 0.0)
        )

        # 3. Elite detection features (9 features)
        elite_features = extract_elite_features(historical_data, current_season)
        features.update(elite_features)

        # 4. Rookie detection features (5 features)
        player_data = current_season.to_dict()
        player_data['Name'] = player_name
        rookie_features = extract_rookie_features(player_data, historical_data, self.player_type)
        features.update(rookie_features)

        # 5. Baseline comparison features (64+ hitters / 128+ pitchers)
        baseline_features = extract_baseline_features(
            historical_data, current_season, self.player_type, injury_data
        )
        features.update(baseline_features)

        # 6. Age curve features (8 features)
        position = current_season.get('Primary_Position', 'OF' if self.player_type == 'hitter' else 'P')
        age_features = extract_age_features(current_season, historical_data, position)
        features.update(age_features)

        # 6b. Baserunning projection features (4 features, hitters only)
        if self.player_type == 'hitter':
            baserunning_features = extract_baserunning_projection_features(current_season, historical_data)
            features.update(baserunning_features)

        # 6c. Defense projection features (5 features, hitters only)
        if self.player_type == 'hitter':
            defense_features = extract_defense_projection_features(current_season, historical_data)
            features.update(defense_features)

        # 7. Injury features (5 features)
        injury_features = extract_injury_features(injury_data, current_date)
        features.update(injury_features)

        # 8. Usage features
        if self.player_type == 'hitter':
            features['PA'] = current_season.get('current_PA', current_season.get('PA'))
            features['Games'] = current_season.get('current_G', current_season.get('G'))
        else:
            features['IP'] = current_season.get('current_IP', current_season.get('IP'))
            features['G'] = current_season.get('current_G', current_season.get('G'))
            features['GS'] = current_season.get('current_GS', current_season.get('GS'))

        # Season timing (critical for flexible ROS predictions)
        features['season_completion_pct'] = current_season.get('season_completion_pct', 0.5)

        # 9. Static covariates (5 hitters / 4 pitchers)
        # Position encoding (simple ordinal for now)
        position_map = {
            'C': 1, '1B': 2, '2B': 3, '3B': 4, 'SS': 5,
            'LF': 6, 'CF': 7, 'RF': 8, 'DH': 9,
            'P': 10, 'SP': 11, 'RP': 12, 'CL': 13
        }
        features['Position_encoded'] = position_map.get(position, 0)

        # Debut age (from rookie features or estimate)
        features['debut_age'] = rookie_features.get('debut_age', current_season.get('Age', 25) - 3)

        # Injury history count
        if injury_data is not None and not injury_data.empty:
            features['injury_history_count'] = len(injury_data)
        else:
            features['injury_history_count'] = 0

        # Career peak WAR (from elite features)
        features['career_peak_WAR'] = elite_features.get('peak_WAR', current_season.get('WAR_per_600', 0.0))

        # Late bloomer (from rookie features)
        if self.player_type == 'hitter':
            features['is_late_bloomer'] = rookie_features.get('is_late_bloomer', 0)
        else:
            # Pitchers: Role encoding instead of is_late_bloomer
            role = current_season.get('Role', 'SP')
            role_map = {'SP': 1, 'RP': 2, 'CL': 3, 'SU': 4}
            features['Role_encoded'] = role_map.get(role, 1)

        return features

    def build_features_batch(
        self,
        current_season_df: pd.DataFrame,
        historical_df: pd.DataFrame,
        injury_df: Optional[pd.DataFrame] = None,
        current_date: datetime = None
    ) -> pd.DataFrame:
        """
        Build features for multiple players (batch mode).

        Args:
            current_season_df: Current season data (all players)
            historical_df: Historical data (all players)
            injury_df: Injury data (optional)
            current_date: Date for injury calculations

        Returns:
            DataFrame with all players and their features

        Example:
            >>> builder = ROSFeatureBuilder('hitter')
            >>> current_df = pd.DataFrame({...})  # All 2025 hitters
            >>> historical_df = pd.DataFrame({...})  # All 2016-2024 data
            >>> feature_df = builder.build_features_batch(current_df, historical_df)
            >>> feature_df.shape
            (250, 86)  # 250 qualified hitters, 86 features each
        """
        # Normalize playerid column name in historical_df
        historical_df = historical_df.copy()
        if 'playerid' not in historical_df.columns:
            for col in ['PlayerId', 'MLBAMID']:
                if col in historical_df.columns:
                    historical_df['playerid'] = historical_df[col]
                    break

        # Normalize playerid column name in current_season_df as well
        current_season_df = current_season_df.copy()
        if 'playerid' not in current_season_df.columns:
            for col in ['PlayerId', 'MLBAMID']:
                if col in current_season_df.columns:
                    current_season_df['playerid'] = current_season_df[col]
                    break

        # Normalize playerid column name in injury_df as well
        if injury_df is not None:
            injury_df = injury_df.copy()
            if 'playerid' not in injury_df.columns:
                for col in ['MLBAMID', 'PlayerId']:
                    if col in injury_df.columns:
                        injury_df['playerid'] = injury_df[col]
                        break

        all_features = []

        for idx, current_row in current_season_df.iterrows():
            player_name = current_row['Name']
            playerid = current_row.get('playerid', None)

            # Get player's historical data
            if playerid is not None and 'playerid' in historical_df.columns:
                player_history = historical_df[historical_df['playerid'] == playerid]
            else:
                player_history = historical_df[historical_df['Name'] == player_name]

            # Get player's injury data
            if injury_df is not None and playerid is not None:
                player_injury = injury_df[injury_df['playerid'] == playerid]
            else:
                player_injury = None

            # Build features
            try:
                features = self.build_features(
                    player_name=player_name,
                    current_season=current_row,
                    historical_data=player_history,
                    injury_data=player_injury,
                    current_date=current_date
                )

                # Add identifiers
                features['Name'] = player_name
                features['playerid'] = playerid
                features['Team'] = current_row.get('Team', '')
                features['Primary_Position'] = current_row.get('Primary_Position', '')

                # Add target and metadata from splits (if present)
                for col in ['WAR', 'current_WAR', 'full_WAR', 'IP', 'current_IP', 'full_IP', 'G', 'current_G', 'full_G', 'GS', 'current_GS', 'full_GS', 'PA', 'current_PA', 'full_PA', 'Games', 'remaining_WAR', 'remaining_PA', 'remaining_IP', 'Year', 'split_point', '_multi_team_current', '_multi_team_stints']:
                    if col in current_row.index:
                        features[col] = current_row[col]

                all_features.append(features)

            except Exception as e:
                print(f"Error processing {player_name}: {e}")
                continue

        return pd.DataFrame(all_features)
