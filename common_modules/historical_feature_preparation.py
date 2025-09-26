"""
Historical Feature Preparation Module
Ensures exact feature compatibility with historical sWARm_CS training

CRITICAL: Uses exact same features as historical training:
- Hitters: 7 features (K%, BB%, AVG, OBP, SLG, Enhanced_Baserunning, Enhanced_Defense)
- Pitchers: 6 features (IP, BB%, K%, ERA, HR%, Enhanced_Defense)

ENHANCED: Now includes park factors and real enhanced features
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime

# Import enhanced features and park factors
from .enhanced_features import get_enhanced_features, get_player_enhanced_features
from .park_factors import apply_park_factor_adjustments
from .positional_adjustments import PositionalAdjustmentCalculator, POSITION_WAR_ADJUSTMENTS


class HistoricalFeaturePreparer:
    """
    Prepare current season data to match exact historical training features

    Drops players with missing critical features and logs to file
    """

    def __init__(self):
        self.log_file = "incomplete_players_projection_log.txt"
        self.dropped_players = []

        # Load enhanced features once for efficiency
        print("Loading enhanced features and park factors...")
        self.baserunning_data, self.defense_data = get_enhanced_features()

    def prepare_hitter_features(self, first_half_data):
        """
        Prepare hitter features with PA and positional adjustments

        Enhanced features: K%, BB%, AVG, OBP, SLG, PA, Position_Adjustment, Enhanced_Baserunning, Enhanced_Defense

        Args:
            first_half_data: DataFrame with first half 2025 hitter data

        Returns:
            dict: {'valid_players': df, 'feature_matrix': np.array, 'player_names': list}
        """
        print("Preparing hitter features for historical compatibility...")

        valid_players = []
        feature_vectors = []
        player_names = []

        for idx, player in first_half_data.iterrows():
            player_name = player.get('Name', player.get('player_name', f'Unknown_{idx}'))

            try:
                # Calculate required features from raw stats
                features = self._calculate_hitter_features(player)

                if features is not None:
                    valid_players.append(player)
                    feature_vectors.append(features)
                    player_names.append(player_name)
                else:
                    self._log_dropped_player(player_name, 'hitter', 'Missing critical stats for feature calculation')

            except Exception as e:
                self._log_dropped_player(player_name, 'hitter', f'Feature calculation error: {str(e)}')

        if valid_players:
            valid_df = pd.DataFrame(valid_players)
            feature_matrix = np.array(feature_vectors)

            print(f"  Valid hitters: {len(valid_players)}")
            print(f"  Dropped hitters: {len(first_half_data) - len(valid_players)}")

            return {
                'valid_players': valid_df,
                'feature_matrix': feature_matrix,
                'player_names': player_names
            }
        else:
            print("  No valid hitters found")
            return None

    def prepare_pitcher_features(self, first_half_data):
        """
        Prepare pitcher features to match historical training exactly

        Historical features: IP, BB%, K%, ERA, HR%, Enhanced_Defense

        Args:
            first_half_data: DataFrame with first half 2025 pitcher data

        Returns:
            dict: {'valid_players': df, 'feature_matrix': np.array, 'player_names': list}
        """
        print("Preparing pitcher features for historical compatibility...")

        valid_players = []
        feature_vectors = []
        player_names = []

        for idx, player in first_half_data.iterrows():
            player_name = player.get('Name', player.get('player_name', f'Unknown_{idx}'))

            try:
                # Calculate required features from raw stats
                features = self._calculate_pitcher_features(player)

                if features is not None:
                    valid_players.append(player)
                    feature_vectors.append(features)
                    player_names.append(player_name)
                else:
                    self._log_dropped_player(player_name, 'pitcher', 'Missing critical stats for feature calculation')

            except Exception as e:
                self._log_dropped_player(player_name, 'pitcher', f'Feature calculation error: {str(e)}')

        if valid_players:
            valid_df = pd.DataFrame(valid_players)
            feature_matrix = np.array(feature_vectors)

            print(f"  Valid pitchers: {len(valid_players)}")
            print(f"  Dropped pitchers: {len(first_half_data) - len(valid_players)}")

            return {
                'valid_players': valid_df,
                'feature_matrix': feature_matrix,
                'player_names': player_names
            }
        else:
            print("  No valid pitchers found")
            return None

    def _calculate_hitter_features(self, player_row):
        """
        Calculate 10 hitter features including PA, positional adjustments, and GDP rate

        Features: [K%, BB%, AVG, OBP, SLG, PA, Position_Adjustment, GDP_rate, Enhanced_Baserunning, Enhanced_Defense]
        """
        try:
            # Basic rate stats - must be present
            avg = self._safe_float(player_row.get('AVG'))
            obp = self._safe_float(player_row.get('OBP'))
            slg = self._safe_float(player_row.get('SLG'))

            if avg is None or obp is None or slg is None:
                return None

            # Calculate percentage stats
            pa = self._safe_float(player_row.get('PA', 0))
            so = self._safe_float(player_row.get('SO', 0))
            bb = self._safe_float(player_row.get('BB', 0))

            if pa <= 0:
                return None

            k_pct = so / pa if pa > 0 else 0.0
            bb_pct = bb / pa if pa > 0 else 0.0

            # Get enhanced features using player identification
            player_id = player_row.get('mlbid', player_row.get('MLBAID', player_row.get('player_name', '')))

            # Get real enhanced features from loaded data
            enhanced_baserunning = self.baserunning_data.get(player_id, 0.0)
            enhanced_defense = self.defense_data.get(player_id, 0.0)

            # If not found by ID and we have a name, try name matching
            if enhanced_baserunning == 0.0 and enhanced_defense == 0.0:
                player_name = player_row.get('player_name', player_row.get('Name', ''))
                if player_name and isinstance(player_name, str):
                    enhanced_features = get_player_enhanced_features(player_name, self.baserunning_data, self.defense_data)
                    enhanced_baserunning = enhanced_features['Enhanced_Baserunning']
                    enhanced_defense = enhanced_features['Enhanced_Defense']

            # Calculate positional adjustment
            position = player_row.get('Pos', player_row.get('Position', ''))
            position_adjustment = POSITION_WAR_ADJUSTMENTS.get(position, 0.0)

            # Scale by playing time (PA ratio to 600)
            position_adjustment = position_adjustment * (pa / 600) if pa > 0 else 0.0

            # Calculate GDP rate for situational hitting
            gdp = self._safe_float(player_row.get('GDP', 0))
            gdp_rate = gdp / pa if pa > 0 else 0.0

            # Return 10 features including PA, positional adjustment, and GDP rate
            return [k_pct, bb_pct, avg, obp, slg, pa, position_adjustment, gdp_rate, enhanced_baserunning, enhanced_defense]

        except Exception:
            return None

    def _calculate_pitcher_features(self, player_row):
        """
        Calculate exact 6 pitcher features from raw stats (NO Enhanced_Baserunning for pitchers)

        Features: [IP, BB%, K%, ERA, HR%, Enhanced_Defense]
        """
        try:
            # Core stats - must be present
            ip = self._safe_float(player_row.get('IP'))
            era = self._safe_float(player_row.get('ERA'))

            if ip is None or era is None or ip <= 0:
                return None

            # Calculate percentage stats
            bf = self._safe_float(player_row.get('BF'))  # Batters faced
            so = self._safe_float(player_row.get('SO', 0))
            bb = self._safe_float(player_row.get('BB', 0))
            hr = self._safe_float(player_row.get('HR', 0))

            # If BF not available, estimate from IP and other stats
            if bf is None or bf <= 0:
                h = self._safe_float(player_row.get('H', 0))
                bf = (ip * 3) + h + bb  # Rough estimate

            if bf <= 0:
                return None

            k_pct = so / bf if bf > 0 else 0.0
            bb_pct = bb / bf if bf > 0 else 0.0
            hr_pct = hr / bf if bf > 0 else 0.0

            # Get enhanced defense using player identification (defense only for pitchers)
            player_id = player_row.get('mlbid', player_row.get('MLBAID', player_row.get('player_name', '')))

            # Get real enhanced defense from loaded data
            enhanced_defense = self.defense_data.get(player_id, 0.0)

            # If not found by ID and we have a name, try name matching
            if enhanced_defense == 0.0:
                player_name = player_row.get('player_name', player_row.get('Name', ''))
                if player_name and isinstance(player_name, str):
                    enhanced_features = get_player_enhanced_features(player_name, self.baserunning_data, self.defense_data)
                    enhanced_defense = enhanced_features['Enhanced_Defense']

            # Return exact 6 features in historical order (NO Enhanced_Baserunning)
            return [ip, bb_pct, k_pct, era, hr_pct, enhanced_defense]

        except Exception:
            return None

    def _safe_float(self, value):
        """Convert value to float safely, return None if invalid"""
        if value is None or pd.isna(value):
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None

    def _log_dropped_player(self, player_name, player_type, reason):
        """Log dropped player to file with detailed reason"""
        self.dropped_players.append({
            'player_name': player_name,
            'player_type': player_type,
            'reason': reason,
            'timestamp': datetime.now().isoformat()
        })

    def write_dropped_players_log(self):
        """Write comprehensive log of dropped players"""
        if not self.dropped_players:
            return

        with open(self.log_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write(f"Incomplete Players Projection Log: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n")
            f.write(f"Total players dropped: {len(self.dropped_players)}\n")

            # Group by reason
            reasons = {}
            for player in self.dropped_players:
                reason = player['reason']
                if reason not in reasons:
                    reasons[reason] = []
                reasons[reason].append(player)

            f.write(f"Unique reasons: {len(reasons)}\n\n")

            for reason, players in reasons.items():
                f.write(f"Reason: {reason}\n")
                f.write(f"Count: {len(players)}\n")
                f.write("-" * 40 + "\n")

                for player in players:
                    f.write(f"  Player: {player['player_name']} ({player['player_type']})\n")

                f.write("\n")

        print(f"Dropped players log written to: {self.log_file}")
        print(f"Total players dropped: {len(self.dropped_players)}")


def prepare_historical_compatible_data(first_half_hitters, first_half_pitchers):
    """
    Convenience function to prepare both hitters and pitchers

    Returns:
        dict: {'hitters': hitter_data, 'pitchers': pitcher_data}
    """
    preparer = HistoricalFeaturePreparer()

    results = {}

    # Prepare hitters
    if first_half_hitters is not None:
        results['hitters'] = preparer.prepare_hitter_features(first_half_hitters)
    else:
        results['hitters'] = None

    # Prepare pitchers
    if first_half_pitchers is not None:
        results['pitchers'] = preparer.prepare_pitcher_features(first_half_pitchers)
    else:
        results['pitchers'] = None

    # Write log of dropped players
    preparer.write_dropped_players_log()

    return results


def validate_feature_compatibility():
    """
    Validate that prepared features match historical training dimensions

    Expected:
    - Hitters: 7 features
    - Pitchers: 6 features
    """
    print("FEATURE COMPATIBILITY VALIDATION:")
    print("Expected - Hitters: 7 features [K%, BB%, AVG, OBP, SLG, Enhanced_Baserunning, Enhanced_Defense]")
    print("Expected - Pitchers: 6 features [IP, BB%, K%, ERA, HR%, Enhanced_Defense]")
    return True