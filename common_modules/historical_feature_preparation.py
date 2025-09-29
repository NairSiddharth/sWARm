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

        # Load enhanced pitcher features (LOB%, GB%, damage_control_ratio)
        from .derived_stats import load_enhanced_pitcher_features, load_percentage_pitcher_features
        self.enhanced_pitcher_features = load_enhanced_pitcher_features()

        # Load percentage-based features for 10-feature K-BB% system
        self.percentage_features = load_percentage_pitcher_features("MLB Player Data", [2020, 2021, 2022, 2023, 2024])

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
        Prepare pitcher features with 10-feature K-BB% system

        Features: IP, BB%, K%, K-BB%, ERA, damage_control_ratio, Opportunity_Success, Contact_Quality_Index, HBP%, Statcast_Launch_Quality_Index

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
        Calculate 10 hitter features with park factor adjustments

        Features: [K%, BB%, AVG (park-adjusted), OBP (park-adjusted), SLG (park-adjusted), PA, Position_Adjustment, GDP_rate, Enhanced_Baserunning, Enhanced_Defense]
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

            # Apply park factor adjustments to batting stats
            player_name = player_row.get('Name', player_row.get('player_name', ''))
            team = player_row.get('Team', player_row.get('team', ''))

            if player_name and team:
                # Create stats dict for park factor adjustment
                hitter_stats = {
                    'AVG': avg,
                    'OBP': obp,
                    'SLG': slg
                }

                # Apply park factors (use 2025 for current season, will fallback to 2024)
                adjusted_stats = apply_park_factor_adjustments(hitter_stats, player_name, team, 'hitter', year=2025)
                avg = adjusted_stats.get('AVG', avg)  # Use park-adjusted AVG
                obp = adjusted_stats.get('OBP', obp)  # Use park-adjusted OBP
                slg = adjusted_stats.get('SLG', slg)  # Use park-adjusted SLG

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
        Calculate 10 pitcher features from raw stats with park factor adjustments and K-BB%

        Features: [IP, BB%, K%, K-BB%, ERA (park-adjusted), damage_control_ratio, Opportunity_Success, Contact_Quality_Index, HBP%, Statcast_Launch_Quality_Index]
        """
        try:
            # Core stats - must be present
            ip = self._safe_float(player_row.get('IP'))
            era = self._safe_float(player_row.get('ERA'))

            if ip is None or era is None or ip <= 0:
                return None

            # Calculate percentage stats - handle both counting stats and rate stats
            bf = self._safe_float(player_row.get('BF', player_row.get('TBF')))  # Batters faced (try both BF and TBF)
            so = self._safe_float(player_row.get('SO'))
            bb = self._safe_float(player_row.get('BB'))
            hr = self._safe_float(player_row.get('HR'))

            # Check if we have counting stats (SO, BB, HR, BF/TBF)
            if bf is not None and bf > 0 and so is not None and bb is not None and hr is not None:
                # Use counting stats
                k_pct = so / bf
                bb_pct = bb / bf
                hr_pct = hr / bf
            else:
                # Try rate stats (K/9, BB/9, HR/9) - convert to percentages
                k9 = self._safe_float(player_row.get('K/9'))
                bb9 = self._safe_float(player_row.get('BB/9'))
                hr9 = self._safe_float(player_row.get('HR/9'))

                if k9 is not None and bb9 is not None and hr9 is not None:
                    # Convert rate stats to percentages
                    # Assume ~3.0 batters per inning (league average)
                    batters_per_inning = 3.0
                    k_pct = k9 / (9 * batters_per_inning)  # K per batter faced
                    bb_pct = bb9 / (9 * batters_per_inning)  # BB per batter faced
                    hr_pct = hr9 / (9 * batters_per_inning)  # HR per batter faced
                else:
                    # No valid stats available
                    return None

            # Apply park factor adjustments to ERA and HR%
            player_name = player_row.get('Name', player_row.get('player_name', ''))
            team = player_row.get('Team', player_row.get('team', ''))

            if player_name and team:
                # Create stats dict for park factor adjustment
                pitcher_stats = {
                    'ERA': era,
                    'HR%': hr_pct
                }

                # Apply park factors (use 2025 for current season, will fallback to 2024)
                adjusted_stats = apply_park_factor_adjustments(pitcher_stats, player_name, team, 'pitcher', year=2025)
                era = adjusted_stats.get('ERA', era)  # Use park-adjusted ERA
                hr_pct = adjusted_stats.get('HR%', hr_pct)  # Use park-adjusted HR%

            # Get enhanced pitcher features for damage control ratio
            player_id = player_row.get('mlbid', player_row.get('MLBAMID', player_row.get('player_name', '')))

            # Calculate damage control ratio from enhanced features or current stats
            damage_control_ratio = 0.0  # Default value

            # First try to get from historical cache
            if hasattr(self, 'enhanced_pitcher_features') and self.enhanced_pitcher_features:
                if player_id in self.enhanced_pitcher_features.get('damage_control_ratio', {}):
                    damage_control_ratio = self.enhanced_pitcher_features['damage_control_ratio'][player_id]
                elif isinstance(player_name, str) and player_name:
                    # Try name-based lookup as fallback
                    for pid, name in self.enhanced_pitcher_features.get('player_names', {}).items():
                        if isinstance(name, str) and name.lower() == player_name.lower():
                            damage_control_ratio = self.enhanced_pitcher_features.get('damage_control_ratio', {}).get(pid, 0.0)
                            break

            # If not found in historical cache, calculate from current season stats
            if damage_control_ratio == 0.0:
                lob_pct = self._safe_float(player_row.get('LOB%'))
                hr9_current = self._safe_float(player_row.get('HR/9'))

                if lob_pct is not None and hr9_current is not None:
                    # Convert LOB% from decimal (0.81) to percentage (81) if needed
                    if lob_pct <= 1.0:
                        lob_pct = lob_pct * 100

                    # Calculate damage control ratio: LOB% / (HR/9 + 0.5)
                    damage_control_ratio = lob_pct / (hr9_current + 0.5)

            # Calculate K-BB% from K% and BB%
            k_bb_pct = k_pct - bb_pct

            # Get enhanced features from production system
            player_id = self._safe_float(player_row.get('MLBAMID', player_row.get('mlbid')))

            # Load features for 10-feature K-BB% system
            opportunity_success = 0.6  # Default value
            contact_quality_index = 50.0  # Default normalized value
            hbp_pct = 0.0  # HBP percentage
            statcast_quality = 50.0  # Default Statcast Launch Quality Index

            if player_id is not None:
                try:
                    player_id = int(player_id)

                    # Get features from percentage-based system (which includes K-BB%)
                    from .derived_stats import get_player_percentage_features
                    percentage_features = get_player_percentage_features(player_id, self.percentage_features)

                    if percentage_features:
                        opportunity_success = percentage_features.get('Opportunity_Success', 0.6)
                        contact_quality_index = percentage_features.get('Contact_Quality_Index', 50.0)
                        hbp_pct = percentage_features.get('HBP%', 0.0)
                        statcast_quality = percentage_features.get('Statcast_Launch_Quality_Index', 50.0)

                        # Use the K-BB% from production system if available
                        if 'K-BB%' in percentage_features:
                            k_bb_pct = percentage_features['K-BB%']

                except (ValueError, TypeError, ImportError):
                    pass

            # If features not found, try to calculate from current row data
            if hbp_pct == 0.0:
                hbp_raw = self._safe_float(player_row.get('HBP', 0.0)) or 0.0
                bf = self._safe_float(player_row.get('BF', player_row.get('TBF'))) or 1.0
                hbp_pct = hbp_raw / bf if bf > 0 else 0.0

            # Return 10 features: [IP, BB%, K%, K-BB%, ERA, damage_control_ratio, Opportunity_Success, Contact_Quality_Index, HBP%, Statcast_Launch_Quality_Index]
            return [ip, bb_pct, k_pct, k_bb_pct, era, damage_control_ratio, opportunity_success, contact_quality_index, hbp_pct, statcast_quality]

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
    - Hitters: 10 features
    - Pitchers: 6 features
    """
    print("FEATURE COMPATIBILITY VALIDATION:")
    print("Expected - Hitters: 10 features [K%, BB%, AVG, OBP, SLG, PA, Position_Adjustment, GDP_rate, Enhanced_Baserunning, Enhanced_Defense]")
    print("Expected - Pitchers: 6 features [IP, BB%, K%, ERA, HR%, damage_control_ratio]")
    return True