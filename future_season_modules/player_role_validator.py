"""
Player Role Validation for Future Projections
=============================================

Validates and filters players based on legitimate role activity levels.
Prevents position players from contaminating pitcher models and vice versa.

Integrates with existing pipeline architecture and TwoWayPlayerModel.
Enhanced with rookie protection bypass for elite talent retention.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import sys
from pathlib import Path

# Add project root to path for rookie protection import
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

@dataclass
class RoleThresholds:
    """Activity thresholds for legitimate role classification."""
    pitcher_ip_min: float = 20.0      # Minimum IP for legitimate pitcher
    pitcher_games_min: int = 10       # Minimum games pitched
    hitter_pa_min: int = 100          # Minimum PA for legitimate hitter
    hitter_games_min: int = 50        # Minimum games played

    # Special thresholds for two-way evaluation
    two_way_pitcher_ip: float = 15.0  # Lower threshold for two-way pitchers
    two_way_hitter_pa: int = 75       # Lower threshold for two-way hitters

class PlayerRoleValidator:
    """
    Validates player roles and filters datasets for projection modeling.

    Ensures data quality by removing players who don't meaningfully
    contribute in a given role while preserving legitimate two-way players.
    """

    def __init__(self,
                 thresholds: RoleThresholds = None,
                 enable_two_way_detection: bool = True,
                 enable_rookie_protection: bool = True):
        """
        Initialize role validator.

        Args:
            thresholds: Activity thresholds for role classification
            enable_two_way_detection: Use sophisticated two-way player detection
            enable_rookie_protection: Enable rookie protection bypass
        """
        self.thresholds = thresholds or RoleThresholds()
        self.enable_two_way_detection = enable_two_way_detection
        self.enable_rookie_protection = enable_rookie_protection

        # Initialize rookie protection system if enabled
        self.rookie_system = None
        if enable_rookie_protection:
            try:
                from common_modules.rookie_elite_protection import RookieEliteProtection
                self.rookie_system = RookieEliteProtection(use_enhanced_system=True)
            except ImportError:
                print("Warning: RookieEliteProtection not available - rookie bypass disabled")
                self.enable_rookie_protection = False

        # Statistics tracking
        self.validation_stats = {
            'pitchers': {'total': 0, 'legitimate': 0, 'filtered': 0, 'two_way': 0},
            'hitters': {'total': 0, 'legitimate': 0, 'filtered': 0, 'two_way': 0}
        }

    def validate_pitcher_eligibility(self,
                                   pitcher_data: pd.DataFrame,
                                   season_context: Dict = None) -> Dict[str, pd.DataFrame]:
        """
        Validate pitcher eligibility and classify player types.

        Args:
            pitcher_data: DataFrame with pitcher performance data
            season_context: Optional context (injuries, COVID, etc.)

        Returns:
            Dictionary with filtered datasets and classifications
        """
        print("Validating pitcher eligibility...")

        # Extract activity metrics
        ip_values = pitcher_data.get('IP', pitcher_data.get('innings_pitched', 0))
        games_pitched = pitcher_data.get('G', pitcher_data.get('games_pitched', 0))

        # Handle missing values
        ip_values = pd.Series(ip_values).fillna(0)
        games_pitched = pd.Series(games_pitched).fillna(0)

        # Apply contextual adjustments
        adjusted_thresholds = self._adjust_thresholds_for_context(
            self.thresholds, season_context, role='pitcher'
        )

        # Basic legitimacy classification
        meets_ip_threshold = ip_values >= adjusted_thresholds.pitcher_ip_min
        meets_games_threshold = games_pitched >= adjusted_thresholds.pitcher_games_min

        # Combined legitimacy (either threshold)
        legitimate_pitchers = meets_ip_threshold | meets_games_threshold

        # Two-way player detection if enabled
        two_way_pitchers = pd.Series([False] * len(pitcher_data))
        if self.enable_two_way_detection:
            two_way_pitchers = self._detect_two_way_pitchers(
                pitcher_data, adjusted_thresholds
            )

        # Rookie protection bypass if enabled
        rookie_protected = pd.Series([False] * len(pitcher_data))
        if self.enable_rookie_protection and self.rookie_system:
            rookie_protected = self._apply_rookie_protection_bypass(
                pitcher_data, legitimate_pitchers | two_way_pitchers
            )

        # Final classification
        keep_as_pitcher = legitimate_pitchers | two_way_pitchers | rookie_protected

        # Create filtered datasets
        legitimate_pitcher_data = pitcher_data[keep_as_pitcher].copy()
        filtered_out_data = pitcher_data[~keep_as_pitcher].copy()

        # Update statistics
        self.validation_stats['pitchers'] = {
            'total': len(pitcher_data),
            'legitimate': legitimate_pitchers.sum(),
            'two_way': two_way_pitchers.sum(),
            'filtered': (~keep_as_pitcher).sum()
        }

        # Analysis summary
        self._log_pitcher_validation_summary(
            ip_values, games_pitched, keep_as_pitcher, adjusted_thresholds
        )

        return {
            'legitimate_pitchers': legitimate_pitcher_data,
            'filtered_out': filtered_out_data,
            'two_way_detected': pitcher_data[two_way_pitchers],
            'classification_mask': keep_as_pitcher,
            'validation_stats': self.validation_stats['pitchers']
        }

    def validate_hitter_eligibility(self,
                                  hitter_data: pd.DataFrame,
                                  season_context: Dict = None) -> Dict[str, pd.DataFrame]:
        """
        Validate hitter eligibility and classify player types.

        Args:
            hitter_data: DataFrame with hitter performance data
            season_context: Optional context (injuries, COVID, etc.)

        Returns:
            Dictionary with filtered datasets and classifications
        """
        print("Validating hitter eligibility...")

        # Extract activity metrics
        pa_values = hitter_data.get('PA', hitter_data.get('plate_appearances', 0))
        games_played = hitter_data.get('G', hitter_data.get('games_played', 0))

        # Handle missing values
        pa_values = pd.Series(pa_values).fillna(0)
        games_played = pd.Series(games_played).fillna(0)

        # Apply contextual adjustments
        adjusted_thresholds = self._adjust_thresholds_for_context(
            self.thresholds, season_context, role='hitter'
        )

        # Basic legitimacy classification
        meets_pa_threshold = pa_values >= adjusted_thresholds.hitter_pa_min
        meets_games_threshold = games_played >= adjusted_thresholds.hitter_games_min

        # Combined legitimacy (either threshold)
        legitimate_hitters = meets_pa_threshold | meets_games_threshold

        # Two-way player detection if enabled
        two_way_hitters = pd.Series([False] * len(hitter_data))
        if self.enable_two_way_detection:
            two_way_hitters = self._detect_two_way_hitters(
                hitter_data, adjusted_thresholds
            )

        # Rookie protection bypass if enabled
        rookie_protected = pd.Series([False] * len(hitter_data))
        if self.enable_rookie_protection and self.rookie_system:
            rookie_protected = self._apply_rookie_protection_bypass(
                hitter_data, legitimate_hitters | two_way_hitters
            )

        # Final classification
        keep_as_hitter = legitimate_hitters | two_way_hitters | rookie_protected

        # Create filtered datasets
        legitimate_hitter_data = hitter_data[keep_as_hitter].copy()
        filtered_out_data = hitter_data[~keep_as_hitter].copy()

        # Update statistics
        self.validation_stats['hitters'] = {
            'total': len(hitter_data),
            'legitimate': legitimate_hitters.sum(),
            'two_way': two_way_hitters.sum(),
            'filtered': (~keep_as_hitter).sum()
        }

        # Analysis summary
        self._log_hitter_validation_summary(
            pa_values, games_played, keep_as_hitter, adjusted_thresholds
        )

        return {
            'legitimate_hitters': legitimate_hitter_data,
            'filtered_out': filtered_out_data,
            'two_way_detected': hitter_data[two_way_hitters],
            'classification_mask': keep_as_hitter,
            'validation_stats': self.validation_stats['hitters']
        }

    def _adjust_thresholds_for_context(self,
                                     base_thresholds: RoleThresholds,
                                     season_context: Dict,
                                     role: str) -> RoleThresholds:
        """
        Adjust thresholds based on seasonal context.

        Args:
            base_thresholds: Base threshold values
            season_context: Season-specific context
            role: 'pitcher' or 'hitter'

        Returns:
            Adjusted thresholds
        """
        if not season_context:
            return base_thresholds

        adjusted = RoleThresholds(
            pitcher_ip_min=base_thresholds.pitcher_ip_min,
            pitcher_games_min=base_thresholds.pitcher_games_min,
            hitter_pa_min=base_thresholds.hitter_pa_min,
            hitter_games_min=base_thresholds.hitter_games_min,
            two_way_pitcher_ip=base_thresholds.two_way_pitcher_ip,
            two_way_hitter_pa=base_thresholds.two_way_hitter_pa
        )

        # COVID-19 2020 season adjustments
        if season_context.get('is_covid_season', False):
            covid_factor = 60 / 162  # 60-game season
            adjusted.pitcher_ip_min *= covid_factor
            adjusted.hitter_pa_min *= covid_factor
            adjusted.pitcher_games_min = max(3, int(adjusted.pitcher_games_min * covid_factor))
            adjusted.hitter_games_min = max(15, int(adjusted.hitter_games_min * covid_factor))

        # Injury-heavy seasons or other contexts
        injury_adjustment = season_context.get('injury_adjustment_factor', 1.0)
        if injury_adjustment != 1.0:
            adjusted.pitcher_ip_min *= injury_adjustment
            adjusted.hitter_pa_min *= injury_adjustment

        return adjusted

    def _detect_two_way_pitchers(self,
                               pitcher_data: pd.DataFrame,
                               thresholds: RoleThresholds) -> pd.Series:
        """
        Detect legitimate two-way players in pitcher dataset.

        Uses sophisticated criteria rather than simple thresholds.
        """
        # This is a simplified version - would integrate with TwoWayPlayerModel
        # For now, use position information as proxy

        if 'Position' in pitcher_data.columns:
            positions = pitcher_data['Position'].fillna('')
            # Two-way if they have non-pitcher positions listed
            is_two_way = positions.str.contains(r'(?:OF|1B|2B|3B|SS|C|DH)', regex=True, na=False)

            # Additional check: meet reduced thresholds
            ip_values = pd.Series(pitcher_data.get('IP', 0)).fillna(0)
            meets_two_way_ip = ip_values >= thresholds.two_way_pitcher_ip

            return is_two_way & meets_two_way_ip

        return pd.Series([False] * len(pitcher_data))

    def _detect_two_way_hitters(self,
                              hitter_data: pd.DataFrame,
                              thresholds: RoleThresholds) -> pd.Series:
        """
        Detect legitimate two-way players in hitter dataset.
        """
        # This is a simplified version - would integrate with TwoWayPlayerModel

        if 'Position' in hitter_data.columns:
            positions = hitter_data['Position'].fillna('')
            # Two-way if they have pitcher positions listed
            is_two_way = positions.str.contains(r'(?:SP|RP|P)', regex=True, na=False)

            # Additional check: meet reduced thresholds
            pa_values = pd.Series(hitter_data.get('PA', 0)).fillna(0)
            meets_two_way_pa = pa_values >= thresholds.two_way_hitter_pa

            return is_two_way & meets_two_way_pa

        return pd.Series([False] * len(hitter_data))

    def _log_pitcher_validation_summary(self,
                                      ip_values: pd.Series,
                                      games_pitched: pd.Series,
                                      keep_mask: pd.Series,
                                      thresholds: RoleThresholds):
        """Log pitcher validation summary."""
        stats = self.validation_stats['pitchers']

        print(f"  Total pitchers: {stats['total']}")
        print(f"  Legitimate pitchers: {stats['legitimate']} ({100*stats['legitimate']/stats['total']:.1f}%)")
        print(f"  Two-way players: {stats['two_way']}")
        print(f"  Filtered out: {stats['filtered']} ({100*stats['filtered']/stats['total']:.1f}%)")

        if stats['filtered'] > 0:
            filtered_ip = ip_values[~keep_mask]
            print(f"  Filtered IP range: {filtered_ip.min():.1f} to {filtered_ip.max():.1f}")
            print(f"  Thresholds used: {thresholds.pitcher_ip_min:.1f} IP, {thresholds.pitcher_games_min} games")

    def _log_hitter_validation_summary(self,
                                     pa_values: pd.Series,
                                     games_played: pd.Series,
                                     keep_mask: pd.Series,
                                     thresholds: RoleThresholds):
        """Log hitter validation summary."""
        stats = self.validation_stats['hitters']

        print(f"  Total hitters: {stats['total']}")
        print(f"  Legitimate hitters: {stats['legitimate']} ({100*stats['legitimate']/stats['total']:.1f}%)")
        print(f"  Two-way players: {stats['two_way']}")
        print(f"  Filtered out: {stats['filtered']} ({100*stats['filtered']/stats['total']:.1f}%)")

        if stats['filtered'] > 0:
            filtered_pa = pa_values[~keep_mask]
            print(f"  Filtered PA range: {filtered_pa.min():.0f} to {filtered_pa.max():.0f}")
            print(f"  Thresholds used: {thresholds.hitter_pa_min} PA, {thresholds.hitter_games_min} games")

    def validate_complete_dataset(self,
                                complete_data: pd.DataFrame,
                                season_context: Dict = None) -> Dict[str, pd.DataFrame]:
        """
        Validate complete dataset with both pitchers and hitters.

        Args:
            complete_data: Complete dataset with DataSource column
            season_context: Season-specific context

        Returns:
            Dictionary with validated datasets by role
        """
        print("Validating complete dataset for role legitimacy...")

        results = {}

        # Separate by data source
        if 'DataSource' in complete_data.columns:
            war_data = complete_data[complete_data['DataSource'] == 'WAR']
            warp_data = complete_data[complete_data['DataSource'] == 'WARP']

            # Separate pitchers and hitters within each source
            pitcher_mask = complete_data['Position'].isin(['SP', 'RP', 'P'])
            hitter_mask = ~pitcher_mask

            # Validate pitchers
            pitcher_data = complete_data[pitcher_mask]
            if len(pitcher_data) > 0:
                pitcher_results = self.validate_pitcher_eligibility(
                    pitcher_data, season_context
                )
                results['pitchers'] = pitcher_results

            # Validate hitters
            hitter_data = complete_data[hitter_mask]
            if len(hitter_data) > 0:
                hitter_results = self.validate_hitter_eligibility(
                    hitter_data, season_context
                )
                results['hitters'] = hitter_results

            # Combine validated datasets
            validated_data_pieces = []

            if 'pitchers' in results:
                validated_data_pieces.append(results['pitchers']['legitimate_pitchers'])

            if 'hitters' in results:
                validated_data_pieces.append(results['hitters']['legitimate_hitters'])

            if validated_data_pieces:
                results['validated_complete_dataset'] = pd.concat(
                    validated_data_pieces, ignore_index=True
                )
        else:
            print("  Warning: No DataSource column found, skipping role-specific validation")
            results['validated_complete_dataset'] = complete_data

        return results

    def get_validation_summary(self) -> Dict:
        """
        Get comprehensive validation summary.

        Returns:
            Summary of validation statistics and recommendations
        """
        summary = {
            'validation_stats': self.validation_stats.copy(),
            'data_quality_impact': {},
            'recommendations': []
        }

        # Calculate data quality impact
        p_stats = self.validation_stats['pitchers']
        h_stats = self.validation_stats['hitters']

        if p_stats['total'] > 0:
            p_retention = p_stats['legitimate'] + p_stats['two_way']
            p_retention_rate = p_retention / p_stats['total']
            summary['data_quality_impact']['pitcher_retention_rate'] = p_retention_rate

        if h_stats['total'] > 0:
            h_retention = h_stats['legitimate'] + h_stats['two_way']
            h_retention_rate = h_retention / h_stats['total']
            summary['data_quality_impact']['hitter_retention_rate'] = h_retention_rate

        # Generate recommendations
        if p_stats.get('filtered', 0) > p_stats.get('total', 1) * 0.1:
            summary['recommendations'].append(
                "High pitcher filtering rate - consider reviewing thresholds"
            )

        if h_stats.get('filtered', 0) > h_stats.get('total', 1) * 0.1:
            summary['recommendations'].append(
                "High hitter filtering rate - consider reviewing thresholds"
            )

        if p_stats.get('two_way', 0) > 0 or h_stats.get('two_way', 0) > 0:
            summary['recommendations'].append(
                "Two-way players detected - integrate with TwoWayPlayerModel"
            )

        return summary

    def _apply_rookie_protection_bypass(self,
                                      player_data: pd.DataFrame,
                                      already_legitimate: pd.Series) -> pd.Series:
        """
        Apply rookie protection bypass for filtered players.

        Args:
            player_data: Player data being validated
            already_legitimate: Boolean series of players already passing validation

        Returns:
            Boolean series of additional players to protect via rookie bypass
        """
        # Start with all False
        rookie_protected = pd.Series([False] * len(player_data))

        # Only consider players that would otherwise be filtered
        filtered_players = player_data[~already_legitimate]

        if len(filtered_players) == 0:
            return rookie_protected

        # Test each filtered player for rookie status
        rookie_count = 0
        for idx, (original_idx, player) in enumerate(filtered_players.iterrows()):
            try:
                player_dict = player.to_dict()

                # Add Season if not present (assume current year)
                if 'Season' not in player_dict:
                    player_dict['Season'] = 2024

                # Validate rookie status (no historical data = assume rookie)
                validation = self.rookie_system.validate_rookie_status(player_dict, pd.DataFrame())

                if validation.get('is_qualifying_rookie', False):
                    rookie_protected.iloc[original_idx] = True
                    rookie_count += 1

            except Exception as e:
                # If rookie validation fails, don't protect (conservative approach)
                continue

        if rookie_count > 0:
            print(f"    Rookie protection: {rookie_count} players restored via rookie bypass")

        return rookie_protected