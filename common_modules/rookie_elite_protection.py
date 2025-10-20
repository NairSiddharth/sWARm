"""
Rookie Elite Protection Module
===============================

Enhanced elite protection for rookie players using MLB official thresholds.
Provides special pathway for qualifying rookies while preventing veteran
"career year" false positives and capturing elite rookie talent.

This module was extracted from the monolithic elite_adjustment.py file.
"""

__version__ = '2.0.0'
__author__ = 'oWAR Development Team'

# Standard library imports
from typing import Dict, Optional

# Third-party imports
import pandas as pd

# Local imports
from .elite_adjustment_base import ElitePlayerAdjuster
from .logging import get_logger

logger = get_logger(__name__)

# MLB Rookie Thresholds (Constants)
PITCHER_IP_THRESHOLD = 50.0    # MLB rookie threshold for pitchers
HITTER_AB_THRESHOLD = 130.0    # MLB rookie threshold for hitters
MIN_CURRENT_IP = 40.0          # Minimum IP to avoid tiny samples
MIN_CURRENT_AB = 200.0         # Minimum AB to avoid tiny samples
MIN_WAR_THRESHOLD = 2.5        # Minimum WAR to trigger rookie protection
ROOKIE_CEILING = 6.0           # Maximum projected WAR for rookies


class RookieEliteProtection(ElitePlayerAdjuster):
    """
    Enhanced elite protection for rookie players using MLB official thresholds.

    Provides special pathway for qualifying rookies (per MLB definition):
    - Pitchers: <50 IP in all previous MLB seasons
    - Hitters: <130 AB in all previous MLB seasons

    Prevents veteran "career year" false positives while capturing elite rookie talent.
    """

    def __init__(self, use_enhanced_system: bool = True):
        """
        Initialize with base elite adjuster functionality.

        Args:
                use_enhanced_system: Use enhanced WAR tier system (recommended)
        """
        super().__init__(use_enhanced_system=use_enhanced_system)

        # MLB official rookie thresholds
        self.rookie_thresholds = {
            'pitcher_ip': PITCHER_IP_THRESHOLD,
            'hitter_ab': HITTER_AB_THRESHOLD,
            'minimum_current_ip': MIN_CURRENT_IP,
            'minimum_current_ab': MIN_CURRENT_AB,
            'minimum_war': MIN_WAR_THRESHOLD
        }

        # Rookie-specific protection matrix (more conservative than veterans)
        self.rookie_protection_matrix = {
            'elite_rookie': {'hitters': 0.30, 'starting_pitchers': 0.25, 'relief_pitchers': 0.35},
            'good_rookie': {'hitters': 0.45, 'starting_pitchers': 0.40, 'relief_pitchers': 0.50},
            'average_rookie': {'hitters': 0.70, 'starting_pitchers': 0.65, 'relief_pitchers': 0.75}
        }

        # Uncertainty factors for small samples
        self.uncertainty_adjustments = {
            'sample_size_penalty': 0.15,  # Additional regression for small samples
            'rookie_ceiling': ROOKIE_CEILING,
            'confidence_discount': 0.9    # Slight confidence discount for inexperience
        }

        logger.info(
            "Initialized RookieEliteProtection with enhanced system: %s",
            use_enhanced_system)

    def validate_rookie_status(self, player_data: Dict,
                               historical_data: Optional[pd.DataFrame] = None) -> Dict:
        """
        Validate rookie status using MLB official thresholds.

        Args:
                player_data: Current season player data (must include Name, IP/AB, Position)
                historical_data: Historical player data for previous seasons validation (optional)

        Returns:
                Dict with rookie validation results

        Raises:
                ValueError: If required fields are missing from player_data
        """
        required_fields = ['Name', 'Position', 'WAR']
        missing_fields = [f for f in required_fields if f not in player_data]
        if missing_fields:
            raise ValueError(f"Missing required fields in player_data: {missing_fields}")

        try:
            player_name = player_data.get('Name', 'Unknown')
            position = player_data.get('Position', 'OF')
            current_war = player_data.get('WAR', 0.0)

            # Get current season stats
            current_ip = player_data.get('IP', 0.0)
            current_ab = player_data.get('AB', 0.0)
            current_pa = player_data.get('PA', 0.0)

            # Determine if player is pitcher or hitter
            is_pitcher = position in ['P', 'SP', 'RP', 'CL'] or current_ip > 0

            # Check minimum thresholds to avoid tiny samples
            meets_minimum = False
            if is_pitcher:
                meets_minimum = current_ip >= self.rookie_thresholds['minimum_current_ip']
            else:
                meets_minimum = (current_ab >= self.rookie_thresholds['minimum_current_ab'] or
                                 current_pa >= self.rookie_thresholds['minimum_current_ab'] * 1.5)

            # Check if performance merits rookie protection
            meets_war_threshold = current_war >= self.rookie_thresholds['minimum_war']

            # Historical validation
            has_previous_experience = False
            total_previous_volume = 0.0

            if historical_data is not None and not historical_data.empty:
                # Look for previous seasons of this player
                previous_seasons = historical_data[
                    (historical_data['Name'] == player_name) &
                    (historical_data['Season'] < player_data.get('Season', 2024))
                ]

                if not previous_seasons.empty:
                    if is_pitcher:
                        total_previous_volume = previous_seasons['IP'].sum()
                        has_previous_experience = total_previous_volume >= self.rookie_thresholds['pitcher_ip']
                    else:
                        total_previous_volume = previous_seasons['AB'].sum()
                        has_previous_experience = total_previous_volume >= self.rookie_thresholds['hitter_ab']

                    logger.debug(
                        f"Player {player_name} historical volume: {
                            total_previous_volume:.1f}")

            # Rookie qualification logic
            is_qualifying_rookie = (
                not has_previous_experience and
                meets_minimum and
                meets_war_threshold
            )

            validation_result = {
                'is_qualifying_rookie': is_qualifying_rookie,
                'is_pitcher': is_pitcher,
                'meets_minimum_volume': meets_minimum,
                'meets_war_threshold': meets_war_threshold,
                'has_previous_experience': has_previous_experience,
                'total_previous_volume': total_previous_volume,
                'current_volume': current_ip if is_pitcher else current_ab,
                'validation_details': {
                    'threshold_used': self.rookie_thresholds['pitcher_ip'] if is_pitcher else self.rookie_thresholds['hitter_ab'],
                    'minimum_current': self.rookie_thresholds['minimum_current_ip'] if is_pitcher else self.rookie_thresholds['minimum_current_ab'],
                    'minimum_war': self.rookie_thresholds['minimum_war']}}

            if is_qualifying_rookie:
                logger.info(f"Player {player_name} qualifies for rookie protection")

            return validation_result

        except Exception as e:
            logger.error(f"Error validating rookie status: {str(e)}", exc_info=True)
            raise

    def classify_rookie_tier(self, war_value: float) -> str:
        """
        Classify rookie into performance tier for protection purposes.

        Args:
                war_value: Current WAR value

        Returns:
                Rookie tier classification
        """
        if war_value >= 4.0:
            return 'elite_rookie'
        elif war_value >= 3.0:
            return 'good_rookie'
        else:
            return 'average_rookie'

    def get_rookie_protection_factor(
            self,
            war_value: float,
            position: str,
            rookie_validation: Dict) -> float:
        """
        Get protection factor for qualifying rookies.

        Args:
                war_value: Current WAR value
                position: Player position
                rookie_validation: Results from validate_rookie_status()

        Returns:
                Protection factor (lower = more protection)
        """
        try:
            if not rookie_validation.get('is_qualifying_rookie', False):
                # Not a qualifying rookie, use standard protection
                return self.get_protection_factor(war_value, position)

            # Rookie-specific protection
            rookie_tier = self.classify_rookie_tier(war_value)
            position_type = self._classify_rookie_position_type(
                position, rookie_validation['is_pitcher'])

            base_protection = self.rookie_protection_matrix[rookie_tier][position_type]

            # Apply uncertainty adjustments for rookie inexperience
            sample_size_factor = 1.0 + self.uncertainty_adjustments['sample_size_penalty']
            confidence_factor = self.uncertainty_adjustments['confidence_discount']

            # Conservative adjustment: slightly higher regression for rookies vs veterans
            adjusted_protection = base_protection * sample_size_factor * confidence_factor

            # Cap at reasonable bounds
            final_protection = max(0.15, min(0.80, adjusted_protection))

            logger.debug(f"Rookie protection factor: {final_protection:.3f} (tier: {rookie_tier})")

            return final_protection

        except Exception as e:
            logger.error(f"Error calculating rookie protection factor: {str(e)}", exc_info=True)
            # Return standard protection on error
            return self.get_protection_factor(war_value, position)

    def _classify_rookie_position_type(self, position: str, is_pitcher: bool) -> str:
        """
        Classify position type for rookie protection purposes.

        Args:
                position: Player position code
                is_pitcher: Whether player is a pitcher

        Returns:
                Position type for protection matrix lookup
        """
        if is_pitcher:
            if position in ['RP', 'CL']:
                return 'relief_pitchers'
            else:
                return 'starting_pitchers'
        else:
            return 'hitters'

    def apply_rookie_adjustments(self, player_data: Dict,
                                 historical_data: Optional[pd.DataFrame] = None) -> Dict:
        """
        Apply rookie-specific projections with uncertainty factoring.

        Args:
                player_data: Current season player data
                historical_data: Historical data for rookie validation (optional)

        Returns:
                Dict with rookie adjustment results

        Raises:
                ValueError: If required fields are missing
        """
        # Validate rookie status
        rookie_validation = self.validate_rookie_status(player_data, historical_data)

        player_name = player_data.get('Name', 'Unknown')
        current_war = player_data.get('WAR', 0.0)
        position = player_data.get('Position', 'OF')

        logger.info(f"Applying rookie adjustments for {player_name}")

        try:
            # Get protection factor
            protection_factor = self.get_rookie_protection_factor(
                current_war, position, rookie_validation
            )

            # Apply ceiling for rookie projections (prevent over-optimism)
            ceiling_applied = False
            projected_war = current_war
            if rookie_validation.get('is_qualifying_rookie', False):
                if projected_war > self.uncertainty_adjustments['rookie_ceiling']:
                    projected_war = self.uncertainty_adjustments['rookie_ceiling']
                    ceiling_applied = True
                    logger.info(
                        f"Applied rookie ceiling for {player_name}: {
                            projected_war:.2f} WAR")

            result = {
                'player_name': player_name,
                'is_qualifying_rookie': rookie_validation.get(
                    'is_qualifying_rookie',
                    False),
                'rookie_tier': self.classify_rookie_tier(current_war) if rookie_validation.get(
                    'is_qualifying_rookie',
                    False) else None,
                'original_war': current_war,
                'projected_war': projected_war,
                'protection_factor': protection_factor,
                'ceiling_applied': ceiling_applied,
                'rookie_validation': rookie_validation,
                'adjustment_type': 'rookie_pathway' if rookie_validation.get(
                    'is_qualifying_rookie',
                    False) else 'standard_pathway'}

            return result

        except Exception as e:
            logger.error(
                f"Error applying rookie adjustments for {player_name}: {
                    str(e)}", exc_info=True)
            raise

    def get_rookie_players_from_data(self,
                                     current_data: pd.DataFrame,
                                     historical_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Identify all qualifying rookies from current season data.

        Args:
                current_data: Current season player data
                historical_data: Historical data for validation (optional)

        Returns:
                DataFrame of qualifying rookie players
        """
        logger.info(f"Identifying rookies from {len(current_data)} players")

        rookie_players = []

        try:
            for _, player in current_data.iterrows():
                player_dict = player.to_dict()

                # Skip if missing required fields
                if 'Name' not in player_dict or 'WAR' not in player_dict:
                    continue

                validation = self.validate_rookie_status(player_dict, historical_data)

                if validation.get('is_qualifying_rookie', False):
                    adjustment_result = self.apply_rookie_adjustments(player_dict, historical_data)

                    rookie_info = {
                        **player_dict,
                        'rookie_tier': adjustment_result['rookie_tier'],
                        'protection_factor': adjustment_result['protection_factor'],
                        'rookie_validation': validation
                    }
                    rookie_players.append(rookie_info)

            result_df = pd.DataFrame(rookie_players) if rookie_players else pd.DataFrame()

            if not result_df.empty:
                logger.info(f"Identified {len(result_df)} qualifying rookies")
            else:
                logger.info("No qualifying rookies identified")

            return result_df

        except Exception as e:
            logger.error(f"Error identifying rookie players: {str(e)}", exc_info=True)
            return pd.DataFrame()

    def generate_rookie_protection_report(self, current_data: pd.DataFrame,
                                          historical_data: Optional[pd.DataFrame] = None) -> Dict:
        """
        Generate comprehensive report on rookie protection system.

        Args:
                current_data: Current season player data
                historical_data: Historical data for validation (optional)

        Returns:
                Dict with rookie protection analysis
        """
        logger.info("Generating rookie protection report")

        try:
            # Identify rookies
            rookies_df = self.get_rookie_players_from_data(current_data, historical_data)

            # Analyze high-WAR players for false positive check
            high_war_players = current_data[current_data['WAR'] > 3.0]

            false_positives = 0
            veteran_high_war = 0

            for _, player in high_war_players.iterrows():
                player_dict = player.to_dict()

                # Skip if missing required fields
                if 'Name' not in player_dict or 'WAR' not in player_dict:
                    continue

                validation = self.validate_rookie_status(player_dict, historical_data)

                if validation.get('is_qualifying_rookie', False):
                    # This is a rookie
                    continue
                else:
                    # This is a veteran with high WAR
                    veteran_high_war += 1

            # Calculate summary statistics
            total_rookies = len(rookies_df)
            elite_rookies = len(rookies_df[rookies_df['WAR'] >= 4.0]) if total_rookies > 0 else 0
            good_rookies = len(rookies_df[(rookies_df['WAR'] >= 3.0) & (
                rookies_df['WAR'] < 4.0)]) if total_rookies > 0 else 0

            false_positive_rate = false_positives / \
                len(high_war_players) if len(high_war_players) > 0 else 0.0

            report = {
                'total_qualifying_rookies': total_rookies,
                'elite_rookies': elite_rookies,
                'good_rookies': good_rookies,
                'total_high_war_players': len(high_war_players),
                'veteran_high_war_players': veteran_high_war,
                'false_positive_count': false_positives,
                'false_positive_rate': false_positive_rate,
                'rookie_details': rookies_df.to_dict('records') if total_rookies > 0 else [],
                'thresholds_used': self.rookie_thresholds,
                'protection_matrix': self.rookie_protection_matrix
            }

            logger.info(f"Rookie protection report complete: {total_rookies} rookies identified")

            return report

        except Exception as e:
            logger.error(f"Error generating rookie protection report: {str(e)}", exc_info=True)
            return {
                'error': str(e),
                'total_qualifying_rookies': 0,
                'elite_rookies': 0,
                'good_rookies': 0
            }
