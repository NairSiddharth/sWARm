"""
Two-Way Player Elite Protection Module
=======================================

Enhanced elite protection for two-way players using component-based approach.
Provides separate protection for pitching and hitting components while
considering cross-role interaction effects and injury-aware adjustments.

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


class TwoWayEliteProtection(ElitePlayerAdjuster):
    """
    Enhanced elite protection for two-way players using component-based approach.

    Provides separate protection for pitching and hitting components while
    considering cross-role interaction effects and injury-aware adjustments.
    """

    def __init__(self, use_enhanced_system: bool = True):
        """
        Initialize with base elite adjuster functionality.

        Args:
                use_enhanced_system: Use enhanced WAR tier system (recommended)
        """
        super().__init__(use_enhanced_system=use_enhanced_system)

        # Component interaction factors
        self.interaction_factors = {
            'dual_role_bonus': 1.1,      # Slight bonus for maintaining both roles
            'injury_compensation': 1.15,  # Extra protection during role-specific injuries
            'focus_shift_penalty': 0.95   # Minor penalty when shifting focus
        }

        logger.info(
            "Initialized TwoWayEliteProtection with enhanced system: %s",
            use_enhanced_system)

    def get_two_way_protection(self,
                               pitcher_war: float,
                               hitter_war: float,
                               injury_status: Optional[Dict] = None) -> Dict:
        """
        Calculate component-based protection factors.

        Args:
                pitcher_war: Pitcher WAR component
                hitter_war: Hitter WAR component
                injury_status: Current injury effects (optional)

        Returns:
                Dictionary with component protection factors

        Example:
                >>> protection = get_two_way_protection(3.5, 2.5, {'pitcher_affected': True})
                >>> print(protection['pitcher_protection'])
                0.35
        """
        try:
            # Base component protections using the enhanced system
            pitcher_protection = self.get_protection_factor(pitcher_war, 'P')
            hitter_protection = self.get_protection_factor(hitter_war, 'DH')

            logger.debug(
                f"Base protection - Pitcher: {pitcher_protection:.3f}, Hitter: {hitter_protection:.3f}")

            # Apply interaction effects
            if injury_status:
                if injury_status.get('pitcher_affected'):
                    # Pitching injury: protect pitcher more, boost hitter slightly
                    pitcher_protection *= self.interaction_factors['injury_compensation']
                    hitter_protection *= 0.95  # Slight hitter boost from focus shift
                    logger.info("Applied pitcher injury compensation")

                elif injury_status.get('hitter_affected'):
                    # Hitting injury: protect hitter more, pitcher slightly less
                    hitter_protection *= self.interaction_factors['injury_compensation']
                    pitcher_protection *= self.interaction_factors['focus_shift_penalty']
                    logger.info("Applied hitter injury compensation")

            # Dual-role bonus (when both components active)
            if pitcher_war > 1.0 and hitter_war > 1.0:
                dual_bonus = self.interaction_factors['dual_role_bonus']
                pitcher_protection *= dual_bonus
                hitter_protection *= dual_bonus
                logger.info("Applied dual-role bonus")

            return {
                'pitcher_protection': pitcher_protection,
                'hitter_protection': hitter_protection,
                'combined_total': (pitcher_protection + hitter_protection) / 2,
                'interaction_effects': injury_status or {}
            }

        except Exception as e:
            logger.error(f"Error calculating two-way protection: {str(e)}", exc_info=True)
            # Return default protection on error
            return {
                'pitcher_protection': 1.0,
                'hitter_protection': 1.0,
                'combined_total': 1.0,
                'interaction_effects': {}
            }

    def apply_two_way_adjustment(self,
                                 pitcher_war: float,
                                 hitter_war: float,
                                 injury_status: Optional[Dict] = None,
                                 player_name: str = 'Two-Way Player') -> Dict:
        """
        Apply two-way player specific adjustments.

        Args:
                pitcher_war: Current pitcher WAR
                hitter_war: Current hitter WAR
                injury_status: Current injury effects (optional)
                player_name: Player name for logging

        Returns:
                Dictionary with two-way adjustment results

        Raises:
                ValueError: If WAR values are negative
        """
        # Validate inputs
        if pitcher_war < 0 or hitter_war < 0:
            raise ValueError("WAR values cannot be negative")

        logger.info(f"Applying two-way adjustment for {player_name}")

        try:
            # Get component protection factors
            protection_factors = self.get_two_way_protection(pitcher_war, hitter_war, injury_status)

            # Calculate adjusted components
            # Protection reduces regression, so lower protection = more value retention
            adjusted_pitcher_war = pitcher_war * \
                (1 + (1 - protection_factors['pitcher_protection']) * 0.5)
            adjusted_hitter_war = hitter_war * \
                (1 + (1 - protection_factors['hitter_protection']) * 0.5)

            # Total adjusted WAR
            total_adjusted = adjusted_pitcher_war + adjusted_hitter_war
            original_total = pitcher_war + hitter_war

            result = {
                'player_name': player_name,
                'original_pitcher_war': pitcher_war,
                'original_hitter_war': hitter_war,
                'original_total_war': original_total,
                'adjusted_pitcher_war': adjusted_pitcher_war,
                'adjusted_hitter_war': adjusted_hitter_war,
                'adjusted_total_war': total_adjusted,
                'pitcher_adjustment': adjusted_pitcher_war - pitcher_war,
                'hitter_adjustment': adjusted_hitter_war - hitter_war,
                'total_adjustment': total_adjusted - original_total,
                'protection_factors': protection_factors
            }

            logger.info(f"Two-way adjustment complete for {player_name}: "
                        f"Total adjustment = {result['total_adjustment']:.2f} WAR")

            return result

        except Exception as e:
            logger.error(
                f"Error applying two-way adjustment for {player_name}: {str(e)}", exc_info=True)
            raise

    def detect_two_way_player(self, player_data: Dict) -> bool:
        """
        Detect if a player qualifies as a two-way player.

        Args:
                player_data: Dictionary with player stats

        Returns:
                True if player qualifies as two-way

        Example:
                >>> data = {'IP': 120.0, 'PA': 450}
                >>> is_two_way = detect_two_way_player(data)
                >>> print(is_two_way)
                True
        """
        try:
            # Check for meaningful contribution in both roles
            pitcher_ip = player_data.get('IP', 0.0)
            hitter_pa = player_data.get('PA', 0.0)

            # Two-way qualification thresholds (based on Ohtani)
            MIN_PITCHER_IP = 50.0    # Meaningful pitching contribution
            MIN_HITTER_PA = 200.0    # Meaningful hitting contribution

            is_two_way = pitcher_ip >= MIN_PITCHER_IP and hitter_pa >= MIN_HITTER_PA

            if is_two_way:
                logger.info(f"Player qualifies as two-way: IP={pitcher_ip:.1f}, PA={hitter_pa:.0f}")

            return is_two_way

        except Exception as e:
            logger.error(f"Error detecting two-way player: {str(e)}", exc_info=True)
            return False

    def process_two_way_players(self, player_df: pd.DataFrame) -> pd.DataFrame:
        """
        Process all two-way players in a DataFrame.

        Args:
                player_df: DataFrame containing player data

        Returns:
                DataFrame with two-way adjustments applied

        Raises:
                ValueError: If required columns are missing
        """
        required_columns = ['IP', 'PA', 'Name']
        missing_columns = set(required_columns) - set(player_df.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        logger.info(f"Processing {len(player_df)} players for two-way adjustments")

        try:
            result = player_df.copy()
            two_way_count = 0

            # Add two-way detection column
            result['is_two_way'] = False

            for idx, row in result.iterrows():
                player_dict = row.to_dict()

                if self.detect_two_way_player(player_dict):
                    result.at[idx, 'is_two_way'] = True
                    two_way_count += 1

                    # Apply adjustments if WAR components are available
                    if 'pitcher_war' in player_dict and 'hitter_war' in player_dict:
                        adjustment = self.apply_two_way_adjustment(
                            pitcher_war=player_dict['pitcher_war'],
                            hitter_war=player_dict['hitter_war'],
                            player_name=player_dict.get('Name', 'Unknown')
                        )

                        # Update the dataframe with adjustments
                        result.at[idx, 'adjusted_pitcher_war'] = adjustment['adjusted_pitcher_war']
                        result.at[idx, 'adjusted_hitter_war'] = adjustment['adjusted_hitter_war']
                        result.at[idx, 'adjusted_total_war'] = adjustment['adjusted_total_war']
                        result.at[idx, 'war_adjustment'] = adjustment['total_adjustment']

            logger.info(f"Identified {two_way_count} two-way players")
            return result

        except Exception as e:
            logger.error(f"Error processing two-way players: {str(e)}", exc_info=True)
            raise
