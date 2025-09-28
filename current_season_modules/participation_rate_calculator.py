"""
Dynamic Participation Rate Calculator

Calculates realistic remaining games projections based on:
1. Current usage patterns
2. Position/role flexibility
3. Performance-based adjustments (hybrid approach)
4. Injury recovery factors

Handles modern baseball realities like multi-positional players, elite season detection,
and position-specific usage patterns.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import os
import sys

# Add project path for imports
project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_path not in sys.path:
    sys.path.append(project_path)


class ParticipationRateCalculator:
    """
    Calculate dynamic participation rates for current season projections.

    Combines current usage patterns, role detection, performance metrics,
    and injury recovery to project realistic remaining games.
    """

    def __init__(self, estimated_team_games: int = 95):
        """
        Initialize participation rate calculator.

        Args:
            estimated_team_games: Estimated team games played to date
        """
        self.estimated_team_games = estimated_team_games
        self.season_length = 162

        # Role-based participation expectations
        self.role_expectations = {
            'everyday_player': {'base_rate': 0.90, 'max_rate': 1.0},
            'versatile_catcher': {'base_rate': 0.85, 'max_rate': 0.96},
            'primary_catcher': {'base_rate': 0.75, 'max_rate': 0.87},
            'flexible_catcher': {'base_rate': 0.70, 'max_rate': 0.85},
            'platoon_catcher': {'base_rate': 0.50, 'max_rate': 0.65},
            'regular_player': {'base_rate': 0.85, 'max_rate': 0.95},
            'platoon_player': {'base_rate': 0.60, 'max_rate': 0.75},
            'backup_player': {'base_rate': 0.40, 'max_rate': 0.60}
        }

        # Performance boost thresholds (WAR-based)
        self.performance_thresholds = {
            'elite': {'war_threshold': 4.0, 'boost_factor': 1.15},
            'very_good': {'war_threshold': 2.5, 'boost_factor': 1.08},
            'above_average': {'war_threshold': 1.5, 'boost_factor': 1.03},
            'average': {'war_threshold': 0.0, 'boost_factor': 1.0},
            'below_average': {'war_threshold': -1.0, 'boost_factor': 0.95}
        }

    def detect_player_role(self, player_data: pd.Series) -> Dict[str, Union[str, float]]:
        """
        Detect player's role based on position, games played, and usage patterns.

        Args:
            player_data: Player statistics row

        Returns:
            Dictionary with role classification and expectations
        """
        games_played = player_data.get('G', 0)
        position_string = str(player_data.get('Pos', ''))
        current_rate = games_played / self.estimated_team_games if self.estimated_team_games > 0 else 0

        # Parse multi-position designations
        positions = position_string.split('/') if '/' in position_string else [position_string]
        primary_position = positions[0].strip()
        is_multi_positional = len(positions) > 1
        has_dh_flexibility = 'DH' in positions or any(pos in ['1B', 'OF'] for pos in positions)

        # Role classification logic
        if primary_position == 'C':
            if is_multi_positional and has_dh_flexibility:
                # C/DH, C/1B types - can play near-everyday
                if current_rate >= 0.80:
                    role = 'versatile_catcher'
                elif current_rate >= 0.60:
                    role = 'flexible_catcher'
                else:
                    role = 'platoon_catcher'
            else:
                # Pure catchers - traditional rest pattern
                if current_rate >= 0.75:
                    role = 'primary_catcher'
                elif current_rate >= 0.50:
                    role = 'flexible_catcher'
                else:
                    role = 'platoon_catcher'

        elif primary_position in ['DH', '1B'] or is_multi_positional:
            # Flexible position players
            if current_rate >= 0.85:
                role = 'everyday_player'
            elif current_rate >= 0.65:
                role = 'regular_player'
            else:
                role = 'platoon_player'

        elif primary_position in ['OF', 'SS', '2B', '3B']:
            # Standard position players
            if current_rate >= 0.85:
                role = 'everyday_player'
            elif current_rate >= 0.70:
                role = 'regular_player'
            elif current_rate >= 0.45:
                role = 'platoon_player'
            else:
                role = 'backup_player'

        else:
            # Default classification
            if current_rate >= 0.80:
                role = 'regular_player'
            elif current_rate >= 0.50:
                role = 'platoon_player'
            else:
                role = 'backup_player'

        expectations = self.role_expectations.get(role, self.role_expectations['regular_player'])

        return {
            'role': role,
            'base_rate': expectations['base_rate'],
            'max_rate': expectations['max_rate'],
            'is_multi_positional': is_multi_positional,
            'has_dh_flexibility': has_dh_flexibility,
            'current_usage_rate': current_rate
        }

    def calculate_performance_boost(self, current_war: float, games_played: int) -> Dict[str, Union[float, str]]:
        """
        Calculate performance-based participation boost using hybrid approach.

        Combines WAR-based performance assessment with usage patterns.

        Args:
            current_war: Current season WAR
            games_played: Games played so far

        Returns:
            Performance boost information
        """
        # Calculate WAR pace (projected full season)
        war_pace = (current_war / games_played * 162) if games_played > 0 else 0

        # Determine performance tier
        performance_tier = 'below_average'
        for tier, config in self.performance_thresholds.items():
            if war_pace >= config['war_threshold']:
                performance_tier = tier
                break

        boost_config = self.performance_thresholds[performance_tier]
        base_boost = boost_config['boost_factor']

        # Additional boost for exceptional usage (Cal Raleigh scenario)
        usage_rate = games_played / self.estimated_team_games if self.estimated_team_games > 0 else 0

        # If usage significantly exceeds expectations, provide additional boost
        usage_boost = 1.0
        if usage_rate >= 0.95:  # Playing 95%+ of games
            usage_boost = 1.10  # 10% additional boost for exceptional usage
        elif usage_rate >= 0.85:  # Playing 85%+ of games
            usage_boost = 1.05  # 5% additional boost

        # Combined boost (capped at reasonable maximum)
        total_boost = min(base_boost * usage_boost, 1.20)  # Cap at 20% boost

        return {
            'performance_tier': performance_tier,
            'war_pace': war_pace,
            'base_boost': base_boost,
            'usage_boost': usage_boost,
            'total_boost': total_boost,
            'reasoning': f"{performance_tier} performance ({war_pace:.1f} WAR pace) + {usage_rate:.1%} usage"
        }

    def calculate_dynamic_participation_rate(self,
                                           player_data: pd.Series,
                                           current_war: float = None,
                                           injury_adjustment: float = 1.0) -> Dict[str, Union[int, float, str]]:
        """
        Calculate dynamic participation rate using hybrid approach.

        Args:
            player_data: Player statistics row
            current_war: Current season WAR (optional)
            injury_adjustment: Injury recovery factor (default 1.0)

        Returns:
            Participation rate calculation results
        """
        games_played = player_data.get('G', 0)

        # Step 1: Detect player role
        role_info = self.detect_player_role(player_data)

        # Step 2: Calculate base participation rate (blend current with expectations)
        current_rate = role_info['current_usage_rate']
        expected_rate = role_info['base_rate']

        # Blend: 70% current usage, 30% role expectation
        blended_rate = 0.7 * current_rate + 0.3 * expected_rate

        # Step 3: Apply performance boost if WAR available
        performance_boost = 1.0
        performance_info = None
        if current_war is not None:
            performance_info = self.calculate_performance_boost(current_war, games_played)
            performance_boost = performance_info['total_boost']

        # Step 4: Apply performance boost to blended rate
        boosted_rate = blended_rate * performance_boost

        # Step 5: Apply role-based cap
        capped_rate = min(boosted_rate, role_info['max_rate'])

        # Step 6: Apply injury adjustment
        final_rate = capped_rate * injury_adjustment

        # Step 7: Calculate remaining games
        remaining_team_games = self.season_length - self.estimated_team_games
        projected_remaining = int(remaining_team_games * final_rate)

        # Calculate projected totals
        projected_total_games = games_played + projected_remaining

        return {
            'games_remaining': projected_remaining,
            'projected_total_games': projected_total_games,
            'participation_rate': final_rate,
            'role_classification': role_info['role'],
            'current_usage_rate': current_rate,
            'expected_rate': expected_rate,
            'blended_rate': blended_rate,
            'performance_boost': performance_boost,
            'injury_adjustment': injury_adjustment,
            'performance_info': performance_info,
            'method': f"Hybrid: {role_info['role']} ({final_rate:.1%} rate, {performance_boost:.2f}x boost)"
        }


def calculate_participation_adjusted_games(player_data: pd.Series,
                                         current_war: float = None,
                                         injury_adjustment: float = 1.0,
                                         estimated_team_games: int = 95) -> Dict[str, Union[int, float, str]]:
    """
    Convenience function for calculating participation-adjusted games.

    Args:
        player_data: Player statistics row
        current_war: Current season WAR (optional)
        injury_adjustment: Injury recovery factor (default 1.0)
        estimated_team_games: Estimated team games played to date

    Returns:
        Participation rate calculation results
    """
    calculator = ParticipationRateCalculator(estimated_team_games)
    return calculator.calculate_dynamic_participation_rate(
        player_data, current_war, injury_adjustment
    )


# Example usage and testing
if __name__ == "__main__":
    # Test with sample data
    sample_player = pd.Series({
        'Name': 'Cal Raleigh',
        'Pos': 'C/DH',
        'G': 95,  # Mid-season estimate
    })

    # Test with high WAR (elite performance)
    result = calculate_participation_adjusted_games(
        sample_player,
        current_war=3.5,  # Elite pace
        injury_adjustment=1.0
    )

    print("Cal Raleigh Projection Example:")
    print(f"Role: {result['role_classification']}")
    print(f"Current usage: {result['current_usage_rate']:.1%}")
    print(f"Final rate: {result['participation_rate']:.1%}")
    print(f"Remaining games: {result['games_remaining']}")
    print(f"Projected total: {result['projected_total_games']}")
    print(f"Method: {result['method']}")