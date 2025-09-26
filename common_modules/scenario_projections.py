"""
Scenario Projections Module - Performance Scenario Modeling
Calculates 5 projection scenarios: 100%, 75%, 50%, 25%, and career average regression
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import sys
import os

# Import WARP calculator and game progress calculator
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class ScenarioProjector:
    """
    Generates multiple performance scenarios for end-of-season projections

    Scenarios:
    1. 100% Performance: Current pace maintained
    2. 75% Performance: 75% of current production rate
    3. 50% Performance: 50% of current production rate
    4. 25% Performance: 25% of current production rate
    5. Career Average: Regression to weighted career average + expected stats
    """

    def __init__(self, season_length=162):
        self.season_length = season_length
        self.scenario_percentages = [1.0, 0.75, 0.50, 0.25]
        self.career_regression_weights = {
            'expected_stats': 0.6,  # Weight for expected stats
            'career_average': 0.4   # Weight for personal career average
        }

    def calculate_all_scenarios(self, current_stats: Dict, games_played: int,
                              career_stats: Optional[Dict] = None,
                              expected_stats: Optional[Dict] = None,
                              player_type: str = 'hitter') -> Dict[str, Dict]:
        """
        Calculate all 5 projection scenarios for a player

        Args:
            current_stats: Current season statistics
            games_played: Games played so far
            career_stats: Historical career averages (optional)
            expected_stats: Expected statistics from Statcast (optional)
            player_type: 'hitter' or 'pitcher'

        Returns:
            Dict with scenario names as keys and projected stats as values
        """
        if games_played >= self.season_length:
            # Season complete, return current stats for all scenarios
            return {
                '100%': current_stats.copy(),
                '75%': current_stats.copy(),
                '50%': current_stats.copy(),
                '25%': current_stats.copy(),
                'career_avg': current_stats.copy()
            }

        games_remaining = self.season_length - games_played
        scenarios = {}

        # Calculate rate-based scenarios (100%, 75%, 50%, 25%)
        for i, percentage in enumerate(self.scenario_percentages):
            scenario_name = f"{int(percentage * 100)}%"
            scenarios[scenario_name] = self._calculate_rate_scenario(
                current_stats, games_played, games_remaining, percentage, player_type
            )

        # Calculate career average regression scenario
        scenarios['career_avg'] = self._calculate_career_regression(
            current_stats, games_played, games_remaining,
            career_stats, expected_stats, player_type
        )

        return scenarios

    def _calculate_rate_scenario(self, current_stats: Dict, games_played: int,
                                games_remaining: int, rate_percentage: float,
                                player_type: str) -> Dict:
        """Calculate projection based on current rate * percentage"""
        projected_stats = current_stats.copy()

        if games_played == 0:
            return projected_stats

        # Calculate per-game rates for key statistics
        if player_type == 'hitter':
            rate_stats = ['HR', 'RBI', 'R', 'SB', 'H', 'BB', 'SO']
            ratio_stats = ['AVG', 'OBP', 'SLG', 'OPS']
        else:  # pitcher
            rate_stats = ['W', 'L', 'SV', 'SO', 'BB', 'H', 'ER', 'HR']
            ratio_stats = ['ERA', 'WHIP', 'K/9', 'BB/9', 'HR/9']

        # Project rate-based stats
        for stat in rate_stats:
            if stat in current_stats and current_stats[stat] is not None:
                current_total = current_stats[stat]
                per_game_rate = current_total / games_played
                projected_additional = per_game_rate * games_remaining * rate_percentage
                projected_stats[stat] = current_total + projected_additional

        # Handle ratio stats (these require more complex calculations)
        projected_stats = self._adjust_ratio_stats(
            projected_stats, current_stats, games_played, games_remaining,
            rate_percentage, player_type
        )

        return projected_stats

    def _adjust_ratio_stats(self, projected_stats: Dict, current_stats: Dict,
                           games_played: int, games_remaining: int,
                           rate_percentage: float, player_type: str) -> Dict:
        """Adjust ratio-based statistics for projections"""

        if player_type == 'hitter':
            # Calculate projected plate appearances
            current_pa = current_stats.get('PA', games_played * 4.0)  # Estimate 4 PA per game
            pa_per_game = current_pa / games_played if games_played > 0 else 4.0
            projected_additional_pa = pa_per_game * games_remaining * rate_percentage
            projected_total_pa = current_pa + projected_additional_pa

            # Adjust batting average, OBP, SLG based on projected hits and PA
            if 'H' in projected_stats and projected_total_pa > 0:
                projected_stats['AVG'] = projected_stats['H'] / projected_total_pa

            if 'H' in projected_stats and 'BB' in projected_stats and projected_total_pa > 0:
                projected_obp = (projected_stats['H'] + projected_stats['BB']) / projected_total_pa
                projected_stats['OBP'] = projected_obp

            # Project OPS if components are available
            if 'OBP' in projected_stats and 'SLG' in current_stats:
                # Estimate SLG based on power stats projection
                if 'HR' in projected_stats and projected_total_pa > 0:
                    # Simplified SLG estimation
                    singles = max(0, projected_stats['H'] - projected_stats.get('2B', 0) -
                                projected_stats.get('3B', 0) - projected_stats['HR'])
                    doubles = projected_stats.get('2B', current_stats.get('2B', 0))
                    triples = projected_stats.get('3B', current_stats.get('3B', 0))
                    total_bases = singles + 2*doubles + 3*triples + 4*projected_stats['HR']
                    projected_stats['SLG'] = total_bases / projected_total_pa

                projected_stats['OPS'] = projected_stats['OBP'] + projected_stats.get('SLG', 0)

        else:  # pitcher
            # Calculate projected innings pitched
            current_ip = current_stats.get('IP', games_played * 6.0)  # Estimate 6 IP per game
            ip_per_game = current_ip / games_played if games_played > 0 else 6.0
            projected_additional_ip = ip_per_game * games_remaining * rate_percentage
            projected_total_ip = current_ip + projected_additional_ip

            # Adjust ERA
            if 'ER' in projected_stats and projected_total_ip > 0:
                projected_stats['ERA'] = (projected_stats['ER'] * 9) / projected_total_ip

            # Adjust WHIP
            if 'H' in projected_stats and 'BB' in projected_stats and projected_total_ip > 0:
                projected_stats['WHIP'] = (projected_stats['H'] + projected_stats['BB']) / projected_total_ip

            # Adjust rate stats per 9 innings
            for stat_per_9 in ['K/9', 'BB/9', 'HR/9']:
                base_stat = stat_per_9.split('/')[0]
                if base_stat in projected_stats and projected_total_ip > 0:
                    if base_stat == 'K':
                        base_stat = 'SO'  # Handle strikeouts
                    projected_stats[stat_per_9] = (projected_stats[base_stat] * 9) / projected_total_ip

        return projected_stats

    def _calculate_career_regression(self, current_stats: Dict, games_played: int,
                                   games_remaining: int, career_stats: Optional[Dict],
                                   expected_stats: Optional[Dict], player_type: str) -> Dict:
        """
        Calculate career regression scenario using weighted blend

        Regression formula: 60% expected stats + 40% career average
        """
        # Start with current stats
        projected_stats = current_stats.copy()

        if games_played == 0 or games_remaining == 0:
            return projected_stats

        # Determine regression target for key stats
        regression_targets = self._calculate_regression_targets(
            career_stats, expected_stats, player_type
        )

        if not regression_targets:
            # No regression data available, use 50% scenario as fallback
            return self._calculate_rate_scenario(
                current_stats, games_played, games_remaining, 0.5, player_type
            )

        # Calculate per-game rates for remaining games based on regression targets
        key_stats = self._get_key_stats_for_regression(player_type)

        for stat in key_stats:
            if stat in regression_targets and stat in current_stats:
                current_value = current_stats[stat]
                target_rate = regression_targets[stat]

                if player_type == 'hitter':
                    # For hitters, calculate based on projected PA
                    current_pa = current_stats.get('PA', games_played * 4.0)
                    pa_per_game = current_pa / games_played
                    additional_pa = pa_per_game * games_remaining

                    if stat in ['AVG', 'OBP', 'SLG', 'OPS']:
                        # Ratio stats - blend current performance with target
                        weight_current = games_played / (games_played + games_remaining)
                        weight_target = games_remaining / (games_played + games_remaining)
                        projected_stats[stat] = (weight_current * current_value +
                                               weight_target * target_rate)
                    else:
                        # Counting stats
                        additional_production = target_rate * additional_pa
                        projected_stats[stat] = current_value + additional_production

                else:  # pitcher
                    # For pitchers, calculate based on projected IP
                    current_ip = current_stats.get('IP', games_played * 6.0)
                    ip_per_game = current_ip / games_played
                    additional_ip = ip_per_game * games_remaining

                    if stat in ['ERA', 'WHIP', 'K/9', 'BB/9', 'HR/9']:
                        # Ratio stats
                        weight_current = current_ip / (current_ip + additional_ip)
                        weight_target = additional_ip / (current_ip + additional_ip)
                        projected_stats[stat] = (weight_current * current_value +
                                               weight_target * target_rate)
                    else:
                        # Counting stats based on IP
                        if stat == 'SO':
                            additional_ks = (target_rate / 9) * additional_ip
                        elif stat == 'BB':
                            additional_bb = (target_rate / 9) * additional_ip
                        elif stat == 'HR':
                            additional_hr = (target_rate / 9) * additional_ip
                        else:
                            additional_production = target_rate * additional_ip

                        projected_stats[stat] = current_value + locals().get(
                            f'additional_{stat.lower()}', additional_production
                        )

        return projected_stats

    def _calculate_regression_targets(self, career_stats: Optional[Dict],
                                    expected_stats: Optional[Dict],
                                    player_type: str) -> Dict:
        """Calculate regression targets using weighted blend of career and expected stats"""
        targets = {}

        if not career_stats and not expected_stats:
            return targets

        key_stats = self._get_key_stats_for_regression(player_type)

        for stat in key_stats:
            career_value = career_stats.get(stat) if career_stats else None
            expected_value = expected_stats.get(stat) if expected_stats else None

            if expected_value is not None and career_value is not None:
                # Use weighted blend
                target_value = (self.career_regression_weights['expected_stats'] * expected_value +
                              self.career_regression_weights['career_average'] * career_value)
                targets[stat] = target_value
            elif expected_value is not None:
                # Use expected stats only
                targets[stat] = expected_value
            elif career_value is not None:
                # Use career average only
                targets[stat] = career_value

        return targets

    def _get_key_stats_for_regression(self, player_type: str) -> List[str]:
        """Get key statistics to use for regression calculations"""
        if player_type == 'hitter':
            return ['AVG', 'OBP', 'SLG', 'HR', 'RBI', 'SB', 'BB', 'SO']
        else:  # pitcher
            return ['ERA', 'WHIP', 'K/9', 'BB/9', 'HR/9', 'SO', 'BB', 'HR']

    def calculate_war_warp_scenarios(self, stat_scenarios: Dict[str, Dict],
                                   player_type: str, warp_calculator=None) -> Dict[str, Dict]:
        """
        Convert statistical scenarios to WAR/WARP projections

        Args:
            stat_scenarios: Dictionary of statistical projections by scenario
            player_type: 'hitter' or 'pitcher'
            warp_calculator: WARPCalculator instance for WARP calculation

        Returns:
            Dictionary with WAR/WARP projections for each scenario
        """
        war_warp_scenarios = {}

        for scenario_name, projected_stats in stat_scenarios.items():
            scenario_result = {
                'projected_stats': projected_stats,
                'war': 0.0,
                'warp': 0.0
            }

            # Calculate WARP using trained models if calculator provided
            if warp_calculator:
                try:
                    warp_result = warp_calculator.calculate_war_warp_ensemble(
                        projected_stats, player_type
                    )
                    scenario_result['war'] = warp_result['war']
                    scenario_result['warp'] = warp_result['warp']
                except Exception as e:
                    print(f"Warning: WARP calculation failed for {scenario_name}: {e}")

            war_warp_scenarios[scenario_name] = scenario_result

        return war_warp_scenarios

    def get_scenario_summary(self, scenarios: Dict[str, Dict]) -> pd.DataFrame:
        """
        Generate summary DataFrame of all scenarios

        Args:
            scenarios: Dictionary of scenario projections

        Returns:
            DataFrame with scenario comparisons
        """
        summary_data = []

        for scenario_name, scenario_data in scenarios.items():
            row = {
                'scenario': scenario_name,
                'war': scenario_data.get('war', 0),
                'warp': scenario_data.get('warp', 0)
            }

            # Add key projected stats
            if 'projected_stats' in scenario_data:
                stats = scenario_data['projected_stats']
                row.update({
                    'HR': stats.get('HR', 0),
                    'RBI': stats.get('RBI', 0),
                    'AVG': stats.get('AVG', 0.250),
                    'OPS': stats.get('OPS', 0.700)
                })

            summary_data.append(row)

        return pd.DataFrame(summary_data)


def project_player_scenarios(current_stats: Dict, games_played: int,
                            career_stats: Optional[Dict] = None,
                            expected_stats: Optional[Dict] = None,
                            player_type: str = 'hitter',
                            warp_calculator=None) -> Dict[str, Dict]:
    """
    Convenience function to generate all scenarios for a player

    Args:
        current_stats: Current season statistics
        games_played: Games played so far
        career_stats: Career averages (optional)
        expected_stats: Expected statistics (optional)
        player_type: 'hitter' or 'pitcher'
        warp_calculator: WARPCalculator for WARP projections

    Returns:
        Dictionary with all scenario projections including WAR/WARP
    """
    projector = ScenarioProjector()

    # Calculate statistical scenarios
    stat_scenarios = projector.calculate_all_scenarios(
        current_stats, games_played, career_stats, expected_stats, player_type
    )

    # Convert to WAR/WARP scenarios
    war_warp_scenarios = projector.calculate_war_warp_scenarios(
        stat_scenarios, player_type, warp_calculator
    )

    return war_warp_scenarios


def validate_scenario_projections(test_data: Dict, actual_outcomes: Dict) -> Dict:
    """
    Validate scenario projection accuracy against actual outcomes

    Args:
        test_data: Historical mid-season data for validation
        actual_outcomes: Actual end-of-season results

    Returns:
        Dictionary with validation metrics
    """
    # This would be implemented to test projection accuracy
    # using historical data where we know both mid-season and final stats
    pass