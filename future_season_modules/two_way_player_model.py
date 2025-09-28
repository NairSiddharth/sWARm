"""
Two-Way Player Modeling System
==============================

Component-based WAR modeling for two-way players with position-specific
injury handling and cross-role interaction effects.

Uses Ohtani's career as the primary case study for empirical modeling.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

@dataclass
class TwoWayPerformance:
    """Two-way player performance components."""
    pitcher_war: float
    hitter_war: float
    total_war: float
    pitcher_ip: float = 0.0
    hitter_pa: float = 0.0
    cross_role_benefits: Dict = None

class TwoWayPlayerModel:
    """
    Models two-way players using component-based approach.

    Handles:
    - Position-specific injury impacts
    - Cross-role interaction effects
    - Role-switching benefits
    - Dual aging curves
    """

    def __init__(self, performance_data: pd.DataFrame, injury_data: pd.DataFrame = None):
        """
        Initialize with performance and injury data.

        Args:
            performance_data: Complete performance dataset
            injury_data: Injury dataset for cross-role impact modeling
        """
        self.performance_data = performance_data
        self.injury_data = injury_data

        # Ohtani's empirical baselines (2022 full two-way, 2024 hitting-only)
        self.ohtani_baselines = self._establish_ohtani_baselines()

        # Cross-role interaction effects
        self.interaction_effects = self._define_interaction_effects()

    def _establish_ohtani_baselines(self) -> Dict:
        """
        Establish Ohtani's performance baselines from available data.

        Returns:
            Dictionary with Ohtani's baseline performance levels
        """
        print("Establishing Ohtani baseline performance levels...")

        # Find Ohtani in the data
        ohtani_data = self.performance_data[
            self.performance_data['Name'].str.contains('Ohtani', case=False, na=False)
        ].copy()

        if len(ohtani_data) == 0:
            print("  Ohtani data not found, using theoretical baselines")
            return self._theoretical_ohtani_baselines()

        print(f"  Found {len(ohtani_data)} Ohtani seasons")

        baselines = {}

        # 2022: Full two-way performance
        season_2022 = ohtani_data[ohtani_data['Season'] == 2022]
        if len(season_2022) > 0:
            baselines['full_two_way_2022'] = {
                'total_war': season_2022['WAR'].iloc[0] if 'WAR' in season_2022.columns else 9.6,
                'season': 2022,
                'note': 'Full two-way performance baseline'
            }

        # 2024: Hitting-only performance (post Tommy John)
        season_2024 = ohtani_data[ohtani_data['Season'] == 2024]
        if len(season_2024) > 0:
            baselines['hitting_only_2024'] = {
                'total_war': season_2024['WAR'].iloc[0] if 'WAR' in season_2024.columns else 9.0,
                'season': 2024,
                'note': 'Hitting-only performance (post pitching injury)'
            }

        # Calculate component estimates
        if 'full_two_way_2022' in baselines and 'hitting_only_2024' in baselines:
            # Estimate 2022 components based on 2024 hitting performance
            total_2022 = baselines['full_two_way_2022']['total_war']
            hitting_2024 = baselines['hitting_only_2024']['total_war']

            # Assume hitting improved in 2024 due to full focus (observed base stealing surge)
            hitting_2022_estimate = hitting_2024 * 0.85  # Estimate 15% improvement when hitting-only
            pitching_2022_estimate = total_2022 - hitting_2022_estimate

            baselines['estimated_components_2022'] = {
                'pitcher_war': pitching_2022_estimate,
                'hitter_war': hitting_2022_estimate,
                'total_war': total_2022,
                'note': 'Estimated component breakdown for 2022'
            }

        return baselines

    def _theoretical_ohtani_baselines(self) -> Dict:
        """
        Theoretical Ohtani baselines if data not available.

        Returns:
            Theoretical baseline dictionary
        """
        return {
            'full_two_way_2022': {
                'total_war': 9.6,  # Known 2022 performance
                'season': 2022,
                'note': 'Theoretical full two-way performance'
            },
            'hitting_only_2024': {
                'total_war': 9.0,  # Known 2024 performance
                'season': 2024,
                'note': 'Theoretical hitting-only performance'
            },
            'estimated_components_2022': {
                'pitcher_war': 4.0,  # Estimated pitching component
                'hitter_war': 5.6,  # Estimated hitting component
                'total_war': 9.6,
                'note': 'Theoretical component breakdown'
            }
        }

    def _define_interaction_effects(self) -> Dict:
        """
        Define cross-role interaction effects based on Ohtani case study.

        Returns:
            Dictionary of interaction effects
        """
        return {
            'pitcher_injury_to_hitter_benefit': {
                'base_stealing_boost': 1.3,  # Observed in Ohtani 2024
                'durability_boost': 1.1,     # Less dual-role wear
                'focus_boost': 1.05,         # Full focus on hitting
                'recovery_time_benefit': 1.2  # Faster recovery without pitching load
            },

            'hitter_injury_to_pitcher_impact': {
                'mechanics_disruption': 0.95,  # Minor impact on pitching mechanics
                'conditioning_impact': 0.98,   # Reduced overall conditioning
                'focus_reduction': 0.97        # Split focus during recovery
            },

            'dual_role_fatigue': {
                'late_season_decline': 0.95,   # Extra fatigue in later months
                'injury_risk_multiplier': 1.15, # Higher injury risk
                'recovery_time_extension': 1.2  # Longer recovery from injuries
            },

            'role_switching_benefits': {
                'pitcher_rest_periods': {
                    'arm_recovery': 1.1,        # Better arm recovery between starts
                    'mental_freshness': 1.05    # Mental break from pitching focus
                },
                'hitter_conditioning': {
                    'athleticism_boost': 1.05,   # Pitching maintains athleticism
                    'competitive_edge': 1.03     # Competitive advantage from dual skills
                }
            }
        }

    def project_two_way_performance(self,
                                  current_performance: Dict,
                                  injury_history: List[Dict],
                                  years_ahead: int = 3,
                                  player_age: int = 30) -> List[TwoWayPerformance]:
        """
        Project two-way player performance with component modeling.

        Args:
            current_performance: Current pitcher and hitter WAR components
            injury_history: List of recent injuries
            years_ahead: Number of years to project
            player_age: Current player age

        Returns:
            List of TwoWayPerformance projections
        """
        print(f"Projecting two-way performance for {years_ahead} years...")

        projections = []

        # Extract current components
        current_pitcher_war = current_performance.get('pitcher_war', 0.0)
        current_hitter_war = current_performance.get('hitter_war', 0.0)

        # Analyze injury impacts
        active_injuries = self._analyze_injury_impacts(injury_history)

        for year in range(1, years_ahead + 1):
            projection_age = player_age + year

            # Base projections with aging
            pitcher_projection = self._project_pitcher_component(
                current_pitcher_war, projection_age, year, active_injuries
            )

            hitter_projection = self._project_hitter_component(
                current_hitter_war, projection_age, year, active_injuries
            )

            # Apply cross-role interactions
            adjusted_projections = self._apply_cross_role_interactions(
                pitcher_projection, hitter_projection, active_injuries, year
            )

            # Calculate cross-role benefits
            cross_benefits = self._calculate_cross_role_benefits(
                adjusted_projections, active_injuries, year
            )

            projections.append(TwoWayPerformance(
                pitcher_war=adjusted_projections['pitcher_war'],
                hitter_war=adjusted_projections['hitter_war'],
                total_war=adjusted_projections['pitcher_war'] + adjusted_projections['hitter_war'],
                cross_role_benefits=cross_benefits
            ))

        return projections

    def _analyze_injury_impacts(self, injury_history: List[Dict]) -> Dict:
        """
        Analyze current and projected injury impacts.

        Args:
            injury_history: List of injury records

        Returns:
            Dictionary of active injury effects
        """
        active_effects = {
            'pitcher_affected': False,
            'hitter_affected': False,
            'recovery_timeline': {},
            'cross_role_benefits': {}
        }

        for injury in injury_history:
            injury_type = injury.get('type', '').lower()
            affected_role = injury.get('affected_role', 'both')
            recovery_years = injury.get('recovery_years', 1.0)

            # Position-specific injury impacts
            if 'tommy john' in injury_type or 'elbow' in injury_type:
                active_effects['pitcher_affected'] = True
                active_effects['recovery_timeline']['pitcher'] = recovery_years

                # Pitching injury benefits hitting (Ohtani 2024 pattern)
                active_effects['cross_role_benefits']['hitter_boost'] = (
                    self.interaction_effects['pitcher_injury_to_hitter_benefit']
                )

            elif 'shoulder' in injury_type and affected_role == 'pitcher':
                active_effects['pitcher_affected'] = True
                active_effects['recovery_timeline']['pitcher'] = recovery_years * 0.8

            elif any(muscle in injury_type for muscle in ['hamstring', 'oblique', 'groin']):
                # Hitting-related injuries
                active_effects['hitter_affected'] = True
                active_effects['recovery_timeline']['hitter'] = recovery_years * 0.5

                # Minor impact on pitching
                if affected_role != 'hitter_only':
                    active_effects['pitcher_affected'] = True

        return active_effects

    def _project_pitcher_component(self,
                                 current_war: float,
                                 age: int,
                                 year: int,
                                 injury_effects: Dict) -> float:
        """
        Project pitcher WAR component.

        Args:
            current_war: Current pitcher WAR
            age: Projected age
            year: Year ahead (1, 2, 3)
            injury_effects: Active injury effects

        Returns:
            Projected pitcher WAR
        """
        # Base aging curve for pitchers
        age_factor = self._calculate_pitcher_aging_factor(age)

        # Injury impacts
        injury_factor = 1.0
        if injury_effects['pitcher_affected']:
            recovery_timeline = injury_effects['recovery_timeline'].get('pitcher', 1.0)

            if year <= recovery_timeline:
                # During recovery period
                recovery_progress = year / recovery_timeline
                injury_factor = min(recovery_progress, 1.0)
            else:
                # Post-recovery, may not return to full strength
                injury_factor = 0.95  # 5% permanent impact

        projected_war = current_war * age_factor * injury_factor

        return max(projected_war, 0.0)

    def _project_hitter_component(self,
                                current_war: float,
                                age: int,
                                year: int,
                                injury_effects: Dict) -> float:
        """
        Project hitter WAR component.

        Args:
            current_war: Current hitter WAR
            age: Projected age
            year: Year ahead (1, 2, 3)
            injury_effects: Active injury effects

        Returns:
            Projected hitter WAR
        """
        # Base aging curve for hitters
        age_factor = self._calculate_hitter_aging_factor(age)

        # Injury impacts
        injury_factor = 1.0
        if injury_effects['hitter_affected']:
            recovery_timeline = injury_effects['recovery_timeline'].get('hitter', 0.5)

            if year <= recovery_timeline:
                injury_factor = 0.85  # Temporary impact
            else:
                injury_factor = 0.98  # Minor permanent impact

        projected_war = current_war * age_factor * injury_factor

        return max(projected_war, 0.0)

    def _calculate_pitcher_aging_factor(self, age: int) -> float:
        """Calculate aging factor for pitching performance."""
        if age <= 26:
            return 1.0
        elif age <= 30:
            return 0.98
        elif age <= 33:
            return 0.95
        elif age <= 36:
            return 0.90
        else:
            return 0.85

    def _calculate_hitter_aging_factor(self, age: int) -> float:
        """Calculate aging factor for hitting performance."""
        if age <= 28:
            return 1.0
        elif age <= 32:
            return 0.98
        elif age <= 35:
            return 0.95
        elif age <= 38:
            return 0.90
        else:
            return 0.85

    def _apply_cross_role_interactions(self,
                                     pitcher_war: float,
                                     hitter_war: float,
                                     injury_effects: Dict,
                                     year: int) -> Dict:
        """
        Apply cross-role interaction effects.

        Args:
            pitcher_war: Base pitcher WAR projection
            hitter_war: Base hitter WAR projection
            injury_effects: Active injury effects
            year: Projection year

        Returns:
            Dictionary with adjusted projections
        """
        adjusted_pitcher = pitcher_war
        adjusted_hitter = hitter_war

        # Apply cross-role benefits from injuries
        if 'hitter_boost' in injury_effects.get('cross_role_benefits', {}):
            boost_effects = injury_effects['cross_role_benefits']['hitter_boost']

            # Base stealing and focus improvements
            focus_boost = boost_effects.get('focus_boost', 1.0)
            durability_boost = boost_effects.get('durability_boost', 1.0)

            adjusted_hitter *= focus_boost * durability_boost

        # Dual-role fatigue effects (when both roles active)
        if pitcher_war > 0 and hitter_war > 0:
            fatigue_effects = self.interaction_effects['dual_role_fatigue']
            late_season_factor = fatigue_effects['late_season_decline']

            adjusted_pitcher *= late_season_factor
            adjusted_hitter *= late_season_factor

        return {
            'pitcher_war': adjusted_pitcher,
            'hitter_war': adjusted_hitter
        }

    def _calculate_cross_role_benefits(self,
                                     projections: Dict,
                                     injury_effects: Dict,
                                     year: int) -> Dict:
        """
        Calculate additional cross-role benefits.

        Args:
            projections: Adjusted WAR projections
            injury_effects: Active injury effects
            year: Projection year

        Returns:
            Dictionary of cross-role benefits
        """
        benefits = {}

        # Base stealing boost from pitching injury (Ohtani 2024 pattern)
        if 'hitter_boost' in injury_effects.get('cross_role_benefits', {}):
            boost_effects = injury_effects['cross_role_benefits']['hitter_boost']
            base_stealing_boost = boost_effects.get('base_stealing_boost', 1.0)

            benefits['base_stealing_improvement'] = base_stealing_boost - 1.0
            benefits['overall_athleticism'] = 'enhanced'

        # Pitching rest benefits
        if projections['pitcher_war'] > 0 and projections['hitter_war'] > 0:
            benefits['pitcher_rest_advantage'] = 'maintained'
            benefits['competitive_edge'] = 'dual_threat'

        return benefits

    def get_ohtani_projection_example(self, current_age: int = 30) -> List[TwoWayPerformance]:
        """
        Generate example projection for Ohtani-type player.

        Args:
            current_age: Current age for projection

        Returns:
            Example two-way projections
        """
        # Use empirical baselines
        if 'estimated_components_2022' in self.ohtani_baselines:
            components = self.ohtani_baselines['estimated_components_2022']
            current_performance = {
                'pitcher_war': components['pitcher_war'],
                'hitter_war': components['hitter_war']
            }
        else:
            current_performance = {
                'pitcher_war': 4.0,
                'hitter_war': 5.6
            }

        # Simulate Tommy John recovery
        injury_history = [{
            'type': 'tommy_john',
            'affected_role': 'pitcher',
            'recovery_years': 1.5,
            'year': 2023
        }]

        return self.project_two_way_performance(
            current_performance,
            injury_history,
            years_ahead=3,
            player_age=current_age
        )