"""
Player Profile Classification System
====================================

Creates generalizable player profiles for enhanced projection modeling:
- Late bloomers (Judge-type)
- Consistent elites (Soto-type)
- Injury-compromised legends (Trout-type)
- Two-way players (Ohtani-type)

Uses historical performance patterns rather than hardcoded player names.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Set
from dataclasses import dataclass

@dataclass
class PlayerProfile:
    """Player profile definition with criteria and adjustments."""
    name: str
    criteria: Dict
    aging_adjustment: float = 1.0
    regression_protection: float = 1.0
    description: str = ""

class PlayerProfileClassifier:
    """
    Classifies players into generalizable profiles based on performance history.

    Provides profile-based adjustments for projection modeling without
    hardcoding specific player names.
    """

    def __init__(self, performance_data: pd.DataFrame):
        """
        Initialize with historical performance data.

        Args:
            performance_data: Complete performance dataset (2016-2024)
        """
        self.performance_data = performance_data
        self.player_histories = self._build_player_histories()

        # Define generalizable profiles
        self.profiles = self._define_profiles()

    def _build_player_histories(self) -> Dict[int, pd.DataFrame]:
        """
        Build individual player history dictionaries.

        Returns:
            Dictionary mapping player_id to their complete history
        """
        print("Building player histories for profile classification...")

        histories = {}
        for player_id in self.performance_data['mlbid'].unique():
            if pd.isna(player_id):
                continue

            player_data = self.performance_data[
                self.performance_data['mlbid'] == player_id
            ].sort_values('Season').copy()

            if len(player_data) > 0:
                histories[int(player_id)] = player_data

        print(f"Built histories for {len(histories)} unique players")
        return histories

    def _define_profiles(self) -> Dict[str, PlayerProfile]:
        """
        Define generalizable player profiles with data-driven criteria.

        Returns:
            Dictionary of profile definitions
        """
        return {
            'late_bloomer': PlayerProfile(
                name='late_bloomer',
                criteria={
                    'debut_age_min': 25,  # Debuted at 25+ (Judge was 25)
                    'peak_war_threshold': 4.0,  # Achieved elite performance
                    'peak_within_years': 4,  # Peak within first 4 MLB years
                    'min_elite_seasons': 2  # At least 2 elite seasons
                },
                aging_adjustment=0.85,  # Age more slowly (less accumulated wear)
                regression_protection=0.7,  # Extra protection from regression
                description="Late-debut players with elite peak performance"
            ),

            'consistent_elite': PlayerProfile(
                name='consistent_elite',
                criteria={
                    'min_elite_seasons': 3,  # 3+ seasons of 4+ WAR
                    'war_threshold': 4.0,
                    'max_down_years': 1,  # Allow 1 off year in elite period
                    'min_career_length': 4,  # At least 4 years to establish pattern
                    'young_elite_bonus_age': 26  # Extra credit if young
                },
                aging_adjustment=0.9,  # Slower aging for proven elites
                regression_protection=0.6,  # Strong protection (proven track record)
                description="Multi-year consistent elite performers"
            ),

            'injury_compromised_legend': PlayerProfile(
                name='injury_compromised_legend',
                criteria={
                    'historical_peak': 6.0,  # Had elite peak (6+ WAR season)
                    'recent_decline_threshold': 0.4,  # Current WAR < 40% of peak
                    'peak_within_last_years': 6,  # Peak was within last 6 years
                    'age_when_declining': 30  # Started declining at 30+
                },
                aging_adjustment=1.2,  # Age faster (injury effects)
                regression_protection=1.1,  # Less protection (injury reality)
                description="Former elite players with injury-related decline"
            ),

            'two_way_player': PlayerProfile(
                name='two_way_player',
                criteria={
                    'pitcher_seasons': 1,  # Pitched in at least 1 season
                    'hitter_seasons': 1,   # Hit in at least 1 season
                    'min_pitcher_ip': 50,  # Meaningful pitching contribution
                    'min_hitter_pa': 200   # Meaningful hitting contribution
                },
                aging_adjustment=1.1,  # Faster aging (dual role wear)
                regression_protection=0.8,  # Moderate protection
                description="Players with significant two-way contributions"
            ),

            'power_specialist': PlayerProfile(
                name='power_specialist',
                criteria={
                    'min_hr_seasons': 3,  # 3+ seasons with 25+ HR
                    'hr_threshold': 25,
                    'age_when_established': 28,  # Established by age 28
                    'position_eligibility': ['1B', '3B', 'OF', 'DH']  # Power positions
                },
                aging_adjustment=1.05,  # Slightly faster aging (power decline)
                regression_protection=0.8,  # Moderate protection
                description="Established power hitters with proven track record"
            )
        }

    def classify_player(self, player_id: int, use_prorated_2020: bool = True) -> List[str]:
        """
        Classify a player into applicable profiles.

        Args:
            player_id: Player's MLB ID
            use_prorated_2020: Use prorated 2020 WAR for classification

        Returns:
            List of applicable profile names
        """
        if player_id not in self.player_histories:
            return []

        history = self.player_histories[player_id].copy()

        # Use prorated 2020 WAR if available and requested
        if use_prorated_2020 and 'prorated_WAR_2020' in history.columns:
            mask_2020 = history['Season'] == 2020
            history.loc[mask_2020, 'WAR'] = history.loc[mask_2020, 'prorated_WAR_2020']

        profiles = []

        # Check each profile
        for profile_name, profile_def in self.profiles.items():
            if self._meets_profile_criteria(history, profile_def):
                profiles.append(profile_name)

        return profiles

    def _meets_profile_criteria(self,
                               player_history: pd.DataFrame,
                               profile: PlayerProfile) -> bool:
        """
        Check if player meets specific profile criteria.

        Args:
            player_history: Player's performance history
            profile: Profile definition to check

        Returns:
            True if player meets profile criteria
        """
        criteria = profile.criteria

        # Late bloomer checks
        if profile.name == 'late_bloomer':
            return self._check_late_bloomer(player_history, criteria)

        # Consistent elite checks
        elif profile.name == 'consistent_elite':
            return self._check_consistent_elite(player_history, criteria)

        # Injury-compromised legend checks
        elif profile.name == 'injury_compromised_legend':
            return self._check_injury_compromised_legend(player_history, criteria)

        # Two-way player checks
        elif profile.name == 'two_way_player':
            return self._check_two_way_player(player_history, criteria)

        # Power specialist checks
        elif profile.name == 'power_specialist':
            return self._check_power_specialist(player_history, criteria)

        return False

    def _check_late_bloomer(self, history: pd.DataFrame, criteria: Dict) -> bool:
        """Check late bloomer criteria."""
        if len(history) == 0:
            return False

        # Check debut age
        first_season = history.iloc[0]
        debut_age = first_season.get('Age', 0)

        if debut_age < criteria['debut_age_min']:
            return False

        # Check for elite peak within first few years
        early_career = history.head(criteria['peak_within_years'])
        peak_war = early_career['WAR'].max()

        if peak_war < criteria['peak_war_threshold']:
            return False

        # Check for multiple elite seasons
        elite_seasons = (history['WAR'] >= criteria['peak_war_threshold']).sum()

        return elite_seasons >= criteria['min_elite_seasons']

    def _check_consistent_elite(self, history: pd.DataFrame, criteria: Dict) -> bool:
        """Check consistent elite criteria."""
        if len(history) < criteria['min_career_length']:
            return False

        # Count elite seasons
        elite_seasons = (history['WAR'] >= criteria['war_threshold']).sum()

        if elite_seasons < criteria['min_elite_seasons']:
            return False

        # Check for consistency (limited down years during elite period)
        # Find elite period
        elite_mask = history['WAR'] >= criteria['war_threshold']
        if elite_mask.sum() == 0:
            return False

        elite_years = history[elite_mask]['Season'].tolist()
        if len(elite_years) < criteria['min_elite_seasons']:
            return False

        # Check if elite seasons are reasonably consecutive
        elite_span = max(elite_years) - min(elite_years) + 1
        down_years_in_span = elite_span - len(elite_years)

        # Young elite bonus
        youngest_elite_age = history[elite_mask]['Age'].min()
        if youngest_elite_age <= criteria['young_elite_bonus_age']:
            return down_years_in_span <= criteria['max_down_years'] + 1

        return down_years_in_span <= criteria['max_down_years']

    def _check_injury_compromised_legend(self, history: pd.DataFrame, criteria: Dict) -> bool:
        """Check injury-compromised legend criteria."""
        if len(history) == 0:
            return False

        # Check for historical peak
        peak_war = history['WAR'].max()
        if peak_war < criteria['historical_peak']:
            return False

        # Check if peak was recent enough
        peak_season = history[history['WAR'] == peak_war]['Season'].iloc[0]
        most_recent_season = history['Season'].max()

        if most_recent_season - peak_season > criteria['peak_within_last_years']:
            return False

        # Check for significant recent decline
        recent_seasons = history[history['Season'] >= most_recent_season - 1]
        if len(recent_seasons) == 0:
            return False

        recent_war = recent_seasons['WAR'].mean()
        decline_ratio = recent_war / peak_war

        if decline_ratio > criteria['recent_decline_threshold']:
            return False

        # Check age when decline started
        peak_age = history[history['WAR'] == peak_war]['Age'].iloc[0]

        return peak_age >= criteria['age_when_declining']

    def _check_two_way_player(self, history: pd.DataFrame, criteria: Dict) -> bool:
        """Check two-way player criteria."""
        # This is simplified - in practice would need pitching/hitting data separation
        # For now, check if player has position data suggesting two-way play

        if 'position' not in history.columns and 'Position' not in history.columns:
            return False

        position_col = 'Position' if 'Position' in history.columns else 'position'
        positions = history[position_col].dropna().unique()

        # Check if player has both pitcher and position player roles
        has_pitcher = any(pos in ['SP', 'RP', 'P'] for pos in positions)
        has_position = any(pos not in ['SP', 'RP', 'P'] for pos in positions)

        return has_pitcher and has_position

    def _check_power_specialist(self, history: pd.DataFrame, criteria: Dict) -> bool:
        """Check power specialist criteria."""
        # This would need HR data - simplified for now
        # Check for sustained elite performance in power-friendly positions

        if 'Position' not in history.columns:
            return False

        positions = history['Position'].dropna().unique()
        eligible_positions = criteria.get('position_eligibility', [])

        # Check position eligibility
        if not any(pos in eligible_positions for pos in positions):
            return False

        # Check for sustained elite performance (proxy for power)
        elite_seasons = (history['WAR'] >= 3.0).sum()  # Lower threshold for power specialists

        return elite_seasons >= criteria.get('min_hr_seasons', 3)

    def get_profile_adjustments(self, player_id: int) -> Dict[str, float]:
        """
        Get aging and regression adjustments for a player's profiles.

        Args:
            player_id: Player's MLB ID

        Returns:
            Dictionary with aging_adjustment and regression_protection factors
        """
        profiles = self.classify_player(player_id)

        if not profiles:
            return {'aging_adjustment': 1.0, 'regression_protection': 1.0}

        # Combine adjustments from multiple profiles
        aging_adjustments = []
        regression_protections = []

        for profile_name in profiles:
            profile = self.profiles[profile_name]
            aging_adjustments.append(profile.aging_adjustment)
            regression_protections.append(profile.regression_protection)

        # Use most favorable adjustments (minimum aging, maximum protection)
        final_aging = min(aging_adjustments) if aging_adjustments else 1.0
        final_protection = min(regression_protections) if regression_protections else 1.0

        return {
            'aging_adjustment': final_aging,
            'regression_protection': final_protection,
            'profiles': profiles
        }

    def analyze_profile_distribution(self) -> Dict:
        """
        Analyze the distribution of profiles across all players.

        Returns:
            Statistics about profile distribution
        """
        print("Analyzing profile distribution across all players...")

        profile_counts = {profile_name: 0 for profile_name in self.profiles.keys()}
        profile_counts['no_profile'] = 0

        player_profile_examples = {profile_name: [] for profile_name in self.profiles.keys()}

        total_players = len(self.player_histories)

        for player_id in self.player_histories.keys():
            profiles = self.classify_player(player_id)

            if profiles:
                for profile in profiles:
                    profile_counts[profile] += 1
                    # Store examples for validation
                    if len(player_profile_examples[profile]) < 5:
                        player_name = self._get_player_name(player_id)
                        player_profile_examples[profile].append({
                            'id': player_id,
                            'name': player_name
                        })
            else:
                profile_counts['no_profile'] += 1

        return {
            'total_players': total_players,
            'profile_counts': profile_counts,
            'profile_examples': player_profile_examples
        }

    def _get_player_name(self, player_id: int) -> str:
        """Get player name for a given ID."""
        if player_id in self.player_histories:
            history = self.player_histories[player_id]
            if 'Name' in history.columns:
                return history['Name'].iloc[0]
        return f"Player_{player_id}"

    def create_enhanced_confidence_scores(self,
                                        confidence_scores: Dict[int, float]) -> Dict[int, float]:
        """
        Enhance confidence scores with profile-based adjustments.

        Args:
            confidence_scores: Base confidence scores

        Returns:
            Enhanced confidence scores with profile adjustments
        """
        enhanced_scores = confidence_scores.copy()

        for player_id, base_confidence in confidence_scores.items():
            adjustments = self.get_profile_adjustments(player_id)

            # Apply regression protection to confidence
            protection_factor = 1.0 / adjustments['regression_protection']
            enhanced_confidence = base_confidence * protection_factor

            # Cap at reasonable maximum
            enhanced_scores[player_id] = min(enhanced_confidence, 8.0)

        return enhanced_scores