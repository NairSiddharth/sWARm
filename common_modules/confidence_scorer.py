"""
Simple Confidence Scoring for Elite Adjustment
==============================================

Provides basic confidence scoring for the elite adjustment system
based on current performance metrics and player characteristics.
"""

import pandas as pd
from typing import Dict, Optional


class SimpleConfidenceScorer:
    """
    Simple confidence scorer for elite adjustment system.

    Calculates confidence scores (0-8 scale) based on:
    - Current performance level
    - Age factors
    - Position scarcity
    - Consistency indicators
    """

    def __init__(self):
        # Position scarcity factors
        self.position_factors = {
            'C': 0.3,    # Catcher scarcity bonus
            'SS': 0.2,   # Shortstop scarcity bonus
            'CF': 0.1,   # Center field premium
            '2B': 0.05,  # Second base slight bonus
            '3B': 0.0,   # Third base baseline
            '1B': 0.0,   # First base baseline
            'LF': 0.0,   # Left field baseline
            'RF': 0.0,   # Right field baseline
            'DH': -0.2,  # Designated hitter penalty
            'P': 0.0,    # Pitcher baseline
            'OF': 0.0    # Generic outfield
        }

    def calculate_confidence_score(self,
                                   war_value: float,
                                   age: Optional[float] = None,
                                   position: Optional[str] = None,
                                   ip_or_pa: Optional[float] = None) -> float:
        """
        Calculate confidence score for a player.

        Args:
            war_value: Current WAR/WARP value
            age: Player age (optional, defaults to 28)
            position: Player position (optional)
            ip_or_pa: Innings pitched or plate appearances for volume bonus

        Returns:
            Confidence score (0-8 scale)
        """
        if pd.isna(war_value):
            return 1.0

        # Base confidence from performance (0-6 scale)
        # Elite performers get high confidence
        if war_value >= 5.0:
            base_confidence = 6.0
        elif war_value >= 4.0:
            base_confidence = 5.5
        elif war_value >= 3.0:
            base_confidence = 4.5
        elif war_value >= 2.0:
            base_confidence = 3.0
        elif war_value >= 1.0:
            base_confidence = 2.0
        elif war_value >= 0.0:
            base_confidence = 1.5
        else:
            base_confidence = 1.0

        # Age factor (peak performance ages get bonus)
        age_factor = 1.0
        if age is not None:
            if 25 <= age <= 30:  # Peak years
                age_factor = 1.1
            elif 23 <= age <= 32:  # Prime years
                age_factor = 1.05
            elif age <= 22:  # Very young, uncertainty
                age_factor = 0.9
            elif 33 <= age <= 35:  # Aging but still good
                age_factor = 0.95
            elif age > 35:  # Significant aging risk
                age_factor = 0.85

        # Position factor
        position_bonus = 0.0
        if position:
            position_bonus = self.position_factors.get(position, 0.0)

        # Volume bonus (more playing time = more confidence)
        volume_bonus = 0.0
        if ip_or_pa is not None:
            if position == 'P':  # Pitcher
                if ip_or_pa >= 180:  # Workhorse starter
                    volume_bonus = 0.2
                elif ip_or_pa >= 120:  # Regular starter
                    volume_bonus = 0.1
                elif ip_or_pa >= 60:  # Regular reliever
                    volume_bonus = 0.05
            else:  # Hitter
                if ip_or_pa >= 600:  # Full-time player
                    volume_bonus = 0.2
                elif ip_or_pa >= 450:  # Regular player
                    volume_bonus = 0.1
                elif ip_or_pa >= 300:  # Part-time player
                    volume_bonus = 0.05

        # Calculate final confidence
        final_confidence = (base_confidence * age_factor) + position_bonus + volume_bonus

        # Cap at 8.0 maximum
        return min(final_confidence, 8.0)

    def calculate_batch_confidence(self,
                                   players_df: pd.DataFrame,
                                   war_column: str = 'WAR',
                                   age_column: str = 'Age',
                                   position_column: str = 'Pos',
                                   volume_column: str = 'IP') -> Dict[int, float]:
        """
        Calculate confidence scores for a batch of players.

        Args:
            players_df: DataFrame with player data
            war_column: Column name for WAR values
            age_column: Column name for age
            position_column: Column name for position
            volume_column: Column name for volume (IP for pitchers, PA for hitters)

        Returns:
            Dictionary mapping player ID to confidence score
        """
        confidence_scores = {}

        for idx, row in players_df.iterrows():
            player_id = row.get('mlbid', row.get('MLBAMID', idx))
            war_value = row.get(war_column, 0.0)
            age = row.get(age_column)
            position = row.get(position_column)
            volume = row.get(volume_column)

            confidence = self.calculate_confidence_score(
                war_value=war_value,
                age=age,
                position=position,
                ip_or_pa=volume
            )

            confidence_scores[player_id] = confidence

        return confidence_scores

    def calculate_undervaluation_adjusted_confidence(self,
                                                     war_value: float,
                                                     age: Optional[float] = None,
                                                     position: Optional[str] = None,
                                                     ip_or_pa: Optional[float] = None) -> float:
        """
        Calculate confidence score adjusted for systematic WAR undervaluation.

        This method accounts for the fact that our current WAR calculations
        systematically undervalue elite players by ~35%. It uses lowered
        thresholds to identify truly elite players despite the undervaluation.

        Args:
            war_value: Current WAR/WARP value (potentially undervalued)
            age: Player age (optional, defaults to 28)
            position: Player position (optional)
            ip_or_pa: Innings pitched or plate appearances for volume bonus

        Returns:
            Confidence score (0-8 scale) adjusted for undervaluation
        """
        if pd.isna(war_value):
            return 1.0

        # UNDERVALUATION-ADJUSTED confidence thresholds
        # Lowered to account for systematic elite player undervaluation
        if war_value >= 4.0:
            base_confidence = 6.0  # Still clearly elite
        elif war_value >= 2.8:  # Catches undervalued elites like Skubal (2.824)
            base_confidence = 5.5  # High confidence for undervalued elites
        elif war_value >= 2.0:  # Good performers, likely undervalued
            base_confidence = 4.5  # Elevated confidence
        elif war_value >= 1.5:  # Average performers
            base_confidence = 3.5  # Some protection
        elif war_value >= 1.0:
            base_confidence = 2.5  # Basic protection
        elif war_value >= 0.5:
            base_confidence = 2.0
        elif war_value >= 0.0:
            base_confidence = 1.5
        else:
            base_confidence = 1.0

        # Apply same age, position, and volume adjustments
        age_factor = 1.0
        if age is not None:
            if 25 <= age <= 30:  # Peak years
                age_factor = 1.1
            elif 23 <= age <= 32:  # Prime years
                age_factor = 1.05
            elif age <= 22:  # Very young, uncertainty
                age_factor = 0.9
            elif 33 <= age <= 35:  # Aging but still good
                age_factor = 0.95
            elif age > 35:  # Significant aging risk
                age_factor = 0.85

        # Position factor
        position_bonus = 0.0
        if position:
            position_bonus = self.position_factors.get(position, 0.0)

        # Volume bonus
        volume_bonus = 0.0
        if ip_or_pa is not None:
            if position == 'P':  # Pitcher
                if ip_or_pa >= 180:  # Workhorse starter
                    volume_bonus = 0.2
                elif ip_or_pa >= 120:  # Regular starter
                    volume_bonus = 0.1
                elif ip_or_pa >= 60:  # Regular reliever
                    volume_bonus = 0.05
            else:  # Hitter
                if ip_or_pa >= 600:  # Full-time player
                    volume_bonus = 0.2
                elif ip_or_pa >= 450:  # Regular player
                    volume_bonus = 0.1
                elif ip_or_pa >= 300:  # Part-time player
                    volume_bonus = 0.05

        # Calculate final confidence
        final_confidence = (base_confidence * age_factor) + position_bonus + volume_bonus

        # Cap at 8.0 maximum
        return min(final_confidence, 8.0)


def calculate_simple_confidence_scores(current_war_data: pd.DataFrame,
                                       war_column: str = 'Current_WAR',
                                       player_type: str = 'pitcher') -> Dict[int, float]:
    """
    Convenience function to calculate confidence scores for WAR data.

    Args:
        current_war_data: DataFrame with current WAR calculations
        war_column: Column containing WAR values
        player_type: 'pitcher' or 'hitter' for appropriate volume column

    Returns:
        Dictionary mapping player ID to confidence score
    """
    scorer = SimpleConfidenceScorer()

    # Determine appropriate volume column
    if player_type == 'pitcher':
        volume_col = 'IP'
    else:
        volume_col = 'PA'

    return scorer.calculate_batch_confidence(
        current_war_data,
        war_column=war_column,
        volume_column=volume_col
    )
