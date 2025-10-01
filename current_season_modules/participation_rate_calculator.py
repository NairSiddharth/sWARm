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

# Standard library imports
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta

# Third-party imports
import pandas as pd
import numpy as np

# Local application imports
from common_modules.config import SEASON_GAMES
from common_modules.logging import get_logger

# Module logger
logger = get_logger(__name__)


def calculate_participation_adjusted_games_with_team_data(
    player_data: pd.Series,
    team_games_dict: Dict[str, int],
    current_war: float = 0.0,
    injury_adjustment: float = 1.0
) -> Dict[str, Union[float, str]]:
    """
    Convenience function to calculate participation-adjusted games using actual team games.

    This is the recommended way to calculate remaining games for pitchers and hitters,
    using actual team games played derived from hitter data.

    Args:
        player_data: Player statistics row
        team_games_dict: Dictionary mapping team to actual games played
        current_war: Player's current WAR for performance boost calculation
        injury_adjustment: Injury recovery factor (0-1)

    Returns:
        Dictionary with games_remaining, method, and other metadata
    """
    calculator = ParticipationRateCalculator(team_games_dict=team_games_dict)

    # Detect role and calculate participation
    role_info = calculator.detect_player_role(player_data)
    perf_boost = calculator.calculate_performance_boost(current_war)

    # Calculate adjusted participation
    base_rate = role_info['expected_rate']
    max_rate = role_info['max_rate']
    current_rate = role_info['current_rate']

    # Blend current usage with expected role
    blended_rate = 0.6 * current_rate + 0.4 * base_rate

    # Apply performance boost
    adjusted_rate = min(blended_rate * perf_boost * injury_adjustment, max_rate)

    # Get actual team games
    team = player_data.get('Team', player_data.get('team', None))
    if team and team in team_games_dict:
        team_games = team_games_dict[team]
    else:
        team_games = calculator.estimated_team_games

    # Calculate remaining games
    games_played = player_data.get('G', 0)
    remaining_team_games = calculator.season_length - team_games
    games_remaining = int(adjusted_rate * remaining_team_games)

    return {
        'games_remaining': games_remaining,
        'method': 'team_data_participation',
        'team_games': team_games,
        'participation_rate': adjusted_rate,
        'current_usage_rate': current_rate,
        'expected_rate': base_rate,
        'performance_boost': perf_boost,
        'injury_factor': injury_adjustment,
        'role_classification': role_info['role']
    }


class ParticipationRateCalculator:
    """
    Calculate dynamic participation rates for current season projections.

    Combines current usage patterns, role detection, performance metrics,
    and injury recovery to project realistic remaining games.
    """

    def __init__(self, team_games_dict: Optional[Dict[str, int]] = None,
                 estimated_team_games: int = 95):
        """
        Initialize participation rate calculator.

        Args:
            team_games_dict: Dictionary mapping team to actual games played
            estimated_team_games: Fallback estimated team games if dict not provided
        """
        self.team_games_dict = team_games_dict
        self.estimated_team_games = estimated_team_games
        self.season_length = SEASON_GAMES

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

        # Get actual team games if available, otherwise use estimate
        team = player_data.get('Team', player_data.get('team', None))
        if self.team_games_dict and team and team in self.team_games_dict:
            team_games = self.team_games_dict[team]
        else:
            team_games = self.estimated_team_games

        current_rate = games_played / team_games if team_games > 0 else 0

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

    def calculate_performance_boost(self, current_war: float,
                                    games_played: int) -> Dict[str, Union[float, str]]:
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
                                             injury_adjustment: float = 1.0) -> Dict[str,
                                                                                     Union[int,
                                                                                           float,
                                                                                           str]]:
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
            'method': f"Hybrid: {
                role_info['role']} ({
                final_rate:.1%} rate, {
                performance_boost:.2f}x boost)"}


def calculate_participation_adjusted_games(player_data: pd.Series,
                                           current_war: float = None,
                                           injury_adjustment: float = 1.0,
                                           estimated_team_games: int = 95) -> Dict[str, Union[int, float, str]]:
    """
    Convenience function for calculating participation-adjusted games.

    NOW USES ACTUAL TEAM GAMES FROM HITTER DATA when available.

    Args:
        player_data: Player statistics row
        current_war: Current season WAR (optional)
        injury_adjustment: Injury recovery factor (default 1.0)
        estimated_team_games: Fallback if team games can't be loaded

    Returns:
        Participation rate calculation results
    """
    # Try to load actual team games for 2025
    try:
        from current_season_modules.current_season_data_loading import calculate_team_games_from_hitters
        team_games_dict = calculate_team_games_from_hitters(2025, 'fangraphs')
        logger.info(f"Loaded actual team games for {len(team_games_dict)} teams")
    except Exception as e:
        logger.warning(f"Could not load team games, using estimate: {e}")
        team_games_dict = None

    # Use actual team games if available
    calculator = ParticipationRateCalculator(
        team_games_dict=team_games_dict,
        estimated_team_games=estimated_team_games
    )
    return calculator.calculate_dynamic_participation_rate(
        player_data, current_war, injury_adjustment
    )


def detect_fielding_status(player_data: pd.Series,
                          games_played: int,
                          defensive_data: pd.DataFrame = None) -> Dict[str, Union[str, float, int]]:
    """
    Detect if player transitioned to DH mid-season using defensive innings data.

    Uses innings_per_game thresholds to determine defensive participation:
    - < 1.0 innings/game: Full-time DH
    - 1.0-4.0 innings/game: Mixed role (likely injury transition)
    - > 4.0 innings/game: Regular fielder

    Args:
        player_data: Player row from hitter data
        games_played: Total games player appeared in
        defensive_data: DataFrame with defensive stats (Pos, Inn columns)

    Returns:
        Dictionary containing:
            - status: 'fielder' | 'DH' | 'mixed'
            - fielding_pct: Percentage of games where player fielded (0.0-1.0)
            - innings_per_game: Average defensive innings per game
            - total_innings: Total defensive innings played
            - primary_position: Position with most innings (if fielder)
            - position_innings: Dict of innings by position
    """
    if games_played == 0:
        logger.warning("detect_fielding_status called with 0 games_played")
        return {
            'status': 'unknown',
            'fielding_pct': 0.0,
            'innings_per_game': 0.0,
            'total_innings': 0.0,
            'primary_position': None,
            'position_innings': {}
        }

    # Get player's MLBAMID for matching
    player_id = player_data.get('MLBAMID', player_data.get('mlbid', None))

    # Try to get innings from defensive data if provided
    total_innings = 0.0
    position_innings = {}

    if defensive_data is not None and player_id is not None:
        # Filter defensive data for this player
        player_def = defensive_data[defensive_data['MLBAMID'] == player_id]

        if len(player_def) > 0:
            # IMPORTANT: Defensive data shows FULL SEASON projections
            # We need to prorate to actual games played (games_played / 162)
            FULL_SEASON_GAMES = 162.0
            proration_factor = games_played / FULL_SEASON_GAMES

            # Sum innings across all position rows (prorated to current season progress)
            for _, row in player_def.iterrows():
                pos = row.get('Pos', '')
                full_season_innings = row.get('Inn', 0)

                try:
                    full_season_innings = float(full_season_innings) if pd.notna(full_season_innings) else 0.0
                    if full_season_innings > 0 and pos:
                        # Prorate innings to games played so far
                        actual_innings = full_season_innings * proration_factor

                        # Clean position (remove DH designation)
                        clean_pos = pos.split('/')[0] if '/' in pos else pos
                        if clean_pos and clean_pos != 'DH':
                            position_innings[clean_pos] = position_innings.get(clean_pos, 0) + actual_innings
                            total_innings += actual_innings
                except (ValueError, TypeError):
                    continue

    # Calculate innings per game
    innings_per_game = total_innings / games_played if games_played > 0 else 0.0

    # Determine primary position (if fielding)
    primary_position = None
    if position_innings:
        primary_position = max(position_innings.items(), key=lambda x: x[1])[0]

    # Estimate games where player actually fielded
    # Assume 9 innings per defensive game as baseline
    INNINGS_PER_GAME = 9.0
    estimated_games_fielded = total_innings / INNINGS_PER_GAME
    fielding_pct = min(estimated_games_fielded / games_played, 1.0) if games_played > 0 else 0.0

    # Determine status based on innings_per_game thresholds
    if innings_per_game < 1.0:
        status = 'DH'
    elif innings_per_game < 4.0:
        status = 'mixed'
    else:
        status = 'fielder'

    logger.debug(
        f"Fielding status for {player_data.get('Name', 'Unknown')}: "
        f"{status} ({total_innings:.1f} inn, {innings_per_game:.2f} inn/g, "
        f"{fielding_pct:.1%} fielding %)"
    )

    return {
        'status': status,
        'fielding_pct': fielding_pct,
        'innings_per_game': innings_per_game,
        'total_innings': total_innings,
        'primary_position': primary_position,
        'position_innings': position_innings
    }


def adjust_defense_for_position(enhanced_defense: float,
                                fielding_status: Dict[str, Union[str, float, int]],
                                positional_adjustment: float,
                                games_played: int) -> Dict[str, float]:
    """
    Adjust Enhanced_Defense based on multi-position fielding participation.

    Calculates weighted positional adjustments for all positions played:
    - Converts innings to games at each position (innings/9)
    - Applies position-specific adjustments weighted by games played
    - Calculates DH time penalty for non-fielding games
    - Prorates Enhanced_Defense by actual fielding participation

    Args:
        enhanced_defense: Raw Enhanced_Defense value from enhanced_features.py
        fielding_status: Output from detect_fielding_status() with position_innings
        positional_adjustment: DEPRECATED - now uses config values per position
        games_played: Total games played

    Returns:
        Dictionary containing:
            - adjusted_defense: Prorated Enhanced_Defense value
            - positional_adjustment: Weighted positional adjustment (runs)
            - total_adjustment: Combined adjustment to WAR (in runs)
            - explanation: Human-readable explanation
            - position_breakdown: Dict of games and adjustments per position
    """
    from common_modules.config import POSITIONAL_ADJUSTMENTS, SEASON_GAMES

    position_innings = fielding_status.get('position_innings', {})
    total_innings = fielding_status.get('total_innings', 0.0)

    if games_played == 0:
        logger.warning("adjust_defense_for_position called with 0 games_played")
        return {
            'adjusted_defense': enhanced_defense,
            'positional_adjustment': 0.0,
            'total_adjustment': 0.0,
            'explanation': 'No games played',
            'position_breakdown': {}
        }

    # Calculate games at each defensive position (innings / 9)
    INNINGS_PER_GAME = 9.0
    position_games = {}
    total_fielding_games = 0.0

    for pos, innings in position_innings.items():
        games_at_pos = innings / INNINGS_PER_GAME
        position_games[pos] = games_at_pos
        total_fielding_games += games_at_pos

    # Calculate DH games (games not spent fielding)
    dh_games = max(0, games_played - total_fielding_games)
    if dh_games > 0:
        position_games['DH'] = dh_games

    # Calculate weighted positional adjustments
    position_breakdown = {}
    total_positional_runs = 0.0

    for pos, games_at_pos in position_games.items():
        if games_at_pos == 0:
            continue

        # Get positional adjustment for this position
        pos_adj_per_162 = POSITIONAL_ADJUSTMENTS.get(pos, 0.0)

        # Scale to games played at this position
        pos_adj_runs = pos_adj_per_162 * (games_at_pos / SEASON_GAMES)
        total_positional_runs += pos_adj_runs

        position_breakdown[pos] = {
            'games': round(games_at_pos, 1),
            'innings': position_innings.get(pos, 0.0) if pos != 'DH' else 0.0,
            'adjustment_per_162': pos_adj_per_162,
            'adjustment_runs': pos_adj_runs
        }

    # Prorate Enhanced_Defense by fielding participation
    fielding_pct = total_fielding_games / games_played if games_played > 0 else 0.0
    adjusted_defense = enhanced_defense * fielding_pct

    # Defense change (negative if player DHs)
    defense_change = adjusted_defense - enhanced_defense

    # Total adjustment combines defense change and positional adjustments
    total_adjustment = defense_change + total_positional_runs

    # Build explanation
    if dh_games > 0.5:
        explanation = (
            f"Multi-position player: {fielding_pct:.0%} fielding, "
            f"{(dh_games/games_played):.0%} DH"
        )
    else:
        explanation = f"Regular fielder across {len(position_innings)} position(s)"

    logger.debug(
        f"Multi-position adjustment: {len(position_games)} positions, "
        f"defense {enhanced_defense:.1f} → {adjusted_defense:.1f}, "
        f"positional adj {total_positional_runs:.1f}, "
        f"total {total_adjustment:.1f} runs"
    )

    return {
        'adjusted_defense': adjusted_defense,
        'positional_adjustment': total_positional_runs,
        'total_adjustment': total_adjustment,
        'explanation': explanation,
        'position_breakdown': position_breakdown
    }


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

    logger.info("Cal Raleigh Projection Example:")
    logger.info(f"Role: {result['role_classification']}")
    logger.info(f"Current usage: {result['current_usage_rate']:.1%}")
    logger.info(f"Final rate: {result['participation_rate']:.1%}")
    logger.info(f"Remaining games: {result['games_remaining']}")
    logger.info(f"Projected total: {result['projected_total_games']}")
    logger.info(f"Method: {result['method']}")
