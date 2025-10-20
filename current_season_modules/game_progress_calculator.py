"""
Game Progress Calculator Module - Games Played and Remaining Logic
Handles calculation of games played, remaining games, and season progress
"""

# Standard library imports
from datetime import datetime, date
from typing import Dict, List, Tuple, Optional, Union

# Third-party imports
import numpy as np
import pandas as pd

# Local application imports
from common_modules.config import (
    CURRENT_SEASON_CONFIG,
    SEASON_MILESTONES,
    PROJECTION_CONFIDENCE,
    SEASON_GAMES,
    CURRENT_SEASON
)
from common_modules.logging import get_logger

# Module logger
logger = get_logger(__name__)


class GameProgressCalculator:
    """
    Calculates games played, games remaining, and season progress for baseball players

    Handles:
    - Standard 162-game MLB season
    - Shortened seasons (like 2020)
    - Player-specific games played (injuries, call-ups, etc.)
    - Season timeline and projections
    """

    def __init__(self, season_length: int = None, season_year: int = None):
        """
        Initialize calculator

        Args:
            season_length: Total games in season (defaults to config)
            season_year: Year of the season (defaults to current year from config)
        """
        self.season_length = season_length or CURRENT_SEASON_CONFIG['season_games']
        self.season_year = season_year or CURRENT_SEASON_CONFIG['season_year']

        # Use dates from configuration
        self.season_start = CURRENT_SEASON_CONFIG['season_start']
        self.season_end = CURRENT_SEASON_CONFIG['season_end']
        self.all_star_break = CURRENT_SEASON_CONFIG['all_star_break']

        logger.debug(f"GameProgressCalculator initialized for {self.season_year} season")

    def calculate_games_remaining(self, games_played: int) -> int:
        """
        Calculate games remaining in season

        Args:
            games_played: Number of games already played

        Returns:
            Number of games remaining
        """
        return max(0, self.season_length - games_played)

    def calculate_season_progress(self, games_played: int) -> float:
        """
        Calculate season progress as percentage

        Args:
            games_played: Number of games already played

        Returns:
            Season progress as decimal (0.0 to 1.0)
        """
        return min(1.0, games_played / self.season_length)

    def estimate_games_by_date(self, current_date: Union[date, str] = None) -> int:
        """
        Estimate games played by date based on typical MLB schedule

        Args:
            current_date: Date to calculate from (defaults to today)

        Returns:
            Estimated games played by this date
        """
        if current_date is None:
            current_date = date.today()
        elif isinstance(current_date, str):
            current_date = datetime.strptime(current_date, '%Y-%m-%d').date()

        if current_date <= self.season_start:
            return 0
        elif current_date >= self.season_end:
            return self.season_length

        # Calculate days into season
        days_elapsed = (current_date - self.season_start).days
        season_total_days = (self.season_end - self.season_start).days

        # MLB typically plays ~162 games over ~180 days (with off days)
        # Approximate games per day rate
        games_per_day_rate = self.season_length / season_total_days

        estimated_games = int(days_elapsed * games_per_day_rate)
        return min(estimated_games, self.season_length)

    def project_season_end_date(self, games_played: int,
                                current_date: Union[date, str] = None) -> date:
        """
        Project when season will end for this player based on current pace

        Args:
            games_played: Current games played
            current_date: Current date (defaults to today)

        Returns:
            Projected season end date
        """
        if current_date is None:
            current_date = date.today()
        elif isinstance(current_date, str):
            current_date = datetime.strptime(current_date, '%Y-%m-%d').date()

        games_remaining = self.calculate_games_remaining(games_played)

        if games_remaining == 0:
            return current_date

        # Calculate current pace (games per day)
        days_elapsed = max(1, (current_date - self.season_start).days)
        games_per_day = games_played / days_elapsed if days_elapsed > 0 else 1

        # Project remaining days
        remaining_days = games_remaining / games_per_day if games_per_day > 0 else 0
        projected_end = current_date + pd.Timedelta(days=remaining_days)

        return projected_end.date()

    def get_season_phase(self, games_played: int) -> str:
        """
        Determine what phase of the season we're in

        Args:
            games_played: Current games played

        Returns:
            Season phase: 'early', 'mid', 'late', 'complete'
        """
        progress = self.calculate_season_progress(games_played)

        if progress >= 1.0:
            return 'complete'
        elif progress >= 0.75:
            return 'late'
        elif progress >= 0.40:
            return 'mid'
        else:
            return 'early'

    def calculate_projection_confidence(self, games_played: int) -> float:
        """
        Calculate confidence level for projections based on sample size

        Args:
            games_played: Games played so far

        Returns:
            Confidence score (0.0 to 1.0)
        """
        if games_played == 0:
            return PROJECTION_CONFIDENCE['minimal']['confidence']

        # Use configuration thresholds
        for level in ['very_high', 'high', 'medium', 'low', 'very_low']:
            if games_played >= PROJECTION_CONFIDENCE[level]['games']:
                return PROJECTION_CONFIDENCE[level]['confidence']

        return PROJECTION_CONFIDENCE['minimal']['confidence']

    def adjust_for_playing_time(self, games_played: int,
                                player_games_in_period: int,
                                team_games_in_period: int) -> Dict:
        """
        Adjust projections based on playing time patterns

        Args:
            games_played: Individual player games played
            player_games_in_period: Player games in recent period
            team_games_in_period: Team games in same period

        Returns:
            Dictionary with playing time adjustments
        """
        if team_games_in_period == 0:
            playing_time_rate = 1.0
        else:
            playing_time_rate = player_games_in_period / team_games_in_period

        # Project future playing time
        games_remaining = self.calculate_games_remaining(games_played)
        projected_future_games = games_remaining * playing_time_rate

        return {
            'current_playing_time_rate': playing_time_rate,
            'projected_future_games': projected_future_games,
            'projected_total_games': games_played + projected_future_games,
            'rest_of_season_adjustment': playing_time_rate
        }

    def get_milestone_games(self) -> Dict[str, int]:
        """
        Get key milestone game numbers for the season

        Returns:
            Dictionary with milestone game numbers
        """
        return SEASON_MILESTONES.copy()

    def format_progress_summary(self, games_played: int,
                                current_date: Union[date, str] = None) -> str:
        """
        Create human-readable progress summary

        Args:
            games_played: Games played so far
            current_date: Current date

        Returns:
            Formatted progress string
        """
        games_remaining = self.calculate_games_remaining(games_played)
        progress_pct = self.calculate_season_progress(games_played) * 100
        phase = self.get_season_phase(games_played)
        confidence = self.calculate_projection_confidence(games_played)

        summary = (
            f"Season Progress: {games_played}/{self.season_length} games ({progress_pct:.1f}%)\n"
            f"Games Remaining: {games_remaining}\n"
            f"Season Phase: {phase.title()}\n"
            f"Projection Confidence: {confidence:.1f}"
        )

        if current_date:
            projected_end = self.project_season_end_date(games_played, current_date)
            summary += f"\nProjected End: {projected_end.strftime('%B %d, %Y')}"

        return summary


class TeamScheduleAnalyzer:
    """
    Analyzes team schedules to better understand individual player game counts

    Useful for understanding why a player might have played fewer games
    (team off days, injuries, call-ups, etc.)
    """

    def __init__(self, season_year: int = None):
        self.season_year = season_year or datetime.now().year

    def estimate_team_games_played(self, date_reference: Union[date, str] = None) -> int:
        """
        Estimate how many games a typical MLB team has played by a given date

        Args:
            date_reference: Date to calculate from

        Returns:
            Estimated team games played
        """
        calculator = GameProgressCalculator(season_year=self.season_year)
        return calculator.estimate_games_by_date(date_reference)

    def calculate_individual_vs_team_pace(self, player_games: int,
                                          team_games: int) -> Dict:
        """
        Compare individual player games to team games

        Args:
            player_games: Games played by individual player
            team_games: Games played by player's team

        Returns:
            Dictionary with pace analysis
        """
        if team_games == 0:
            participation_rate = 1.0
        else:
            participation_rate = player_games / team_games

        analysis = {
            'participation_rate': participation_rate,
            'games_missed': max(0, team_games - player_games),
            'pace_status': 'full' if participation_rate >= 0.95 else
            'regular' if participation_rate >= 0.8 else
            'part_time' if participation_rate >= 0.5 else
            'limited'
        }

        return analysis


def calculate_games_and_projections(current_stats: Dict,
                                    player_name: str = None,
                                    current_date: Union[date, str] = None,
                                    season_year: int = None) -> Dict:
    """
    Convenience function to calculate all game-related metrics for a player

    Args:
        current_stats: Dictionary containing current season stats (must include 'G' for games)
        player_name: Player name (for display)
        current_date: Current date for calculations
        season_year: Season year

    Returns:
        Dictionary with comprehensive game progress information
    """
    games_played = current_stats.get('G', current_stats.get('games_played', 0))

    if games_played == 0:
        logger.warning(f"No games played data available for {player_name or 'unknown player'}")
        return {
            'error': 'No games played data available',
            'games_played': 0,
            'games_remaining': SEASON_GAMES,
            'season_progress': 0.0
        }

    calculator = GameProgressCalculator(season_year=season_year)

    result = {
        'player_name': player_name,
        'games_played': games_played,
        'games_remaining': calculator.calculate_games_remaining(games_played),
        'season_progress': calculator.calculate_season_progress(games_played),
        'season_phase': calculator.get_season_phase(games_played),
        'projection_confidence': calculator.calculate_projection_confidence(games_played),
        'progress_summary': calculator.format_progress_summary(games_played, current_date)
    }

    # Add date-based estimates if current_date provided
    if current_date:
        estimated_team_games = calculator.estimate_games_by_date(current_date)
        if estimated_team_games > 0:
            analyzer = TeamScheduleAnalyzer(season_year)
            pace_analysis = analyzer.calculate_individual_vs_team_pace(
                games_played, estimated_team_games
            )
            result['pace_analysis'] = pace_analysis

    return result


def validate_games_played_data(stats_dict: Dict) -> Tuple[bool, str]:
    """
    Validate that games played data is reasonable

    Args:
        stats_dict: Dictionary of player statistics

    Returns:
        Tuple of (is_valid, error_message)
    """
    games_played = stats_dict.get('G', stats_dict.get('games_played'))

    if games_played is None:
        logger.debug("Validation failed: No games played data found")
        return False, "No games played data found"

    if games_played < 0:
        logger.debug(f"Validation failed: Negative games played ({games_played})")
        return False, "Games played cannot be negative"

    if games_played > SEASON_GAMES:
        logger.debug(
            f"Validation failed: Games played ({games_played}) exceeds season maximum ({SEASON_GAMES})")
        return False, f"Games played ({games_played}) exceeds season maximum ({SEASON_GAMES})"

    # Check for consistency with other stats
    plate_appearances = stats_dict.get('PA', 0)
    if plate_appearances > 0 and games_played > 0:
        pa_per_game = plate_appearances / games_played

        # Define reasonable bounds for PA per game
        MAX_PA_PER_GAME = 8.0
        MIN_PA_PER_GAME = 1.0

        if pa_per_game > MAX_PA_PER_GAME:
            logger.debug(f"Validation warning: PA per game ({pa_per_game:.1f}) unusually high")
            return False, f"PA per game ({pa_per_game:.1f}) seems unusually high"
        elif pa_per_game < MIN_PA_PER_GAME:
            logger.debug(f"Validation warning: PA per game ({pa_per_game:.1f}) unusually low")
            return False, f"PA per game ({pa_per_game:.1f}) seems unusually low"

    logger.debug("Games played data validated successfully")
    return True, "Games played data validated successfully"
