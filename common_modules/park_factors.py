"""
Park Factors Module for sWARm Current Season Analysis.

Uses official FanGraphs park factors (3yr column) instead of calculating from game data.
Ensures consistent park factor integration across the historical training pipeline.
"""

__version__ = '2.0.0'
__author__ = 'oWAR Development Team'

# Standard library imports
import json
from pathlib import Path
from typing import Dict, Optional

# Third-party imports
import pandas as pd

# Local imports
from .config import CACHE_DIR, DATA_DIR
from .logging import get_logger

# Initialize logger
logger = get_logger(__name__)

# Global cache for park factors to avoid repeated loading
_PARK_FACTORS_CACHE: Dict[int, Dict[str, float]] = {}

# FanGraphs team name to abbreviation mapping
FANGRAPHS_TO_ABBREV = {
    'Angels': 'LAA', 'Astros': 'HOU', 'Athletics': 'OAK', 'Mariners': 'SEA', 'Rangers': 'TEX',
    'Yankees': 'NYY', 'Red Sox': 'BOS', 'Blue Jays': 'TOR', 'Orioles': 'BAL', 'Rays': 'TB',
    'White Sox': 'CWS', 'Guardians': 'CLE', 'Tigers': 'DET', 'Royals': 'KC', 'Twins': 'MIN',
    'Braves': 'ATL', 'Marlins': 'MIA', 'Mets': 'NYM', 'Phillies': 'PHI', 'Nationals': 'WSN',
    'Cubs': 'CHC', 'Reds': 'CIN', 'Brewers': 'MIL', 'Pirates': 'PIT', 'Cardinals': 'STL',
    'Diamondbacks': 'ARI', 'Rockies': 'COL', 'Dodgers': 'LAD', 'Padres': 'SD', 'Giants': 'SF'
}

# Park factor adjustment constants
NEUTRAL_PARK_FACTOR = 100.0
DEFAULT_YEAR = 2024
FALLBACK_YEAR = 2024  # Used when current year not available

# Public API
__all__ = [
    'load_park_factors',
    'apply_park_factor_adjustments',
]


def load_park_factors(year: int = DEFAULT_YEAR) -> Dict[str, Dict[str, float]]:
    """
    Load official FanGraphs park factors with multiple components.

    Args:
        year: Year to load park factors for

    Returns:
        Dictionary mapping team abbreviation to dict of park factors:
        {
            'NYY': {
                '3yr': 95.0,
                'HR': 92.0,
                'FB': 97.0,
                ...
            }
        }

    Raises:
        FileNotFoundError: If park factors file cannot be found
        ValueError: If required columns are missing from data

    Example:
        >>> park_factors = load_park_factors(2024)
        >>> park_factors['NYY']['3yr']
        95.0
    """
    # Check global cache first (fastest)
    if year in _PARK_FACTORS_CACHE:
        logger.debug(f"Returning cached park factors for {year}")
        return _PARK_FACTORS_CACHE[year]

    cache_file = CACHE_DIR / f'fangraphs_park_factors_{year}.json'

    # Try to load from file cache
    if cache_file.exists():
        try:
            with open(cache_file, 'r') as f:
                cached_data = json.load(f)
            logger.info(
                f"Loaded cached FanGraphs park factors for {year} ({
                    len(cached_data)} teams)")
            _PARK_FACTORS_CACHE[year] = cached_data
            return cached_data
        except Exception as e:
            logger.warning(f"Failed to load cache file: {e}")

    logger.info(f"Loading FanGraphs park factors for {year}")

    # Try to load the specified year first
    park_factors_file = DATA_DIR / 'FanGraphs_Data' / \
        'park_factors' / f'fangraphs_parkfactors_{year}.csv'

    # If year file doesn't exist and it's recent, fallback to previous year
    if not park_factors_file.exists() and year >= 2025:
        logger.warning(f"{year} park factors not found, falling back to {FALLBACK_YEAR}")
        park_factors_file = DATA_DIR / 'FanGraphs_Data' / \
            'park_factors' / f'fangraphs_parkfactors_{FALLBACK_YEAR}.csv'
        year = FALLBACK_YEAR
        cache_file = CACHE_DIR / f'fangraphs_park_factors_{year}.json'

    if not park_factors_file.exists():
        logger.error(f"Park factors file not found: {park_factors_file}")
        raise FileNotFoundError(f"Park factors file not found: {park_factors_file}")

    try:
        park_df = pd.read_csv(park_factors_file)

        # Validate required columns
        required_columns = ['Team', '3yr']
        missing_columns = set(required_columns) - set(park_df.columns)
        if missing_columns:
            raise ValueError(f"Required columns missing: {missing_columns}")

        # Park factor columns we want to extract
        factor_columns = ['3yr', 'HR', 'FB', 'GB', 'LD']

        # Create team -> park factors mapping using abbreviations
        park_factors = {}
        for _, row in park_df.iterrows():
            team_name = row['Team']

            if pd.notna(team_name):
                # Convert FanGraphs team name to abbreviation
                team_abbrev = FANGRAPHS_TO_ABBREV.get(str(team_name).strip())
                if team_abbrev:
                    team_factors = {}
                    for col in factor_columns:
                        if col in park_df.columns and pd.notna(row[col]):
                            team_factors[col] = float(row[col])

                    if team_factors:  # Only add if we got at least some factors
                        park_factors[team_abbrev] = team_factors

        # Save to cache
        try:
            with open(cache_file, 'w') as f:
                json.dump(park_factors, f, indent=2)
            logger.debug(f"Saved park factors to cache: {cache_file}")
        except Exception as e:
            logger.warning(f"Could not save park factors cache: {e}")

        # Store in global cache for future calls
        _PARK_FACTORS_CACHE[year] = park_factors
        logger.info(f"Loaded FanGraphs park factors for {len(park_factors)} teams from {year}")
        return park_factors

    except Exception as e:
        logger.error(f"Error loading park factors from {park_factors_file}: {e}")
        raise


def apply_park_factor_adjustments(
    player_stats: Dict[str, float],
    player_name: str,
    team: str,
    player_type: str = 'hitter',
    year: int = DEFAULT_YEAR
) -> Dict[str, float]:
    """
    Apply park factor adjustments to player statistics using FanGraphs data.

    For pitchers, applies specific adjustments:
    - ERA: 3yr park factor
    - HR%: HR park factor
    - FB%: FB park factor
    - Hard%, Soft%, Med%: 3yr park factor

    Args:
        player_stats: Dictionary of player statistics to adjust
        player_name: Player name for tracking/logging
        team: Team abbreviation
        player_type: Either 'hitter' or 'pitcher'
        year: Year for park factors

    Returns:
        Dictionary with park_factor_adjustment added and stats adjusted

    Raises:
        ValueError: If player_type is not 'hitter' or 'pitcher'

    Example:
        >>> stats = {'AVG': .300, 'OBP': .400, 'SLG': .500}
        >>> adjusted = apply_park_factor_adjustments(stats, 'Judge', 'NYY', 'hitter')
        >>> adjusted['park_factor_adjustment']
        1.05
    """
    if player_type not in ['hitter', 'pitcher']:
        raise ValueError(f"player_type must be 'hitter' or 'pitcher', got {player_type}")

    try:
        park_factors = load_park_factors(year)
    except (FileNotFoundError, ValueError) as e:
        logger.warning(f"Could not load park factors: {e}. Using neutral adjustment.")
        player_stats['park_factor_adjustment'] = 1.0
        return player_stats

    if not park_factors:
        # No park factors available, return neutral
        player_stats['park_factor_adjustment'] = 1.0
        return player_stats

    # Get team's park factors
    team_abbrev = str(team).upper().strip()
    if team_abbrev not in park_factors:
        logger.debug(f"Team {team_abbrev} not found in park factors, using neutral")
        player_stats['park_factor_adjustment'] = 1.0
        return player_stats

    team_park_factors = park_factors[team_abbrev]

    # Apply park factor adjustment
    if player_type == 'hitter':
        # For hitters: park factor > 100 helps offense, < 100 hurts offense
        # Adjustment is inverse: if park helps hitters, adjust stats down
        park_factor_3yr = team_park_factors.get('3yr', NEUTRAL_PARK_FACTOR)
        park_adjustment = NEUTRAL_PARK_FACTOR / park_factor_3yr

        # Apply adjustments to offensive stats
        _adjust_hitter_stats(player_stats, park_adjustment)

        player_stats['park_factor_adjustment'] = park_adjustment
        logger.debug(f"Applied park factor {park_factor_3yr} (adj: {park_adjustment:.3f}) to {player_name}")

    else:  # pitcher
        # For pitchers: apply specific park factors for different stats
        _adjust_pitcher_stats(player_stats, team_park_factors, player_name)

    return player_stats


def _adjust_hitter_stats(player_stats: Dict[str, float], park_adjustment: float) -> None:
    """
    Apply park adjustments to hitter statistics.

    Args:
        player_stats: Dictionary of statistics to modify in-place
        park_adjustment: Adjustment factor to apply
    """
    hitter_stats_to_adjust = ['AVG', 'OBP', 'SLG', 'OPS', 'ISO']

    for stat in hitter_stats_to_adjust:
        if stat in player_stats and player_stats[stat] is not None:
            player_stats[stat] = player_stats[stat] * park_adjustment


def _adjust_pitcher_stats(player_stats: Dict[str, float], team_park_factors: Dict[str, float], player_name: str) -> None:
    """
    Apply park adjustments to pitcher statistics using specific park factors.

    Adjustments:
    - ERA: 3yr park factor
    - HR%: HR park factor (if available, else 3yr)
    - FB%: FB park factor (if available, else 3yr)
    - Hard%, Soft%, Med%: 3yr park factor

    Args:
        player_stats: Dictionary of statistics to modify in-place
        team_park_factors: Dictionary of park factors for the team
        player_name: Player name for logging
    """
    # Get park factors with fallbacks
    park_3yr = team_park_factors.get('3yr', NEUTRAL_PARK_FACTOR)
    park_hr = team_park_factors.get('HR', park_3yr)  # Fallback to 3yr if HR not available
    park_fb = team_park_factors.get('FB', park_3yr)  # Fallback to 3yr if FB not available

    # Calculate effective park factors (blend with neutral for ~50% home games)
    # Formula: (park_factor + 100) / 2 to account for home/road split
    effective_park_3yr = (park_3yr + NEUTRAL_PARK_FACTOR) / 2
    effective_park_hr = (park_hr + NEUTRAL_PARK_FACTOR) / 2
    effective_park_fb = (park_fb + NEUTRAL_PARK_FACTOR) / 2

    # Calculate adjustments (inverse: higher park factor = worse for pitcher = lower stat to credit pitcher)
    adj_3yr = NEUTRAL_PARK_FACTOR / effective_park_3yr
    adj_hr = NEUTRAL_PARK_FACTOR / effective_park_hr
    adj_fb = NEUTRAL_PARK_FACTOR / effective_park_fb

    # Apply specific adjustments
    adjustments_made = []

    # ERA: 3yr park factor
    if 'ERA' in player_stats and player_stats['ERA'] is not None:
        player_stats['ERA'] = player_stats['ERA'] * adj_3yr
        adjustments_made.append(f"ERA({park_3yr:.1f})")

    # HR%: HR park factor
    if 'HR%' in player_stats and player_stats['HR%'] is not None:
        player_stats['HR%'] = player_stats['HR%'] * adj_hr
        adjustments_made.append(f"HR%({park_hr:.1f})")

    # FB%: FB park factor
    if 'FB%' in player_stats and player_stats['FB%'] is not None:
        player_stats['FB%'] = player_stats['FB%'] * adj_fb
        adjustments_made.append(f"FB%({park_fb:.1f})")

    # Hard%, Soft%, Med%: DO NOT park adjust - these are pitcher skill metrics
    # Park factors would destroy variance since contact quality is more skill than park

    # Store the primary adjustment for tracking (3yr for ERA)
    player_stats['park_factor_adjustment'] = adj_3yr

    if adjustments_made:
        logger.debug(f"Park adjustments for {player_name}: {', '.join(adjustments_made)}")
    else:
        logger.debug(f"No park adjustments applied for {player_name} (no matching stats)")


def normalize_team_abbreviation(team: Optional[str]) -> Optional[str]:
    """
    Normalize team abbreviation for consistent lookup.

    Args:
        team: Team name or abbreviation

    Returns:
        Normalized team abbreviation or None if invalid

    Example:
        >>> normalize_team_abbreviation('nyy')
        'NYY'
        >>> normalize_team_abbreviation('Yankees')
        'NYY'
    """
    if pd.isna(team) or team is None:
        return None

    team_upper = str(team).upper().strip()

    # Check if it's already an abbreviation
    if team_upper in FANGRAPHS_TO_ABBREV.values():
        return team_upper

    # Check if it's a full team name
    for full_name, abbrev in FANGRAPHS_TO_ABBREV.items():
        if full_name.upper() == team_upper:
            return abbrev

    logger.warning(f"Unknown team: {team}")
    return team_upper
