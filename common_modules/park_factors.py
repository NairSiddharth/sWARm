"""
Park Factors Module for sWARm Current Season Analysis

Uses official FanGraphs park factors (3yr column) instead of calculating from game data.
Ensures consistent park factor integration across the historical training pipeline.
"""

import os
import pandas as pd
import json
from pathlib import Path

# Constants
DATA_DIR = r"C:\Users\nairs\Documents\GithubProjects\oWAR\MLB Player Data"
CACHE_DIR = r"C:\Users\nairs\Documents\GithubProjects\oWAR\cache"

# Ensure cache directory exists
os.makedirs(CACHE_DIR, exist_ok=True)

# Global cache for park factors to avoid repeated loading
_PARK_FACTORS_CACHE = {}

def load_fangraphs_park_factors(year=2024):
    """
    Load official FanGraphs park factors using 3yr column.

    Args:
        year: Year to load park factors for

    Returns:
        dict: Team abbreviation -> park factor (per 100)
    """
    # Check global cache first (fastest)
    if year in _PARK_FACTORS_CACHE:
        return _PARK_FACTORS_CACHE[year]

    cache_file = os.path.join(CACHE_DIR, f'fangraphs_park_factors_{year}.json')

    # Try to load from file cache
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r') as f:
                cached_data = json.load(f)
            print(f"Loaded cached FanGraphs park factors for {year} ({len(cached_data)} teams)")
            # Store in global cache for future calls
            _PARK_FACTORS_CACHE[year] = cached_data
            return cached_data
        except Exception:
            pass

    print(f"=== LOADING FANGRAPHS PARK FACTORS ({year}) ===")

    # Try to load the specified year first
    park_factors_file = os.path.join(DATA_DIR, 'FanGraphs_Data', 'park_factors', f'fangraphs_parkfactors_{year}.csv')

    # If year file doesn't exist and it's 2025, fallback to 2024
    if not os.path.exists(park_factors_file) and year == 2025:
        print(f"2025 park factors not found, falling back to 2024")
        park_factors_file = os.path.join(DATA_DIR, 'FanGraphs_Data', 'park_factors', f'fangraphs_parkfactors_2024.csv')
        year = 2024  # Update cache file name
        cache_file = os.path.join(CACHE_DIR, f'fangraphs_park_factors_{year}.json')

    if not os.path.exists(park_factors_file):
        print(f"Warning: Park factors file not found: {park_factors_file}")
        return {}

    try:
        park_df = pd.read_csv(park_factors_file)

        # Check if required columns exist
        if 'Team' not in park_df.columns or '3yr' not in park_df.columns:
            print(f"Warning: Required columns ('Team', '3yr') not found in {park_factors_file}")
            return {}

        # Create team -> park factor mapping using abbreviations
        park_factors = {}
        for _, row in park_df.iterrows():
            team_name = row['Team']
            park_factor_3yr = row['3yr']

            if pd.notna(team_name) and pd.notna(park_factor_3yr):
                # Convert FanGraphs team name to abbreviation
                team_abbrev = FANGRAPHS_TO_ABBREV.get(str(team_name).strip())
                if team_abbrev:
                    park_factors[team_abbrev] = float(park_factor_3yr)

        # Save to cache
        try:
            with open(cache_file, 'w') as f:
                json.dump(park_factors, f, indent=2)
            print(f"Loaded FanGraphs park factors for {len(park_factors)} teams from {year}")
        except Exception as e:
            print(f"Warning: Could not save park factors cache: {e}")

        # Store in global cache for future calls
        _PARK_FACTORS_CACHE[year] = park_factors
        return park_factors

    except Exception as e:
        print(f"Error loading park factors from {park_factors_file}: {e}")
        return {}


def apply_park_factor_adjustments(player_stats, player_name, team, player_type='hitter', year=2024):
    """
    Apply park factor adjustments to player statistics using FanGraphs data

    Args:
        player_stats: dict of player statistics
        player_name: player name for tracking
        team: team abbreviation
        player_type: 'hitter' or 'pitcher'
        year: year for park factors (default 2024)

    Returns:
        dict: player_stats with park_factor_adjustment added
    """
    park_factors = load_fangraphs_park_factors(year)

    if not park_factors:
        # No park factors available, return neutral
        player_stats['park_factor_adjustment'] = 1.0
        return player_stats

    # Get team's park factor
    team_abbrev = str(team).upper().strip()
    if team_abbrev not in park_factors:
        # Team not found in park factors, return neutral
        player_stats['park_factor_adjustment'] = 1.0
        return player_stats

    park_factor = park_factors[team_abbrev]

    # Apply park factor adjustment
    if player_type == 'hitter':
        # For hitters: park factor > 100 helps offense, < 100 hurts offense
        # Adjustment is inverse: if park helps hitters, adjust stats down
        park_adjustment = 100 / park_factor

        # Apply adjustments to offensive stats
        if 'AVG' in player_stats:
            player_stats['AVG'] = player_stats['AVG'] * park_adjustment
        if 'OBP' in player_stats:
            player_stats['OBP'] = player_stats['OBP'] * park_adjustment
        if 'SLG' in player_stats:
            player_stats['SLG'] = player_stats['SLG'] * park_adjustment

    else:  # pitcher
        # For pitchers: park factor > 100 hurts pitchers, < 100 helps pitchers
        # Adjustment is direct: if park helps hitters, credit pitcher more
        park_adjustment = park_factor / 100

        # Apply adjustments to pitching stats
        if 'ERA' in player_stats:
            player_stats['ERA'] = player_stats['ERA'] * park_adjustment
        if 'HR%' in player_stats:
            player_stats['HR%'] = player_stats['HR%'] * park_adjustment

    player_stats['park_factor_adjustment'] = park_adjustment
    return player_stats


# FanGraphs team name to abbreviation mapping
FANGRAPHS_TO_ABBREV = {
    'Angels': 'LAA', 'Astros': 'HOU', 'Athletics': 'OAK', 'Mariners': 'SEA', 'Rangers': 'TEX',
    'Yankees': 'NYY', 'Red Sox': 'BOS', 'Blue Jays': 'TOR', 'Orioles': 'BAL', 'Rays': 'TB',
    'White Sox': 'CWS', 'Guardians': 'CLE', 'Tigers': 'DET', 'Royals': 'KC', 'Twins': 'MIN',
    'Braves': 'ATL', 'Marlins': 'MIA', 'Mets': 'NYM', 'Phillies': 'PHI', 'Nationals': 'WSN',
    'Cubs': 'CHC', 'Reds': 'CIN', 'Brewers': 'MIL', 'Pirates': 'PIT', 'Cardinals': 'STL',
    'Diamondbacks': 'ARI', 'Rockies': 'COL', 'Dodgers': 'LAD', 'Padres': 'SD', 'Giants': 'SF'
}


def normalize_team_abbreviation(team):
    """Normalize team abbreviation for consistent lookup."""
    if pd.isna(team):
        return None

    team_upper = str(team).upper().strip()
    return TEAM_MAPPINGS.get(team_upper, team_upper)


# For backward compatibility, create calculate_park_factors as alias
def calculate_park_factors(year=2024):
    """Backward compatibility function."""
    return load_fangraphs_park_factors(year)


if __name__ == "__main__":
    # Test park factor loading
    print("Testing FanGraphs park factor loading...")

    # Test 2024 factors
    factors_2024 = load_fangraphs_park_factors(2024)
    print(f"2024 factors loaded: {len(factors_2024)} teams")

    # Test 2025 factors (should fallback to 2024)
    factors_2025 = load_fangraphs_park_factors(2025)
    print(f"2025 factors loaded: {len(factors_2025)} teams")

    # Test specific team lookup
    if factors_2024:
        sample_teams = ['DET', 'LAA', 'NYY']
        for team in sample_teams:
            factor = factors_2024.get(team, 'Not found')
            print(f"{team}: {factor}")