"""
Park Factors Module for oWAR Analysis

This module handles all park factor calculations and adjustments including:
- Base park factor calculations using home/away methodology
- Enhanced park factors with stronger effects (1.5x amplification)
- Regular season park factor filtering
- Player-specific park adjustments for hitters and pitchers
- Stadium name cleaning and normalization
"""

import os
import pandas as pd
import json
from pathlib import Path

# Import configuration from parent module
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Constants
DATA_DIR = r"C:\Users\nairs\Documents\GithubProjects\oWAR\MLB Player Data"
CACHE_DIR = r"C:\Users\nairs\Documents\GithubProjects\oWAR\cache"

# Ensure cache directory exists
os.makedirs(CACHE_DIR, exist_ok=True)

__all__ = [
    'calculate_park_factors',
    'calculate_regular_season_park_factors',
    'get_player_park_adjustment',
    'apply_enhanced_hitter_park_adjustments',
    'apply_enhanced_pitcher_park_adjustments',
    'get_player_park_games',
    'clean_stadium_name',
    'get_regular_season_stadiums'
]

def clean_stadium_name(stadium_name):
    """
    Clean and normalize stadium names for consistent matching
    FIXED: Handles renamed stadiums, eliminates duplicates, removes HTML/formatting

    Args:
        stadium_name: Raw stadium name from data

    Returns:
        str: Cleaned and normalized stadium name
    """
    if pd.isna(stadium_name):
        return None

    # Basic cleaning and remove HTML/formatting
    cleaned = str(stadium_name).strip()

    # Remove HTML tags, embedded content, newlines, and extra whitespace
    import re
    cleaned = re.sub(r'<[^>]*>', '', cleaned)  # Remove HTML tags
    cleaned = re.sub(r'\s*Coverage:.*$', '', cleaned, flags=re.MULTILINE)  # Remove coverage info
    cleaned = re.sub(r'\s*\([^)]*\)', '', cleaned)  # Remove parenthetical content like (Spring Training)
    cleaned = re.sub(r'\s+', ' ', cleaned)  # Normalize whitespace
    cleaned = cleaned.strip()

    # CRITICAL: Handle renamed stadiums to eliminate duplicates
    stadium_mappings = {
        # Milwaukee Brewers - Miller Park renamed to American Family Field
        'Miller Park': 'American Family Field',

        # Miami Marlins - Marlins Park renamed to loanDepot park
        'Marlins Park': 'loanDepot park',

        # Texas Rangers - Moved from Globe Life Park to Globe Life Field
        'Globe Life Park in Arlington': 'Globe Life Field',
        'The Ballpark in Arlington': 'Globe Life Field',  # Even older name

        # Handle other common variations
        'Citizens Bank Park': 'Citizens Bank Park',  # Keep as-is (Phillies)
        'Guaranteed Rate Field': 'Guaranteed Rate Field',  # Keep as-is (White Sox)
        'Kauffman Stadium': 'Kauffman Stadium',  # Keep as-is (Royals)
        'PNC Park': 'PNC Park',  # Keep as-is (Pirates)
    }

    # Apply stadium mappings first
    if cleaned in stadium_mappings:
        cleaned = stadium_mappings[cleaned]

    # Preserve names that should keep their full titles
    preserve_names = {
        'Citizens Bank Park', 'Guaranteed Rate Field', 'Kauffman Stadium',
        'PNC Park', 'loanDepot park', 'Globe Life Field', 'American Family Field',
        'Citi Field', 'Busch Stadium', 'Coors Field', 'Chase Field', 'Target Field',
        'Fenway Park', 'Wrigley Field', 'Dodger Stadium', 'Yankee Stadium',
        'Tropicana Field', 'Progressive Field', 'Comerica Park', 'Angel Stadium',
        'Petco Park', 'Oracle Park', 'Truist Park', 'Nationals Park',
        'Great American Ball Park', 'Oriole Park at Camden Yards'
    }

    if cleaned not in preserve_names:
        # Apply limited cleaning for non-preserved stadiums only
        replacements = {
            ' Ballpark': '',  # Only remove "Ballpark" suffix
        }

        for old, new in replacements.items():
            if old in cleaned:
                cleaned = cleaned.replace(old, new)

    return cleaned.strip()

def get_regular_season_stadiums():
    """
    Get list of regular season MLB stadiums (30 teams) + special event venues
    FIXED: Added missing stadiums, removed duplicates, included special venues

    Returns:
        set: Set of regular season stadium names plus special event venues
    """
    # 30 regular season MLB stadiums (current as of 2024)
    regular_stadiums = {
        # AL West
        'Angel Stadium',          # Los Angeles Angels
        'Minute Maid Park',       # Houston Astros
        'Oakland Coliseum',       # Oakland Athletics
        'T-Mobile Park',          # Seattle Mariners
        'Globe Life Field',       # Texas Rangers (NEW stadium)

        # AL East
        'Tropicana Field',        # Tampa Bay Rays
        'Rogers Centre',          # Toronto Blue Jays
        'Oriole Park at Camden Yards',  # Baltimore Orioles
        'Fenway Park',           # Boston Red Sox
        'Yankee Stadium',        # New York Yankees

        # AL Central
        'Progressive Field',      # Cleveland Guardians
        'Comerica Park',         # Detroit Tigers
        'Guaranteed Rate Field', # Chicago White Sox
        'Kauffman Stadium',      # Kansas City Royals
        'Target Field',          # Minnesota Twins

        # NL West
        'Coors Field',           # Colorado Rockies
        'Chase Field',           # Arizona Diamondbacks
        'Dodger Stadium',        # Los Angeles Dodgers
        'Petco Park',            # San Diego Padres
        'Oracle Park',           # San Francisco Giants

        # NL East
        'Truist Park',           # Atlanta Braves
        'loanDepot park',        # Miami Marlins (RENAMED from Marlins Park)
        'Citi Field',            # New York Mets (MISSING - ADDED)
        'Citizens Bank Park',    # Philadelphia Phillies
        'Nationals Park',        # Washington Nationals

        # NL Central
        'Wrigley Field',         # Chicago Cubs
        'Great American Ball Park',  # Cincinnati Reds
        'American Family Field', # Milwaukee Brewers (RENAMED from Miller Park)
        'PNC Park',              # Pittsburgh Pirates
        'Busch Stadium',         # St. Louis Cardinals (MISSING - ADDED)
    }

    # Special event venues (valid for regular season games)
    special_venues = {
        "BB&T Ballpark at Historic Bowman Field",  # Little League Classic
        "Fort Bragg Field",                        # 2016 Military Event
        "Field of Dreams",                         # 2021+ Field of Dreams Game
        "Sahlen Field",                           # Blue Jays COVID-19 home (2020-2021)
    }

    # Combine regular and special venues
    all_valid_stadiums = regular_stadiums.union(special_venues)

    return all_valid_stadiums

def calculate_park_factors():
    """
    Enhanced park factors using home/away run scoring methodology with stronger effects

    Returns:
        dict: {stadium: park_factor} where >100 = hitter-friendly, <100 = pitcher-friendly
    """

    # Check cache first
    cache_file = os.path.join(CACHE_DIR, "enhanced_park_factors.json")
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                cached_data = json.load(f)
            print(f"Loaded cached enhanced park factors ({len(cached_data)} stadiums)")
            return cached_data
        except:
            pass

    print("=== CALCULATING ENHANCED PARK FACTORS ===")

    # Load games data
    games_df = pd.read_csv(os.path.join(DATA_DIR, 'games(team_data).csv'))
    games_df['Stadium_Clean'] = games_df['Stadium'].apply(clean_stadium_name)
    games_df = games_df.dropna(subset=['Stadium_Clean'])

    park_stats = {}

    for stadium in games_df['Stadium_Clean'].unique():
        if not stadium:
            continue

        stadium_games = games_df[games_df['Stadium_Clean'] == stadium]
        home_team = stadium_games['home'].mode().iloc[0] if len(stadium_games['home'].mode()) > 0 else None

        if not home_team:
            continue

        # Calculate home/away splits for the home team
        home_games = stadium_games[stadium_games['home'] == home_team]
        away_games = games_df[(games_df['away'] == home_team) & (games_df['Stadium_Clean'] != stadium)]

        if len(home_games) < 10 or len(away_games) < 10:  # Need sufficient sample size
            continue

        # Calculate run scoring rates
        home_runs_for = home_games['home-score'].sum()
        home_runs_against = home_games['away-score'].sum()
        home_games_count = len(home_games)

        away_runs_for = away_games['away-score'].sum()
        away_runs_against = away_games['home-score'].sum()
        away_games_count = len(away_games)

        if home_games_count > 0 and away_games_count > 0:
            # Runs per game rates
            home_rpg_for = home_runs_for / home_games_count
            home_rpg_against = home_runs_against / home_games_count
            home_total_rpg = (home_runs_for + home_runs_against) / home_games_count

            away_rpg_for = away_runs_for / away_games_count
            away_rpg_against = away_runs_against / away_games_count
            away_total_rpg = (away_runs_for + away_runs_against) / away_games_count

            # Park factor calculation (amplified effect)
            if away_total_rpg > 0:
                park_factor = (home_total_rpg / away_total_rpg) * 100
                # Amplify the effect by 1.5x to make park differences more pronounced
                park_factor = 100 + (park_factor - 100) * 1.2
                park_stats[stadium] = round(park_factor, 1)

    # Cache the result
    try:
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(park_stats, f, indent=2)
        print(f"Cached enhanced park factors ({len(park_stats)} stadiums)")
    except Exception as e:
        print(f"Warning: Could not cache park factors: {e}")

    print(f"Calculated park factors for {len(park_stats)} stadiums")
    return park_stats

def calculate_regular_season_park_factors():
    """
    Enhanced park factors for REGULAR SEASON stadiums only, filtering out:
    - Spring training facilities
    - Special event venues
    - International/exhibition stadiums

    Returns:
        dict: {stadium: park_factor} for 30 MLB regular season stadiums only
    """

    # Check cache first
    cache_file = os.path.join(CACHE_DIR, "regular_season_park_factors.json")
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                cached_data = json.load(f)
            print(f"Loaded cached regular season park factors ({len(cached_data)} stadiums)")
            return cached_data
        except:
            pass

    print("=== CALCULATING REGULAR SEASON PARK FACTORS ===")

    # Get all park factors
    all_park_factors = calculate_park_factors()

    # Get regular season stadiums list
    regular_stadiums = get_regular_season_stadiums()

    # Filter to regular season only
    regular_park_factors = {}
    for stadium, factor in all_park_factors.items():
        if stadium in regular_stadiums:
            regular_park_factors[stadium] = factor

    print(f"Filtered from {len(all_park_factors)} total stadiums to {len(regular_park_factors)} regular season stadiums")
    print("Excluded non-regular season venues:")
    excluded = set(all_park_factors.keys()) - set(regular_park_factors.keys())
    for venue in sorted(excluded):
        venue_type = 'Spring Training' if any(x in venue for x in ['Stadium', 'Park', 'Field']) and venue not in regular_stadiums else 'Special/International'
        print(f"  - {venue} ({venue_type})")

    # Cache the result
    try:
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(regular_park_factors, f, indent=2)
        print(f"Cached regular season park factors")
    except Exception as e:
        print(f"Warning: Could not cache park factors: {e}")

    return regular_park_factors

def get_player_park_adjustment(player_name, team):
    """Get weighted park factor for a player based on games played"""

    park_factors = calculate_park_factors()

    # Load games data
    games_df = pd.read_csv(os.path.join(DATA_DIR, 'games(team_data).csv'))
    games_df['Stadium_Clean'] = games_df['Stadium'].apply(clean_stadium_name)

    # Get player's games
    player_games = games_df[(games_df['home'] == team) | (games_df['away'] == team)]

    park_games = {}

    # Count games at each stadium
    home_games = player_games[player_games['home'] == team]
    for stadium in home_games['Stadium_Clean'].dropna():
        park_games[stadium] = park_games.get(stadium, 0) + 1

    away_games = player_games[player_games['away'] == team]
    for stadium in away_games['Stadium_Clean'].dropna():
        park_games[stadium] = park_games.get(stadium, 0) + 1

    if not park_games:
        return 100.0  # Neutral

    # Calculate weighted average
    total_games = sum(park_games.values())
    weighted_park_factor = 0

    for stadium, games in park_games.items():
        if stadium in park_factors:
            weight = games / total_games
            weighted_park_factor += park_factors[stadium] * weight

    return weighted_park_factor if weighted_park_factor > 0 else 100.0

def get_player_park_games(player_name, team, year=None):
    """
    Get breakdown of games played at different ballparks for a player

    Returns:
        dict: {stadium: games_played}
    """

    games_df = pd.read_csv(os.path.join(DATA_DIR, 'games(team_data).csv'))
    games_df['Stadium_Clean'] = games_df['Stadium'].apply(clean_stadium_name)

    # Filter by team
    player_games = games_df[(games_df['home'] == team) | (games_df['away'] == team)]

    if year:
        # Filter by year if provided (extract year from date)
        player_games['Year'] = pd.to_datetime(player_games['Date']).dt.year
        player_games = player_games[player_games['Year'] == year]

    # Count games at each stadium
    park_games = {}

    # Home games
    home_games = player_games[player_games['home'] == team]
    for stadium in home_games['Stadium_Clean'].dropna():
        park_games[stadium] = park_games.get(stadium, 0) + 1

    # Away games
    away_games = player_games[player_games['away'] == team]
    for stadium in away_games['Stadium_Clean'].dropna():
        park_games[stadium] = park_games.get(stadium, 0) + 1

    return park_games

def apply_enhanced_hitter_park_adjustments(hitter_stats, player_name, team, park_factors=None):
    """
    Apply enhanced park factor adjustments to hitter statistics with stronger effects

    Args:
        hitter_stats: dict of hitter stats
        player_name: player name
        team: team abbreviation
        park_factors: Pre-calculated park factors (optional, will calculate if None)

    Returns:
        dict: adjusted hitter stats with stronger park effects
    """

    if park_factors is None:
        park_factors = calculate_park_factors()
    park_games = get_player_park_games(player_name, team)

    if not park_games:
        return hitter_stats

    # Calculate weighted park factor based on games played
    total_games = sum(park_games.values())
    weighted_park_factor = 0

    for stadium, games in park_games.items():
        if stadium in park_factors:
            weight = games / total_games
            weighted_park_factor += park_factors[stadium] * weight

    if weighted_park_factor == 0:
        weighted_park_factor = 100  # Neutral if no park data

    # Apply STRONGER adjustment (inverse of park factor for hitters)
    # If park helps hitters (>100), reduce stats MORE. If park hurts hitters (<100), boost stats MORE
    base_adjustment = 100 / weighted_park_factor

    # Amplify the adjustment effect for stronger park factors
    if weighted_park_factor > 100:
        # Hitter-friendly park - apply larger penalty
        amplified_adjustment = 1 - (1 - base_adjustment) * 1.5
    else:
        # Pitcher-friendly park - apply larger boost
        amplified_adjustment = 1 + (base_adjustment - 1) * 1.5

    adjusted_stats = hitter_stats.copy()

    # Adjust key offensive stats with amplified effects
    offensive_stats = ['AVG', 'OBP', 'SLG']
    for stat in offensive_stats:
        if stat in adjusted_stats:
            adjusted_stats[f'{stat}_park_adj'] = adjusted_stats[stat] * amplified_adjustment

    adjusted_stats['park_factor'] = weighted_park_factor
    adjusted_stats['park_adjustment'] = amplified_adjustment
    adjusted_stats['park_effect_strength'] = 'ENHANCED'

    return adjusted_stats

def apply_enhanced_pitcher_park_adjustments(pitcher_stats, player_name, team, park_factors=None):
    """
    Apply enhanced park factor adjustments to pitcher performance evaluation

    Args:
        pitcher_stats: dict of pitcher stats
        player_name: player name
        team: team abbreviation
        park_factors: Pre-calculated park factors (optional, will calculate if None)

    Returns:
        dict: adjusted pitcher stats with enhanced park context
    """

    if park_factors is None:
        park_factors = calculate_park_factors()
    park_games = get_player_park_games(player_name, team)

    if not park_games:
        return pitcher_stats

    # Calculate weighted park factor
    total_games = sum(park_games.values())
    weighted_park_factor = 0

    for stadium, games in park_games.items():
        if stadium in park_factors:
            weight = games / total_games
            weighted_park_factor += park_factors[stadium] * weight

    if weighted_park_factor == 0:
        weighted_park_factor = 100

    # Calculate ENHANCED pitcher park adjustment
    # If park helps hitters (>100), pitcher deserves MORE extra credit
    # If park helps pitchers (<100), pitcher gets MORE penalty
    if weighted_park_factor > 100:
        # Hitter-friendly park - boost pitcher performance more
        pitcher_boost = 1 + ((weighted_park_factor - 100) / 150)  # Stronger effect: Max 1.67x boost
    else:
        # Pitcher-friendly park - larger penalty
        pitcher_boost = 1 - ((100 - weighted_park_factor) / 300)  # Stronger penalty: Max 0.67x

    adjusted_stats = pitcher_stats.copy()
    adjusted_stats['park_factor'] = weighted_park_factor
    adjusted_stats['park_boost'] = pitcher_boost
    adjusted_stats['park_effect_strength'] = 'ENHANCED'

    # Apply ERA adjustment
    if 'ERA' in adjusted_stats:
        # For ERA, lower park factor (pitcher-friendly) should result in higher adjusted ERA
        # Higher park factor (hitter-friendly) should result in lower adjusted ERA
        era_adjustment = 100 / weighted_park_factor
        adjusted_stats['ERA_park_adj'] = adjusted_stats['ERA'] * era_adjustment

    return adjusted_stats