"""
Park Factors Module for sWARm Current Season Analysis

This module applies park factor adjustments exactly as used in the monolithic sWARm system,
ensuring consistent park factor integration across the historical training pipeline.

Ported from: current_season_modules/park_factors.py
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

def clean_stadium_name(stadium_name):
    """
    Clean and normalize stadium names for consistent matching
    Handles renamed stadiums, eliminates duplicates, removes HTML/formatting
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

    # Handle renamed stadiums to eliminate duplicates
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

    return cleaned.strip()

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
                # Amplify the effect by 1.2x to make park differences more pronounced
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

def apply_park_factor_adjustments(player_stats, player_name, team, player_type='hitter'):
    """
    Apply park factor adjustments to player statistics

    Args:
        player_stats: dict of player statistics
        player_name: player name for tracking
        team: team abbreviation
        player_type: 'hitter' or 'pitcher'

    Returns:
        dict: player_stats with park_factor_adjustment added
    """
    park_factors = calculate_park_factors()

    # Load games data to find team's home stadium
    games_df = pd.read_csv(os.path.join(DATA_DIR, 'games(team_data).csv'))
    games_df['Stadium_Clean'] = games_df['Stadium'].apply(clean_stadium_name)

    # Get team's home stadium (most frequent home venue)
    team_home_games = games_df[games_df['home'] == team]
    if len(team_home_games) == 0:
        # No home games found, return neutral factor
        player_stats['park_factor_adjustment'] = 1.0
        return player_stats

    home_stadium = team_home_games['Stadium_Clean'].mode().iloc[0] if len(team_home_games['Stadium_Clean'].mode()) > 0 else None

    if home_stadium not in park_factors:
        # Stadium not found in park factors, return neutral
        player_stats['park_factor_adjustment'] = 1.0
        return player_stats

    park_factor = park_factors[home_stadium]

    # Apply park factor adjustment
    if player_type == 'hitter':
        # For hitters: park factor > 100 helps offense, < 100 hurts offense
        # Adjustment is inverse: if park helps hitters, adjust stats down
        park_adjustment = 100 / park_factor
    else:  # pitcher
        # For pitchers: park factor > 100 hurts pitchers, < 100 helps pitchers
        # Adjustment is direct: if park helps hitters, credit pitcher more
        park_adjustment = park_factor / 100

    player_stats['park_factor_adjustment'] = park_adjustment
    return player_stats