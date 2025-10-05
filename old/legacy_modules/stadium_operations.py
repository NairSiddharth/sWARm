"""
Stadium Operations Module

This module handles stadium name cleaning, standardization, and park factor integration.
Includes comprehensive stadium name mapping to handle renames and variations.
Extracted from cleanedDataParser.py for better modularity.
"""

import re
import pandas as pd

def clean_stadium_name(stadium_str):
    """Clean stadium names from HTML formatting and standardize to official names"""
    if pd.isna(stadium_str):
        return None

    clean_name = re.sub(r'<[^>]+>', '', str(stadium_str))
    clean_name = re.sub(r'\n+', ' ', clean_name)
    clean_name = re.sub(r'\t+', ' ', clean_name)
    clean_name = re.sub(r'\s+', ' ', clean_name).strip()

    if 'Coverage:' in clean_name:
        clean_name = clean_name.split('Coverage:')[0].strip()

    if not clean_name:
        return None

    # FIXED: Standardize stadium names to handle renames and variations
    # This fixes the 69 stadiums issue by mapping all variations to official names
    STADIUM_STANDARDIZATION = {
        # Houston variations
        'Minute Maid Park': 'Minute Maid Park',
        'Astros Field': 'Minute Maid Park',

        # Texas Rangers variations (Globe Life Park renamed to Globe Life Field)
        'Globe Life Park in Arlington': 'Globe Life Field',
        'Globe Life Park': 'Globe Life Field',
        'Globe Life Field': 'Globe Life Field',
        'Rangers Ballpark in Arlington': 'Globe Life Field',

        # Milwaukee variations (Miller Park renamed to American Family Field)
        'Miller Park': 'American Family Field',
        'American Family Field': 'American Family Field',

        # Miami variations (Marlins Park renamed to loanDepot park)
        'Marlins Park': 'loanDepot Park',
        'loanDepot Park': 'loanDepot Park',
        'loanDepot park': 'loanDepot Park',

        # Atlanta variations (Turner Field -> Truist Park)
        'Turner Field': 'Truist Park',
        'Truist Park': 'Truist Park',
        'SunTrust Park': 'Truist Park',

        # Chicago White Sox variations
        'Guaranteed Rate Field': 'Guaranteed Rate Field',
        'U.S. Cellular Field': 'Guaranteed Rate Field',
        'Comiskey Park': 'Guaranteed Rate Field',

        # Cleveland variations
        'Progressive Field': 'Progressive Field',
        'Jacobs Field': 'Progressive Field',

        # Standard mappings for exact matches
        'Angel Stadium': 'Angel Stadium',
        'Chase Field': 'Chase Field',
        'Citi Field': 'Citi Field',
        'Comerica Park': 'Comerica Park',
        'Coors Field': 'Coors Field',
        'Dodger Stadium': 'Dodger Stadium',
        'Fenway Park': 'Fenway Park',
        'Great American Ball Park': 'Great American Ball Park',
        'Kauffman Stadium': 'Kauffman Stadium',
        'Nationals Park': 'Nationals Park',
        'Oakland Coliseum': 'Oakland Coliseum',
        'Oriole Park at Camden Yards': 'Oriole Park at Camden Yards',
        'Petco Park': 'Petco Park',
        'PNC Park': 'PNC Park',
        'Rogers Centre': 'Rogers Centre',
        'T-Mobile Park': 'T-Mobile Park',
        'Target Field': 'Target Field',
        'Tropicana Field': 'Tropicana Field',
        'Wrigley Field': 'Wrigley Field',
        'Yankee Stadium': 'Yankee Stadium',
        'Busch Stadium': 'Busch Stadium',
        'Oracle Park': 'Oracle Park',
        'Citizens Bank Park': 'Citizens Bank Park'
    }

    # Check for exact matches first
    if clean_name in STADIUM_STANDARDIZATION:
        return STADIUM_STANDARDIZATION[clean_name]

    # Check for partial matches (case insensitive)
    clean_lower = clean_name.lower()
    for stadium_variant, official_name in STADIUM_STANDARDIZATION.items():
        if stadium_variant.lower() in clean_lower or clean_lower in stadium_variant.lower():
            return official_name

    # Return original name if no standardization found
    return clean_name

def get_regular_season_stadiums():
    """
    Returns the list of MLB regular season stadiums only, excluding:
    - Spring training facilities
    - Special event venues (Field of Dreams, London, etc.)
    - International stadiums
    - Minor league parks
    """
    regular_season_stadiums = {
        # 30 MLB regular season stadiums
        'American Family Field',      # Milwaukee Brewers (formerly Miller Park)
        'Angel Stadium',              # Los Angeles Angels
        'Busch Stadium',             # St. Louis Cardinals
        'Chase Field',               # Arizona Diamondbacks
        'Citi Field',                # New York Mets
        'Citizens Bank Park',        # Philadelphia Phillies
        'Comerica Park',             # Detroit Tigers
        'Coors Field',               # Colorado Rockies
        'Dodger Stadium',            # Los Angeles Dodgers
        'Fenway Park',               # Boston Red Sox
        'Globe Life Field',          # Texas Rangers
        'Great American Ball Park',  # Cincinnati Reds
        'Guaranteed Rate Field',     # Chicago White Sox
        'Kauffman Stadium',          # Kansas City Royals
        'loanDepot Park',            # Miami Marlins
        'Minute Maid Park',          # Houston Astros
        'Nationals Park',            # Washington Nationals
        'Oakland Coliseum',          # Oakland Athletics
        'Oracle Park',               # San Francisco Giants
        'Oriole Park at Camden Yards', # Baltimore Orioles
        'Petco Park',                # San Diego Padres
        'PNC Park',                  # Pittsburgh Pirates
        'Progressive Field',         # Cleveland Guardians
        'Rogers Centre',             # Toronto Blue Jays
        'T-Mobile Park',             # Seattle Mariners
        'Target Field',              # Minnesota Twins
        'Tropicana Field',           # Tampa Bay Rays
        'Truist Park',               # Atlanta Braves
        'Wrigley Field',             # Chicago Cubs
        'Yankee Stadium'             # New York Yankees
    }
    return regular_season_stadiums

def is_regular_season_stadium(stadium_name):
    """Check if a stadium is a regular season MLB venue"""
    regular_stadiums = get_regular_season_stadiums()
    cleaned_name = clean_stadium_name(stadium_name)
    return cleaned_name in regular_stadiums

def get_stadium_team_mapping():
    """Get mapping of stadiums to their home teams"""
    stadium_team_mapping = {
        'American Family Field': 'MIL',
        'Angel Stadium': 'LAA',
        'Busch Stadium': 'STL',
        'Chase Field': 'ARI',
        'Citi Field': 'NYM',
        'Citizens Bank Park': 'PHI',
        'Comerica Park': 'DET',
        'Coors Field': 'COL',
        'Dodger Stadium': 'LAD',
        'Fenway Park': 'BOS',
        'Globe Life Field': 'TEX',
        'Great American Ball Park': 'CIN',
        'Guaranteed Rate Field': 'CWS',
        'Kauffman Stadium': 'KC',
        'loanDepot Park': 'MIA',
        'Minute Maid Park': 'HOU',
        'Nationals Park': 'WSN',
        'Oakland Coliseum': 'OAK',
        'Oracle Park': 'SF',
        'Oriole Park at Camden Yards': 'BAL',
        'Petco Park': 'SD',
        'PNC Park': 'PIT',
        'Progressive Field': 'CLE',
        'Rogers Centre': 'TOR',
        'T-Mobile Park': 'SEA',
        'Target Field': 'MIN',
        'Tropicana Field': 'TB',
        'Truist Park': 'ATL',
        'Wrigley Field': 'CHC',
        'Yankee Stadium': 'NYY'
    }
    return stadium_team_mapping

def get_team_for_stadium(stadium_name):
    """Get the home team for a given stadium"""
    cleaned_name = clean_stadium_name(stadium_name)
    team_mapping = get_stadium_team_mapping()
    return team_mapping.get(cleaned_name, 'UNK')

def validate_stadium_data(stadium_list):
    """Validate and clean a list of stadium names"""
    cleaned_stadiums = []
    regular_stadiums = get_regular_season_stadiums()

    for stadium in stadium_list:
        cleaned = clean_stadium_name(stadium)
        if cleaned and cleaned in regular_stadiums:
            cleaned_stadiums.append(cleaned)

    return cleaned_stadiums

def get_stadium_rename_history():
    """Get history of stadium renames for reference"""
    rename_history = {
        'Miller Park': {'renamed_to': 'American Family Field', 'year': 2021},
        'Globe Life Park': {'renamed_to': 'Globe Life Field', 'year': 2020},
        'Marlins Park': {'renamed_to': 'loanDepot Park', 'year': 2021},
        'SunTrust Park': {'renamed_to': 'Truist Park', 'year': 2020},
        'U.S. Cellular Field': {'renamed_to': 'Guaranteed Rate Field', 'year': 2016},
        'Turner Field': {'renamed_to': 'Truist Park', 'year': 2017}
    }
    return rename_history