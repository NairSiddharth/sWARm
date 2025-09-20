"""
WARP (Wins Above Replacement Player) Processing Module

This module handles WARP data cleaning and processing for both hitters and pitchers.
Includes legacy functions and enhanced yearly processing with team mapping.
Extracted from cleanedDataParser.py for better modularity.
"""

import os
import pandas as pd
from modules.data_loading import get_primary_dataframes, load_yearly_bp_data
from modules.defensive_metrics import create_player_team_mapping

# Path configuration
CACHE_DIR = r"C:\Users\nairs\Documents\GithubProjects\oWAR\cache"

def clean_warp_hitter():
    """Legacy function - uses only 2021 data"""
    dataframes = get_primary_dataframes()
    warp_hitter_df = dataframes.get('warp_hitter_df')

    df = warp_hitter_df.drop(['bpid', 'mlbid', 'Age', 'DRC+', '+/-', 'PA', 'R', 'RBI',
                              'ISO', 'K%', 'BB%', 'Whiff%'], axis=1)
    return df.sort_values(by='WARP')

def clean_warp_pitcher():
    """Legacy function - uses only 2021 data"""
    dataframes = get_primary_dataframes()
    warp_pitcher_df = dataframes.get('warp_pitcher_df')

    df = warp_pitcher_df.drop(['bpid', 'mlbid', 'DRA-', 'DRA', 'DRA SD', 'cFIP',
                               'GS', 'W', 'L', 'ERA', 'RA9', 'Whiff%'], axis=1)
    return df.sort_values(by='WARP')

def clean_yearly_warp_hitter():
    """
    Enhanced function - uses all available years of BP hitter data (2016-2024)
    Returns expanded dataset for improved training
    """
    cache_file = os.path.join(CACHE_DIR, "yearly_warp_hitter_cleaned.json")

    # Check cache first
    if os.path.exists(cache_file):
        try:
            cached_df = pd.read_json(cache_file, orient='records')
            print(f"Loaded cached yearly WARP hitter data ({len(cached_df)} player-seasons)")
            return cached_df
        except:
            pass

    print("Preparing yearly WARP hitter data...")
    bp_data = load_yearly_bp_data()
    team_mapping = create_player_team_mapping()
    hitter_records = []

    for player_year, warp in bp_data['hitters'].items():
        name, year = player_year.rsplit('_', 1)

        # Try to get team from game data mapping using multiple name formats
        team = 'UNK'

        # Try exact match first
        if player_year in team_mapping:
            team = team_mapping[player_year]
        else:
            # Try abbreviated format: "Juan Soto" -> "J. Soto"
            name_parts = name.split()
            if len(name_parts) >= 2:
                abbreviated_name = f"{name_parts[0][0]}. {name_parts[-1]}"
                abbreviated_key = f"{abbreviated_name}_{year}"
                if abbreviated_key in team_mapping:
                    team = team_mapping[abbreviated_key]

        hitter_records.append({
            'Name': name,
            'Year': int(year),
            'WARP': warp,
            'Team': team
        })

    df = pd.DataFrame(hitter_records)

    # Cache the result
    try:
        df.to_json(cache_file, orient='records', indent=2)
        print(f"Cached yearly WARP hitter data ({len(df)} player-seasons)")
    except Exception as e:
        print(f"Warning: Could not cache data: {e}")

    return df.sort_values(by='WARP')

def clean_yearly_warp_pitcher():
    """
    Enhanced function - uses all available years of BP pitcher data (2016-2024)
    Returns expanded dataset for improved training with dual CSV support for 2022-2024
    """
    cache_file = os.path.join(CACHE_DIR, "yearly_warp_pitcher_cleaned_v2.json")

    # Check cache first
    if os.path.exists(cache_file):
        try:
            cached_df = pd.read_json(cache_file, orient='records')
            print(f"Loaded cached yearly WARP pitcher data ({len(cached_df)} player-seasons)")
            return cached_df
        except:
            pass

    print("Preparing yearly WARP pitcher data...")
    bp_data = load_yearly_bp_data()
    team_mapping = create_player_team_mapping()
    pitcher_records = []

    for player_year, warp in bp_data['pitchers'].items():
        name, year = player_year.rsplit('_', 1)

        # Try to get team from game data mapping using multiple name formats
        team = 'UNK'

        # Try exact match first
        if player_year in team_mapping:
            team = team_mapping[player_year]
        else:
            # Try abbreviated format: "Juan Soto" -> "J. Soto"
            name_parts = name.split()
            if len(name_parts) >= 2:
                abbreviated_name = f"{name_parts[0][0]}. {name_parts[-1]}"
                abbreviated_key = f"{abbreviated_name}_{year}"
                if abbreviated_key in team_mapping:
                    team = team_mapping[abbreviated_key]

        pitcher_records.append({
            'Name': name,
            'Year': int(year),
            'WARP': warp,
            'Team': team
        })

    df = pd.DataFrame(pitcher_records)

    # Cache the result
    try:
        df.to_json(cache_file, orient='records', indent=2)
        print(f"Cached yearly WARP pitcher data ({len(df)} player-seasons)")
    except Exception as e:
        print(f"Warning: Could not cache data: {e}")

    return df.sort_values(by='WARP')

def get_warp_for_player(player_name, year=None, player_type='hitter'):
    """Get WARP value for a specific player and year"""
    if player_type == 'hitter':
        df = clean_yearly_warp_hitter()
    else:
        df = clean_yearly_warp_pitcher()

    player_data = df[df['Name'] == player_name]

    if year:
        player_data = player_data[player_data['Year'] == year]

    if not player_data.empty:
        return player_data['WARP'].iloc[0]

    return None

def get_top_warp_players(n=10, year=None, player_type='hitter'):
    """Get top N players by WARP for a given year"""
    if player_type == 'hitter':
        df = clean_yearly_warp_hitter()
    else:
        df = clean_yearly_warp_pitcher()

    if year:
        df = df[df['Year'] == year]

    return df.nlargest(n, 'WARP')[['Name', 'Year', 'WARP', 'Team']]