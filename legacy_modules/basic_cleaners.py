"""
Basic Data Cleaners Module

This module contains core data cleaning and aggregation functions for hitter and pitcher data.
Extracted from cleanedDataParser.py for better modularity and maintainability.
"""

import pandas as pd
import numpy as np
from current_season_modules.data_loading import get_primary_dataframes

def clean_sorted_hitter():
    """Aggregate game-level hitter data to season-level for proper matching"""
    # Get data from the data loading module
    dataframes = get_primary_dataframes()
    hitter_by_game_df = dataframes.get('hitter_by_game_df')

    df = hitter_by_game_df.drop(['H-AB', 'AB', 'H', '#P', 'Game', 'Team', 'Hitter Id'], axis=1)

    # Convert numeric columns and handle missing/invalid values
    numeric_cols = ['K', 'BB', 'AVG', 'OBP', 'SLG']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        # Replace infinite values with reasonable defaults
        df[col] = df[col].replace([np.inf, -np.inf], 0)
        # Cap extreme values
        if col in ['AVG', 'OBP', 'SLG']:
            df[col] = df[col].clip(0, 1.0)  # Batting stats should be 0-1
        elif col in ['K', 'BB']:
            df[col] = df[col].clip(0, 300)  # Reasonable season maximums

    # OPTIMIZATION 4: Filter by game count to focus on meaningful players
    # Count games per player first
    game_counts = df.groupby('Hitters').size()
    qualified_players = game_counts[game_counts >= 10].index  # At least 10 games
    df_filtered = df[df['Hitters'].isin(qualified_players)]

    # Aggregate by player name to get season totals/averages
    aggregated = df_filtered.groupby('Hitters').agg({
        'K': 'sum',        # Total strikeouts
        'BB': 'sum',       # Total walks
        'AVG': 'mean',     # Average batting average
        'OBP': 'mean',     # Average on-base percentage
        'SLG': 'mean'      # Average slugging percentage
    }).reset_index()

    print(f"Aggregated hitter data: {len(df)} game records -> {len(aggregated)} qualified players (10+ games)")
    return aggregated.sort_values(by='Hitters')

def clean_sorted_pitcher():
    """Aggregate game-level pitcher data to season-level for proper matching"""
    # Get data from the data loading module
    dataframes = get_primary_dataframes()
    pitcher_by_game_df = dataframes.get('pitcher_by_game_df')

    df = pitcher_by_game_df.drop(['R', 'ER', 'PC', 'Game', 'Team', 'Extra', 'Pitcher Id'], axis=1)

    # Convert numeric columns and handle missing/invalid values
    numeric_cols = ['IP', 'BB', 'K', 'HR', 'ERA']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        # Replace infinite values with reasonable defaults
        df[col] = df[col].replace([np.inf, -np.inf], 0)
        # Cap extreme values
        if col == 'ERA':
            df[col] = df[col].clip(0, 30.0)  # Reasonable ERA range
        elif col in ['BB', 'K', 'HR']:
            df[col] = df[col].clip(0, 500)  # Reasonable season maximums
        elif col == 'IP':
            df[col] = df[col].clip(0, 300)  # Max innings in a season

    # Aggregate by player name to get season totals/averages
    aggregated = df.groupby('Pitchers').agg({
        'IP': 'sum',       # Total innings pitched
        'BB': 'sum',       # Total walks allowed
        'K': 'sum',        # Total strikeouts
        'HR': 'sum',       # Total home runs allowed
        'ERA': 'mean'      # Average ERA
    }).reset_index()

    print(f"Aggregated pitcher data: {len(df)} game records -> {len(aggregated)} unique players")
    return aggregated.sort_values(by='Pitchers')

def clean_defensive_players_fast():
    """Fast version - only catcher framing, skip traditional fielding"""
    from legacy_modules.catcher_framing import clean_catcher_framing

    print("Using fast mode - only catcher framing data")
    framing_values = clean_catcher_framing()
    defensive_values = {}
    for player, framing_runs in framing_values.items():
        defensive_values[player] = framing_runs / 10.0
    return defensive_values

def clean_sorted_baserunning_fast():
    """Fast version - return empty baserunning data"""
    print("Using fast mode - skipping baserunning data")
    return {}

def clean_defensive_players():
    """
    Combine traditional fielding and catcher framing data
    Uses existing defensive_metrics module for core calculations
    """
    from legacy_modules.defensive_metrics import calculate_outs_above_average_from_fielding_notes
    from legacy_modules.catcher_framing import clean_catcher_framing

    # Get OAA values from defensive metrics module
    oaa_values = calculate_outs_above_average_from_fielding_notes()

    # Get catcher framing values
    framing_values = clean_catcher_framing()

    # Combine both metrics
    combined_defensive = {}

    # Add OAA values (converted to defensive runs)
    for player_key, oaa_data in oaa_values.items():
        if isinstance(oaa_data, dict):
            oaa_value = oaa_data.get('oaa', 0)
            player_name = oaa_data.get('player_name', player_key)
        else:
            oaa_value = oaa_data
            player_name = player_key

        # Convert OAA to defensive runs (rough conversion: 1 OAA ≈ 2 defensive runs)
        defensive_runs = oaa_value * 2.0
        combined_defensive[player_name] = defensive_runs

    # Add catcher framing values
    for player, framing_runs in framing_values.items():
        if player in combined_defensive:
            combined_defensive[player] += framing_runs
        else:
            combined_defensive[player] = framing_runs

    print(f"Combined defensive metrics: {len(combined_defensive)} players")
    print(f"  OAA players: {len(oaa_values)}")
    print(f"  Framing players: {len(framing_values)}")

    return combined_defensive