"""
WAR (Wins Above Replacement) Processing Module

This module handles WAR data cleaning and processing.
Extracted from cleanedDataParser.py for better modularity.
"""

import pandas as pd
from modules.data_loading import get_primary_dataframes

def clean_war():
    """
    Clean WAR data while keeping positional information for adjustments
    FIXED: Keep 'Pos' column for positional adjustments
    """
    dataframes = get_primary_dataframes()
    war_df = dataframes.get('war_df')

    # Drop unnecessary columns but keep 'Pos' for positional adjustments
    df = war_df.drop(['playerid', 'Team'], axis=1)
    return df.sort_values(by='Total WAR')

def get_war_for_player(player_name):
    """Get WAR value for a specific player"""
    df = clean_war()
    player_data = df[df['Name'] == player_name]

    if not player_data.empty:
        return {
            'total_war': player_data['Total WAR'].iloc[0],
            'primary_war': player_data.get('Primary WAR', pd.Series([None])).iloc[0],
            'position': player_data.get('Pos', pd.Series([None])).iloc[0]
        }

    return None

def get_top_war_players(n=10):
    """Get top N players by total WAR"""
    df = clean_war()
    return df.nlargest(n, 'Total WAR')[['Name', 'Total WAR', 'Primary WAR', 'Pos']]

def get_war_by_position(position=None):
    """Get WAR data filtered by position"""
    df = clean_war()

    if position:
        df = df[df['Pos'] == position]

    return df[['Name', 'Total WAR', 'Primary WAR', 'Pos']].sort_values(by='Total WAR', ascending=False)

def calculate_war_components(player_name):
    """Calculate WAR components for a player"""
    player_data = get_war_for_player(player_name)

    if player_data:
        total_war = player_data['total_war']
        primary_war = player_data['primary_war']

        if primary_war is not None:
            # For two-way players, separate hitting and pitching components
            hitting_war = total_war - primary_war  # Hitting + fielding + baserunning
            pitching_war = primary_war  # Primary pitching WAR
            return {
                'total_war': total_war,
                'hitting_war': hitting_war,
                'pitching_war': pitching_war,
                'is_two_way': True
            }
        else:
            # Single-role player
            return {
                'total_war': total_war,
                'hitting_war': total_war if player_data['position'] != 'P' else 0,
                'pitching_war': total_war if player_data['position'] == 'P' else 0,
                'is_two_way': False
            }

    return None