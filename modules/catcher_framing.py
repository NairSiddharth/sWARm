"""
Catcher Framing Module

This module handles catcher framing data processing and analysis.
Extracted from cleanedDataParser.py for better modularity.
"""

import pandas as pd
from modules.data_loading import load_yearly_catcher_framing_data

def clean_catcher_framing():
    """Extract catcher framing data and convert to player-based dictionary"""
    # Load catcher framing data using the data loading module
    yearly_framing_data = load_yearly_catcher_framing_data()

    # Handle if data is returned as dict instead of DataFrame
    if isinstance(yearly_framing_data, dict):
        # Convert to DataFrame if it's a dictionary
        yearly_framing_data = pd.DataFrame.from_dict(yearly_framing_data, orient='index').reset_index()

    framing_values = {}
    for _, row in yearly_framing_data.iterrows():
        # Combine first and last name for matching (note the space in column name)
        first_name = str(row.get(' first_name', '')).strip()
        last_name = str(row.get('last_name', '')).strip()

        # Handle missing names or ID-only rows
        if (first_name == 'nan' or first_name == '' or
            last_name == 'nan' or last_name == '' or
            first_name.isdigit()):
            continue

        player_name = f"{first_name} {last_name}"

        # Get framing runs value
        framing_runs = row.get('runs_extra_strikes', 0)
        if pd.notna(framing_runs) and framing_runs != 0:
            framing_values[player_name] = float(framing_runs)

    print(f"Loaded framing data for {len(framing_values)} catchers")
    return framing_values

def get_catcher_framing_value(player_name):
    """Get framing value for a specific player"""
    framing_data = clean_catcher_framing()
    return framing_data.get(player_name, 0.0)

def get_top_framers(n=10):
    """Get top N catchers by framing runs"""
    framing_data = clean_catcher_framing()
    sorted_framers = sorted(framing_data.items(), key=lambda x: x[1], reverse=True)
    return sorted_framers[:n]