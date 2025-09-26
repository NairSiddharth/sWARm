"""
Enhanced Features Module for sWARm Analysis

This module calculates Enhanced_Baserunning and Enhanced_Defense features using
comprehensive data sources as specified:

Baserunning:
- BP Data: SB, CS, SB%, PO, XBT%
- Statcast: seconds_since_hit_090 for speed calculations (90ft = home to first)

Defense:
- FanGraphs Standard: Pos, Inn, PO, A, E, DPS, DPT, DPF, Scp
- FanGraphs Statcast: Throwing, Blocking, Framing, Arm (catchers)
- Statcast Catch Probability: 5-star rating system with position-specific adjustments

Key matching: MLBAID (primary) → name fallback
"""

import os
import pandas as pd
import numpy as np
import glob
from datetime import datetime

# Constants
DATA_DIR = r"C:\Users\nairs\Documents\GithubProjects\oWAR\MLB Player Data"
CACHE_DIR = r"C:\Users\nairs\Documents\GithubProjects\oWAR\cache"

# Ensure cache directory exists
os.makedirs(CACHE_DIR, exist_ok=True)

def load_baserunning_data():
    """
    Load baserunning data from BP and Statcast sources

    Returns:
        dict: {player_id/name: baserunning_value}
    """
    print("=== LOADING BASERUNNING DATA ===")

    # Check cache first
    cache_file = os.path.join(CACHE_DIR, "enhanced_baserunning_values.json")
    if os.path.exists(cache_file):
        try:
            import json
            with open(cache_file, 'r') as f:
                cached_data = json.load(f)
            print(f"Loaded cached baserunning data ({len(cached_data)} players)")
            return cached_data
        except:
            pass

    baserunning_values = {}

    # Load BP baserunning data
    bp_baserunning_dir = os.path.join(DATA_DIR, "BP_Data", "baserunning")
    if os.path.exists(bp_baserunning_dir):
        bp_files = glob.glob(os.path.join(bp_baserunning_dir, "*.csv"))
        print(f"Found {len(bp_files)} BP baserunning files")

        for file in bp_files:
            try:
                df = pd.read_csv(file)
                year = os.path.basename(file).split('_')[-1].replace('.csv', '')

                # Key BP baserunning features: SB, CS, SB%, PO, XBT%
                required_cols = ['SB', 'CS', 'SB%', 'PO', 'XBT%']
                available_cols = [col for col in required_cols if col in df.columns]

                if available_cols:
                    for _, row in df.iterrows():
                        player_key = row.get('mlbid', row.get('Name', ''))
                        if player_key:
                            # Calculate baserunning value from BP data
                            sb = row.get('SB', 0)
                            cs = row.get('CS', 0)
                            sb_pct = row.get('SB%', 0) / 100 if row.get('SB%', 0) > 1 else row.get('SB%', 0)
                            xbt_pct = row.get('XBT%', 0) / 100 if row.get('XBT%', 0) > 1 else row.get('XBT%', 0)

                            # Baserunning value calculation
                            # Reward efficient stealing (>75% success rate) and extra base taking
                            steal_value = sb * sb_pct if sb_pct >= 0.75 else sb * sb_pct - cs * 0.5
                            xbt_value = xbt_pct * 10  # Scale XBT% contribution

                            baserunning_value = steal_value + xbt_value
                            baserunning_values[player_key] = baserunning_value

                print(f"Processed BP baserunning data for {year}: {len(df)} records")

            except Exception as e:
                print(f"Error loading BP baserunning file {file}: {e}")

    # Load Statcast running splits for speed calculations
    statcast_running_dir = os.path.join(DATA_DIR, "Statcast_Data", "running_splits")
    if os.path.exists(statcast_running_dir):
        statcast_files = glob.glob(os.path.join(statcast_running_dir, "*.csv"))
        print(f"Found {len(statcast_files)} Statcast running files")

        for file in statcast_files:
            try:
                df = pd.read_csv(file)

                if 'seconds_since_hit_090' in df.columns:
                    for _, row in df.iterrows():
                        player_key = row.get('player_id', row.get('player_name', ''))
                        seconds_090 = row.get('seconds_since_hit_090', 0)

                        if player_key and seconds_090 > 0:
                            # Calculate speed: 90 feet in seconds_090 time
                            # 90 feet = 27.432 meters
                            # Speed in ft/sec = 90 / seconds_090
                            speed_ft_per_sec = 90 / seconds_090

                            # Convert to mph: ft/sec * 0.681818
                            speed_mph = speed_ft_per_sec * 0.681818

                            # Scale speed into baserunning value (faster = better)
                            # MLB average sprint speed ~27 ft/sec, elite ~30+ ft/sec
                            speed_value = max(0, (speed_ft_per_sec - 24) * 2)  # Scale starting from 24 ft/sec

                            # Add to existing baserunning value or create new
                            if player_key in baserunning_values:
                                baserunning_values[player_key] += speed_value
                            else:
                                baserunning_values[player_key] = speed_value

                print(f"Processed Statcast running data: {len(df)} records")

            except Exception as e:
                print(f"Error loading Statcast running file {file}: {e}")

    print(f"Enhanced baserunning calculated for {len(baserunning_values)} players")

    # Cache the results
    try:
        import json
        with open(cache_file, 'w') as f:
            json.dump(baserunning_values, f, indent=2)
        print("Cached enhanced baserunning values")
    except Exception as e:
        print(f"Warning: Could not cache baserunning data: {e}")

    return baserunning_values

def load_defense_data():
    """
    Load comprehensive defense data from FanGraphs and Statcast sources

    Returns:
        dict: {player_id/name: defense_value}
    """
    print("=== LOADING DEFENSE DATA ===")

    # Check cache first
    cache_file = os.path.join(CACHE_DIR, "enhanced_defense_values.json")
    if os.path.exists(cache_file):
        try:
            import json
            with open(cache_file, 'r') as f:
                cached_data = json.load(f)
            print(f"Loaded cached defense data ({len(cached_data)} players)")
            return cached_data
        except:
            pass

    defense_values = {}

    # Load FanGraphs defensive standard data
    fg_defensive_dir = os.path.join(DATA_DIR, "FanGraphs_Data", "defensive")

    # Process FanGraphs standard defensive data
    if os.path.exists(fg_defensive_dir):
        standard_files = glob.glob(os.path.join(fg_defensive_dir, "fangraphs_defensive_standard_*.csv"))
        print(f"Found {len(standard_files)} FanGraphs standard defensive files")

        for file in standard_files:
            try:
                df = pd.read_csv(file)
                year = os.path.basename(file).split('_')[-1].replace('.csv', '')

                # Standard defensive features: Pos, Inn, PO, A, E, DPS, DPT, DPF, Scp
                for _, row in df.iterrows():
                    player_key = row.get('MLBAID', row.get('Name', ''))
                    if not player_key:
                        continue

                    position = row.get('Pos', '')
                    innings = row.get('Inn', 0)
                    putouts = row.get('PO', 0)
                    assists = row.get('A', 0)
                    errors = row.get('E', 0)
                    dp_started = row.get('DPS', 0)
                    dp_turned = row.get('DPT', 0)
                    dp_finished = row.get('DPF', 0)
                    scoops = row.get('Scp', 0)  # First base specific

                    if innings > 0:
                        # Calculate fielding percentage
                        total_chances = putouts + assists + errors
                        fielding_pct = (putouts + assists) / total_chances if total_chances > 0 else 0

                        # Position-specific defensive value
                        # Base defensive value from fielding percentage above/below average
                        base_value = (fielding_pct - 0.980) * 100  # League average ~.980

                        # Add double play contributions (middle infield bonus)
                        if position in ['2B', 'SS']:
                            dp_value = (dp_started + dp_turned + dp_finished) * 0.5
                            base_value += dp_value

                        # First base scoop bonus
                        if position == '1B' and scoops > 0:
                            base_value += scoops * 0.2

                        # Scale by innings played
                        defense_value = base_value * (innings / 1000)  # Scale to reasonable range

                        defense_values[player_key] = defense_values.get(player_key, 0) + defense_value

                print(f"Processed FanGraphs standard defensive data for {year}: {len(df)} records")

            except Exception as e:
                print(f"Error loading FanGraphs standard file {file}: {e}")

        # Process FanGraphs Statcast defensive data (catchers)
        statcast_files = glob.glob(os.path.join(fg_defensive_dir, "fangraphs_defensive_statcast_*.csv"))
        print(f"Found {len(statcast_files)} FanGraphs Statcast defensive files")

        for file in statcast_files:
            try:
                df = pd.read_csv(file)
                year = os.path.basename(file).split('_')[-1].replace('.csv', '')

                # Statcast features for catchers: Throwing, Blocking, Framing, Arm
                for _, row in df.iterrows():
                    player_key = row.get('MLBAID', row.get('Name', ''))
                    if not player_key:
                        continue

                    throwing = row.get('Throwing', 0)
                    blocking = row.get('Blocking', 0)
                    framing = row.get('Framing', 0)
                    arm = row.get('Arm', 0)

                    # Catcher-specific defensive value
                    catcher_value = (throwing + blocking + framing + arm) / 4  # Average the components

                    if catcher_value != 0:
                        defense_values[player_key] = defense_values.get(player_key, 0) + catcher_value

                print(f"Processed FanGraphs Statcast defensive data for {year}: {len(df)} records")

            except Exception as e:
                print(f"Error loading FanGraphs Statcast file {file}: {e}")

    # Load Statcast catch probability data (outfielders)
    statcast_catch_dir = os.path.join(DATA_DIR, "Statcast_Data", "catch_probability")
    if os.path.exists(statcast_catch_dir):
        catch_files = glob.glob(os.path.join(statcast_catch_dir, "catch_probability_*.csv"))
        print(f"Found {len(catch_files)} Statcast catch probability files")

        # Position-specific league averages for catch probability
        position_averages = {
            'RF': {'5star': 0.10, '4star': 0.35, '3star': 0.60, '2star': 0.80, '1star': 0.95},
            'CF': {'5star': 0.15, '4star': 0.40, '3star': 0.65, '2star': 0.85, '1star': 0.97},
            'LF': {'5star': 0.08, '4star': 0.30, '3star': 0.55, '2star': 0.75, '1star': 0.93}
        }

        for file in catch_files:
            try:
                df = pd.read_csv(file)

                # 5-star catch probability features
                star_columns = [
                    'n_fieldout_5stars', 'n_opp_5stars', 'n_5star_percent',
                    'n_fieldout_4stars', 'n_opp_4stars', 'n_4star_percent',
                    'n_fieldout_3stars', 'n_opp_3stars', 'n_3star_percent',
                    'n_fieldout_2stars', 'n_opp_2stars', 'n_2star_percent',
                    'n_fieldout_1stars', 'n_opp_1stars', 'n_1star_percent'
                ]

                for _, row in df.iterrows():
                    player_key = row.get('player_id', row.get('player_name', ''))
                    position = row.get('position', 'CF')  # Default to CF if not specified

                    if not player_key or position not in position_averages:
                        continue

                    # Calculate catch probability value by difficulty
                    total_value = 0
                    for star_level in ['5star', '4star', '3star', '2star', '1star']:
                        catches = row.get(f'n_fieldout_{star_level}', 0)
                        opportunities = row.get(f'n_opp_{star_level}', 0)

                        if opportunities > 0:
                            player_pct = catches / opportunities
                            expected_pct = position_averages[position][star_level]

                            # Value = (actual - expected) * opportunities * difficulty weight
                            difficulty_weights = {'5star': 5, '4star': 4, '3star': 3, '2star': 2, '1star': 1}
                            value = (player_pct - expected_pct) * opportunities * difficulty_weights[star_level]
                            total_value += value

                    if total_value != 0:
                        defense_values[player_key] = defense_values.get(player_key, 0) + total_value

                print(f"Processed Statcast catch probability data: {len(df)} records")

            except Exception as e:
                print(f"Error loading Statcast catch file {file}: {e}")

    print(f"Enhanced defense calculated for {len(defense_values)} players")

    # Cache the results
    try:
        import json
        with open(cache_file, 'w') as f:
            json.dump(defense_values, f, indent=2)
        print("Cached enhanced defense values")
    except Exception as e:
        print(f"Warning: Could not cache defense data: {e}")

    return defense_values

def get_enhanced_features():
    """
    Get both enhanced baserunning and defense features

    Returns:
        tuple: (baserunning_dict, defense_dict)
    """
    print("Loading enhanced features...")

    baserunning_data = load_baserunning_data()
    defense_data = load_defense_data()

    return baserunning_data, defense_data

def get_player_enhanced_features(player_identifier, baserunning_data=None, defense_data=None):
    """
    Get enhanced features for a specific player using MLBAID or name fallback

    Args:
        player_identifier: MLBAID (preferred) or player name
        baserunning_data: Pre-loaded baserunning data (optional)
        defense_data: Pre-loaded defense data (optional)

    Returns:
        dict: {'Enhanced_Baserunning': value, 'Enhanced_Defense': value}
    """
    if baserunning_data is None or defense_data is None:
        baserunning_data, defense_data = get_enhanced_features()

    # Try direct lookup first (MLBAID)
    baserunning_value = baserunning_data.get(player_identifier, 0.0)
    defense_value = defense_data.get(player_identifier, 0.0)

    # If not found and identifier looks like a name, try name matching
    if (baserunning_value == 0.0 and defense_value == 0.0 and
        isinstance(player_identifier, str) and ' ' in player_identifier):

        # Simple name matching - could be enhanced with fuzzy matching
        for key, value in baserunning_data.items():
            if isinstance(key, str) and player_identifier.lower() in key.lower():
                baserunning_value = value
                break

        for key, value in defense_data.items():
            if isinstance(key, str) and player_identifier.lower() in key.lower():
                defense_value = value
                break

    return {
        'Enhanced_Baserunning': baserunning_value,
        'Enhanced_Defense': defense_value
    }