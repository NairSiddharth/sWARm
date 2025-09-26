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
    Calculate per-year values, then use most recent year or average

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

    # Store per-year values: {player_id: {year: baserunning_value}}
    player_year_values = {}

    # Load BP baserunning data
    bp_baserunning_dir = os.path.join(DATA_DIR, "BP_Data", "baserunning")
    if os.path.exists(bp_baserunning_dir):
        bp_files = glob.glob(os.path.join(bp_baserunning_dir, "*.csv"))
        print(f"Found {len(bp_files)} BP baserunning files")

        for file in bp_files:
            try:
                df = pd.read_csv(file)
                # Extract year from filename (e.g., bp_baserunning_2024.csv -> 2024)
                year_str = os.path.basename(file).split('_')[-1].replace('.csv', '')
                year = int(year_str) if year_str.isdigit() else 2020  # fallback

                # Key BP baserunning features: SB, CS, SB%, PO, XBT%
                required_cols = ['SB', 'CS', 'SB%', 'PO', 'XBT%']
                available_cols = [col for col in required_cols if col in df.columns]

                if available_cols:
                    for _, row in df.iterrows():
                        player_key = row.get('mlbid', row.get('Name', ''))
                        if player_key:
                            # Add null safety for baserunning calculations
                            try:
                                sb = float(row.get('SB', 0)) if pd.notna(row.get('SB')) else 0
                                cs = float(row.get('CS', 0)) if pd.notna(row.get('CS')) else 0
                                sb_pct_raw = row.get('SB%', 0)
                                xbt_pct_raw = row.get('XBT%', 0)

                                # Handle percentage conversion safely
                                if pd.notna(sb_pct_raw):
                                    sb_pct = float(sb_pct_raw) / 100 if float(sb_pct_raw) > 1 else float(sb_pct_raw)
                                else:
                                    sb_pct = 0

                                if pd.notna(xbt_pct_raw):
                                    xbt_pct = float(xbt_pct_raw) / 100 if float(xbt_pct_raw) > 1 else float(xbt_pct_raw)
                                else:
                                    xbt_pct = 0

                                # Baserunning value calculation using proper run values (PER YEAR)
                                # Stolen base run values: ~0.2 runs per SB, -0.4 runs per CS
                                steal_runs = (sb * 0.2) - (cs * 0.4)

                                # XBT% scaled to reasonable run impact (max ~2-3 runs per season)
                                # Elite XBT% ~50%, league average ~40%, scale difference * 10
                                xbt_runs = (xbt_pct - 0.40) * 10 if xbt_pct > 0 else 0

                                # Cap XBT contribution to reasonable range
                                xbt_runs = max(-3, min(3, xbt_runs))

                                baserunning_value = steal_runs + xbt_runs

                                # Store per-year value
                                if pd.notna(baserunning_value) and np.isfinite(baserunning_value):
                                    player_key_str = str(player_key)
                                    if player_key_str not in player_year_values:
                                        player_year_values[player_key_str] = {}
                                    player_year_values[player_key_str][year] = baserunning_value

                            except (ValueError, TypeError, ZeroDivisionError) as e:
                                # Skip invalid calculations
                                continue

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

                # Extract year from Statcast file if available
                file_year = 2020  # Default fallback
                for potential_year in range(2016, 2025):
                    if str(potential_year) in file:
                        file_year = potential_year
                        break

                if 'seconds_since_hit_090' in df.columns:
                    for _, row in df.iterrows():
                        player_key = row.get('player_id', row.get('player_name', ''))

                        # Add null safety for speed calculations
                        try:
                            seconds_090 = float(row.get('seconds_since_hit_090', 0)) if pd.notna(row.get('seconds_since_hit_090')) else 0

                            if player_key and seconds_090 > 0:
                                # Calculate speed: 90 feet in seconds_090 time
                                # Speed in ft/sec = 90 / seconds_090
                                speed_ft_per_sec = 90 / seconds_090

                                # Scale speed into baserunning value using realistic run impact
                                # MLB average sprint speed ~27 ft/sec, elite ~30+ ft/sec
                                # Speed impact: +/- 1-2 runs per season based on speed difference
                                speed_diff = speed_ft_per_sec - 27.0  # Difference from league average
                                speed_value = speed_diff * 0.5  # Scale to ~+/-2 runs max

                                # Cap speed contribution to reasonable range
                                speed_value = max(-2, min(2, speed_value))

                                # Store per-year speed value
                                if pd.notna(speed_value) and np.isfinite(speed_value):
                                    player_key_str = str(player_key)
                                    if player_key_str not in player_year_values:
                                        player_year_values[player_key_str] = {}

                                    # Add speed to existing year value or create new
                                    if file_year in player_year_values[player_key_str]:
                                        player_year_values[player_key_str][file_year] += speed_value
                                    else:
                                        player_year_values[player_key_str][file_year] = speed_value

                        except (ValueError, TypeError, ZeroDivisionError) as e:
                            # Skip invalid calculations
                            continue

                print(f"Processed Statcast running data: {len(df)} records")

            except Exception as e:
                print(f"Error loading Statcast running file {file}: {e}")

    # Convert per-year values to final baserunning values
    # Use most recent 3 years average, or most recent year if limited data
    baserunning_values = {}
    current_year = 2024  # Most recent year in dataset

    for player_id, year_values in player_year_values.items():
        if not year_values:
            continue

        # Get the most recent years (prioritize last 3 years)
        sorted_years = sorted(year_values.keys(), reverse=True)
        recent_years = sorted_years[:3]  # Last 3 years max

        # Calculate average of recent years (weighted toward most recent)
        if len(recent_years) >= 3:
            # 3+ years: weighted average (50%, 30%, 20%)
            weights = [0.5, 0.3, 0.2]
            final_value = sum(year_values[year] * weight for year, weight in zip(recent_years, weights))
        elif len(recent_years) == 2:
            # 2 years: weighted average (70%, 30%)
            final_value = year_values[recent_years[0]] * 0.7 + year_values[recent_years[1]] * 0.3
        else:
            # 1 year: use that year
            final_value = year_values[recent_years[0]]

        # Cap final value to target range [-7, 10]
        final_value = max(-7, min(10, final_value))
        baserunning_values[player_id] = final_value

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
    Calculate per-year values, then use most recent year or average

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

    # Store per-year values: {player_id: {year: defense_value}}
    player_year_values = {}

    # Load FanGraphs defensive standard data
    fg_defensive_dir = os.path.join(DATA_DIR, "FanGraphs_Data", "defensive")

    # Process FanGraphs standard defensive data
    if os.path.exists(fg_defensive_dir):
        standard_files = glob.glob(os.path.join(fg_defensive_dir, "fangraphs_defensive_standard_*.csv"))
        print(f"Found {len(standard_files)} FanGraphs standard defensive files")

        for file in standard_files:
            try:
                df = pd.read_csv(file)
                # Extract year from filename
                year_str = os.path.basename(file).split('_')[-1].replace('.csv', '')
                year = int(year_str) if year_str.isdigit() else 2020  # fallback

                # Standard defensive features: Pos, Inn, PO, A, E, DPS, DPT, DPF, Scp
                for _, row in df.iterrows():
                    # Fix ID mapping: Use MLBAMID instead of MLBAID
                    player_key = row.get('MLBAMID', row.get('Name', ''))
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

                    # Add null safety checks
                    try:
                        innings = float(innings) if pd.notna(innings) else 0
                        putouts = float(putouts) if pd.notna(putouts) else 0
                        assists = float(assists) if pd.notna(assists) else 0
                        errors = float(errors) if pd.notna(errors) else 0
                        dp_started = float(dp_started) if pd.notna(dp_started) else 0
                        dp_turned = float(dp_turned) if pd.notna(dp_turned) else 0
                        dp_finished = float(dp_finished) if pd.notna(dp_finished) else 0
                        scoops = float(scoops) if pd.notna(scoops) else 0

                        if innings > 0:
                            # Calculate fielding percentage
                            total_chances = putouts + assists + errors
                            fielding_pct = (putouts + assists) / total_chances if total_chances > 0 else 0

                            # Defensive value using realistic run impact scaling
                            # Fielding percentage impact: typical range .970-.990, league average ~.980
                            # Scale difference to ~10-20 runs per full season
                            fielding_runs = (fielding_pct - 0.980) * 1000  # Scale percentage to run impact

                            # Position-specific adjustments (runs per season)
                            position_runs = 0
                            if position in ['2B', 'SS']:
                                # Middle infield: double play value (~0.8 runs per DP)
                                dp_total = dp_started + dp_turned + dp_finished
                                position_runs = dp_total * 0.8
                            elif position == '1B':
                                # First base: scoop value (~0.5 runs per scoop)
                                position_runs = scoops * 0.5

                            # Combine components
                            total_runs = fielding_runs + position_runs

                            # Scale to full season (1400 innings ≈ full season)
                            defense_value = total_runs * (innings / 1400)

                            # Cap to realistic range [-25, 50] runs per season equivalent
                            defense_value = max(-25, min(50, defense_value))

                            # Store per-year value
                            if pd.notna(defense_value) and np.isfinite(defense_value):
                                player_key_str = str(player_key)
                                if player_key_str not in player_year_values:
                                    player_year_values[player_key_str] = {}

                                # Add to existing year value or create new
                                if year in player_year_values[player_key_str]:
                                    player_year_values[player_key_str][year] += defense_value
                                else:
                                    player_year_values[player_key_str][year] = defense_value

                    except (ValueError, TypeError, ZeroDivisionError) as e:
                        # Skip invalid calculations
                        continue

                print(f"Processed FanGraphs standard defensive data for {year}: {len(df)} records")

            except Exception as e:
                print(f"Error loading FanGraphs standard file {file}: {e}")

        # Process FanGraphs Statcast defensive data (catchers)
        statcast_files = glob.glob(os.path.join(fg_defensive_dir, "fangraphs_defensive_statcast_*.csv"))
        print(f"Found {len(statcast_files)} FanGraphs Statcast defensive files")

        for file in statcast_files:
            try:
                df = pd.read_csv(file)
                # Extract year from filename
                year_str = os.path.basename(file).split('_')[-1].replace('.csv', '')
                year = int(year_str) if year_str.isdigit() else 2020  # fallback

                # Statcast features for catchers: Throwing, Blocking, Framing, Arm
                for _, row in df.iterrows():
                    # Fix ID mapping: Use MLBAMID instead of MLBAID
                    player_key = row.get('MLBAMID', row.get('Name', ''))
                    if not player_key:
                        continue

                    # Add null safety for catcher stats
                    try:
                        throwing = float(row.get('Throwing', 0)) if pd.notna(row.get('Throwing')) else 0
                        blocking = float(row.get('Blocking', 0)) if pd.notna(row.get('Blocking')) else 0
                        framing = float(row.get('Framing', 0)) if pd.notna(row.get('Framing')) else 0
                        arm = float(row.get('Arm', 0)) if pd.notna(row.get('Arm')) else 0

                        # Catcher-specific defensive value
                        catcher_value = (throwing + blocking + framing + arm) / 4  # Average the components

                        # Store per-year catcher value
                        if pd.notna(catcher_value) and np.isfinite(catcher_value) and catcher_value != 0:
                            player_key_str = str(player_key)
                            if player_key_str not in player_year_values:
                                player_year_values[player_key_str] = {}

                            # Add to existing year value or create new
                            if year in player_year_values[player_key_str]:
                                player_year_values[player_key_str][year] += catcher_value
                            else:
                                player_year_values[player_key_str][year] = catcher_value

                    except (ValueError, TypeError, ZeroDivisionError) as e:
                        # Skip invalid calculations
                        continue

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

                    # Store per-year catch probability value
                    if total_value != 0:
                        player_key_str = str(player_key)
                        if player_key_str not in player_year_values:
                            player_year_values[player_key_str] = {}

                        # Add to existing year value or create new (need to extract year from file)
                        file_year = 2020  # Default fallback
                        for potential_year in range(2016, 2025):
                            if str(potential_year) in file:
                                file_year = potential_year
                                break

                        if file_year in player_year_values[player_key_str]:
                            player_year_values[player_key_str][file_year] += total_value
                        else:
                            player_year_values[player_key_str][file_year] = total_value

                print(f"Processed Statcast catch probability data: {len(df)} records")

            except Exception as e:
                print(f"Error loading Statcast catch file {file}: {e}")

    # Convert per-year values to final defense values
    # Use most recent 3 years average, or most recent year if limited data
    defense_values = {}
    current_year = 2024  # Most recent year in dataset

    for player_id, year_values in player_year_values.items():
        if not year_values:
            continue

        # Get the most recent years (prioritize last 3 years)
        sorted_years = sorted(year_values.keys(), reverse=True)
        recent_years = sorted_years[:3]  # Last 3 years max

        # Calculate average of recent years (weighted toward most recent)
        if len(recent_years) >= 3:
            # 3+ years: weighted average (50%, 30%, 20%)
            weights = [0.5, 0.3, 0.2]
            final_value = sum(year_values[year] * weight for year, weight in zip(recent_years, weights))
        elif len(recent_years) == 2:
            # 2 years: weighted average (70%, 30%)
            final_value = year_values[recent_years[0]] * 0.7 + year_values[recent_years[1]] * 0.3
        else:
            # 1 year: use that year
            final_value = year_values[recent_years[0]]

        # Cap final value to target range [-25, 50]
        final_value = max(-25, min(50, final_value))
        defense_values[player_id] = final_value

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