"""
Baserunning Analytics Module

This module handles enhanced baserunning calculations using run expectancy matrices
and situational adjustments. Includes both simple and enhanced baserunning systems.
Extracted from cleanedDataParser.py for better modularity.
"""

import os
import re
import json
import pandas as pd
import numpy as np
from modules.data_loading import get_primary_dataframes, load_yearly_bp_baserunning_data

# Path configuration
CACHE_DIR = r"C:\Users\nairs\Documents\GithubProjects\oWAR\cache"
DATA_DIR = r"C:\Users\nairs\Documents\GithubProjects\oWAR\MLB Player Data"

# Regex for extracting player names
capitalized_words = r"((?:[A-Z][a-z']+ ?)+)"

def clean_sorted_baserunning():
    """Basic baserunning data processing with simple +/- scoring"""
    # Check cache first
    cache_file = os.path.join(CACHE_DIR, "baserunning_values.json")
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                cached_data = json.load(f)
            print(f"Loaded cached baserunning data ({len(cached_data)} players)")
            return cached_data
        except:
            pass

    print("Processing baserunning data (this may take a moment)...")
    dataframes = get_primary_dataframes()
    baserunning_by_game_df = dataframes.get('baserunning_by_game_df')

    df = baserunning_by_game_df.drop(['Game'], axis=1)
    sorted_df = df.sort_values(by='Stat')

    baserunning_values = {}

    for _, row in sorted_df.iterrows():
        statlines = str(row['Stat']).split(',')
        if not statlines:
            continue

        if statlines[0] == 'SB':
            players = re.findall(capitalized_words, str(row.get('Data', '')))
            for p in players:
                baserunning_values[p] = baserunning_values.get(p, 0) + (1 / 3)

        elif statlines[0] in ['CS', 'Picked Off']:
            players = re.findall(capitalized_words, str(row.get('Data', '')))
            for p in players:
                baserunning_values[p] = baserunning_values.get(p, 0) - (1 / 3)

    # Cache the result
    try:
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(baserunning_values, f, indent=2)
        print(f"Cached baserunning data ({len(baserunning_values)} players)")
    except:
        pass

    return baserunning_values

def calculate_steal_run_value(from_base, to_base, outs, success=True):
    """
    Calculate run value of a steal attempt using run expectancy matrix

    Args:
        from_base: Starting base (1=1st, 2=2nd, 3=3rd)
        to_base: Target base (2=2nd, 3=3rd, 4=home)
        outs: Number of outs when steal occurred
        success: Whether steal was successful

    Returns:
        float: Run value of the steal attempt
    """
    # MLB Run Expectancy Matrix (approximate values)
    # [base_state][outs] = expected runs
    run_expectancy = {
        'empty': [0.48, 0.25, 0.10],
        '1st': [0.85, 0.50, 0.22],
        '2nd': [1.06, 0.64, 0.32],
        '3rd': [1.21, 0.82, 0.37],
        '1st_2nd': [1.44, 0.89, 0.42],
        '1st_3rd': [1.78, 1.13, 0.51],
        '2nd_3rd': [1.96, 1.29, 0.63],
        'loaded': [2.29, 1.54, 0.80]
    }

    # Simplified for steal situations (runner on single base)
    if from_base == 1:  # 1st to 2nd
        before_state = '1st'
        if success:
            after_state = '2nd'
            after_outs = outs
        else:
            after_state = 'empty'
            after_outs = outs + 1
    elif from_base == 2:  # 2nd to 3rd
        before_state = '2nd'
        if success:
            after_state = '3rd'
            after_outs = outs
        else:
            after_state = 'empty'
            after_outs = outs + 1
    elif from_base == 3:  # 3rd to home
        before_state = '3rd'
        if success:
            return 1.0  # One run scored
        else:
            after_state = 'empty'
            after_outs = outs + 1
    else:
        return 0.0

    # Prevent out-of-bounds for outs
    outs = min(outs, 2)
    after_outs = min(after_outs, 2)

    if from_base == 3 and success:
        return 1.0  # Run scored

    before_value = run_expectancy.get(before_state, [0, 0, 0])[outs]
    after_value = run_expectancy.get(after_state, [0, 0, 0])[after_outs]

    return after_value - before_value

def parse_baserunning_event(data_str):
    """
    Parse baserunning event data string to extract player and steal information

    Args:
        data_str: Raw data string from baserunning events

    Returns:
        dict: Parsed event data with player, bases, etc.
    """
    if pd.isna(data_str):
        return {}

    # Extract player name (before first parenthesis)
    player_match = re.match(r'\s*([^(]+)', data_str)
    player_name = player_match.group(1).strip() if player_match else ""

    # Extract steal number and base information
    base_pattern = r'\((\d+),\s*(\w+)\s+base'
    base_match = re.search(base_pattern, data_str)

    if base_match:
        steal_number = int(base_match.group(1))
        target_base = base_match.group(2).lower()

        # Convert base names to numbers
        base_mapping = {'2nd': 2, '3rd': 3, 'home': 4}
        to_base = base_mapping.get(target_base, 0)
        from_base = to_base - 1 if to_base > 1 else 1

        return {
            'player': player_name,
            'steal_number': steal_number,
            'from_base': from_base,
            'to_base': to_base,
            'raw_data': data_str
        }

    return {'player': player_name, 'raw_data': data_str}

def extract_year_from_game_id(game_id):
    """
    Extract year from ESPN game ID format:
    2016: 36XXXXXXX (e.g., 360403107)
    2017: 37XXXXXXX (e.g., 370403119)
    2018: 38XXXXXXX (e.g., 380329114)
    2019: 401XXXXXX (e.g., 401074733)
    2020: 401XXXXXX (e.g., 401225674) - pandemic season
    2021: 401XXXXXX (e.g., 401227058)
    """
    game_id_str = str(game_id)

    # Check first 2 digits for 2016-2018
    if game_id_str.startswith('36'):
        return 2016
    elif game_id_str.startswith('37'):
        return 2017
    elif game_id_str.startswith('38'):
        return 2018
    elif game_id_str.startswith('401'):
        # Need to distinguish 2019, 2020, 2021 by date ranges in game ID
        # 2019: 401074733 to 401169053
        # 2020: 401225674 to 401226568 (pandemic season)
        # 2021: 401227058 to 401229475
        game_id_int = int(game_id_str)

        if 401074733 <= game_id_int <= 401169053:
            return 2019
        elif 401225674 <= game_id_int <= 401226568:
            return 2020
        elif 401227058 <= game_id_int <= 401229475:
            return 2021
        else:
            # For any 401 IDs outside known ranges, try to infer from ID value
            if game_id_int < 401200000:
                return 2019  # Earlier 401 IDs likely 2019
            elif game_id_int < 401227000:
                return 2020  # Mid-range likely 2020
            else:
                return 2021  # Later IDs likely 2021

    return None  # Unknown format

def calculate_enhanced_baserunning_values():
    """
    Calculate enhanced baserunning values using BP data and speed adjustments

    This replaces the original play-by-play system with BP's cleaner data
    combined with Statcast speed data for more accurate, speed-adjusted values.

    Returns:
        dict: {player_name: speed_adjusted_baserunning_value}
    """
    print("=== CALCULATING ENHANCED BASERUNNING VALUES ===")
    print("Using BP data with Statcast speed adjustments and quintile-based expectations")

    # Use the new speed-adjusted calculation system
    return calculate_speed_adjusted_baserunning_values()

def calculate_defensive_baserunning_impact():
    """Calculate defensive impact of baserunning (for catchers, pitchers)"""
    # This could be expanded to include defensive baserunning metrics
    print("Defensive baserunning impact calculation not yet implemented")
    return {}

def get_baserunning_for_player(player_name, enhanced=True):
    """Get baserunning value for a specific player"""
    if enhanced:
        data = calculate_enhanced_baserunning_values()
    else:
        data = clean_sorted_baserunning()

    return data.get(player_name, 0.0)

def calculate_speed_adjusted_baserunning_values():
    """
    Calculate speed-adjusted baserunning values using BP data and Statcast running speeds

    Uses quintile-based speed adjustments for SB% and XBT% expectations:
    - Fast players (top quintile): Higher expected SB% threshold, more expected XBT%
    - Slow players (bottom quintile): Lower expected thresholds

    Returns:
        dict: {player_name: average_speed_adjusted_baserunning_value}
    """
    print("=== CALCULATING SPEED-ADJUSTED BASERUNNING VALUES ===")
    print("Using BP data with Statcast speed adjustments and quintile-based expectations")

    # Check cache first
    cache_file = os.path.join(CACHE_DIR, "speed_adjusted_baserunning_values.json")
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                cached_data = json.load(f)
            print(f"Loaded cached speed-adjusted baserunning values ({len(cached_data)} players)")
            return cached_data
        except:
            pass

    # Load BP baserunning and speed data
    bp_data = load_yearly_bp_baserunning_data()
    baserunning_data = bp_data['baserunning']
    speed_data = bp_data['running_speed']

    if not baserunning_data:
        print("⚠️  No BP baserunning data available")
        return {}

    print(f"Processing {len(baserunning_data)} baserunning records and {len(speed_data)} speed records")

    # Calculate speed quintiles for players with speed data
    speed_values = []
    player_speeds = {}

    for key, data in speed_data.items():
        speed = data.get('speed_ft_per_sec')
        if speed and pd.notna(speed) and speed > 0:
            player_speeds[key] = speed
            speed_values.append(speed)

    if len(speed_values) < 5:
        print("WARNING: Insufficient speed data for quintile calculation, using fixed thresholds")
        speed_quintiles = [20, 22, 24, 26, 30]  # Fixed ft/sec thresholds
    else:
        speed_quintiles = np.percentile(speed_values, [20, 40, 60, 80, 100])

    print(f"Speed quintiles (ft/sec): {[f'{q:.1f}' for q in speed_quintiles]}")

    # Define expected thresholds by speed quintile
    sb_thresholds = [70, 72, 75, 77, 80]  # SB% thresholds by quintile (slow to fast)
    xbt_bonuses = [1.2, 1.1, 1.0, 0.9, 0.8]  # XBT% value multipliers (reward slow, neutral fast)

    # Process each player-season
    player_season_values = {}
    missing_speed_count = 0

    for key, br_data in baserunning_data.items():
        player_name = br_data['name']
        year = br_data['year']

        # Get speed data if available
        speed = None
        speed_quintile = 2  # Default to middle quintile if no speed data

        if key in player_speeds:
            speed = player_speeds[key]
            # Determine speed quintile (0 = slowest, 4 = fastest)
            speed_quintile = 0
            for i, threshold in enumerate(speed_quintiles[:-1]):  # Exclude 100th percentile
                if speed <= threshold:
                    speed_quintile = i
                    break
            else:
                speed_quintile = 4  # Fastest quintile
        else:
            missing_speed_count += 1

        # Calculate speed-adjusted baserunning value
        baserunning_value = 0.0

        # Stolen Base Component
        sb = br_data.get('SB', 0) or 0
        cs = br_data.get('CS', 0) or 0
        sb_attempts = sb + cs

        if sb_attempts > 0:
            actual_sb_pct = (sb / sb_attempts) * 100
            expected_sb_pct = sb_thresholds[speed_quintile]

            # Calculate SB run value based on performance vs expectation
            sb_run_value = 0.0

            # Successful steals: base value + bonus for exceeding expectations
            if actual_sb_pct >= expected_sb_pct:
                # Base value for successful steals
                sb_run_value += sb * 0.2  # ~0.2 runs per successful steal
                # Bonus for exceeding speed-based expectations
                excess_pct = actual_sb_pct - expected_sb_pct
                sb_run_value += (excess_pct / 100) * sb_attempts * 0.1
            else:
                # Penalty for underperforming expectations
                shortfall_pct = expected_sb_pct - actual_sb_pct
                sb_run_value += sb * 0.2  # Base value
                sb_run_value -= (shortfall_pct / 100) * sb_attempts * 0.15

            # Caught stealing penalty
            sb_run_value -= cs * 0.3  # ~0.3 runs lost per caught stealing

            baserunning_value += sb_run_value

        # Extra Base Taking Component
        xbt_pct = br_data.get('XBT_pct')
        if xbt_pct and pd.notna(xbt_pct):
            # Convert percentage to decimal if needed
            xbt_decimal = float(xbt_pct) / 100 if float(xbt_pct) > 1 else float(xbt_pct)

            # Speed-adjusted XBT value
            xbt_multiplier = xbt_bonuses[speed_quintile]
            xbt_value = xbt_decimal * 0.15 * xbt_multiplier  # Base XBT run value adjusted by speed
            baserunning_value += xbt_value

        # Pick-off penalty
        po = br_data.get('PO', 0) or 0
        baserunning_value -= po * 0.25  # Penalty for pick-offs

        # Use BP's derived metrics as additional factors
        traa = br_data.get('TRAA', 0) or 0
        if pd.notna(traa) and traa != 0:
            baserunning_value += float(traa) * 0.5  # Weight BP's total baserunning metric

        # Store player-season value
        player_season_values[key] = {
            'player_name': player_name,
            'year': year,
            'baserunning_value': baserunning_value,
            'speed_quintile': speed_quintile,
            'speed_ft_per_sec': speed,
            'sb_attempts': sb_attempts,
            'actual_sb_pct': (sb / sb_attempts * 100) if sb_attempts > 0 else None,
            'expected_sb_pct': sb_thresholds[speed_quintile]
        }

    if missing_speed_count > 0:
        print(f"WARNING: {missing_speed_count} player-seasons missing speed data (used middle quintile)")

    # Aggregate by player (average across seasons)
    player_aggregated = {}
    player_season_counts = {}

    for data in player_season_values.values():
        player_name = data['player_name']

        if player_name not in player_aggregated:
            player_aggregated[player_name] = 0.0
            player_season_counts[player_name] = 0

        player_aggregated[player_name] += data['baserunning_value']
        player_season_counts[player_name] += 1

    # Calculate final averages
    summary_values = {}
    for player_name in player_aggregated:
        avg_value = player_aggregated[player_name] / player_season_counts[player_name]
        summary_values[player_name] = avg_value

    # Cache the results
    try:
        cache_data = summary_values.copy()
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, indent=2)
        print(f"Cached speed-adjusted baserunning values ({len(summary_values)} players)")
    except Exception as e:
        print(f"Warning: Could not cache data: {e}")

    print(f"SUCCESS: Calculated speed-adjusted baserunning values for {len(summary_values)} players")

    # Print some sample results for verification
    if summary_values:
        sorted_results = sorted(summary_values.items(), key=lambda x: x[1], reverse=True)[:5]
        print("Top 5 speed-adjusted baserunners:")
        for name, value in sorted_results:
            print(f"  {name}: {value:.3f}")

    return summary_values

def get_top_baserunners(n=10, enhanced=True):
    """Get top N baserunners"""
    if enhanced:
        data = calculate_enhanced_baserunning_values()
    else:
        data = clean_sorted_baserunning()

    sorted_runners = sorted(data.items(), key=lambda x: x[1], reverse=True)
    return sorted_runners[:n]