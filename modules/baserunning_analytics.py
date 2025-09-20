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
from modules.data_loading import get_primary_dataframes

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
    Calculate enhanced baserunning values using run expectancy with SEASONAL totals (FIXED!)

    This fixes the multi-year aggregation issue - now calculates realistic seasonal values
    instead of career aggregates that were inflating totals like Altuve's "50 SB"

    Returns:
        dict: {player_name: average_seasonal_baserunning_value}
    """
    print("=== CALCULATING ENHANCED BASERUNNING VALUES ===")
    print("Using run expectancy matrix and situational adjustments")

    # Check cache first
    cache_file = os.path.join(CACHE_DIR, "enhanced_baserunning_values.json")
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                cached_data = json.load(f)
            print(f"Loaded cached enhanced baserunning values ({len(cached_data)} players)")
            return cached_data
        except:
            pass

    # Load baserunning events
    baserunning_file = os.path.join(DATA_DIR, 'baserunningNotes(player_offense_data).csv')
    if not os.path.exists(baserunning_file):
        print(f"⚠️  Baserunning file not found: {baserunning_file}")
        return {}

    df = pd.read_csv(baserunning_file)
    print(f"Loaded {len(df)} baserunning events")

    # Initialize seasonal player values
    player_season_values = {}

    # Process each event with seasonal separation
    for _, row in df.iterrows():
        game_id = row['Game']
        team = row['Team']
        stat_type = row['Stat']
        data_str = row['Data']

        # Extract year from game ID
        year = extract_year_from_game_id(game_id)
        if year is None:
            continue  # Skip unrecognized game IDs

        # Parse the event
        event = parse_baserunning_event(data_str)
        player_name = event.get('player', '')

        if not player_name:
            continue

        # Create seasonal key
        season_key = f"{player_name}_{year}"

        # Initialize player-season if not seen
        if season_key not in player_season_values:
            player_season_values[season_key] = {
                'total_value': 0.0,
                'steals': 0,
                'caught_stealing': 0,
                'picked_offs': 0,
                'year': year,
                'player_name': player_name
            }

        # Calculate run value based on event type
        run_value = 0.0

        if stat_type == 'SB' and 'from_base' in event and 'to_base' in event:
            # Successful stolen base
            outs = 1  # Average assumption
            run_value = calculate_steal_run_value(
                event['from_base'], event['to_base'], outs, success=True
            )
            player_season_values[season_key]['steals'] += 1

        elif stat_type == 'CS' and 'from_base' in event:
            # Caught stealing
            outs = 1  # Average assumption
            to_base = event.get('to_base', event['from_base'] + 1)
            run_value = calculate_steal_run_value(
                event['from_base'], to_base, outs, success=False
            )
            player_season_values[season_key]['caught_stealing'] += 1

        elif stat_type == 'Picked Off':
            # Picked off - significant negative value
            run_value = -0.3  # Approximate value
            player_season_values[season_key]['picked_offs'] += 1

        # Add situational bonuses/penalties
        situational_multiplier = 1.0
        if stat_type == 'SB':
            if event.get('to_base') == 3:
                situational_multiplier = 1.2  # Stealing 3rd more valuable
            elif event.get('to_base') == 4:
                situational_multiplier = 2.0   # Stealing home extremely valuable

        final_value = run_value * situational_multiplier
        player_season_values[season_key]['total_value'] += final_value

    # Aggregate by player (average across seasons for current use)
    player_aggregated = {}
    player_season_counts = {}

    for season_key, data in player_season_values.items():
        player_name = data['player_name']

        if player_name not in player_aggregated:
            player_aggregated[player_name] = 0.0
            player_season_counts[player_name] = 0

        player_aggregated[player_name] += data['total_value']
        player_season_counts[player_name] += 1

    # Calculate averages
    summary_values = {}

    for player_name in player_aggregated:
        # Use average seasonal value for now
        avg_value = player_aggregated[player_name] / player_season_counts[player_name]
        summary_values[player_name] = avg_value

    # Cache the result
    try:
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(summary_values, f, indent=2)
        print(f"Cached enhanced baserunning values ({len(summary_values)} players)")
    except Exception as e:
        print(f"Warning: Could not cache data: {e}")

    print(f"✅ Calculated enhanced baserunning values for {len(summary_values)} players")
    return summary_values

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

def get_top_baserunners(n=10, enhanced=True):
    """Get top N baserunners"""
    if enhanced:
        data = calculate_enhanced_baserunning_values()
    else:
        data = clean_sorted_baserunning()

    sorted_runners = sorted(data.items(), key=lambda x: x[1], reverse=True)
    return sorted_runners[:n]