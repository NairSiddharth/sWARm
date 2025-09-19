"""
Defensive Metrics Module for oWAR Analysis

This module handles all defensive calculations and evaluations including:
- Fielding analysis from game notes with contextual weighting
- Outs Above Average (OAA) calculations using play-by-play data
- Position-specific defensive adjustments and replacement levels
- Double play contribution analysis with role-based weighting
- Enhanced defensive metrics combining multiple data sources
- Catcher framing value integration
- Defensive WAR component calculations with realistic bounds
"""

import os
import pandas as pd
import numpy as np
import json
import re
from pathlib import Path

# Import configuration from parent module
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import required functions from other modules
from modules.data_loading import load_official_oaa_data, load_yearly_catcher_framing_data

# Constants
DATA_DIR = r"C:\Users\nairs\Documents\GithubProjects\oWAR\MLB Player Data"
CACHE_DIR = r"C:\Users\nairs\Documents\GithubProjects\oWAR\cache"

# Ensure cache directory exists
os.makedirs(CACHE_DIR, exist_ok=True)

# Global data loading (from parent context)
try:
    fielding_df = pd.read_csv(os.path.join(DATA_DIR, 'fieldingNotes(player_defensive_data).csv'))
    print(f"Loaded fielding data: {len(fielding_df)} rows, columns: {list(fielding_df.columns)}")
except Exception as e:
    print(f"Error loading fielding data: {e}")
    fielding_df = pd.DataFrame()  # Fallback if file not found

# Pattern for finding capitalized words (player names)
capitalized_words = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'

__all__ = [
    'classify_defensive_play',
    'analyze_double_play_contributions',
    'get_positional_defensive_weights',
    'estimate_player_position',
    'extract_year_unified',
    'extract_year_from_game_id',
    'create_player_season_key',
    'calculate_outs_above_average_from_fielding_notes',
    'compare_oaa_calculations',
    'create_player_team_mapping',
    'get_catcher_framing_value',
    'clean_enhanced_defensive_players',
    'get_defensive_war_component',
    'get_defensive_war_component_simple',
    'get_all_defensive_data'
]

def extract_year_unified(data_source, **kwargs):
    """
    Unified year extraction from multiple data sources with different formats:

    Args:
        data_source: String indicating the data source type
        **kwargs: Flexible arguments depending on data source

    Supported data sources:
        - 'direct': Direct year from row data (Year/Season columns)
        - 'espn_game_id': ESPN Game ID requiring conversion
        - 'csv_row': DataFrame row with Year/Season columns
        - 'filename': Extract from filename pattern

    Returns:
        int: Year (2016-2021+) or None if extraction fails
    """
    try:
        if data_source == 'direct':
            # Direct year value
            year_value = kwargs.get('year_value')
            if year_value is not None:
                return int(year_value)

        elif data_source == 'espn_game_id':
            # ESPN Game ID conversion
            game_id = kwargs.get('game_id')
            return extract_year_from_game_id(game_id)

        elif data_source == 'csv_row':
            # DataFrame row with Year/Season columns
            row = kwargs.get('row')
            if row is not None:
                # Try multiple column names in order of preference
                for col_name in ['Year', 'Season', 'year', 'season']:
                    if col_name in row and pd.notna(row[col_name]):
                        return int(row[col_name])

        elif data_source == 'filename':
            # Extract from filename pattern (e.g., "data_2021.csv")
            filename = kwargs.get('filename', '')
            year_match = re.search(r'(20\d{2})', filename)
            if year_match:
                return int(year_match.group(1))

        # Default fallback
        default_year = kwargs.get('default_year', 2021)
        return default_year

    except (ValueError, TypeError, KeyError):
        return kwargs.get('default_year', 2021)

def extract_year_from_game_id(game_id):
    """
    Extract year from ESPN Game ID based on correct mapping:
    2016: 36xxxxxxx (360403107 - 361002107)
    2017: 37xxxxxxx (370403119 - 371001127)
    2018: 38xxxxxxx (380329114 - 381001116)
    2019: 401xxxxxx (401074733 - 401169053)
    2020: 401xxxxxx (401225674 - 401226568)
    2021: 401xxxxxx (401227058 - 401229475)
    """
    try:
        game_str = str(game_id).strip()
        if not game_str.isdigit():
            return None

        game_int = int(game_str)

        # Map game ID ranges to years based on ESPN's actual encoding
        if 360000000 <= game_int <= 369999999:
            return 2016
        elif 370000000 <= game_int <= 379999999:
            return 2017
        elif 380000000 <= game_int <= 389999999:
            return 2018
        elif 390000000 <= game_int <= 399999999:
            return 2019  # Just in case, though examples start at 401
        elif 400000000 <= game_int <= 409999999:
            # Need to distinguish 2019, 2020, 2021 within 40xxxxxxx range
            if game_int <= 401169999:  # 2019 ends around 401169053
                return 2019
            elif game_int <= 401226999:  # 2020 range 401225674 - 401226568
                return 2020
            else:  # 2021 starts around 401227058
                return 2021
        else:
            # For future years or unknown ranges
            return None

    except (ValueError, TypeError):
        return None

def classify_defensive_play(data_text, play_type):
    """Extract contextual information from play data"""
    data_lower = str(data_text).lower()
    context = {
        'difficulty_multiplier': 1.0,
        'play_subtype': 'standard',
        'position_context': None
    }

    # Play difficulty based on ball type
    if 'line drive' in data_lower:
        context['difficulty_multiplier'] = 1.4  # Harder reaction time
        context['play_subtype'] = 'line_drive'
    elif 'ground ball' in data_lower:
        context['difficulty_multiplier'] = 1.0  # Standard
        context['play_subtype'] = 'ground_ball'
    elif 'fly ball' in data_lower:
        context['difficulty_multiplier'] = 0.9  # Easier tracking
        context['play_subtype'] = 'fly_ball'
    elif 'throw' in data_lower and play_type == 'E':
        context['difficulty_multiplier'] = 1.3  # Throwing errors more costly
        context['play_subtype'] = 'throwing_error'
    elif 'catch' in data_lower:
        context['difficulty_multiplier'] = 1.1
        context['play_subtype'] = 'catching_error'
    elif 'bobble' in data_lower:
        context['difficulty_multiplier'] = 0.8  # Less severe than clean miss
        context['play_subtype'] = 'bobble'

    # Position context for assists and double plays
    if 'at home' in data_lower or 'at 1st' in data_lower:
        context['position_context'] = 'base_play'
    elif '2nd base' in data_lower or '3rd base' in data_lower:
        context['position_context'] = 'middle_infield'

    return context

def analyze_double_play_contributions(data_text):
    """Analyze individual contributions to double plays"""
    contributions = {}

    # Extract player sequences from parentheses like "(Carpenter-Wong-Adams, Grichuk-Wong)"
    dp_patterns = re.findall(r'\(([^)]+)\)', str(data_text))

    for pattern in dp_patterns:
        # Handle multiple DPs separated by commas
        dp_sequences = [seq.strip() for seq in pattern.split(',')]

        for sequence in dp_sequences:
            players = [p.strip() for p in sequence.split('-')]

            for i, player in enumerate(players):
                if player not in contributions:
                    contributions[player] = []

                if len(players) >= 2:
                    if i == 0:
                        contributions[player].append('initiate_dp')  # Start the DP
                    elif i == len(players) - 1:
                        contributions[player].append('complete_dp')  # Finish the DP
                    else:
                        contributions[player].append('turn_dp')     # Turn the DP (hardest)

    return contributions

def get_positional_defensive_weights():
    """Position-specific defensive expectations and weights"""
    return {
        # Infield positions (higher opportunity, higher expectations)
        'SS': {'weight': 1.2, 'replacement_adj': 0.05},  # Shortstop: hardest position
        '2B': {'weight': 1.1, 'replacement_adj': 0.03},  # Second base: many DPs
        '3B': {'weight': 1.1, 'replacement_adj': 0.02},  # Third base: reaction time
        '1B': {'weight': 0.9, 'replacement_adj': -0.02}, # First base: easier fielding
        'C':  {'weight': 1.0, 'replacement_adj': 0.0},   # Catcher: separate framing

        # Outfield positions (fewer opportunities in our data)
        'LF': {'weight': 0.8, 'replacement_adj': -0.05},
        'CF': {'weight': 1.0, 'replacement_adj': 0.0},
        'RF': {'weight': 0.9, 'replacement_adj': -0.03},

        # Pitchers (minimal defensive plays in our data)
        'P':  {'weight': 0.7, 'replacement_adj': -0.10},
    }

def estimate_player_position(player_name, player_stats):
    """Estimate primary position based on play patterns"""
    # Heuristics based on play types
    dp_rate = player_stats.get('double_plays', 0) / max(player_stats.get('total_defensive_plays', 1), 1)
    assist_rate = player_stats.get('assists', 0) / max(player_stats.get('total_defensive_plays', 1), 1)

    # High DP involvement suggests middle infield
    if dp_rate > 0.6:
        return 'SS' if assist_rate > 0.3 else '2B'
    elif dp_rate > 0.3:
        return '3B' if assist_rate > 0.2 else '1B'
    elif assist_rate > 0.4:
        return 'C'  # Catchers have many assists but fewer DPs
    else:
        return 'OF'  # Default for outfielders/pitchers

def create_player_season_key(player_name, team, year):
    """Create unique identifier for player-season combinations"""
    return f"{player_name}_{team}_{year}"

def calculate_outs_above_average_from_fielding_notes():
    """
    Season-based OAA calculation with proper player identification
    Each player-season is treated as separate entity to avoid aggregation issues
    """
    # Check cache first (new version for season-based calculation)
    cache_file = os.path.join(CACHE_DIR, "fielding_oaa_values_v4_seasonal.json")
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                cached_data = json.load(f)
            print(f"Loaded cached seasonal fielding OAA data ({len(cached_data)} player-seasons)")
            return cached_data
        except:
            pass

    print("Calculating seasonal outs above average from fielding notes...")

    # Check if fielding data is available
    if fielding_df.empty or 'Game' not in fielding_df.columns:
        print("No fielding data available or missing 'Game' column. Returning empty defensive values.")
        return {}

    df = fielding_df.copy()

    # Add year column
    df['Year'] = df['Game'].apply(extract_year_from_game_id)
    df = df.dropna(subset=['Year'])  # Remove rows with unparseable game IDs

    # Enhanced player-season tracking with contextual weights
    player_season_stats = {}
    positional_weights = get_positional_defensive_weights()

    for _, row in df.iterrows():
        statlines = str(row['Stat']).split(',')
        if not statlines:
            continue

        play_type = statlines[0].strip()
        data_text = str(row.get('Data', ''))
        team = row['Team']
        year = int(row['Year'])

        # Get contextual information about the play
        play_context = classify_defensive_play(data_text, play_type)

        if play_type == 'DP':
            # Enhanced double play analysis
            dp_contributions = analyze_double_play_contributions(data_text)
            for player, roles in dp_contributions.items():
                player_key = create_player_season_key(player, team, year)

                if player_key not in player_season_stats:
                    player_season_stats[player_key] = {
                        'player_name': player,
                        'team': team,
                        'year': year,
                        'weighted_dp_value': 0,
                        'weighted_assist_value': 0,
                        'weighted_error_penalty': 0,
                        'total_defensive_plays': 0,
                        'play_contexts': [],
                        'raw_counts': {'double_plays': 0, 'assists': 0, 'errors': 0}
                    }

                player_season_stats[player_key]['total_defensive_plays'] += 1
                player_season_stats[player_key]['raw_counts']['double_plays'] += 1

                # Weight double play contributions differently
                for role in roles:
                    if role == 'turn_dp':
                        dp_value = 1.5 * play_context['difficulty_multiplier']  # Hardest role
                    elif role == 'initiate_dp':
                        dp_value = 1.2 * play_context['difficulty_multiplier']  # Start the play
                    else:  # complete_dp
                        dp_value = 1.0 * play_context['difficulty_multiplier']  # Finish the play

                    player_season_stats[player_key]['weighted_dp_value'] += dp_value
                    player_season_stats[player_key]['play_contexts'].append(play_context['play_subtype'])

        else:
            # Handle non-DP plays
            players = re.findall(capitalized_words, data_text)
            for player in players:
                player_key = create_player_season_key(player, team, year)

                if player_key not in player_season_stats:
                    player_season_stats[player_key] = {
                        'player_name': player,
                        'team': team,
                        'year': year,
                        'weighted_dp_value': 0,
                        'weighted_assist_value': 0,
                        'weighted_error_penalty': 0,
                        'total_defensive_plays': 0,
                        'play_contexts': [],
                        'raw_counts': {'double_plays': 0, 'assists': 0, 'errors': 0}
                    }

                player_season_stats[player_key]['total_defensive_plays'] += 1

                if play_type == 'Assists':
                    assist_value = 0.6 * play_context['difficulty_multiplier']
                    if play_context['position_context'] == 'base_play':
                        assist_value *= 1.2  # Assists to bases are more valuable

                    player_season_stats[player_key]['weighted_assist_value'] += assist_value
                    player_season_stats[player_key]['raw_counts']['assists'] += 1

                elif play_type == 'E':
                    # For errors, count each player only once per error entry
                    # The data format is: "Player (error_num, type)" - don't double count
                    error_penalty = play_context['difficulty_multiplier']
                    player_season_stats[player_key]['weighted_error_penalty'] += error_penalty
                    player_season_stats[player_key]['raw_counts']['errors'] += 1

                player_season_stats[player_key]['play_contexts'].append(play_context['play_subtype'])

    # Calculate position-adjusted replacement levels per season
    seasons = list(set(stats['year'] for stats in player_season_stats.values()))
    position_baselines = {}

    for season in seasons:
        season_players = [key for key, stats in player_season_stats.items()
                         if stats['year'] == season and stats['total_defensive_plays'] >= 5]

        position_replacement_levels = {}

        for player_key in season_players:
            stats = player_season_stats[player_key]
            estimated_pos = estimate_player_position(stats['player_name'], stats)

            if estimated_pos not in position_replacement_levels:
                position_replacement_levels[estimated_pos] = []

            # Calculate weighted defensive value per play
            total_value = (stats['weighted_dp_value'] +
                          stats['weighted_assist_value'] -
                          stats['weighted_error_penalty'])
            value_per_play = total_value / stats['total_defensive_plays']
            position_replacement_levels[estimated_pos].append(value_per_play)

        # Calculate 40th percentile as replacement level for each position in this season
        for pos, values in position_replacement_levels.items():
            if values and len(values) >= 3:  # Minimum sample size
                position_baselines[f"{pos}_{season}"] = np.percentile(values, 40)

    print(f"Calculated replacement levels for {len(position_baselines)} position-season combinations")

    # Calculate enhanced OAA for each player-season
    fielding_oaa_values = {}

    for player_key, stats in player_season_stats.items():
        if stats['total_defensive_plays'] < 5:  # Lower threshold for seasonal data
            continue

        # Estimate position and get positional adjustments
        estimated_pos = estimate_player_position(stats['player_name'], stats)
        pos_weight = positional_weights.get(estimated_pos, {'weight': 1.0, 'replacement_adj': 0.0})

        # Get replacement level for this position-season
        baseline_key = f"{estimated_pos}_{stats['year']}"
        replacement_level = position_baselines.get(baseline_key, 0.0)

        # Calculate weighted defensive value
        total_weighted_value = (stats['weighted_dp_value'] +
                               stats['weighted_assist_value'] -
                               stats['weighted_error_penalty'])

        value_per_play = total_weighted_value / stats['total_defensive_plays']

        # Calculate above replacement
        above_replacement = (value_per_play - replacement_level - pos_weight['replacement_adj']) * pos_weight['weight']

        # Convert to season-level OAA
        # Scale notable plays to expected seasonal impact
        # Assume 20-40 notable plays per season = normal, scale to OAA range of ±20
        season_scale_factor = stats['total_defensive_plays'] / 30.0  # 30 notable plays = average
        final_oaa = above_replacement * season_scale_factor * 15.0  # Scale to ±15 OAA range

        # Apply realistic bounds (no season should be >±25 OAA)
        final_oaa = max(-25.0, min(25.0, final_oaa))

        fielding_oaa_values[player_key] = {
            'oaa': round(final_oaa, 1),
            'player_name': stats['player_name'],
            'team': stats['team'],
            'year': stats['year'],
            'total_plays': stats['total_defensive_plays'],
            'estimated_position': estimated_pos,
            'weighted_dp_value': round(stats['weighted_dp_value'], 2),
            'weighted_assist_value': round(stats['weighted_assist_value'], 2),
            'weighted_error_penalty': round(stats['weighted_error_penalty'], 2),
            'value_per_play': round(value_per_play, 3),
            'above_replacement': round(above_replacement, 3),
            'raw_counts': stats['raw_counts'],
            'primary_contexts': list(set(stats['play_contexts']))[:3]
        }

    # Cache the result
    try:
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(fielding_oaa_values, f, indent=2)
        print(f"Cached seasonal fielding OAA data ({len(fielding_oaa_values)} player-seasons)")
    except:
        pass

    return fielding_oaa_values

def compare_oaa_calculations():
    """
    Compare our fielding notes OAA calculation with official OAA data
    Returns comparison statistics and calibration suggestions
    """
    # Import required function from parent context
    from cleanedDataParser import create_name_mapping

    our_oaa = calculate_outs_above_average_from_fielding_notes()
    official_oaa = load_official_oaa_data()

    # Create name mapping between datasets
    our_names = list(our_oaa.keys())
    official_names = list(official_oaa.keys())
    name_mapping = create_name_mapping(our_names, official_names)

    comparisons = []
    for our_name, official_name in name_mapping.items():
        if our_name in our_oaa and official_name in official_oaa:
            our_value = our_oaa[our_name]['oaa']
            official_value = official_oaa[official_name]['official_oaa']

            comparisons.append({
                'player': our_name,
                'our_oaa': our_value,
                'official_oaa': official_value,
                'difference': our_value - official_value,
                'position': official_oaa[official_name]['position'],
                'total_plays': our_oaa[our_name]['total_plays']
            })

    if not comparisons:
        print("No matching players found between datasets")
        return {}

    # Calculate correlation and statistics
    our_values = [c['our_oaa'] for c in comparisons]
    official_values = [c['official_oaa'] for c in comparisons]

    correlation = np.corrcoef(our_values, official_values)[0, 1] if len(our_values) > 1 else 0
    mean_difference = np.mean([c['difference'] for c in comparisons])
    std_difference = np.std([c['difference'] for c in comparisons])

    print(f"\n=== OAA COMPARISON RESULTS ===")
    print(f"Players compared: {len(comparisons)}")
    print(f"Correlation: {correlation:.3f}")
    print(f"Mean difference (our - official): {mean_difference:.3f}")
    print(f"Std difference: {std_difference:.3f}")

    # Show top positive and negative differences
    sorted_comparisons = sorted(comparisons, key=lambda x: x['difference'])

    print(f"\nTOP 5 PLAYERS WE RATE HIGHER:")
    for comp in sorted_comparisons[-5:]:
        print(f"  {comp['player']}: Our {comp['our_oaa']:.1f}, Official {comp['official_oaa']:.1f} (+{comp['difference']:.1f})")

    print(f"\nTOP 5 PLAYERS WE RATE LOWER:")
    for comp in sorted_comparisons[:5]:
        print(f"  {comp['player']}: Our {comp['our_oaa']:.1f}, Official {comp['official_oaa']:.1f} ({comp['difference']:.1f})")

    # Calculate suggested calibration multiplier
    suggested_multiplier = 1.0
    if correlation > 0.3 and len(comparisons) > 10:
        # Use linear regression to find best scaling factor
        slope = np.polyfit(our_values, official_values, 1)[0]
        suggested_multiplier = slope
        print(f"\nSUGGESTED CALIBRATION MULTIPLIER: {suggested_multiplier:.3f}")

    return {
        'comparisons': comparisons,
        'correlation': correlation,
        'mean_difference': mean_difference,
        'std_difference': std_difference,
        'suggested_multiplier': suggested_multiplier
    }

def create_player_team_mapping():
    """
    Create a mapping of player names to teams by year using game-level data

    Returns:
        dict: {player_name_year: team} e.g., {'Juan Soto_2021': 'WSH'}
    """
    cache_file = os.path.join(CACHE_DIR, "player_team_mapping.json")

    # Check cache first
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                cached_data = json.load(f)
            print(f"Loaded cached player-team mapping ({len(cached_data)} player-seasons)")
            return cached_data
        except:
            pass

    print("Creating player-team mapping from game data...")

    # Load game-level data
    hitter_df = pd.read_csv(os.path.join(DATA_DIR, "hittersByGame(player_offense_data).csv"), low_memory=False)
    pitcher_df = pd.read_csv(os.path.join(DATA_DIR, "pitchersByGame(pitcher_data).csv"), low_memory=False)

    player_team_map = {}

    # Extract year from game IDs and map hitters to teams
    for _, row in hitter_df.iterrows():
        game_id = row.get('Game', '')
        year = extract_year_from_game_id(game_id)
        if year:
            player_name = str(row.get('Hitters', '')).strip()
            team = str(row.get('Team', '')).strip()
            if player_name and team:
                key = f"{player_name}_{year}"
                player_team_map[key] = team

    # Extract year from game IDs and map pitchers to teams
    for _, row in pitcher_df.iterrows():
        game_id = row.get('Game', '')
        year = extract_year_from_game_id(game_id)
        if year:
            player_name = str(row.get('Pitchers', '')).strip()
            team = str(row.get('Team', '')).strip()
            if player_name and team:
                key = f"{player_name}_{year}"
                player_team_map[key] = team

    # Cache the result
    try:
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(player_team_map, f, indent=2)
        print(f"Cached player-team mapping ({len(player_team_map)} player-seasons)")
    except Exception as e:
        print(f"Warning: Could not cache mapping: {e}")

    return player_team_map

def get_catcher_framing_value(player_name, year):
    """
    Get catcher framing run value for a specific player-year

    Args:
        player_name: Player name (First Last format)
        year: Season year (2016-2021)

    Returns:
        float: Framing run value (positive = good framing, negative = poor framing)
    """
    framing_data = load_yearly_catcher_framing_data()
    player_year_key = f"{player_name}_{year}"

    return framing_data.get(player_year_key, 0.0)

def clean_enhanced_defensive_players():
    """
    Enhanced defensive metric combining seasonal data with official metrics
    Returns data organized by player-season keys for proper oWAR integration
    """
    cache_file = os.path.join(CACHE_DIR, "enhanced_defensive_values_v2_seasonal.json")
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                cached_data = json.load(f)
            print(f"Loaded cached enhanced defensive data ({len(cached_data)} player-seasons)")
            return cached_data
        except:
            pass

    print("Calculating enhanced seasonal defensive metrics...")

    # Get all our data sources
    our_seasonal_oaa = calculate_outs_above_average_from_fielding_notes()
    official_oaa = load_official_oaa_data()
    yearly_framing_values = load_yearly_catcher_framing_data()

    enhanced_defensive_values = {}

    # Process each player-season from our fielding notes
    for player_season_key, our_data in our_seasonal_oaa.items():
        player_name = our_data['player_name']
        team = our_data['team']
        year = our_data['year']
        our_oaa_value = our_data['oaa']

        # Try to find official OAA for this player (note: official OAA doesn't have seasons)
        official_oaa_value = None
        # Create possible name variations for matching
        name_variations = [player_name, f"{player_name.split()[0]} {player_name.split()[-1]}" if ' ' in player_name else player_name]

        for name_var in name_variations:
            if name_var in official_oaa:
                official_oaa_value = official_oaa[name_var]['official_oaa']
                break

        # Combine OAA values with seasonal context
        if official_oaa_value is not None:
            # Weight: 70% official OAA, 30% our seasonal OAA
            combined_oaa = 0.7 * official_oaa_value + 0.3 * our_oaa_value
        else:
            # Only our seasonal OAA available
            combined_oaa = our_oaa_value

        # Add catcher framing if applicable (year-specific)
        framing_bonus = 0
        framing_runs_value = 0

        # Try direct match first (in case names are exactly the same)
        player_year_key = f"{player_name}_{year}"
        if player_year_key in yearly_framing_values:
            framing_runs_value = yearly_framing_values[player_year_key]
            framing_bonus = framing_runs_value / 1.0
        else:
            # Try fuzzy matching by last name for this year
            player_last_name = player_name.split()[-1].strip()
            for framing_key in yearly_framing_values.keys():
                if framing_key.endswith(f"_{year}"):
                    framing_name = framing_key.rsplit('_', 1)[0]  # Remove the year
                    if player_last_name in framing_name:
                        framing_runs_value = yearly_framing_values[framing_key]
                        framing_bonus = framing_runs_value / 1.0
                        break

        # Final defensive value
        final_defensive_value = combined_oaa + framing_bonus

        # Apply realistic bounds (-20 to +25 OAA reasonable for a season)
        final_defensive_value = max(-20.0, min(25.0, final_defensive_value))

        enhanced_defensive_values[player_season_key] = {
            'enhanced_def_value': round(final_defensive_value, 1),
            'player_name': player_name,
            'team': team,
            'year': year,
            'our_oaa': our_oaa_value,
            'official_oaa': official_oaa_value,
            'combined_oaa': round(combined_oaa, 1),
            'framing_runs': framing_runs_value,
            'framing_bonus': round(framing_bonus, 1),
            'total_plays': our_data['total_plays'],
            'estimated_position': our_data.get('estimated_position', 'Unknown'),
            'has_official_oaa': official_oaa_value is not None
        }

    # Cache the result
    try:
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(enhanced_defensive_values, f, indent=2)
        print(f"Cached enhanced defensive data ({len(enhanced_defensive_values)} player-seasons)")

        # Summary stats
        with_official = sum(1 for v in enhanced_defensive_values.values() if v['has_official_oaa'])
        with_framing = sum(1 for v in enhanced_defensive_values.values() if v['framing_runs'] != 0)
        years = set(v['year'] for v in enhanced_defensive_values.values())

        print(f"  - Seasons covered: {sorted(years)}")
        print(f"  - Player-seasons with official OAA: {with_official}")
        print(f"  - Player-seasons with framing data: {with_framing}")
    except Exception as e:
        print(f"Warning: Could not cache enhanced defensive data: {e}")

    return enhanced_defensive_values

def get_defensive_war_component(player_name, team, year):
    """
    Get defensive WAR component for a specific player-season

    Args:
        player_name: Player's last name (as appears in fielding notes)
        team: Team abbreviation
        year: Season year

    Returns:
        Defensive WAR component (float) with realistic bounds (-2.0 to +3.0)
    """
    enhanced_values = clean_enhanced_defensive_players()

    # Create player-season key
    player_season_key = f"{player_name}_{team}_{year}"

    if player_season_key not in enhanced_values:
        return 0.0

    def_value = enhanced_values[player_season_key]['enhanced_def_value']

    # Convert OAA to WAR: roughly 10 OAA = 1 WAR
    defensive_war = def_value / 10.0

    # Apply realistic bounds for defensive WAR as requested
    # Cap at -2.0 WAR (worst realistic defense) and +3.0 WAR (elite defense)
    defensive_war = max(-2.0, min(3.0, defensive_war))

    return round(defensive_war, 2)

def get_defensive_war_component_simple(player_season_key):
    """
    Simplified interface - pass the full player_season_key directly

    Args:
        player_season_key: Format "PlayerName_TEAM_YEAR" (e.g. "Devers_BOS_2019")

    Returns:
        Defensive WAR component (float) with realistic bounds (-2.0 to +3.0)
    """
    enhanced_values = clean_enhanced_defensive_players()

    if player_season_key not in enhanced_values:
        return 0.0

    def_value = enhanced_values[player_season_key]['enhanced_def_value']
    defensive_war = def_value / 10.0
    defensive_war = max(-2.0, min(3.0, defensive_war))

    return round(defensive_war, 2)

def get_all_defensive_data():
    """
    Get complete defensive data for integration with other oWAR components

    Returns:
        Dictionary with all player-season defensive data including:
        - Enhanced defensive value (OAA + framing)
        - Defensive WAR component
        - Breakdown of contributing factors
    """
    return clean_enhanced_defensive_players()