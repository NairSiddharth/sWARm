# Player Game Data Parser
import os
import pandas as pd
import re
import numpy as np

# ====== PATH CONFIG ======
DATA_DIR = r"C:\Users\nairs\Documents\GithubProjects\oWAR\MLB Player Data"
CACHE_DIR = r"C:\Users\nairs\Documents\GithubProjects\oWAR\cache"

# ====== REGEX ======
capitalized_words = r"((?:[A-Z][a-z']+ ?)+)"  # regex to get capitalized words in sentence

# ====== NAME MATCHING UTILITIES ======
import difflib
import unicodedata
import json
import hashlib
from pathlib import Path

# OPTIMIZATION: Cache for name mappings
_name_mapping_cache = {}

# Ensure cache directory exists
os.makedirs(CACHE_DIR, exist_ok=True)

def normalize_name(name):
    """Normalize name for better matching"""
    if pd.isna(name) or not isinstance(name, str):
        return ""

    # Remove accents and special characters
    name = unicodedata.normalize('NFD', name).encode('ascii', 'ignore').decode('ascii')
    # Remove common suffixes and clean
    name = name.replace(' Jr.', '').replace(' Sr.', '').replace(' III', '').replace(' II', '')
    name = name.strip().title()
    return name

def extract_last_name(name):
    """Extract last name from various name formats"""
    if pd.isna(name) or not isinstance(name, str):
        return ""

    name = normalize_name(name)
    parts = name.split()
    if len(parts) == 0:
        return ""

    # Handle abbreviated first names like "M. Carpenter"
    if len(parts) >= 2 and parts[0].endswith('.'):
        return parts[1]

    # Return last part for full names
    return parts[-1] if parts else ""

def fuzzy_match_names(target_name, name_list, cutoff=0.7):
    """Find best match using fuzzy string matching"""
    if not target_name or not name_list:
        return None

    target_normalized = normalize_name(target_name)
    target_last = extract_last_name(target_name)

    best_match = None
    best_score = 0

    for candidate in name_list:
        if pd.isna(candidate):
            continue

        candidate_normalized = normalize_name(candidate)
        candidate_last = extract_last_name(candidate)

        # Try different matching strategies
        scores = [
            difflib.SequenceMatcher(None, target_normalized, candidate_normalized).ratio(),
            difflib.SequenceMatcher(None, target_last, candidate_last).ratio(),
        ]

        # If last names match well, boost the score
        if target_last and candidate_last and target_last == candidate_last:
            scores.append(0.9)

        max_score = max(scores)
        if max_score > best_score and max_score >= cutoff:
            best_score = max_score
            best_match = candidate

    return best_match

def create_name_mapping(source_names, target_names):
    """Create mapping between two sets of player names with persistent caching"""
    # OPTIMIZATION 5: Check persistent file cache first
    cached_mapping = load_mapping_from_file(source_names, target_names)
    if cached_mapping is not None:
        return cached_mapping

    # OPTIMIZATION 3: Check in-memory cache
    cache_key = (tuple(sorted(source_names)), tuple(sorted(target_names)))
    if cache_key in _name_mapping_cache:
        print(f"Using in-memory cached mapping: {len(source_names)} -> {len(target_names)}")
        return _name_mapping_cache[cache_key]

    mapping = {}
    unmatched = []

    # OPTIMIZATION 1: Pre-filter by last name for speed
    print(f"Computing name mapping: {len(source_names)} -> {len(target_names)}")

    # Create last name indices for fast lookup
    target_by_lastname = {}
    for target_name in target_names:
        if pd.isna(target_name):
            continue
        last_name = extract_last_name(target_name).lower()
        if last_name not in target_by_lastname:
            target_by_lastname[last_name] = []
        target_by_lastname[last_name].append(target_name)

    # OPTIMIZATION 2: Only check candidates with matching last names
    all_matches = []
    for source_name in source_names:
        if pd.isna(source_name):
            continue

        source_last = extract_last_name(source_name).lower()

        # Get candidates with same last name (much smaller set)
        candidates = target_by_lastname.get(source_last, [])

        # If no exact last name match, try partial matches (slower fallback)
        if not candidates:
            candidates = [t for t in target_names if pd.notna(t) and
                         source_last in extract_last_name(t).lower()]

        # Score only the relevant candidates
        scored_candidates = []
        for target_name in candidates:
            score = get_enhanced_similarity_score(source_name, target_name)
            if score >= 0.6:
                scored_candidates.append((target_name, score))

        # Sort by score and take top candidates
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        for target_name, score in scored_candidates[:3]:  # Top 3 candidates
            all_matches.append({
                'source': source_name,
                'target': target_name,
                'score': score
            })

    # Sort all matches by score (best first)
    all_matches.sort(key=lambda x: x['score'], reverse=True)

    # Assign matches ensuring 1:1 mapping
    used_sources = set()
    used_targets = set()
    conflicts = []

    for match in all_matches:
        source = match['source']
        target = match['target']

        if source not in used_sources and target not in used_targets:
            mapping[source] = target
            used_sources.add(source)
            used_targets.add(target)
        else:
            conflicts.append(match)

    # Track unmatched
    for source_name in source_names:
        if pd.notna(source_name) and source_name not in mapping:
            unmatched.append(source_name)

    print(f"Matched {len(mapping)} players, {len(unmatched)} unmatched, {len(conflicts)} conflicts resolved")
    if unmatched and len(unmatched) <= 10:
        print(f"Unmatched: {unmatched}")
    if conflicts and len(conflicts) <= 5:
        print(f"Conflicts (first 5): {[c['source'] + ' -> ' + c['target'] + ' (' + str(round(c['score'], 2)) + ')' for c in conflicts[:5]]}")

    # OPTIMIZATION 3: Cache the result in memory
    _name_mapping_cache[cache_key] = mapping

    # OPTIMIZATION 5: Save to persistent cache
    save_mapping_to_file(mapping, source_names, target_names)

    return mapping

def get_enhanced_similarity_score(source_name, target_name):
    """Enhanced similarity scoring with baseball-specific heuristics"""
    if not source_name or not target_name:
        return 0

    source_norm = normalize_name(source_name)
    target_norm = normalize_name(target_name)
    source_last = extract_last_name(source_name)
    target_last = extract_last_name(target_name)

    # Base similarity score
    base_score = difflib.SequenceMatcher(None, source_norm, target_norm).ratio()

    # Bonus for exact last name match (very important for baseball)
    if source_last and target_last and source_last == target_last:
        base_score += 0.2

    # Bonus for similar name lengths (avoid matching "A. Smith" to "Alexander Smitherson")
    length_diff = abs(len(source_name) - len(target_name))
    if length_diff <= 2:
        base_score += 0.1
    elif length_diff <= 5:
        base_score += 0.05

    # Bonus for first initial match
    source_parts = source_norm.split()
    target_parts = target_norm.split()
    if (source_parts and target_parts and
        source_parts[0][0].lower() == target_parts[0][0].lower()):
        base_score += 0.05

    return min(base_score, 1.0)  # Cap at 1.0

def get_cache_filename(source_names, target_names):
    """Generate a unique cache filename based on the input data"""
    # Create a hash of the sorted names to ensure consistency
    source_str = '|'.join(sorted([str(x) for x in source_names if pd.notna(x)]))
    target_str = '|'.join(sorted([str(x) for x in target_names if pd.notna(x)]))
    combined = f"{source_str}||{target_str}"

    # Create hash and truncate for filename
    hash_obj = hashlib.md5(combined.encode())
    hash_str = hash_obj.hexdigest()[:16]

    return f"name_mapping_{hash_str}.json"

def save_mapping_to_file(mapping, source_names, target_names):
    """Save name mapping to persistent file"""
    filename = get_cache_filename(source_names, target_names)
    filepath = os.path.join(CACHE_DIR, filename)

    # Create metadata for cache validation
    cache_data = {
        'mapping': mapping,
        'metadata': {
            'source_count': len([x for x in source_names if pd.notna(x)]),
            'target_count': len([x for x in target_names if pd.notna(x)]),
            'created_timestamp': pd.Timestamp.now().isoformat(),
            'mapping_count': len(mapping)
        }
    }

    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, indent=2, ensure_ascii=False)
        print(f"Saved mapping to {filename}")
    except Exception as e:
        print(f"Warning: Could not save mapping cache: {e}")

def load_mapping_from_file(source_names, target_names):
    """Load name mapping from persistent file"""
    filename = get_cache_filename(source_names, target_names)
    filepath = os.path.join(CACHE_DIR, filename)

    if not os.path.exists(filepath):
        return None

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            cache_data = json.load(f)

        # Validate cache is still current
        metadata = cache_data.get('metadata', {})
        source_count = len([x for x in source_names if pd.notna(x)])
        target_count = len([x for x in target_names if pd.notna(x)])

        if (metadata.get('source_count') == source_count and
            metadata.get('target_count') == target_count):

            print(f"Loaded cached mapping from {filename}")
            print(f"   Created: {metadata.get('created_timestamp', 'Unknown')}")
            print(f"   Mappings: {metadata.get('mapping_count', 0)}")
            return cache_data['mapping']
        else:
            print(f"Cache invalid (data changed), will regenerate mapping")
            return None

    except Exception as e:
        print(f"Warning: Could not load mapping cache: {e}")
        return None

def clear_mapping_cache():
    """Clear all cached name mappings (useful when data changes)"""
    try:
        import glob
        cache_files = glob.glob(os.path.join(CACHE_DIR, "name_mapping_*.json"))
        for file in cache_files:
            os.remove(file)
        print(f"Cleared {len(cache_files)} cached mapping files")
    except Exception as e:
        print(f"Warning: Could not clear cache: {e}")

def clear_all_cache():
    """Clear all cached data (mappings, baserunning, defensive)"""
    try:
        import glob
        cache_files = glob.glob(os.path.join(CACHE_DIR, "*.json"))
        for file in cache_files:
            os.remove(file)
        print(f"Cleared all {len(cache_files)} cache files")
    except Exception as e:
        print(f"Warning: Could not clear cache: {e}")

def clean_defensive_players_fast():
    """Fast version - only catcher framing, skip traditional fielding"""
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

def create_name_mapping_simple(source_names, target_names):
    """Ultra-fast exact matching only for testing"""
    print(f"Using simple exact matching: {len(source_names)} -> {len(target_names)}")
    mapping = {}

    # Create set for O(1) lookup
    target_set = set(target_names)

    for source in source_names:
        if pd.notna(source) and source in target_set:
            mapping[source] = source

    print(f"Simple mapping found {len(mapping)} exact matches")
    return mapping

def list_cached_mappings():
    """List all cached mappings with metadata"""
    try:
        import glob
        cache_files = glob.glob(os.path.join(CACHE_DIR, "name_mapping_*.json"))

        print(f"\nFound {len(cache_files)} cached mappings:")
        for filepath in cache_files:
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                metadata = cache_data.get('metadata', {})
                filename = os.path.basename(filepath)
                print(f"   {filename}")
                print(f"     Created: {metadata.get('created_timestamp', 'Unknown')}")
                print(f"     Mappings: {metadata.get('mapping_count', 0)}")
                print(f"     Size: {metadata.get('source_count', 0)} -> {metadata.get('target_count', 0)}")
            except:
                print(f"   {os.path.basename(filepath)} (corrupted)")
        print()
    except Exception as e:
        print(f"Warning: Could not list cache: {e}")

# ====== LOAD DATA ======
hitter_by_game_df = pd.read_csv(os.path.join(DATA_DIR, "hittersByGame(player_offense_data).csv"), low_memory=False)
pitcher_by_game_df = pd.read_csv(os.path.join(DATA_DIR, "pitchersByGame(pitcher_data).csv"), low_memory=False)
baserunning_by_game_df = pd.read_csv(os.path.join(DATA_DIR, "baserunningNotes(player_offense_data).csv"))
fielding_by_game_df = pd.read_csv(os.path.join(DATA_DIR, "fieldingNotes(player_defensive_data).csv"))

warp_hitter_df = pd.read_csv(os.path.join(DATA_DIR, "bp_hitters_2021.csv"))
warp_pitcher_df = pd.read_csv(os.path.join(DATA_DIR, "bp_pitchers_2021.csv"))
oaa_hitter_df = pd.read_csv(os.path.join(DATA_DIR, "outs_above_average.csv"))
catcher_framing_df = pd.read_csv(os.path.join(DATA_DIR, "catcher-framing.csv"))

fielding_df = pd.read_csv(os.path.join(DATA_DIR, "fieldingNotes(player_defensive_data).csv"))
baserunning_df = pd.read_csv(os.path.join(DATA_DIR, "baserunningNotes(player_offense_data).csv"))
war_df = pd.read_csv(os.path.join(DATA_DIR, "FanGraphs Leaderboard.csv"))

# ====== CLEANERS ======

def clean_sorted_hitter():
    """Aggregate game-level hitter data to season-level for proper matching"""
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

def clean_warp_hitter():
    df = warp_hitter_df.drop(['bpid', 'mlbid', 'Age', 'DRC+', '+/-', 'PA', 'R', 'RBI',
                              'ISO', 'K%', 'BB%', 'Whiff%'], axis=1)
    return df.sort_values(by='WARP')

def clean_warp_pitcher():
    df = warp_pitcher_df.drop(['bpid', 'mlbid', 'DRA-', 'DRA', 'DRA SD', 'cFIP',
                               'GS', 'W', 'L', 'ERA', 'RA9', 'Whiff%'], axis=1)
    return df.sort_values(by='WARP')

def clean_war():
    df = war_df.drop(['playerid', 'Team', 'Pos'], axis=1)
    return df.sort_values(by='Total WAR')

def clean_sorted_baserunning():
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
    df = baserunning_by_game_df.drop(['Game'], axis=1)
    sorted_df = df.sort_values(by='Stat')

    baserunning_values = {}

    for _, row in sorted_df.iterrows():
        statlines = str(row['Stat']).split(',')
        if not statlines:
            continue

        if statlines[0] == 'SB':
            players = re.findall(capitalized_words, str(row.get('Data', '')))  # Changed from 'Play' to 'Data'
            for p in players:
                baserunning_values[p] = baserunning_values.get(p, 0) + (1 / 3)

        elif statlines[0] in ['CS', 'Picked Off']:
            players = re.findall(capitalized_words, str(row.get('Data', '')))  # Changed from 'Play' to 'Data'
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

def clean_defensive_players():
    # Check cache first
    cache_file = os.path.join(CACHE_DIR, "defensive_values.json")
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                cached_data = json.load(f)
            print(f"Loaded cached defensive data ({len(cached_data)} players)")
            return cached_data
        except:
            pass

    print("Processing defensive data (this may take a moment)...")
    df = fielding_df.drop(['Game', 'Team'], axis=1)
    sorted_df = df.sort_values(by='Stat')

    defensive_values = {}

    for _, row in sorted_df.iterrows():
        statlines = str(row['Stat']).split(',')
        if not statlines:
            continue

        players = re.findall(capitalized_words, str(row.get('Data', '')))  # Changed from 'Play' to 'Data'

        if statlines[0] == 'DP':
            for p in players:
                defensive_values[p] = defensive_values.get(p, 0) + (1 / 3)

        elif statlines[0] == 'Assists':
            for p in players:
                defensive_values[p] = defensive_values.get(p, 0) + (0.5 / 3)

        elif statlines[0] == 'E':
            for p in players:
                defensive_values[p] = defensive_values.get(p, 0) - (1 / 3)

    # Add catcher framing data
    framing_values = clean_catcher_framing()
    for player, framing_runs in framing_values.items():
        defensive_values[player] = defensive_values.get(player, 0) + (framing_runs / 10.0)  # Scale to match defensive metric

    # Cache the result
    try:
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(defensive_values, f, indent=2)
        print(f"Cached defensive data ({len(defensive_values)} players)")
    except:
        pass

    return defensive_values

def clean_catcher_framing():
    """Extract catcher framing data and convert to player-based dictionary"""
    framing_values = {}

    for _, row in catcher_framing_df.iterrows():
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


