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

# ====== DATA LOADING MODULE ======
from modules.data_loading import (
    load_primary_datasets, get_primary_dataframes,
    load_mapping_from_file, clear_mapping_cache, clear_all_cache,
    load_official_oaa_data, load_yearly_bp_data, load_yearly_catcher_framing_data
)

# ====== PARK FACTORS MODULE ======
from modules.park_factors import (
    calculate_park_factors, calculate_regular_season_park_factors,
    get_player_park_adjustment, apply_enhanced_hitter_park_adjustments,
    apply_enhanced_pitcher_park_adjustments, get_player_park_games,
    clean_stadium_name, get_regular_season_stadiums
)

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

def create_enhanced_name_mapping_with_context(source_data, target_data, context_col='Team'):
    """
    Enhanced name mapping that uses context to disambiguate multiple matches

    Args:
        source_data: DataFrame with 'Name' and context columns
        target_data: DataFrame with 'Name' and 'PA'/'IP' columns
        context_col: Column to use for disambiguation context
    """
    mapping = {}

    # Convert to dictionaries for easier lookup
    source_names = source_data['Name'].tolist()
    target_names = target_data['Name'].tolist()

    # Create normalized target lookup
    target_normalized = {}
    for idx, name in enumerate(target_names):
        normalized = normalize_name(name)
        if normalized not in target_normalized:
            target_normalized[normalized] = []
        target_normalized[normalized].append((name, idx))

    exact_matches = 0
    normalized_matches = 0
    context_disambiguated = 0

    for _, source_row in source_data.iterrows():
        source_name = source_row['Name']

        # Step 1: Check for exact matches (including multiple)
        exact_candidates = [(name, idx) for idx, name in enumerate(target_names) if name == source_name]

        if len(exact_candidates) == 1:
            # Single exact match
            mapping[source_name] = exact_candidates[0][0]
            exact_matches += 1
            continue
        elif len(exact_candidates) > 1:
            # Multiple exact matches - need disambiguation using PA/IP context
            best_candidate = None
            best_score = -1

            for cand_name, cand_idx in exact_candidates:
                target_row = target_data.iloc[cand_idx]
                pa = target_row.get('PA', None)
                ip = target_row.get('IP', None)

                # Score candidates: hitters (PA not null, IP null) get higher scores
                score = 0
                if pd.notna(pa) and pd.isna(ip):
                    score = 100 + pa  # Hitter with plate appearances
                elif pd.isna(pa) and pd.notna(ip):
                    score = 10  # Pitcher (lower priority for hitter matching)
                else:
                    score = 1  # Unknown type

                if score > best_score:
                    best_score = score
                    best_candidate = cand_name

            if best_candidate:
                mapping[source_name] = best_candidate
                context_disambiguated += 1
                continue

        # Step 2: Normalized match with context disambiguation
        normalized_source = normalize_name(source_name)
        if normalized_source in target_normalized:
            candidates = target_normalized[normalized_source]

            if len(candidates) == 1:
                mapping[source_name] = candidates[0][0]
                normalized_matches += 1
            else:
                # Multiple candidates - use context for disambiguation
                # For hitters, prefer candidates with PA (plate appearances) over IP (innings pitched)
                best_candidate = None
                best_score = -1

                for cand_name, cand_idx in candidates:
                    target_row = target_data.iloc[cand_idx]
                    pa = target_row.get('PA', None)
                    ip = target_row.get('IP', None)

                    # Score candidates: hitters (PA not null, IP null) get higher scores
                    score = 0
                    if pd.notna(pa) and pd.isna(ip):
                        score = 100 + pa  # Hitter with plate appearances
                    elif pd.isna(pa) and pd.notna(ip):
                        score = 10  # Pitcher (lower priority for hitter matching)
                    else:
                        score = 1  # Unknown type

                    if score > best_score:
                        best_score = score
                        best_candidate = cand_name

                if best_candidate:
                    mapping[source_name] = best_candidate
                    context_disambiguated += 1
                else:
                    # Fallback to first candidate
                    mapping[source_name] = candidates[0][0]
                    normalized_matches += 1

    print(f"Enhanced context-aware mapping results:")
    print(f"  Exact matches: {exact_matches}")
    print(f"  Normalized matches: {normalized_matches}")
    print(f"  Context disambiguated: {context_disambiguated}")
    print(f"  Total: {len(mapping)}/{len(source_names)}")

    return mapping

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

# load_mapping_from_file function moved to modules/data_loading.py

# clear_mapping_cache function moved to modules/data_loading.py

# clear_all_cache function moved to modules/data_loading.py

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
# Load all primary datasets using the data_loading module
_dataframes = get_primary_dataframes()

# Create individual dataframe references for backward compatibility
hitter_by_game_df = _dataframes.get('hitter_by_game_df')
pitcher_by_game_df = _dataframes.get('pitcher_by_game_df')
baserunning_by_game_df = _dataframes.get('baserunning_by_game_df')
fielding_by_game_df = _dataframes.get('fielding_by_game_df')
warp_hitter_df = _dataframes.get('warp_hitter_df')
warp_pitcher_df = _dataframes.get('warp_pitcher_df')
oaa_hitter_df = _dataframes.get('oaa_hitter_df')
fielding_df = _dataframes.get('fielding_df')
baserunning_df = _dataframes.get('baserunning_df')
war_df = _dataframes.get('war_df')

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
    """Legacy function - uses only 2021 data"""
    df = warp_hitter_df.drop(['bpid', 'mlbid', 'Age', 'DRC+', '+/-', 'PA', 'R', 'RBI',
                              'ISO', 'K%', 'BB%', 'Whiff%'], axis=1)
    return df.sort_values(by='WARP')

def clean_warp_pitcher():
    """Legacy function - uses only 2021 data"""
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
    Enhanced function - uses all available years of BP pitcher data (2016-2021)
    Returns expanded dataset for improved training
    """
    cache_file = os.path.join(CACHE_DIR, "yearly_warp_pitcher_cleaned.json")

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

def clean_war():
    # FIXED: Keep 'Pos' column for positional adjustments
    df = war_df.drop(['playerid', 'Team'], axis=1)
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
    import re
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

def extract_year_from_game_id(game_id):
    """Extract season year from ESPN game ID using exact ranges"""
    game_id_int = int(str(game_id))

    # ESPN Game ID ranges by season (from confirmed ESPN data)
    if 360403107 <= game_id_int <= 361002107:
        return 2016
    elif 370403119 <= game_id_int <= 371001127:
        return 2017
    elif 380329114 <= game_id_int <= 381001116:
        return 2018
    elif 401074733 <= game_id_int <= 401169053:
        return 2019
    elif 401225674 <= game_id_int <= 401226568:
        return 2020
    elif 401227058 <= game_id_int <= 401229475:
        return 2021

    return None

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

# load_official_oaa_data function moved to modules/data_loading.py

def compare_oaa_calculations():
    """
    Compare our fielding notes OAA calculation with official OAA data
    Returns comparison statistics and calibration suggestions
    """
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
        'suggested_multiplier': suggested_multiplier if 'suggested_multiplier' in locals() else 1.0
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

# load_yearly_bp_data function moved to modules/data_loading.py

# load_yearly_catcher_framing_data function moved to modules/data_loading.py

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

# ========== PARK FACTOR INTEGRATION ==========

def clean_stadium_name(stadium_str):
    """Clean stadium names from HTML formatting and standardize to official names"""
    if pd.isna(stadium_str):
        return None

    import re
    clean_name = re.sub(r'<[^>]+>', '', str(stadium_str))
    clean_name = re.sub(r'\n+', ' ', clean_name)
    clean_name = re.sub(r'\t+', ' ', clean_name)
    clean_name = re.sub(r'\s+', ' ', clean_name).strip()

    if 'Coverage:' in clean_name:
        clean_name = clean_name.split('Coverage:')[0].strip()

    if not clean_name:
        return None

    # FIXED: Standardize stadium names to handle renames and variations
    # This fixes the 69 stadiums issue by mapping all variations to official names
    STADIUM_STANDARDIZATION = {
        # Houston variations
        'Minute Maid Park': 'Minute Maid Park',
        'Astros Field': 'Minute Maid Park',

        # Texas Rangers variations (Globe Life Park renamed to Globe Life Field)
        'Globe Life Park in Arlington': 'Globe Life Field',
        'Globe Life Park': 'Globe Life Field',
        'Globe Life Field': 'Globe Life Field',
        'Rangers Ballpark in Arlington': 'Globe Life Field',

        # Milwaukee variations (Miller Park renamed to American Family Field)
        'Miller Park': 'American Family Field',
        'American Family Field': 'American Family Field',

        # Miami variations (Marlins Park renamed to loanDepot park)
        'Marlins Park': 'loanDepot Park',
        'loanDepot Park': 'loanDepot Park',
        'loanDepot park': 'loanDepot Park',

        # Atlanta variations (Turner Field -> Truist Park)
        'Turner Field': 'Truist Park',
        'Truist Park': 'Truist Park',
        'SunTrust Park': 'Truist Park',

        # Chicago White Sox variations
        'Guaranteed Rate Field': 'Guaranteed Rate Field',
        'U.S. Cellular Field': 'Guaranteed Rate Field',
        'Comiskey Park': 'Guaranteed Rate Field',

        # Standard mappings for exact matches
        'Angel Stadium': 'Angel Stadium',
        'Chase Field': 'Chase Field',
        'Citi Field': 'Citi Field',
        'Comerica Park': 'Comerica Park',
        'Coors Field': 'Coors Field',
        'Dodger Stadium': 'Dodger Stadium',
        'Fenway Park': 'Fenway Park',
        'Great American Ball Park': 'Great American Ball Park',
        'Kauffman Stadium': 'Kauffman Stadium',
        'Marlins Park': 'loanDepot Park',  # Handle old name
        'Nationals Park': 'Nationals Park',
        'Oakland Coliseum': 'Oakland Coliseum',
        'Oriole Park at Camden Yards': 'Oriole Park at Camden Yards',
        'Petco Park': 'Petco Park',
        'PNC Park': 'PNC Park',
        'Progressive Field': 'Progressive Field',
        'Rogers Centre': 'Rogers Centre',
        'T-Mobile Park': 'T-Mobile Park',
        'Target Field': 'Target Field',
        'Tropicana Field': 'Tropicana Field',
        'Wrigley Field': 'Wrigley Field',
        'Yankee Stadium': 'Yankee Stadium',
        'Busch Stadium': 'Busch Stadium',
        'Oracle Park': 'Oracle Park',
        'Citizens Bank Park': 'Citizens Bank Park'
    }

    # Check for exact matches first
    if clean_name in STADIUM_STANDARDIZATION:
        return STADIUM_STANDARDIZATION[clean_name]

    # Check for partial matches (case insensitive)
    clean_lower = clean_name.lower()
    for stadium_variant, official_name in STADIUM_STANDARDIZATION.items():
        if stadium_variant.lower() in clean_lower or clean_lower in stadium_variant.lower():
            return official_name

    # Return original name if no standardization found
    return clean_name

def get_regular_season_stadiums():
    """
    Returns the list of MLB regular season stadiums only, excluding:
    - Spring training facilities
    - Special event venues (Field of Dreams, London, etc.)
    - International stadiums
    - Minor league parks
    """
    regular_season_stadiums = {
        # 30 MLB regular season stadiums
        'American Family Field',      # Milwaukee Brewers (formerly Miller Park)
        'Angel Stadium',              # Los Angeles Angels
        'Busch Stadium',             # St. Louis Cardinals
        'Chase Field',               # Arizona Diamondbacks
        'Citi Field',                # New York Mets
        'Citizens Bank Park',        # Philadelphia Phillies
        'Comerica Park',             # Detroit Tigers
        'Coors Field',               # Colorado Rockies
        'Dodger Stadium',            # Los Angeles Dodgers
        'Fenway Park',               # Boston Red Sox
        'Globe Life Field',          # Texas Rangers
        'Great American Ball Park',  # Cincinnati Reds
        'Guaranteed Rate Field',     # Chicago White Sox
        'Kauffman Stadium',          # Kansas City Royals
        'loanDepot Park',            # Miami Marlins
        'Minute Maid Park',          # Houston Astros
        'Nationals Park',            # Washington Nationals
        'Oakland Coliseum',          # Oakland Athletics
        'Oracle Park',               # San Francisco Giants
        'Oriole Park at Camden Yards', # Baltimore Orioles
        'Petco Park',                # San Diego Padres
        'PNC Park',                  # Pittsburgh Pirates
        'Progressive Field',         # Cleveland Guardians
        'Rogers Centre',             # Toronto Blue Jays
        'T-Mobile Park',             # Seattle Mariners
        'Target Field',              # Minnesota Twins
        'Tropicana Field',           # Tampa Bay Rays
        'Truist Park',               # Atlanta Braves
        'Wrigley Field',             # Chicago Cubs
        'Yankee Stadium'             # New York Yankees
    }
    return regular_season_stadiums

# Park factor functions moved to modules/park_factors.py
# This includes:
# - calculate_park_factors()
# - calculate_regular_season_park_factors()
# - get_player_park_adjustment()
# - apply_enhanced_hitter_park_adjustments()
# - apply_enhanced_pitcher_park_adjustments()
# - get_player_park_games()

# ===== MISSING CRITICAL IMPROVEMENTS FROM STANDALONE FILES =====

def create_optimized_name_mapping_with_indices(source_data, target_data):
    """
    SUPERIOR enhanced name mapping that:
    1. Returns INDICES instead of names (handles duplicates correctly) - from fixed_enhanced_mapping.py
    2. Uses optimized performance techniques - from optimized_name_matching.py
    3. Has better conflict resolution - from multiple_matches_handling.py

    This is the MISSING CRITICAL IMPROVEMENT that properly handles duplicate names.
    """
    mapping = {}
    target_names = target_data['Name'].tolist()

    # Performance optimization: Create exact match lookup table
    exact_lookup = {}
    for idx, name in enumerate(target_names):
        if pd.notna(name):
            if name not in exact_lookup:
                exact_lookup[name] = []
            exact_lookup[name].append(idx)

    # Performance optimization: Create normalized lookup table
    normalized_lookup = {}
    for idx, name in enumerate(target_names):
        if pd.notna(name):
            normalized = normalize_name(name)
            if normalized not in normalized_lookup:
                normalized_lookup[normalized] = []
            normalized_lookup[normalized].append((name, idx))

    exact_matches = 0
    normalized_matches = 0
    context_disambiguated = 0

    for _, source_row in source_data.iterrows():
        source_name = source_row['Name']
        if pd.isna(source_name):
            continue

        # OPTIMIZATION: Fast exact lookup first
        if source_name in exact_lookup:
            candidates = exact_lookup[source_name]

            if len(candidates) == 1:
                # Single exact match - store INDEX not name
                mapping[source_name] = candidates[0]
                exact_matches += 1
                continue
            else:
                # Multiple exact matches - use enhanced scoring for disambiguation
                best_idx = None
                best_score = -1

                for cand_idx in candidates:
                    target_row = target_data.iloc[cand_idx]
                    pa = target_row.get('PA', None)
                    ip = target_row.get('IP', None)

                    # Enhanced scoring with multiple factors
                    score = 0
                    if pd.notna(pa) and pd.isna(ip):
                        score = 1000 + pa  # Hitter priority
                    elif pd.isna(pa) and pd.notna(ip):
                        score = 100 + ip   # Pitcher secondary
                    else:
                        score = 10  # Unknown type

                    if score > best_score:
                        best_score = score
                        best_idx = cand_idx

                if best_idx is not None:
                    mapping[source_name] = best_idx
                    context_disambiguated += 1
                    continue

        # OPTIMIZATION: Fast normalized lookup
        normalized_source = normalize_name(source_name)
        if normalized_source in normalized_lookup:
            candidates = normalized_lookup[normalized_source]

            if len(candidates) == 1:
                mapping[source_name] = candidates[0][1]  # Store INDEX
                normalized_matches += 1
            else:
                # Multiple normalized candidates - enhanced disambiguation
                best_idx = None
                best_score = -1

                for cand_name, cand_idx in candidates:
                    target_row = target_data.iloc[cand_idx]
                    pa = target_row.get('PA', None)
                    ip = target_row.get('IP', None)

                    # Primary scoring
                    score = 0
                    if pd.notna(pa) and pd.isna(ip):
                        score = 1000 + pa
                    elif pd.isna(pa) and pd.notna(ip):
                        score = 100 + ip
                    else:
                        score = 10

                    # Enhanced scoring: Last name exact match bonus
                    source_parts = source_name.split()
                    target_parts = cand_name.split()
                    if (len(source_parts) > 0 and len(target_parts) > 0 and
                        source_parts[-1].lower() == target_parts[-1].lower()):
                        score += 100

                    # Length similarity bonus
                    length_diff = abs(len(source_name) - len(cand_name))
                    if length_diff <= 2:
                        score += 50

                    if score > best_score:
                        best_score = score
                        best_idx = cand_idx

                if best_idx is not None:
                    mapping[source_name] = best_idx
                    context_disambiguated += 1

    print(f"SUPERIOR optimized mapping with indices (FIXES DUPLICATE NAMES):")
    print(f"  Exact matches: {exact_matches}")
    print(f"  Normalized matches: {normalized_matches}")
    print(f"  Context disambiguated: {context_disambiguated}")
    print(f"  Total: {len(mapping)}/{len(source_data)}")

    return mapping

def validate_and_clean_data_enhanced(X, y):
    """
    ENHANCED data cleaning from complete_fix_integration.py
    Specifically designed to fix neural network issues with extreme values
    """
    X = np.array(X)
    y = np.array(y)

    # Replace infinite values with NaN, then fill with median
    X = np.where(np.isinf(X), np.nan, X)
    for col in range(X.shape[1]):
        median_val = np.nanmedian(X[:, col])
        X[:, col] = np.where(np.isnan(X[:, col]), median_val, X[:, col])

    # Cap extreme outliers (beyond 5 standard deviations)
    for col in range(X.shape[1]):
        mean_val = np.mean(X[:, col])
        std_val = np.std(X[:, col])
        if std_val > 0:
            X[:, col] = np.clip(X[:, col], mean_val - 5*std_val, mean_val + 5*std_val)

    # CRITICAL FIX: Enhanced y-value cleaning for neural networks
    y = np.where(np.isinf(y), np.nan, y)
    y_median = np.nanmedian(y)
    y = np.where(np.isnan(y), y_median, y)

    # NEURAL NETWORK FIX: Cap extreme WAR values that break Keras training
    # From complete_fix_integration.py - this fixes the Keras performance issues
    y = np.clip(y, -5.0, 10.0)  # Reasonable WAR bounds for neural networks

    return X.tolist(), y.tolist()

# ===== ENHANCED BASERUNNING SYSTEM WITH RUN EXPECTANCY =====

# Run Expectancy Matrix (2016-2021 average)
RUN_EXPECTANCY_MATRIX = {
    ('000', 0): 0.481, ('000', 1): 0.254, ('000', 2): 0.095,
    ('100', 0): 0.859, ('100', 1): 0.509, ('100', 2): 0.214,
    ('010', 0): 1.100, ('010', 1): 0.664, ('010', 2): 0.305,
    ('001', 0): 1.361, ('001', 1): 0.815, ('001', 2): 0.413,
    ('110', 0): 1.437, ('110', 1): 0.908, ('110', 2): 0.424,
    ('101', 0): 1.758, ('101', 1): 1.140, ('101', 2): 0.471,
    ('011', 0): 2.052, ('011', 1): 1.424, ('011', 2): 0.623,
    ('111', 0): 2.292, ('111', 1): 1.541, ('111', 2): 0.736,
}

def calculate_steal_run_value(from_base, to_base, outs, success=True):
    """
    Calculate run value of a stolen base attempt using run expectancy

    Args:
        from_base: Starting base (1, 2, 3)
        to_base: Target base (2, 3, 4=home)
        outs: Number of outs (0, 1, 2)
        success: True for successful steal, False for caught stealing

    Returns:
        float: Run value of the play
    """

    # Build base state strings (1st, 2nd, 3rd)
    if from_base == 1:
        before_state = '100'
        after_state_success = '010'
        after_state_caught = '000'
    elif from_base == 2:
        before_state = '010'
        after_state_success = '001'
        after_state_caught = '000'
    elif from_base == 3:
        before_state = '001'
        after_state_success = '000'  # Scored
        after_state_caught = '000'
    else:
        return 0.0  # Invalid base

    # Get run expectancies
    re_before = RUN_EXPECTANCY_MATRIX.get((before_state, outs), 0)

    if success:
        if to_base == 4:  # Steal of home - immediate run scored
            re_after = RUN_EXPECTANCY_MATRIX.get((after_state_success, outs), 0)
            return (re_after - re_before) + 1.0  # +1 for run scored
        else:
            re_after = RUN_EXPECTANCY_MATRIX.get((after_state_success, outs), 0)
            return re_after - re_before
    else:
        # Caught stealing - usually increases outs
        if outs < 2:
            re_after = RUN_EXPECTANCY_MATRIX.get((after_state_caught, outs + 1), 0)
        else:
            re_after = 0  # Inning over
        return re_after - re_before

def parse_baserunning_event(data_str):
    """
    Parse baserunning event data to extract details

    Args:
        data_str: Event data like " Escobar (1, 2nd base off Harvey/d'Arnaud)"

    Returns:
        dict: Parsed event details
    """
    import re

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
    baserunning_file = 'MLB Player Data/baserunningNotes(player_offense_data).csv'
    if not os.path.exists(baserunning_file):
        print(f"⚠️  Baserunning file not found: {baserunning_file}")
        return {}

    df = pd.read_csv(baserunning_file)
    print(f"Loaded {len(df)} baserunning events")

    # Extract year from game ID function
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

    # Calculate averages and display sample seasonal data
    summary_values = {}
    display_data = []

    for player_name in player_aggregated:
        # Use average seasonal value for now
        avg_value = player_aggregated[player_name] / player_season_counts[player_name]
        summary_values[player_name] = avg_value

        # Collect seasonal data for display
        for season_key, data in player_season_values.items():
            if data['player_name'] == player_name:
                # Only show notable seasonal performances
                if data['steals'] > 10 or abs(data['total_value']) > 1.0:
                    display_data.append((
                        f"{player_name} ({data['year']})",
                        data['total_value'],
                        data['steals'],
                        data['caught_stealing'],
                        data['picked_offs']
                    ))

    # Display realistic seasonal totals (not inflated career aggregates)
    display_data.sort(key=lambda x: x[1])  # Sort by total value
    for player_season, value, sb, cs, po in display_data[:50]:  # Show top 50
        print(f"  {player_season}: {value:.2f} runs ({sb} SB, {cs} CS, {po} PO)")

    print(f"Calculated enhanced baserunning values for {len(summary_values)} players")
    print(f"Fixed multi-year aggregation issue - now showing realistic seasonal totals")

    if summary_values:
        print(f"Total run value range: {min(summary_values.values()):.2f} to {max(summary_values.values()):.2f}")

    # Cache the results
    try:
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(summary_values, f, indent=2)
        print(f"Cached enhanced baserunning values")
    except:
        pass

    return summary_values

def calculate_defensive_baserunning_impact():
    """
    Calculate defensive impact on baserunning events using game matching

    Returns:
        dict: {player_name: defensive_baserunning_impact}
    """
    print("=== CALCULATING DEFENSIVE BASERUNNING IMPACT ===")

    # Load both datasets
    baserunning_file = 'MLB Player Data/baserunningNotes(player_offense_data).csv'
    fielding_file = 'MLB Player Data/fieldingNotes(player_defensive_data).csv'

    if not os.path.exists(baserunning_file) or not os.path.exists(fielding_file):
        print("⚠️  Required files not found for defensive impact calculation")
        return {}

    baserunning_df = pd.read_csv(baserunning_file)
    fielding_df = pd.read_csv(fielding_file)

    # Group by game for matching
    baserunning_by_game = baserunning_df.groupby('Game')
    fielding_by_game = fielding_df.groupby('Game')

    defensive_impact = {}

    # For each game, look for baserunning events and defensive responses
    for game_id in baserunning_by_game.groups.keys():
        if game_id not in fielding_by_game.groups:
            continue

        br_events = baserunning_by_game.get_group(game_id)
        field_events = fielding_by_game.get_group(game_id)

        # Look for caught stealing events and match with defensive players
        for _, br_row in br_events.iterrows():
            if br_row['Stat'] == 'CS':
                event = parse_baserunning_event(br_row['Data'])
                # Extract catcher from CS data if available
                # This would need more sophisticated parsing of the fielding events
                # For now, create basic framework
                pass

    return defensive_impact

def calculate_pitcher_context_bonus(player_name, team):
    """Calculate pitcher bonuses for depth and leverage"""

    # Load pitcher data
    pitcher_data = pd.read_csv('MLB Player Data/pitchersByGame(pitcher_data).csv')
    games_data = pd.read_csv('MLB Player Data/games(team_data).csv')

    player_games = pitcher_data[pitcher_data['Pitchers'] == player_name]

    if len(player_games) == 0:
        return 0.0

    total_bonus = 0.0

    # Starter depth bonus
    starts = player_games[player_games['IP'] >= 3.0]
    if len(starts) > 0:
        avg_ip = starts['IP'].mean()
        if avg_ip >= 7.0:
            total_bonus += 0.5
        elif avg_ip >= 6.0:
            total_bonus += 0.3
        elif avg_ip >= 5.0:
            total_bonus += 0.1

    # Leverage context bonus (relief appearances)
    relief_apps = player_games[player_games['IP'] < 3.0]
    high_leverage_count = 0

    for _, game in relief_apps.iterrows():
        game_info = games_data[games_data['Game'] == game['Game']]
        if len(game_info) > 0:
            game_row = game_info.iloc[0]
            score_diff = abs(game_row['away-score'] - game_row['home-score'])
            if score_diff <= 3:  # Close game
                high_leverage_count += 1

    if len(relief_apps) > 0:
        leverage_pct = high_leverage_count / len(relief_apps)
        if leverage_pct >= 0.7:
            total_bonus += 0.4
        elif leverage_pct >= 0.5:
            total_bonus += 0.2

    # Park factor bonus for pitchers
    park_factor = get_player_park_adjustment(player_name, team)
    if park_factor > 100:  # Hitter-friendly park
        park_bonus = (park_factor - 100) / 200  # Max 0.5 bonus
        total_bonus += park_bonus
    elif park_factor < 100:  # Pitcher-friendly park
        park_penalty = (100 - park_factor) / 400  # Max 0.25 penalty
        total_bonus -= park_penalty

    return round(total_bonus, 2)

def get_enhanced_pitcher_war_component(player_name, team):
    """Get pitcher WAR component with park and context adjustments"""

    # This would integrate with existing pitcher WAR calculation
    # For now, return the context bonus which can be added to base WAR
    return calculate_pitcher_context_bonus(player_name, team)


