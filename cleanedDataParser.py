# Player Game Data Parser
import os
import pandas as pd
import re

# ====== PATH CONFIG ======
DATA_DIR = r"C:\Users\nairs\Documents\GithubProjects\oWAR\MLB Player Data"

# ====== REGEX ======
capitalized_words = r"((?:[A-Z][a-z']+ ?)+)"  # regex to get capitalized words in sentence

# ====== NAME MATCHING UTILITIES ======
import difflib
import unicodedata

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
    """Create mapping between two sets of player names with conflict resolution"""
    mapping = {}
    unmatched = []

    # Get all potential matches first
    all_matches = []

    for source_name in source_names:
        if pd.isna(source_name):
            continue

        # Get multiple potential matches
        candidates = []
        for target_name in target_names:
            if pd.isna(target_name):
                continue

            score = get_enhanced_similarity_score(source_name, target_name)
            if score >= 0.6:
                candidates.append((target_name, score))

        # Sort by score and take top candidates
        candidates.sort(key=lambda x: x[1], reverse=True)
        for target_name, score in candidates[:3]:  # Top 3 candidates
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
        print(f"Conflicts (first 5): {[f\"{c['source']} -> {c['target']} ({c['score']:.2f})\" for c in conflicts[:5]]}")

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

# ====== LOAD DATA ======
hitter_by_game_df = pd.read_csv(os.path.join(DATA_DIR, "hittersByGame(player_offense_data).csv"), low_memory=False)
pitcher_by_game_df = pd.read_csv(os.path.join(DATA_DIR, "pitchersByGame(pitcher_data).csv"), low_memory=False)
baserunning_by_game_df = pd.read_csv(os.path.join(DATA_DIR, "baserunningNotes(player_offense_data).csv"))
fielding_by_game_df = pd.read_csv(os.path.join(DATA_DIR, "fieldingNotes(player_defensive_data).csv"))

warp_hitter_df = pd.read_csv(os.path.join(DATA_DIR, "bp_hitters_2021.csv"))
warp_pitcher_df = pd.read_csv(os.path.join(DATA_DIR, "bp_pitchers_2021.csv"))
oaa_hitter_df = pd.read_csv(os.path.join(DATA_DIR, "outs_above_average.csv"))

fielding_df = pd.read_csv(os.path.join(DATA_DIR, "fieldingNotes(player_defensive_data).csv"))
baserunning_df = pd.read_csv(os.path.join(DATA_DIR, "baserunningNotes(player_offense_data).csv"))
war_df = pd.read_csv(os.path.join(DATA_DIR, "FanGraphs Leaderboard.csv"))

# ====== CLEANERS ======

def clean_sorted_hitter():
    """Aggregate game-level hitter data to season-level for proper matching"""
    df = hitter_by_game_df.drop(['H-AB', 'AB', 'H', '#P', 'Game', 'Team', 'Hitter Id'], axis=1)

    # Aggregate by player name to get season totals/averages
    aggregated = df.groupby('Hitters').agg({
        'K': 'sum',        # Total strikeouts
        'BB': 'sum',       # Total walks
        'AVG': 'mean',     # Average batting average
        'OBP': 'mean',     # Average on-base percentage
        'SLG': 'mean'      # Average slugging percentage
    }).reset_index()

    print(f"Aggregated hitter data: {len(df)} game records → {len(aggregated)} unique players")
    return aggregated.sort_values(by='Hitters')

def clean_sorted_pitcher():
    """Aggregate game-level pitcher data to season-level for proper matching"""
    df = pitcher_by_game_df.drop(['R', 'ER', 'PC', 'Game', 'Team', 'Extra', 'Pitcher Id'], axis=1)

    # Aggregate by player name to get season totals/averages
    aggregated = df.groupby('Pitchers').agg({
        'IP': 'sum',       # Total innings pitched
        'BB': 'sum',       # Total walks allowed
        'K': 'sum',        # Total strikeouts
        'HR': 'sum',       # Total home runs allowed
        'ERA': 'mean'      # Average ERA
    }).reset_index()

    print(f"Aggregated pitcher data: {len(df)} game records → {len(aggregated)} unique players")
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
    df = baserunning_by_game_df.drop(['Game'], axis=1)
    sorted_df = df.sort_values(by='Stat')

    baserunning_values = {}

    for _, row in sorted_df.iterrows():
        statlines = str(row['Stat']).split(',')
        if not statlines:
            continue

        if statlines[0] == 'SB':
            players = re.findall(capitalized_words, str(row.get('Play', '')))
            for p in players:
                baserunning_values[p] = baserunning_values.get(p, 0) + (1 / 3)

        elif statlines[0] in ['CS', 'Picked Off']:
            players = re.findall(capitalized_words, str(row.get('Play', '')))
            for p in players:
                baserunning_values[p] = baserunning_values.get(p, 0) - (1 / 3)

    return baserunning_values

def clean_defensive_players():
    df = fielding_df.drop(['Game', 'Team'], axis=1)
    sorted_df = df.sort_values(by='Stat')

    defensive_values = {}

    for _, row in sorted_df.iterrows():
        statlines = str(row['Stat']).split(',')
        if not statlines:
            continue

        players = re.findall(capitalized_words, str(row.get('Play', '')))

        if statlines[0] == 'DP':
            for p in players:
                defensive_values[p] = defensive_values.get(p, 0) + (1 / 3)

        elif statlines[0] == 'Assists':
            for p in players:
                defensive_values[p] = defensive_values.get(p, 0) + (0.5 / 3)

        elif statlines[0] == 'E':
            for p in players:
                defensive_values[p] = defensive_values.get(p, 0) - (1 / 3)

    return defensive_values


