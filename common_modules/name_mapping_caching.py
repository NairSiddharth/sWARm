import difflib
import hashlib
import json
import os
import unicodedata
import pandas as pd

# Removed circular import - load_mapping_from_file will be imported locally when needed
CACHE_DIR = r"C:\Users\nairs\Documents\GithubProjects\oWAR\cache"

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
    from current_season_modules.data_loading import load_mapping_from_file  # Import locally to avoid circular import
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
