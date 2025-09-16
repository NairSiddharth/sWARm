# HANDLING MULTIPLE MATCHES IN NAME MAPPING

import difflib
from rapidfuzz import fuzz, process
import pandas as pd
from collections import defaultdict

# =============================================
# PROBLEM: Multiple matches can occur when:
# =============================================
# 1. Similar names: "Mike Smith" vs "Michael Smith"
# 2. Common names: Multiple "John Johnson"s
# 3. Abbreviations: "J. Rodriguez" could match "Jose Rodriguez", "Juan Rodriguez"
# 4. Nicknames: "Alex" could be "Alexander" or "Alexandra"

# =============================================
# SOLUTION 1: One-to-One Mapping with Conflict Resolution
# =============================================

def create_name_mapping_no_duplicates(source_names, target_names, cutoff=70):
    """
    Ensures 1:1 mapping - each target can only match one source
    Resolves conflicts by keeping the best score
    """
    # Get all potential matches first
    all_matches = []

    for source_idx, source_name in enumerate(source_names):
        if pd.isna(source_name):
            continue

        # Get top N matches instead of just top 1
        matches = process.extract(
            source_name.strip().title(),
            [name.strip().title() for name in target_names if pd.notna(name)],
            scorer=fuzz.ratio,
            score_cutoff=cutoff,
            limit=5  # Get top 5 potential matches
        )

        for match_text, score, target_idx in matches:
            all_matches.append({
                'source_name': source_name,
                'source_idx': source_idx,
                'target_name': target_names[target_idx],
                'target_idx': target_idx,
                'score': score
            })

    # Sort by score (best matches first)
    all_matches.sort(key=lambda x: x['score'], reverse=True)

    # Assign matches ensuring 1:1 mapping
    used_sources = set()
    used_targets = set()
    final_mapping = {}
    conflicts = []

    for match in all_matches:
        source = match['source_name']
        target = match['target_name']

        if source not in used_sources and target not in used_targets:
            final_mapping[source] = target
            used_sources.add(source)
            used_targets.add(target)
        else:
            conflicts.append(match)

    print(f"One-to-one mapping: {len(final_mapping)} matches, {len(conflicts)} conflicts resolved")
    return final_mapping, conflicts

# =============================================
# SOLUTION 2: Many-to-One Mapping (Allow multiple sources to same target)
# =============================================

def create_name_mapping_many_to_one(source_names, target_names, cutoff=70):
    """
    Allows multiple source names to map to same target
    Useful when you have different spellings/abbreviations of same person
    """
    mapping = {}
    target_usage = defaultdict(list)  # Track what maps to each target

    for source_name in source_names:
        if pd.isna(source_name):
            continue

        match = process.extractOne(
            source_name.strip().title(),
            [name.strip().title() for name in target_names if pd.notna(name)],
            scorer=fuzz.ratio,
            score_cutoff=cutoff
        )

        if match:
            target_name = target_names[match[2]]
            mapping[source_name] = target_name
            target_usage[target_name].append((source_name, match[1]))

    # Report multiple mappings
    multiple_mappings = {k: v for k, v in target_usage.items() if len(v) > 1}
    if multiple_mappings:
        print(f"Multiple source names mapping to same target:")
        for target, sources in multiple_mappings.items():
            print(f"  {target}: {[f'{s[0]} ({s[1]:.1f})' for s in sources]}")

    return mapping, multiple_mappings

# =============================================
# SOLUTION 3: Smart Conflict Resolution
# =============================================

def create_name_mapping_smart_resolution(source_names, target_names, cutoff=70):
    """
    Smart conflict resolution using additional heuristics:
    1. Exact last name match gets priority
    2. Longer name match gets priority (more specific)
    3. Higher similarity score wins
    """

    def get_resolution_score(source, target, base_score):
        """Calculate enhanced score for conflict resolution"""
        source_parts = source.strip().split()
        target_parts = target.strip().split()

        bonus = 0

        # Exact last name match bonus
        if source_parts[-1].lower() == target_parts[-1].lower():
            bonus += 20

        # Length similarity bonus (prefer matching full names)
        length_diff = abs(len(source) - len(target))
        if length_diff <= 2:
            bonus += 10
        elif length_diff <= 5:
            bonus += 5

        # First name initial match
        if (len(source_parts) > 0 and len(target_parts) > 0 and
            source_parts[0][0].lower() == target_parts[0][0].lower()):
            bonus += 5

        return base_score + bonus

    # Get all potential matches with enhanced scoring
    all_matches = []

    for source_name in source_names:
        if pd.isna(source_name):
            continue

        matches = process.extract(
            source_name.strip().title(),
            [name.strip().title() for name in target_names if pd.notna(name)],
            scorer=fuzz.ratio,
            score_cutoff=cutoff,
            limit=3
        )

        for match_text, score, target_idx in matches:
            enhanced_score = get_resolution_score(
                source_name,
                target_names[target_idx],
                score
            )

            all_matches.append({
                'source_name': source_name,
                'target_name': target_names[target_idx],
                'original_score': score,
                'enhanced_score': enhanced_score
            })

    # Sort by enhanced score and create 1:1 mapping
    all_matches.sort(key=lambda x: x['enhanced_score'], reverse=True)

    used_sources = set()
    used_targets = set()
    final_mapping = {}

    for match in all_matches:
        source = match['source_name']
        target = match['target_name']

        if source not in used_sources and target not in used_targets:
            final_mapping[source] = target
            used_sources.add(source)
            used_targets.add(target)

    return final_mapping

# =============================================
# SOLUTION 4: Interactive/Manual Resolution
# =============================================

def create_name_mapping_with_review(source_names, target_names, cutoff=70, review_threshold=85):
    """
    Automatically maps high-confidence matches, flags others for manual review
    """
    auto_mapping = {}
    manual_review = []

    for source_name in source_names:
        if pd.isna(source_name):
            continue

        matches = process.extract(
            source_name.strip().title(),
            [name.strip().title() for name in target_names if pd.notna(name)],
            scorer=fuzz.ratio,
            score_cutoff=cutoff,
            limit=3
        )

        if not matches:
            continue

        best_match = matches[0]

        if best_match[1] >= review_threshold:
            # High confidence - auto map
            auto_mapping[source_name] = target_names[best_match[2]]
        else:
            # Lower confidence or multiple similar matches - flag for review
            manual_review.append({
                'source': source_name,
                'candidates': [(target_names[m[2]], m[1]) for m in matches]
            })

    print(f"Auto-mapped: {len(auto_mapping)}, Manual review needed: {len(manual_review)}")

    # Display manual review cases
    for review in manual_review[:5]:  # Show first 5
        print(f"'{review['source']}' -> {review['candidates']}")

    return auto_mapping, manual_review

# =============================================
# SOLUTION 5: Statistical Validation
# =============================================

def validate_mapping_quality(mapping, source_names, target_names):
    """
    Analyze mapping quality and detect potential issues
    """
    stats = {
        'total_sources': len([n for n in source_names if pd.notna(n)]),
        'mapped': len(mapping),
        'unmapped': 0,
        'duplicate_targets': 0,
        'low_confidence': 0
    }

    stats['unmapped'] = stats['total_sources'] - stats['mapped']

    # Check for duplicate targets
    target_counts = defaultdict(int)
    for target in mapping.values():
        target_counts[target] += 1

    stats['duplicate_targets'] = sum(1 for count in target_counts.values() if count > 1)

    # Re-score all mappings to find low confidence
    for source, target in mapping.items():
        score = fuzz.ratio(source.strip().title(), target.strip().title())
        if score < 80:
            stats['low_confidence'] += 1

    print("Mapping Quality Report:")
    print(f"  Mapped: {stats['mapped']}/{stats['total_sources']} ({stats['mapped']/stats['total_sources']*100:.1f}%)")
    print(f"  Duplicate targets: {stats['duplicate_targets']}")
    print(f"  Low confidence matches: {stats['low_confidence']}")

    return stats

# =============================================
# RECOMMENDED USAGE FOR YOUR BASEBALL DATA
# =============================================

def recommended_baseball_name_mapping(source_names, target_names):
    """
    Recommended approach for baseball player name matching
    """
    # Step 1: Smart resolution with baseball-specific heuristics
    mapping = create_name_mapping_smart_resolution(source_names, target_names, cutoff=75)

    # Step 2: Validate quality
    stats = validate_mapping_quality(mapping, source_names, target_names)

    # Step 3: Manual review for low-confidence cases if needed
    if stats['low_confidence'] > len(mapping) * 0.1:  # More than 10% low confidence
        print("High number of low-confidence matches - consider manual review")
        auto_mapping, manual_review = create_name_mapping_with_review(source_names, target_names)
        return auto_mapping, manual_review

    return mapping, []