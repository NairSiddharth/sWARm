"""
Comprehensive Duplicate Name Disambiguation Module

This module handles all known cases of MLB players with identical names,
providing team-based and position-based disambiguation to ensure accurate
player identification in the oWAR analysis system.

Known Duplicate Name Cases (2022+):
- Will Smith: Catcher (LAD) vs Pitcher (Multi-team)
- Diego Castillo: Pitcher (TB/SEA) vs Infielder (PIT/ARI)
- Luis Castillo: Pitcher (CIN/SEA) vs Pitcher (DET)
- Luis Garcia: 3 players - Reliever (SD), Infielder (WSH), Starter (HOU)

Plus historical Luis Castillo (2B, 1996-2010)
"""

import pandas as pd
from typing import Dict, List, Tuple, Optional

__all__ = [
    'get_duplicate_name_cases',
    'disambiguate_duplicate_players',
    'create_team_position_key',
    'apply_duplicate_name_disambiguation',
    'detect_unknown_duplicates',
    'generic_duplicate_disambiguation',
    'analyze_statistical_patterns',
    'create_career_signature'
]

def get_duplicate_name_cases() -> Dict[str, List[Dict]]:
    """
    Get comprehensive list of all known duplicate name cases with disambiguation criteria

    Returns:
        Dict mapping player names to list of player profiles for disambiguation
    """
    duplicate_cases = {
        'Will Smith': [
            {
                'player_id': 'will_smith_catcher',
                'primary_position': 'Catcher',
                'teams': ['LAD'],  # Primarily Dodgers
                'years_active': [2019, 2020, 2021, 2022, 2023, 2024],
                'player_type': 'hitter',
                'disambiguation_criteria': {
                    'strong_hitting': True,  # Notable offensive production
                    'primary_team': 'LAD',
                    'position_indicator': ['C', 'Catcher']
                }
            },
            {
                'player_id': 'will_smith_pitcher',
                'primary_position': 'Pitcher',
                'teams': ['SF', 'MIL', 'ATL', 'TEX'],  # Multiple teams
                'years_active': [2016, 2017, 2018, 2019, 2020, 2021],
                'player_type': 'pitcher',
                'disambiguation_criteria': {
                    'reliever': True,
                    'multi_team': True,
                    'position_indicator': ['P', 'Pitcher', 'RP']
                }
            }
        ],

        'Diego Castillo': [
            {
                'player_id': 'diego_castillo_pitcher',
                'primary_position': 'Pitcher',
                'teams': ['TB', 'SEA'],  # Tampa Bay to Seattle
                'years_active': [2018, 2019, 2020, 2021, 2022],
                'player_type': 'pitcher',
                'country': 'Dominican Republic',
                'disambiguation_criteria': {
                    'reliever': True,
                    'saves_leader': True,  # Had 7 saves in 2022
                    'tb_sea_connection': True,
                    'no_middle_name': True,
                    'nickname': 'Samana'
                }
            },
            {
                'player_id': 'diego_castillo_infielder',
                'primary_position': 'Infielder',
                'teams': ['PIT', 'ARI'],  # Pittsburgh to Arizona
                'years_active': [2022, 2023],
                'player_type': 'hitter',
                'country': 'Venezuela',
                'full_name': 'Diego Alejandro Castillo',
                'disambiguation_criteria': {
                    'infielder': True,
                    'venezuelan': True,
                    'young_debut': True,  # 2022 MLB debut
                    'pit_ari_connection': True,
                    'middle_name': 'Alejandro'
                }
            }
        ],

        'Luis Castillo': [
            {
                'player_id': 'luis_castillo_pitcher_cin_sea',
                'primary_position': 'Pitcher',
                'teams': ['CIN', 'SEA'],  # Cincinnati to Seattle
                'years_active': [2017, 2018, 2019, 2020, 2021, 2022, 2023],
                'player_type': 'pitcher',
                'full_name': 'Luis Miguel Castillo',
                'age': 30,  # As of 2023
                'disambiguation_criteria': {
                    'starter': True,
                    'ace_caliber': True,  # 2.99 ERA, traded to Mariners
                    'cin_sea_trade': True,
                    'middle_name': 'Miguel',
                    'high_strikeouts': True
                }
            },
            {
                'player_id': 'luis_castillo_pitcher_det',
                'primary_position': 'Pitcher',
                'teams': ['DET'],  # Detroit Tigers
                'years_active': [2022],
                'player_type': 'pitcher',
                'full_name': 'Luis Felipe Castillo',
                'age': 27,  # As of 2022
                'disambiguation_criteria': {
                    'reliever': True,
                    'limited_appearances': True,  # Only 3 games in 2022
                    'tigers_only': True,
                    'middle_name': 'Felipe',
                    'triple_a_success': True
                }
            },
            {
                'player_id': 'luis_castillo_infielder_historical',
                'primary_position': 'Second Base',
                'teams': ['FLA', 'MIN', 'NYM'],  # Historical player
                'years_active': list(range(1996, 2011)),  # 1996-2010
                'player_type': 'hitter',
                'full_name': 'Luis Antonio Castillo',
                'disambiguation_criteria': {
                    'historical_player': True,  # Before current era
                    'second_baseman': True,
                    'speed_defense': True,  # Known for glove and speed
                    'marlins_twins_mets': True,
                    'middle_name': 'Antonio'
                }
            }
        ],

        'Luis Garcia': [
            {
                'player_id': 'luis_garcia_reliever_sd',
                'primary_position': 'Pitcher',
                'teams': ['PHI', 'SD'],  # Phillies to Padres
                'years_active': list(range(2013, 2024)),  # Long career
                'player_type': 'pitcher',
                'full_name': 'Luis Amado Garcia',
                'age': 35,  # As of 2022
                'disambiguation_criteria': {
                    'reliever': True,
                    'veteran': True,  # 35 years old, debuted 2013
                    'padres_connection': True,
                    'middle_name': 'Amado',
                    'long_career': True
                }
            },
            {
                'player_id': 'luis_garcia_infielder_wsh',
                'primary_position': 'Infielder',
                'teams': ['WSH'],  # Washington Nationals
                'years_active': [2020, 2021, 2022],
                'player_type': 'hitter',
                'full_name': 'Luis Victoriano Garcia',
                'age': 22,  # As of 2022
                'disambiguation_criteria': {
                    'infielder': True,
                    'young_player': True,  # 22 years old
                    'nationals_only': True,
                    'middle_name': 'Victoriano',
                    'recent_debut': True  # 2020 debut
                }
            },
            {
                'player_id': 'luis_garcia_starter_hou',
                'primary_position': 'Pitcher',
                'teams': ['HOU'],  # Houston Astros
                'years_active': [2020, 2021, 2022, 2023],
                'player_type': 'pitcher',
                'full_name': 'Luis Heibardo Garcia',
                'age': 26,  # As of 2022
                'disambiguation_criteria': {
                    'starter': True,
                    'astros_only': True,
                    'win_leader': True,  # 15-8 record in 2022
                    'middle_name': 'Heibardo',
                    'workhorse': True  # 157.1 innings
                }
            }
        ]
    }

    return duplicate_cases

def create_team_position_key(player_name: str, team: str, year: int, position: str = None) -> str:
    """
    Create a unique key for players using team and position information

    Args:
        player_name: Player's name
        team: Team abbreviation
        year: Season year
        position: Position information (optional)

    Returns:
        Unique key string for disambiguation
    """
    key_parts = [player_name.replace(' ', '_').lower(), team.upper(), str(year)]
    if position:
        key_parts.append(position.upper())
    return '_'.join(key_parts)

def disambiguate_duplicate_players(player_name: str, team: str, year: int,
                                 player_data: Dict, context: Dict = None) -> Optional[str]:
    """
    Disambiguate players with duplicate names using team, year, and performance context

    Args:
        player_name: Name of the player
        team: Team abbreviation
        year: Season year
        player_data: Player's statistical data
        context: Additional context (position, stats, etc.)

    Returns:
        Unique player identifier or None if cannot disambiguate
    """
    duplicate_cases = get_duplicate_name_cases()

    if player_name not in duplicate_cases:
        return None  # Not a known duplicate case

    candidates = duplicate_cases[player_name]
    best_match = None
    best_score = 0

    for candidate in candidates:
        score = 0

        # Team matching
        if team in candidate['teams']:
            score += 100

        # Year matching
        if year in candidate['years_active']:
            score += 50

        # Position/type matching from context
        if context:
            player_type = context.get('player_type', '')
            if player_type == candidate['player_type']:
                score += 75

            # Specific position matching
            position = context.get('position', '')
            if position in candidate['disambiguation_criteria'].get('position_indicator', []):
                score += 25

        # Statistical pattern matching
        if player_data:
            # Hitting vs pitching stats
            has_hitting_stats = any(k in player_data for k in ['PA', 'AB', 'H', 'HR', 'RBI'])
            has_pitching_stats = any(k in player_data for k in ['IP', 'ERA', 'WHIP', 'SO', 'SV'])

            if candidate['player_type'] == 'hitter' and has_hitting_stats and not has_pitching_stats:
                score += 30
            elif candidate['player_type'] == 'pitcher' and has_pitching_stats and not has_hitting_stats:
                score += 30

        # Special case logic for each duplicate set
        if player_name == 'Will Smith':
            score += _score_will_smith_disambiguation(candidate, team, year, player_data, context)
        elif player_name == 'Diego Castillo':
            score += _score_diego_castillo_disambiguation(candidate, team, year, player_data, context)
        elif player_name == 'Luis Castillo':
            score += _score_luis_castillo_disambiguation(candidate, team, year, player_data, context)
        elif player_name == 'Luis Garcia':
            score += _score_luis_garcia_disambiguation(candidate, team, year, player_data, context)

        if score > best_score:
            best_score = score
            best_match = candidate['player_id']

    return best_match if best_score > 50 else None  # Require minimum confidence

def _score_will_smith_disambiguation(candidate: Dict, team: str, year: int,
                                   player_data: Dict, context: Dict) -> int:
    """Specific disambiguation logic for Will Smith players"""
    score = 0

    if candidate['player_id'] == 'will_smith_catcher':
        # Catcher Will Smith: Strong with LAD, recent years
        if team == 'LAD':
            score += 50
        if year >= 2019:
            score += 25
        if player_data and player_data.get('PA', 0) > 400:  # Regular player
            score += 20

    elif candidate['player_id'] == 'will_smith_pitcher':
        # Pitcher Will Smith: Multiple teams, earlier years
        if team in ['SF', 'MIL', 'ATL', 'TEX']:
            score += 50
        if year <= 2021:
            score += 25
        if player_data and player_data.get('IP', 0) > 0:
            score += 20

    return score

def _score_diego_castillo_disambiguation(candidate: Dict, team: str, year: int,
                                       player_data: Dict, context: Dict) -> int:
    """Specific disambiguation logic for Diego Castillo players"""
    score = 0

    if candidate['player_id'] == 'diego_castillo_pitcher':
        # Pitcher: TB/SEA, reliever with saves
        if team in ['TB', 'SEA']:
            score += 50
        if player_data and player_data.get('SV', 0) > 0:  # Has saves
            score += 30
        if year >= 2018:
            score += 20

    elif candidate['player_id'] == 'diego_castillo_infielder':
        # Infielder: PIT/ARI, recent debut
        if team in ['PIT', 'ARI']:
            score += 50
        if year >= 2022:  # Recent debut
            score += 30
        if player_data and player_data.get('PA', 0) > 200:  # Regular playing time
            score += 20

    return score

def _score_luis_castillo_disambiguation(candidate: Dict, team: str, year: int,
                                      player_data: Dict, context: Dict) -> int:
    """Specific disambiguation logic for Luis Castillo players"""
    score = 0

    if candidate['player_id'] == 'luis_castillo_pitcher_cin_sea':
        # Ace pitcher: CIN/SEA, high innings/strikeouts
        if team in ['CIN', 'SEA']:
            score += 50
        if player_data and player_data.get('IP', 0) > 100:  # Starter workload
            score += 30
        if player_data and player_data.get('SO', 0) > 150:  # High strikeouts
            score += 20

    elif candidate['player_id'] == 'luis_castillo_pitcher_det':
        # Tigers reliever: Limited appearances
        if team == 'DET':
            score += 50
        if year == 2022:
            score += 30
        if player_data and player_data.get('IP', 0) < 10:  # Limited usage
            score += 20

    elif candidate['player_id'] == 'luis_castillo_infielder_historical':
        # Historical 2B: 1996-2010 era
        if year <= 2010:
            score += 50
        if team in ['FLA', 'MIN', 'NYM']:
            score += 30

    return score

def _score_luis_garcia_disambiguation(candidate: Dict, team: str, year: int,
                                    player_data: Dict, context: Dict) -> int:
    """Specific disambiguation logic for Luis Garcia players"""
    score = 0

    if candidate['player_id'] == 'luis_garcia_reliever_sd':
        # Veteran reliever: SD, long career
        if team in ['PHI', 'SD']:
            score += 50
        if year >= 2013:  # Long career
            score += 20
        if player_data and 0 < player_data.get('IP', 0) < 80:  # Reliever workload
            score += 30

    elif candidate['player_id'] == 'luis_garcia_infielder_wsh':
        # Young infielder: WSH only
        if team == 'WSH':
            score += 50
        if year >= 2020:  # Recent debut
            score += 30
        if player_data and player_data.get('PA', 0) > 200:  # Regular playing time
            score += 20

    elif candidate['player_id'] == 'luis_garcia_starter_hou':
        # Astros starter: High wins/innings
        if team == 'HOU':
            score += 50
        if player_data and player_data.get('IP', 0) > 120:  # Starter workload
            score += 30
        if player_data and player_data.get('W', 0) > 10:  # Win leader
            score += 20

    return score

def apply_duplicate_name_disambiguation(source_data: pd.DataFrame, target_data: pd.DataFrame,
                                      mapping: Dict[str, int]) -> Dict[str, int]:
    """
    Apply duplicate name disambiguation to an existing name mapping

    Args:
        source_data: Source dataset (e.g., WARP data)
        target_data: Target dataset (e.g., WAR data)
        mapping: Existing name mapping to enhance

    Returns:
        Enhanced mapping with duplicate name disambiguation
    """
    print("Applying comprehensive duplicate name disambiguation...")

    duplicate_cases = get_duplicate_name_cases()
    enhanced_mapping = mapping.copy()
    disambiguated_count = 0

    for _, source_row in source_data.iterrows():
        source_name = source_row['Name']
        team = source_row.get('Team', '')
        year = source_row.get('Year', source_row.get('Season', 2021))

        if source_name in duplicate_cases:
            # This is a known duplicate name case
            player_data = source_row.to_dict()
            context = {
                'player_type': 'hitter' if 'PA' in source_row else 'pitcher',
                'position': player_data.get('Pos', '')
            }

            unique_id = disambiguate_duplicate_players(
                source_name, team, year, player_data, context
            )

            if unique_id:
                # Find best target match for this specific player
                target_candidates = target_data[target_data['Name'] == source_name]

                if len(target_candidates) > 1:
                    # Multiple targets with same name - need disambiguation
                    best_target_idx = None
                    best_score = 0

                    for target_idx, target_row in target_candidates.iterrows():
                        target_data_dict = target_row.to_dict()
                        target_context = {
                            'player_type': 'hitter' if 'PA' in target_row else 'pitcher'
                        }

                        target_unique_id = disambiguate_duplicate_players(
                            source_name, team, year, target_data_dict, target_context
                        )

                        if target_unique_id == unique_id:
                            best_target_idx = target_idx
                            break

                    if best_target_idx is not None:
                        enhanced_mapping[source_name] = best_target_idx
                        disambiguated_count += 1

    # ENHANCED: Apply generic disambiguation for unknown duplicate cases
    unknown_duplicates = detect_unknown_duplicates(source_data, target_data, duplicate_cases)
    if unknown_duplicates:
        print(f"  Detected {len(unknown_duplicates)} unknown duplicate name cases")
        generic_mapping = apply_generic_disambiguation(source_data, target_data,
                                                     enhanced_mapping, unknown_duplicates)
        enhanced_mapping.update(generic_mapping)
        print(f"  Applied generic disambiguation to: {list(unknown_duplicates.keys())}")

    print(f"  Disambiguated {disambiguated_count} known duplicate name cases")
    print(f"  Known duplicate cases: {list(duplicate_cases.keys())}")

    return enhanced_mapping

def detect_unknown_duplicates(source_data: pd.DataFrame, target_data: pd.DataFrame,
                            known_cases: Dict[str, List[Dict]]) -> Dict[str, List[int]]:
    """
    Automatically detect unknown duplicate name cases using statistical analysis

    Args:
        source_data: Source dataset
        target_data: Target dataset
        known_cases: Already handled duplicate cases

    Returns:
        Dict mapping player names to lists of potential duplicate indices
    """
    unknown_duplicates = {}

    # Find names that appear multiple times but aren't in known cases
    name_counts = source_data['Name'].value_counts()
    potential_duplicates = name_counts[name_counts > 1].index.tolist()

    for name in potential_duplicates:
        if name in known_cases:
            continue  # Skip known cases

        # Get all instances of this name
        name_instances = source_data[source_data['Name'] == name]

        if len(name_instances) < 2:
            continue

        # Analyze if they represent different players using career signatures
        career_signatures = []
        indices = []

        for idx, row in name_instances.iterrows():
            signature = create_career_signature(row)
            career_signatures.append(signature)
            indices.append(idx)

        # Check if signatures are sufficiently different to indicate different players
        if _are_different_players(career_signatures):
            unknown_duplicates[name] = indices

    return unknown_duplicates

def create_career_signature(player_row: pd.Series) -> Dict[str, any]:
    """
    Create a statistical signature for a player to identify distinct careers

    Args:
        player_row: Single player's data row

    Returns:
        Career signature dictionary
    """
    signature = {
        'name': player_row.get('Name', ''),
        'team': player_row.get('Team', ''),
        'year': player_row.get('Year', player_row.get('Season', 0)),
        'player_type': 'hitter' if player_row.get('PA', 0) > 0 else 'pitcher',
        'primary_stats': {},
        'usage_pattern': '',
        'career_stage': ''
    }

    # Hitting signature
    if signature['player_type'] == 'hitter':
        signature['primary_stats'] = {
            'pa': player_row.get('PA', 0),
            'games': player_row.get('G', 0),
            'avg': player_row.get('AVG', 0),
            'hr': player_row.get('HR', 0),
            'rbi': player_row.get('RBI', 0)
        }

        # Determine usage pattern
        pa = signature['primary_stats']['pa']
        if pa > 500:
            signature['usage_pattern'] = 'regular'
        elif pa > 200:
            signature['usage_pattern'] = 'platoon'
        else:
            signature['usage_pattern'] = 'bench'

    # Pitching signature
    else:
        signature['primary_stats'] = {
            'ip': player_row.get('IP', 0),
            'games': player_row.get('G', 0),
            'era': player_row.get('ERA', 0),
            'whip': player_row.get('WHIP', 0),
            'saves': player_row.get('SV', 0),
            'wins': player_row.get('W', 0)
        }

        # Determine usage pattern
        ip = signature['primary_stats']['ip']
        saves = signature['primary_stats']['saves']
        if ip > 140:
            signature['usage_pattern'] = 'starter'
        elif saves > 5:
            signature['usage_pattern'] = 'closer'
        elif ip > 40:
            signature['usage_pattern'] = 'reliever'
        else:
            signature['usage_pattern'] = 'spot'

    # Career stage estimation based on year and performance
    year = signature['year']
    if year >= 2020:
        signature['career_stage'] = 'current'
    elif year >= 2015:
        signature['career_stage'] = 'recent'
    else:
        signature['career_stage'] = 'historical'

    return signature

def _are_different_players(signatures: List[Dict[str, any]]) -> bool:
    """
    Determine if career signatures represent different players

    Args:
        signatures: List of career signatures to compare

    Returns:
        True if signatures indicate different players
    """
    if len(signatures) < 2:
        return False

    # Check for clear player type differences
    player_types = [sig['player_type'] for sig in signatures]
    if len(set(player_types)) > 1:
        return True  # Hitter vs pitcher = different players

    # Check for distinct team patterns
    teams = [sig['team'] for sig in signatures]
    if len(set(teams)) == len(signatures):
        # All different teams could indicate different players
        usage_patterns = [sig['usage_pattern'] for sig in signatures]
        if len(set(usage_patterns)) > 1:
            return True  # Different teams + different usage = likely different players

    # Check for era differences (historical vs current)
    career_stages = [sig['career_stage'] for sig in signatures]
    if 'historical' in career_stages and 'current' in career_stages:
        return True  # Likely different generations

    # Check for vastly different performance levels
    if signatures[0]['player_type'] == 'hitter':
        pa_values = [sig['primary_stats']['pa'] for sig in signatures]
        if max(pa_values) > 400 and min(pa_values) < 100:
            return True  # Regular vs bench player
    else:
        ip_values = [sig['primary_stats']['ip'] for sig in signatures]
        if max(ip_values) > 100 and min(ip_values) < 20:
            return True  # Starter vs reliever

    return False

def apply_generic_disambiguation(source_data: pd.DataFrame, target_data: pd.DataFrame,
                               existing_mapping: Dict[str, int],
                               unknown_duplicates: Dict[str, List[int]]) -> Dict[str, int]:
    """
    Apply generic disambiguation logic to unknown duplicate cases

    Args:
        source_data: Source dataset
        target_data: Target dataset
        existing_mapping: Existing name mapping
        unknown_duplicates: Detected unknown duplicate cases

    Returns:
        Additional mappings for unknown duplicates
    """
    generic_mapping = {}

    for player_name, source_indices in unknown_duplicates.items():
        # Get corresponding target candidates
        target_candidates = target_data[target_data['Name'] == player_name]

        if len(target_candidates) == 0:
            continue

        # For each source instance, find best target match
        for source_idx in source_indices:
            source_row = source_data.iloc[source_idx]
            source_signature = create_career_signature(source_row)

            best_target_idx = None
            best_score = 0

            for target_idx, target_row in target_candidates.iterrows():
                target_signature = create_career_signature(target_row)

                # Score compatibility between signatures
                score = _score_signature_compatibility(source_signature, target_signature)

                if score > best_score:
                    best_score = score
                    best_target_idx = target_idx

            # Only map if we have good confidence
            if best_target_idx is not None and best_score > 50:
                # Create unique key for this specific player instance
                unique_key = f"{player_name}_{source_signature['team']}_{source_signature['year']}"
                generic_mapping[unique_key] = best_target_idx

    return generic_mapping

def _score_signature_compatibility(source_sig: Dict[str, any], target_sig: Dict[str, any]) -> int:
    """
    Score how compatible two career signatures are

    Args:
        source_sig: Source player signature
        target_sig: Target player signature

    Returns:
        Compatibility score (0-100)
    """
    score = 0

    # Player type match
    if source_sig['player_type'] == target_sig['player_type']:
        score += 40
    else:
        return 0  # Different types = incompatible

    # Team match
    if source_sig['team'] == target_sig['team']:
        score += 30

    # Year proximity
    year_diff = abs(source_sig['year'] - target_sig['year'])
    if year_diff == 0:
        score += 20
    elif year_diff <= 1:
        score += 15
    elif year_diff <= 2:
        score += 10

    # Usage pattern match
    if source_sig['usage_pattern'] == target_sig['usage_pattern']:
        score += 10

    # Statistical similarity
    if source_sig['player_type'] == 'hitter':
        pa_diff = abs(source_sig['primary_stats']['pa'] - target_sig['primary_stats']['pa'])
        if pa_diff < 50:
            score += 5
    else:
        ip_diff = abs(source_sig['primary_stats']['ip'] - target_sig['primary_stats']['ip'])
        if ip_diff < 20:
            score += 5

    return score

def analyze_statistical_patterns(player_data: List[pd.Series]) -> Dict[str, any]:
    """
    Analyze statistical patterns to help identify distinct players

    Args:
        player_data: List of player data rows with same name

    Returns:
        Analysis results for disambiguation
    """
    analysis = {
        'player_count': len(player_data),
        'distinct_teams': set(),
        'year_range': [float('inf'), 0],
        'player_types': set(),
        'usage_patterns': set(),
        'separation_confidence': 0
    }

    for player_row in player_data:
        # Collect team information
        team = player_row.get('Team', '')
        if team:
            analysis['distinct_teams'].add(team)

        # Track year range
        year = player_row.get('Year', player_row.get('Season', 0))
        if year > 0:
            analysis['year_range'][0] = min(analysis['year_range'][0], year)
            analysis['year_range'][1] = max(analysis['year_range'][1], year)

        # Determine player type and usage
        signature = create_career_signature(player_row)
        analysis['player_types'].add(signature['player_type'])
        analysis['usage_patterns'].add(signature['usage_pattern'])

    # Calculate separation confidence
    confidence_factors = []

    # Multiple player types = high confidence of different players
    if len(analysis['player_types']) > 1:
        confidence_factors.append(80)

    # Multiple distinct teams = medium confidence
    if len(analysis['distinct_teams']) >= len(player_data):
        confidence_factors.append(60)

    # Large year gap = medium confidence
    year_span = analysis['year_range'][1] - analysis['year_range'][0]
    if year_span > 10:
        confidence_factors.append(50)

    # Multiple usage patterns = low-medium confidence
    if len(analysis['usage_patterns']) > 1:
        confidence_factors.append(30)

    analysis['separation_confidence'] = max(confidence_factors) if confidence_factors else 0

    return analysis