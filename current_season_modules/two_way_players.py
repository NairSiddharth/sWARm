"""
Two-Way Player Identification and Classification

This module handles the identification of true two-way players using comprehensive
FanGraphs and BP data with proper MLB rolling window criteria.

MLB Two-Way Player Criteria:
- Pitching: At least 20 Major League innings in current season
- Hitting: At least 20 games as position player/DH with 3+ PA each (60+ PA total)
- Must meet both criteria in same season for official designation

Functions:
    identify_two_way_players_comprehensive(): Uses FanGraphs/BP data with MLB criteria
    filter_blowout_pitching(): Filters emergency pitching appearances
    analyze_two_way_candidates(): Detailed analysis of potential two-way players
"""

import os
import json
import pandas as pd
import numpy as np
from current_season_modules.data_loading import load_comprehensive_fangraphs_data, load_yearly_bp_data
from shared_modules.bp_derived_stats import load_fixed_bp_data


def identify_two_way_players_comprehensive():
    """
    Identify two-way players using comprehensive FanGraphs and BP data with proper MLB criteria.

    MLB Two-Way Player Criteria (rolling window within season):
    - Pitching: At least 20 Major League innings pitched
    - Hitting: At least 20 games as position player/DH with 3+ PA each (60+ PA total)
    - Must meet both criteria in same season for official designation

    Returns:
        dict: {
            'qualified_two_way': {player_name_year: detailed_stats},
            'candidates': {player_name_year: reason_for_disqualification},
            'summary': overall_statistics
        }
    """
    print("=== COMPREHENSIVE TWO-WAY PLAYER ANALYSIS ===")
    print("Using MLB criteria: 20+ IP pitching AND 20+ games/60+ PA hitting")

    # Load comprehensive datasets
    print("Loading comprehensive FanGraphs and BP data...")
    fg_data = load_comprehensive_fangraphs_data()
    bp_hitters, bp_pitchers = load_fixed_bp_data()

    two_way_results = {
        'qualified_two_way': {},
        'candidates': {},
        'summary': {'years_analyzed': [], 'total_candidates': 0, 'qualified_players': 0}
    }

    # Analyze each year from 2016-2024
    for year in range(2016, 2025):
        print(f"\nAnalyzing {year} season...")
        year_candidates = {}

        # Get FanGraphs hitting data for this year
        year_hitters = {k: v for k, v in fg_data['hitters'].items() if k.endswith(f'_{year}')}
        year_pitchers = {k: v for k, v in fg_data['pitchers'].items() if k.endswith(f'_{year}')}

        # Get BP data for this year
        bp_hitters_year = bp_hitters[bp_hitters['Season'] == year] if len(bp_hitters) > 0 else pd.DataFrame()
        bp_pitchers_year = bp_pitchers[bp_pitchers['Season'] == year] if len(bp_pitchers) > 0 else pd.DataFrame()

        # Create MLBID-based mappings for FanGraphs data
        hitter_mlbids = {}
        pitcher_mlbids = {}

        # Extract MLBIDs from FanGraphs hitter data
        for name_year_key, data in year_hitters.items():
            mlbid = data.get('MLBAMID')
            if mlbid and pd.notna(mlbid):
                mlbid = int(mlbid)
                hitter_mlbids[mlbid] = name_year_key

        # Extract MLBIDs from FanGraphs pitcher data
        for name_year_key, data in year_pitchers.items():
            mlbid = data.get('MLBAMID')
            if mlbid and pd.notna(mlbid):
                mlbid = int(mlbid)
                pitcher_mlbids[mlbid] = name_year_key

        # Find MLBIDs that appear in both hitting and pitching datasets
        potential_two_way_mlbids = set(hitter_mlbids.keys()).intersection(set(pitcher_mlbids.keys()))

        print(f"  Found {len(potential_two_way_mlbids)} MLBIDs appearing in both hitting and pitching data")

        for mlbid in potential_two_way_mlbids:
            # Get the name_year keys for this MLBID
            hitter_key = hitter_mlbids[mlbid]
            pitcher_key = pitcher_mlbids[mlbid]

            # Get FanGraphs data
            hitting_data = year_hitters.get(hitter_key, {})
            pitching_data = year_pitchers.get(pitcher_key, {})

            # Extract player name from hitting data (should be same for both)
            player_name = hitting_data.get('name', hitting_data.get('Name', f'Player_{mlbid}'))

            # Extract key metrics for MLB criteria
            # Hitting metrics (need 20+ games with 3+ PA each = 60+ PA)
            games_played = hitting_data.get('G', 0)
            plate_appearances = hitting_data.get('PA', 0)

            # Pitching metrics (need 20+ innings)
            innings_pitched = pitching_data.get('IP', 0)
            games_pitched = pitching_data.get('G', 0)

            # Convert innings to float if it's a string (e.g., "45.1" -> 45.33)
            if isinstance(innings_pitched, str):
                try:
                    innings_pitched = float(innings_pitched.replace('.1', '.33').replace('.2', '.67'))
                except:
                    innings_pitched = 0

            # Get WARP data from BP if available using MLBID
            bp_hitting_warp = 0
            bp_pitching_warp = 0

            if len(bp_hitters_year) > 0 and 'mlbid' in bp_hitters_year.columns:
                player_bp_hit = bp_hitters_year[bp_hitters_year['mlbid'] == mlbid]
                if len(player_bp_hit) > 0:
                    bp_hitting_warp = player_bp_hit['WARP'].iloc[0] if 'WARP' in player_bp_hit.columns else 0
                else:
                    # Fallback to name matching if MLBID not found
                    player_bp_hit = bp_hitters_year[bp_hitters_year['Name'] == player_name]
                    if len(player_bp_hit) > 0:
                        bp_hitting_warp = player_bp_hit['WARP'].iloc[0] if 'WARP' in player_bp_hit.columns else 0

            if len(bp_pitchers_year) > 0 and 'mlbid' in bp_pitchers_year.columns:
                player_bp_pitch = bp_pitchers_year[bp_pitchers_year['mlbid'] == mlbid]
                if len(player_bp_pitch) > 0:
                    bp_pitching_warp = player_bp_pitch['WARP'].iloc[0] if 'WARP' in player_bp_pitch.columns else 0
                else:
                    # Fallback to name matching if MLBID not found
                    player_bp_pitch = bp_pitchers_year[bp_pitchers_year['Name'] == player_name]
                    if len(player_bp_pitch) > 0:
                        bp_pitching_warp = player_bp_pitch['WARP'].iloc[0] if 'WARP' in player_bp_pitch.columns else 0

            # Apply MLB criteria
            meets_hitting_criteria = games_played >= 20 and plate_appearances >= 60
            meets_pitching_criteria = innings_pitched >= 20.0

            # Create MLBID-based key for proper tracking
            mlbid_year_key = f"{mlbid}_{year}"

            candidate_info = {
                'mlbid': mlbid,
                'name': player_name,
                'year': year,
                'games_played': games_played,
                'plate_appearances': plate_appearances,
                'innings_pitched': innings_pitched,
                'games_pitched': games_pitched,
                'fg_war_hitting': hitting_data.get('WAR', 0),
                'fg_war_pitching': pitching_data.get('WAR', 0),
                'bp_warp_hitting': bp_hitting_warp,
                'bp_warp_pitching': bp_pitching_warp,
                'meets_hitting_criteria': meets_hitting_criteria,
                'meets_pitching_criteria': meets_pitching_criteria,
                'is_qualified_two_way': meets_hitting_criteria and meets_pitching_criteria
            }

            year_candidates[mlbid_year_key] = candidate_info

            # Categorize results
            if candidate_info['is_qualified_two_way']:
                two_way_results['qualified_two_way'][mlbid_year_key] = candidate_info
                print(f"  QUALIFIED: {player_name} (MLBID: {mlbid}) - {games_played}G/{plate_appearances}PA hitting, {innings_pitched:.1f}IP pitching")
            else:
                # Determine reason for disqualification
                reasons = []
                if not meets_hitting_criteria:
                    reasons.append(f"hitting ({games_played}G/{plate_appearances}PA, need 20G/60PA)")
                if not meets_pitching_criteria:
                    reasons.append(f"pitching ({innings_pitched:.1f}IP, need 20IP)")

                candidate_info['disqualification_reason'] = "; ".join(reasons)
                two_way_results['candidates'][mlbid_year_key] = candidate_info

                # Only print significant candidates (not just emergency appearances)
                if plate_appearances >= 30 or innings_pitched >= 10:
                    print(f"  CANDIDATE: {player_name} (MLBID: {mlbid}) - {'; '.join(reasons)}")

        two_way_results['summary']['years_analyzed'].append(year)
        two_way_results['summary']['total_candidates'] += len(year_candidates)

    # Generate summary
    qualified_count = len(two_way_results['qualified_two_way'])
    candidate_count = len(two_way_results['candidates'])

    two_way_results['summary']['qualified_players'] = qualified_count
    two_way_results['summary']['total_candidates'] = candidate_count

    print(f"\n=== FINAL RESULTS ===")
    print(f"Qualified two-way players: {qualified_count}")
    print(f"Total candidates analyzed: {candidate_count}")

    if qualified_count > 0:
        print(f"\nQUALIFIED TWO-WAY PLAYERS:")
        for player_key, data in two_way_results['qualified_two_way'].items():
            name = data['name']
            year = data['year']
            fg_hitting_war = data['fg_war_hitting']
            fg_pitching_war = data['fg_war_pitching']
            total_war = fg_hitting_war + fg_pitching_war
            print(f"  {name} ({year}): {fg_hitting_war:.1f} hitting WAR + {fg_pitching_war:.1f} pitching WAR = {total_war:.1f} total WAR")

    return two_way_results

# Backward compatibility alias
def identify_two_way_players():
    """Backward compatibility wrapper for existing code"""
    results = identify_two_way_players_comprehensive()

    # Convert to old format for compatibility (convert MLBID_year keys to name_year keys)
    old_format = {}
    for mlbid_year_key, data in results['qualified_two_way'].items():
        # Create name_year key for backward compatibility
        name_year_key = f"{data['name']}_{data['year']}"
        old_format[name_year_key] = {
            'hitting_warp': data['bp_warp_hitting'],
            'pitching_warp': data['bp_warp_pitching'],
            'total_warp': data['bp_warp_hitting'] + data['bp_warp_pitching'],
            'is_official_two_way': True,
            'mlbid': data['mlbid']  # Include MLBID for reference
        }

    return old_format


def filter_blowout_pitching():
    """
    Identify and filter out position players' emergency/blowout relief pitching using comprehensive data.
    This prevents players like Rizzo, France from having their WAR hurt by emergency pitching.

    Criteria for emergency pitching:
    - Player has significant hitting stats (20+ games, 60+ PA) indicating position player
    - But minimal pitching (<5 innings or <0.5 WAR) indicating emergency use
    - High hitting WAR (>1.0) with low pitching WAR (<0.5) confirms emergency role

    Returns:
        dict: {
            'emergency_pitching': list of players with emergency pitching filtered,
            'legitimate_pitching': list of players with legitimate pitching,
            'summary': filtering statistics
        }
    """
    print("Filtering emergency pitching using comprehensive FanGraphs/BP data...")

    # Get comprehensive analysis results
    results = identify_two_way_players_comprehensive()
    candidates = results['candidates']

    emergency_pitching = []
    legitimate_pitching = []

    for player_key, data in candidates.items():
        # Emergency pitching criteria
        is_emergency = (
            data['games_played'] >= 20 and  # Significant hitting presence
            data['plate_appearances'] >= 60 and  # Meaningful hitting role
            data['innings_pitched'] < 5.0 and  # Minimal pitching
            abs(data['bp_warp_hitting']) > 1.0 and  # Good hitting performance
            abs(data['bp_warp_pitching']) < 0.5  # Poor/minimal pitching
        )

        player_info = {
            'mlbid': data['mlbid'],
            'name': data['name'],
            'year': data['year'],
            'games_played': data['games_played'],
            'plate_appearances': data['plate_appearances'],
            'innings_pitched': data['innings_pitched'],
            'hitting_warp': data['bp_warp_hitting'],
            'pitching_warp': data['bp_warp_pitching'],
            'is_emergency': is_emergency
        }

        if is_emergency:
            emergency_pitching.append(player_info)
            print(f"  EMERGENCY: {data['name']} (MLBID: {data['mlbid']}) ({data['year']}) - {data['games_played']}G/{data['plate_appearances']}PA hitting, {data['innings_pitched']:.1f}IP pitching")
        else:
            legitimate_pitching.append(player_info)

    print(f"\nFiltered {len(emergency_pitching)} emergency pitching appearances")
    print(f"Identified {len(legitimate_pitching)} legitimate dual-role players")

    return {
        'emergency_pitching': emergency_pitching,
        'legitimate_pitching': legitimate_pitching,
        'summary': {
            'emergency_count': len(emergency_pitching),
            'legitimate_count': len(legitimate_pitching),
            'total_analyzed': len(candidates)
        }
    }

def analyze_two_way_candidates():
    """
    Detailed analysis of all potential two-way players with breakdown by criteria.

    Returns:
        dict: Comprehensive analysis with multiple categories
    """
    print("=== DETAILED TWO-WAY PLAYER CANDIDATE ANALYSIS ===")

    results = identify_two_way_players_comprehensive()
    emergency_results = filter_blowout_pitching()

    analysis = {
        'qualified_two_way': results['qualified_two_way'],
        'near_misses': {},
        'emergency_only': emergency_results['emergency_pitching'],
        'historical_candidates': {},
        'summary': {
            'total_qualified': len(results['qualified_two_way']),
            'total_emergency': len(emergency_results['emergency_pitching']),
            'years_with_qualified': []
        }
    }

    # Identify near misses (close to qualifying)
    for player_key, data in results['candidates'].items():
        # Near miss criteria: close to thresholds
        hitting_close = data['games_played'] >= 15 and data['plate_appearances'] >= 45
        pitching_close = data['innings_pitched'] >= 15.0

        if hitting_close and pitching_close and not data['is_qualified_two_way']:
            analysis['near_misses'][player_key] = data
            print(f"NEAR MISS: {data['name']} ({data['year']}) - {data['disqualification_reason']}")

    # Track years with qualified players
    qualified_years = set()
    for player_key in results['qualified_two_way'].keys():
        year = int(player_key.rsplit('_', 1)[1])
        qualified_years.add(year)

    analysis['summary']['years_with_qualified'] = sorted(list(qualified_years))

    print(f"\n=== ANALYSIS SUMMARY ===")
    print(f"Qualified two-way players: {analysis['summary']['total_qualified']}")
    print(f"Near misses: {len(analysis['near_misses'])}")
    print(f"Emergency pitching only: {analysis['summary']['total_emergency']}")
    print(f"Years with qualified players: {analysis['summary']['years_with_qualified']}")

    return analysis

# Legacy function compatibility
def filter_blowout_pitching_from_warp():
    """Legacy compatibility wrapper"""
    return filter_blowout_pitching()


def separate_will_smith_players():
    """
    Separate the two different Will Smith players who are incorrectly merged in the data:
    - Will Smith (Catcher): LAD, strong hitting 2020-2024, primary position is catcher
    - Will Smith (Pitcher): Journeyman reliever 2016-2021, multiple teams

    This addresses the name collision issue where two completely different players
    have the same name and get incorrectly combined in analysis.

    Returns:
        dict: {
            'will_smith_catcher': {player_year: stats for catcher Will Smith},
            'will_smith_pitcher': {player_year: stats for pitcher Will Smith}
        }

    Example:
        >>> separated = separate_will_smith_players()
        >>> catcher_seasons = len(separated['will_smith_catcher'])
        >>> pitcher_seasons = len(separated['will_smith_pitcher'])
        >>> print(f"Separated {catcher_seasons} catcher seasons and {pitcher_seasons} pitcher seasons")
    """
    print("Separating Will Smith catcher from Will Smith pitcher...")

    bp_data = load_yearly_bp_data()
    team_mapping = create_player_team_mapping()

    will_smith_data = {
        'will_smith_catcher': {},
        'will_smith_pitcher': {}
    }

    # Find all Will Smith entries
    will_smith_keys = [k for k in bp_data['hitters'].keys() if 'Will Smith' in k]
    will_smith_keys.extend([k for k in bp_data['pitchers'].keys() if 'Will Smith' in k])
    will_smith_keys = list(set(will_smith_keys))  # Remove duplicates

    for key in will_smith_keys:
        name, year = key.rsplit('_', 1)
        year = int(year)

        # Get team if available
        team = team_mapping.get(key, 'UNK')

        # Separation logic based on career patterns:
        # Catcher Will Smith: Strong hitting 2020+, primarily LAD
        # Pitcher Will Smith: Pitching 2016-2021, multiple teams

        if key in bp_data['hitters']:
            hitting_warp = bp_data['hitters'][key]

            # Strong hitting seasons (2020+) are likely the catcher
            if year >= 2020 and hitting_warp > 1.0:
                will_smith_data['will_smith_catcher'][key] = {
                    'hitting_warp': hitting_warp,
                    'team': team,
                    'year': year,
                    'type': 'catcher'
                }
            # Weak hitting seasons (especially early years) could be pitcher hitting
            elif hitting_warp < 0.5:
                will_smith_data['will_smith_pitcher'][key] = {
                    'hitting_warp': hitting_warp,
                    'team': team,
                    'year': year,
                    'type': 'pitcher_hitting'
                }

        if key in bp_data['pitchers']:
            pitching_warp = bp_data['pitchers'][key]
            # All pitching performances go to pitcher Will Smith
            will_smith_data['will_smith_pitcher'][key] = {
                'pitching_warp': pitching_warp,
                'team': team,
                'year': year,
                'type': 'pitcher'
            }

    # Display separation results
    print("Will Smith Catcher (LAD):")
    for key, data in will_smith_data['will_smith_catcher'].items():
        print(f"  {key}: {data.get('hitting_warp', 0):.1f} hitting WARP, Team: {data['team']}")

    print("Will Smith Pitcher (Multi-team):")
    for key, data in will_smith_data['will_smith_pitcher'].items():
        hitting = data.get('hitting_warp', 0)
        pitching = data.get('pitching_warp', 0)
        print(f"  {key}: {hitting:.2f} hitting, {pitching:.1f} pitching WARP, Team: {data['team']}")

    return will_smith_data


def get_cleaned_two_way_data():
    """
    Comprehensive function that runs complete two-way player analysis using modern MLB criteria.

    This function orchestrates the complete two-way player analysis workflow:
    1. Identifies qualified two-way players using MLB criteria (20+ IP, 20+ games/60+ PA)
    2. Filters out emergency/blowout relief pitching
    3. Provides detailed candidate analysis
    4. Maintains backward compatibility with existing code

    Returns:
        dict: {
            'two_way_players': Qualified two-way players (backward compatible format),
            'comprehensive_analysis': Full analysis with all categories,
            'filtered_data': Emergency pitching filtered out,
            'summary': Overall statistics
        }
    """
    print("=== COMPREHENSIVE TWO-WAY PLAYER ANALYSIS ===")
    print("Using MLB criteria with FanGraphs and BP comprehensive data")

    # Run comprehensive analysis
    comprehensive_results = identify_two_way_players_comprehensive()
    detailed_analysis = analyze_two_way_candidates()
    filtered_data = filter_blowout_pitching()

    # Convert to backward compatible format
    legacy_two_way = {}
    for player_key, data in comprehensive_results['qualified_two_way'].items():
        legacy_two_way[player_key] = {
            'hitting_warp': data['bp_warp_hitting'],
            'pitching_warp': data['bp_warp_pitching'],
            'total_warp': data['bp_warp_hitting'] + data['bp_warp_pitching'],
            'is_official_two_way': True,
            'fg_hitting_war': data['fg_war_hitting'],
            'fg_pitching_war': data['fg_war_pitching'],
            'meets_mlb_criteria': True
        }

    # Summary
    print(f"\n=== FINAL SUMMARY ===")
    print(f"Qualified two-way players (MLB criteria): {len(legacy_two_way)}")
    print(f"Emergency pitching filtered: {filtered_data['summary']['emergency_count']}")
    print(f"Near misses: {len(detailed_analysis['near_misses'])}")
    print(f"Years analyzed: {len(comprehensive_results['summary']['years_analyzed'])}")

    if len(legacy_two_way) > 0:
        print(f"\nQUALIFIED PLAYERS:")
        for player_key, data in legacy_two_way.items():
            name, year = player_key.rsplit('_', 1)
            fg_total = data['fg_hitting_war'] + data['fg_pitching_war']
            print(f"  {name} ({year}): {fg_total:.1f} total WAR (FG), {data['total_warp']:.1f} total WARP (BP)")

    return {
        'two_way_players': legacy_two_way,
        'comprehensive_analysis': detailed_analysis,
        'filtered_data': filtered_data,
        'summary': {
            'qualified_count': len(legacy_two_way),
            'emergency_count': filtered_data['summary']['emergency_count'],
            'years_analyzed': comprehensive_results['summary']['years_analyzed'],
            'total_candidates': comprehensive_results['summary']['total_candidates']
        }
    }


# Convenience exports for easier importing
__all__ = [
    'identify_two_way_players',
    'identify_two_way_players_comprehensive',
    'filter_blowout_pitching',
    'filter_blowout_pitching_from_warp',
    'analyze_two_way_candidates',
    'separate_will_smith_players',
    'get_cleaned_two_way_data'
]