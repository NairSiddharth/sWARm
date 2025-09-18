"""
Two-Way Player Identification and Classification

This module handles the identification of true two-way players (like Shohei Ohtani),
filtering out position players who pitched in blowouts, and separating players
with identical names (like the two Will Smiths).

Functions:
    identify_two_way_players(): Identifies officially designated MLB two-way players
    filter_blowout_pitching_from_warp(): Filters emergency pitching appearances
    separate_will_smith_players(): Separates Will Smith catcher from Will Smith pitcher
"""

import os
import json
import pandas as pd
from cleanedDataParser import load_yearly_bp_data, create_player_team_mapping, CACHE_DIR


def identify_two_way_players():
    """
    Identify TRUE two-way players using strict MLB criteria:
    - Only officially designated two-way players (Shohei Ohtani)
    - Excludes position players pitching in blowouts (Rizzo, France, etc.)
    - Excludes name conflicts (Will Smith catcher vs Will Smith pitcher)

    MLB Two-Way Criteria:
    - Pitching: At least 20 Major League innings
    - Hitting: At least 20 games as position player/DH with 3+ PA each
    - Must be officially classified by MLB

    Returns:
        dict: {player_name_year: {'hitting_warp': X, 'pitching_warp': Y, 'total_warp': Z}}

    Example:
        >>> two_way_players = identify_two_way_players()
        >>> print(two_way_players)
        {'Shohei Ohtani_2021': {'hitting_warp': 5.0, 'pitching_warp': 3.0, 'total_warp': 8.0}}
    """
    print("Identifying ONLY officially designated MLB two-way players...")

    # Get both datasets
    bp_data = load_yearly_bp_data()

    # STRICT: Only Ohtani is officially designated as two-way player by MLB
    official_two_way_players = {
        'Shohei Ohtani_2018': True,
        'Shohei Ohtani_2021': True,  # Only years where he appears in both datasets significantly
    }

    two_way_players = {}

    for player_key in official_two_way_players:
        if player_key in bp_data['hitters'] and player_key in bp_data['pitchers']:
            hitting_warp = bp_data['hitters'][player_key]
            pitching_warp = bp_data['pitchers'][player_key]

            # Additional validation: both WARPs should be meaningful (>0.5)
            if hitting_warp > 0.5 and pitching_warp > 0.5:
                name, year = player_key.rsplit('_', 1)
                print(f"Official two-way player: {name} ({year}) - {hitting_warp:.1f} hitting, {pitching_warp:.1f} pitching WARP")
                two_way_players[player_key] = {
                    'hitting_warp': hitting_warp,
                    'pitching_warp': pitching_warp,
                    'total_warp': hitting_warp + pitching_warp,
                    'is_official_two_way': True
                }

    # Show examples of what we're filtering OUT
    print(f"\nFiltered OUT (not true two-way players):")

    # Position players with tiny pitching WARP (blowout relief)
    blowout_pitchers = []
    for hitter_key, hitting_warp in bp_data['hitters'].items():
        if hitter_key in bp_data['pitchers'] and hitter_key not in two_way_players:
            pitching_warp = bp_data['pitchers'][hitter_key]
            if hitting_warp > 1.0 and pitching_warp < 0.5:  # Good hitter, minimal pitching
                name, year = hitter_key.rsplit('_', 1)
                blowout_pitchers.append(f"{name} ({year}): {hitting_warp:.1f} hitting, {pitching_warp:.2f} pitching")

    if blowout_pitchers:
        print("  Position players pitching in blowouts:")
        for example in blowout_pitchers[:5]:
            print(f"    {example}")
        if len(blowout_pitchers) > 5:
            print(f"    ... and {len(blowout_pitchers) - 5} more position players with minimal pitching")

    # NL pitchers who had to hit
    nl_pitchers = []
    for hitter_key, hitting_warp in bp_data['hitters'].items():
        if hitter_key in bp_data['pitchers'] and hitter_key not in two_way_players:
            pitching_warp = bp_data['pitchers'][hitter_key]
            if pitching_warp > 1.0 and hitting_warp < 1.0:  # Good pitcher, minimal hitting
                name, year = hitter_key.rsplit('_', 1)
                nl_pitchers.append(f"{name} ({year}): {hitting_warp:.2f} hitting, {pitching_warp:.1f} pitching")

    if nl_pitchers:
        print("  NL pitchers who had to hit:")
        for example in nl_pitchers[:5]:
            print(f"    {example}")
        if len(nl_pitchers) > 5:
            print(f"    ... and {len(nl_pitchers) - 5} more NL pitchers")

    return two_way_players


def filter_blowout_pitching_from_warp():
    """
    Identify and filter out position players' blowout relief pitching from WARP calculations.
    This prevents players like Rizzo, France from having their WAR hurt by emergency pitching.

    Criteria for blowout relief appearance:
    - Player has significant hitting WARP (>1.0) indicating they're a position player
    - But minimal pitching WARP (<0.5) indicating limited/emergency pitching
    - This suggests they pitched only in blowout games or emergencies

    Returns:
        dict: {
            'filtered_hitters': cleaned hitter WARP data (unchanged),
            'filtered_pitchers': cleaned pitcher WARP data (emergency pitching removed),
            'blowout_appearances': list of filtered entries with details
        }

    Example:
        >>> filtered = filter_blowout_pitching_from_warp()
        >>> print(f"Filtered {len(filtered['blowout_appearances'])} emergency pitching appearances")
    """
    print("Filtering out blowout relief pitching from position players...")

    bp_data = load_yearly_bp_data()
    filtered_data = {
        'filtered_hitters': bp_data['hitters'].copy(),
        'filtered_pitchers': {},
        'blowout_appearances': []
    }

    # Criteria for blowout relief appearance:
    # 1. Player has significant hitting WARP (>1.0)
    # 2. But minimal pitching WARP (<0.5)
    # 3. This suggests emergency/blowout pitching

    for pitcher_key, pitching_warp in bp_data['pitchers'].items():
        if pitcher_key in bp_data['hitters']:
            hitting_warp = bp_data['hitters'][pitcher_key]

            # Check if this looks like blowout relief
            is_blowout_relief = (
                hitting_warp > 1.0 and  # Good position player
                pitching_warp < 0.5 and  # Minimal pitching contribution
                pitching_warp > -0.5     # Not completely terrible (just limited)
            )

            if is_blowout_relief:
                name, year = pitcher_key.rsplit('_', 1)
                filtered_data['blowout_appearances'].append({
                    'player': name,
                    'year': year,
                    'hitting_warp': hitting_warp,
                    'pitching_warp': pitching_warp,
                    'reason': 'Position player emergency pitching'
                })
                print(f"  Filtered: {name} ({year}) - {hitting_warp:.1f} hitting, {pitching_warp:.2f} pitching (blowout relief)")
            else:
                # Keep legitimate pitching performance
                filtered_data['filtered_pitchers'][pitcher_key] = pitching_warp
        else:
            # Pure pitcher, keep as-is
            filtered_data['filtered_pitchers'][pitcher_key] = pitching_warp

    print(f"Filtered {len(filtered_data['blowout_appearances'])} blowout relief appearances")
    print(f"Kept {len(filtered_data['filtered_pitchers'])} legitimate pitching seasons")

    return filtered_data


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
    Convenience function that runs all two-way player analysis and returns clean data.

    This function orchestrates the complete two-way player analysis workflow:
    1. Identifies true two-way players (only Ohtani)
    2. Filters out blowout relief pitching
    3. Separates Will Smith players

    Returns:
        dict: {
            'two_way_players': Official two-way players,
            'filtered_data': Data with blowout pitching removed,
            'will_smith_separated': Separated Will Smith players
        }
    """
    print("=== COMPLETE TWO-WAY PLAYER ANALYSIS ===")

    # Run all analyses
    two_way_players = identify_two_way_players()
    filtered_data = filter_blowout_pitching_from_warp()
    will_smith_data = separate_will_smith_players()

    # Summary
    print(f"\n=== SUMMARY ===")
    print(f"True two-way players: {len(two_way_players)}")
    print(f"Blowout appearances filtered: {len(filtered_data['blowout_appearances'])}")
    print(f"Will Smith catcher seasons: {len(will_smith_data['will_smith_catcher'])}")
    print(f"Will Smith pitcher seasons: {len(will_smith_data['will_smith_pitcher'])}")

    return {
        'two_way_players': two_way_players,
        'filtered_data': filtered_data,
        'will_smith_separated': will_smith_data
    }


# Convenience exports for easier importing
__all__ = [
    'identify_two_way_players',
    'filter_blowout_pitching_from_warp',
    'separate_will_smith_players',
    'get_cleaned_two_way_data'
]