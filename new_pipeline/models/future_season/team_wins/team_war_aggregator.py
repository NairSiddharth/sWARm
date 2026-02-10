"""
Team WAR Aggregator - Merge rosters with projections, aggregate by team.

Handles the join between roster data and individual WAR projections,
including edge cases like missing projections, multi-team players,
and two-way players. Optionally integrates FanGraphs Future Value (FV)
prospect grades for rookies/prospects.
"""

from typing import Dict, Optional

import numpy as np
import pandas as pd

from new_pipeline.models.future_season.team_wins.constants import REPLACEMENT_ROLES
from new_pipeline.models.future_season.team_wins.playing_time_estimator import (
    allocate_team_playing_time,
    adjust_war_for_playing_time
)


def merge_roster_with_projections(
    roster_df: pd.DataFrame,
    hitter_projections: pd.DataFrame,
    pitcher_projections: pd.DataFrame,
    war_column: str,
    fv_lookup: Optional[Dict[int, dict]] = None,
    career_usage: Optional[Dict[int, dict]] = None,
) -> pd.DataFrame:
    """
    Map rostered players to their WAR projections.

    Joins on playerid. Players with no projection match get 0.0 WAR
    unless FV prospect data provides a fallback. Rookies with projections
    can have their WAR blended with FV-based estimates.

    Replacement-role players always get 0.0 WAR regardless of projection.

    Args:
        roster_df: Validated roster DataFrame (from load_roster).
        hitter_projections: Hitter projection CSV data.
        pitcher_projections: Pitcher projection CSV data.
        war_column: Which WAR column to use (e.g., 'war_2026').
        fv_lookup: Optional dict from match_prospects_to_roster().
            Maps MLBAM ID -> {fv, risk, fv_war, is_pitcher, match_method}.
        career_usage: Optional dict from build_career_usage_lookup().
            Maps MLBAM ID -> {career_pa, career_ip}.

    Returns:
        Enriched roster DataFrame with columns:
            rate_war, age, allocated_pa_ip, playing_time_factor,
            adjusted_war, projection_source
    """
    roster = roster_df.copy()

    # Build a combined projection lookup: playerid -> (war, age)
    proj_lookup = {}

    usage_col_map = {'hitter': 'PA', 'pitcher': 'IP'}

    for df_proj, ptype in [(hitter_projections, 'hitter'), (pitcher_projections, 'pitcher')]:
        if war_column not in df_proj.columns:
            print(f"Warning: {war_column} not found in {ptype} projections. Available: {list(df_proj.columns)}")
            continue

        usage_col = usage_col_map[ptype]

        for _, row in df_proj.iterrows():
            pid = row.get('playerid')
            if pd.isna(pid):
                continue
            pid = int(pid)
            war_val = row.get(war_column, 0.0)
            age_val = row.get('Age', np.nan)
            base_usage = row.get(usage_col, np.nan)
            if pd.isna(war_val):
                war_val = 0.0
            if pd.isna(base_usage):
                base_usage = 0.0
            if pid in proj_lookup:
                # Two-way player: sum WAR, keep metadata from higher-WAR role
                existing = proj_lookup[pid]
                existing['war'] += war_val
                if war_val > existing.get('_primary_war', 0):
                    existing['age'] = age_val
                    existing['base_pa_ip'] = float(base_usage)
                    existing['player_type_proj'] = ptype
                    existing['_primary_war'] = war_val
            else:
                proj_lookup[pid] = {
                    'war': war_val,
                    'age': age_val,
                    'base_pa_ip': float(base_usage),
                    'player_type_proj': ptype,
                    '_primary_war': war_val,
                }

    # Lazy import to avoid circular dependency at module level
    if fv_lookup:
        from new_pipeline.models.future_season.team_wins.fv_prospect_integrator import (
            calculate_blend_alpha,
            blend_war,
        )

    # Merge projections into roster
    rate_wars = []
    ages = []
    base_usages = []
    sources = []
    fv_blend_details = []  # track blend info for reporting

    for _, row in roster.iterrows():
        pid = row.get('playerid')
        role = row.get('role', '')
        manual_war = row.get('manual_war')
        has_manual = pd.notna(manual_war) if manual_war is not None else False

        # Replacement roles always get 0 WAR
        if role in REPLACEMENT_ROLES:
            rate_wars.append(0.0)
            ages.append(np.nan)
            base_usages.append(0.0)
            sources.append('replacement_level')
            fv_blend_details.append(None)
            continue

        # No playerid: use manual_war if available, else 0
        if pd.isna(pid):
            if has_manual:
                rate_wars.append(float(manual_war))
                ages.append(np.nan)
                base_usages.append(0.0)
                sources.append('manual')
            else:
                rate_wars.append(0.0)
                ages.append(np.nan)
                base_usages.append(0.0)
                sources.append('no_playerid')
            fv_blend_details.append(None)
            continue

        pid = int(pid)
        if pid in proj_lookup:
            proj_war = proj_lookup[pid]['war']
            proj_age = proj_lookup[pid]['age']
            proj_usage = proj_lookup[pid]['base_pa_ip']
            proj_type = proj_lookup[pid]['player_type_proj']

            # Check if this player is a rookie with FV data for blending
            if fv_lookup and career_usage and pid in fv_lookup:
                fv_info = fv_lookup[pid]
                usage_info = career_usage.get(pid, {'career_pa': 0.0, 'career_ip': 0.0})
                is_pitcher = proj_type == 'pitcher'
                career_val = usage_info['career_ip'] if is_pitcher else usage_info['career_pa']
                alpha = calculate_blend_alpha(career_val, is_pitcher)

                if alpha < 1.0:
                    fv_war = fv_info['fv_war']
                    blended = blend_war(proj_war, fv_war, alpha)
                    rate_wars.append(blended)
                    ages.append(proj_age)
                    base_usages.append(proj_usage)
                    sources.append('projected+fv')
                    fv_blend_details.append({
                        'proj_war': proj_war, 'fv_war': fv_war,
                        'alpha': alpha, 'fv': fv_info['fv'],
                    })
                    continue

            # Pure projection (no FV blend needed or alpha = 1.0)
            rate_wars.append(proj_war)
            ages.append(proj_age)
            base_usages.append(proj_usage)
            sources.append('projected')
            fv_blend_details.append(None)

        elif has_manual:
            rate_wars.append(float(manual_war))
            ages.append(np.nan)
            base_usages.append(0.0)
            sources.append('manual')
            fv_blend_details.append(None)

        elif fv_lookup and pid in fv_lookup:
            # No projection, but FV data available -- use FV WAR
            fv_info = fv_lookup[pid]
            rate_wars.append(fv_info['fv_war'])
            ages.append(np.nan)
            base_usages.append(0.0)
            sources.append('fv_only')
            fv_blend_details.append({
                'fv': fv_info['fv'], 'fv_war': fv_info['fv_war'],
            })

        else:
            rate_wars.append(0.0)
            ages.append(np.nan)
            base_usages.append(0.0)
            sources.append('not_found')
            fv_blend_details.append(None)

    roster['rate_war'] = rate_wars
    roster['age'] = ages
    roster['base_pa_ip'] = base_usages
    roster['projection_source'] = sources

    # Report manual WAR players
    manual = roster[roster['projection_source'] == 'manual']
    if len(manual) > 0:
        print(f"\nPlayers using manual WAR ({len(manual)}):")
        for _, row in manual.iterrows():
            print(f"  {row['Team']}: {row['Name']} -> {row['rate_war']:.1f} WAR (manual)")

    # Report FV-blended projections
    fv_blended = roster[roster['projection_source'] == 'projected+fv']
    if len(fv_blended) > 0:
        print(f"\nPlayers with FV-blended projections ({len(fv_blended)}):")
        for i, (_, row) in enumerate(roster.iterrows()):
            if row.get('projection_source') == 'projected+fv':
                detail = fv_blend_details[i]
                if detail:
                    print(f"  {row['Team']}: {row['Name']} -> "
                          f"{row['rate_war']:.2f} WAR "
                          f"(projected+fv, alpha={detail['alpha']:.2f})")

    # Report FV-only players
    fv_only = roster[roster['projection_source'] == 'fv_only']
    if len(fv_only) > 0:
        print(f"\nPlayers using FV-only WAR ({len(fv_only)}):")
        for i, (_, row) in enumerate(roster.iterrows()):
            if row.get('projection_source') == 'fv_only':
                detail = fv_blend_details[i]
                if detail:
                    print(f"  {row['Team']}: {row['Name']} -> "
                          f"{row['rate_war']:.2f} WAR "
                          f"(fv_only, FV={detail['fv']})")

    # Report missing projections
    not_found = roster[roster['projection_source'] == 'not_found']
    if len(not_found) > 0:
        print(f"\nPlayers on roster without projections ({len(not_found)}):")
        for _, row in not_found.iterrows():
            print(f"  {row['Team']}: {row['Name']} (playerid={row.get('playerid', 'N/A')}) -> 0.0 WAR")

    no_pid = roster[roster['projection_source'] == 'no_playerid']
    if len(no_pid) > 0:
        print(f"\nPlayers with no playerid and no manual WAR ({len(no_pid)}):")
        for _, row in no_pid.iterrows():
            print(f"  {row['Team']}: {row['Name']} -> 0.0 WAR (set manual_war in roster CSV)")

    return roster


def allocate_and_adjust(roster_df: pd.DataFrame) -> pd.DataFrame:
    """
    Run playing time allocation and WAR adjustment on the merged roster.

    Args:
        roster_df: Merged roster from merge_roster_with_projections().

    Returns:
        Roster with allocated_pa_ip, playing_time_factor, and adjusted_war.
    """
    # Allocate playing time per team
    roster = allocate_team_playing_time(roster_df)

    # Calculate adjusted WAR
    roster['adjusted_war'] = roster.apply(
        lambda row: adjust_war_for_playing_time(
            row['rate_war'], row['playing_time_factor']
        ),
        axis=1
    )

    return roster


def aggregate_team_war(enriched_roster: pd.DataFrame) -> pd.DataFrame:
    """
    Sum adjusted WAR by team, split by hitter/pitcher.

    Args:
        enriched_roster: Roster with adjusted_war column.

    Returns:
        DataFrame with per-team totals:
            Team, hitter_war, pitcher_war, total_war,
            num_hitters, num_pitchers, num_replacement, num_not_found
    """
    teams = []

    for team in sorted(enriched_roster['Team'].unique()):
        team_data = enriched_roster[enriched_roster['Team'] == team]
        hitters = team_data[team_data['player_type'] == 'hitter']
        pitchers = team_data[team_data['player_type'] == 'pitcher']

        hitter_war = hitters['adjusted_war'].sum()
        pitcher_war = pitchers['adjusted_war'].sum()

        teams.append({
            'Team': team,
            'hitter_war': round(hitter_war, 2),
            'pitcher_war': round(pitcher_war, 2),
            'total_war': round(hitter_war + pitcher_war, 2),
            'num_hitters': len(hitters),
            'num_pitchers': len(pitchers),
            'num_replacement': len(team_data[team_data['role'].isin(REPLACEMENT_ROLES)]),
            'num_manual': len(team_data[team_data['projection_source'] == 'manual']),
            'num_not_found': len(team_data[team_data['projection_source'] == 'not_found']),
            'num_fv_blended': len(team_data[team_data['projection_source'] == 'projected+fv']),
            'num_fv_only': len(team_data[team_data['projection_source'] == 'fv_only']),
        })

    result = pd.DataFrame(teams)

    # Report summary
    total_war = result['total_war'].sum()
    avg_war = result['total_war'].mean()
    print("\nTeam WAR aggregation complete:")
    print(f"  Total league WAR: {total_war:.1f}")
    print(f"  Average team WAR: {avg_war:.1f}")
    print(f"  Highest: {result.loc[result['total_war'].idxmax(), 'Team']} "
          f"({result['total_war'].max():.1f})")
    print(f"  Lowest:  {result.loc[result['total_war'].idxmin(), 'Team']} "
          f"({result['total_war'].min():.1f})")

    return result


def generate_team_war_breakdown(
    enriched_roster: pd.DataFrame,
    team: str
) -> pd.DataFrame:
    """
    Generate a detailed per-player WAR breakdown for a single team.

    Args:
        enriched_roster: Full enriched roster.
        team: Team abbreviation to filter.

    Returns:
        Sorted DataFrame showing each player's contribution.
    """
    team_data = enriched_roster[enriched_roster['Team'] == team].copy()

    display_cols = [
        'Name', 'role', 'position', 'rate_war', 'allocated_pa_ip',
        'playing_time_factor', 'adjusted_war', 'projection_source'
    ]
    available_cols = [c for c in display_cols if c in team_data.columns]

    breakdown = team_data[available_cols].sort_values('adjusted_war', ascending=False)
    breakdown = breakdown.round({
        'rate_war': 2, 'allocated_pa_ip': 1,
        'playing_time_factor': 3, 'adjusted_war': 2
    })

    return breakdown
