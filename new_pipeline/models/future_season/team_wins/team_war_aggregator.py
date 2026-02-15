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

from new_pipeline.common.constants import FANGRAPHS_HITTER_DIR, FANGRAPHS_PITCHER_DIR
from new_pipeline.models.future_season.team_wins.constants import REPLACEMENT_ROLES
from new_pipeline.models.future_season.team_wins.playing_time_estimator import (
    allocate_team_playing_time,
    adjust_war_for_playing_time
)


# Recovery factors by injury classification (year 1 post-return)
INJURY_RECOVERY_FACTORS = {
    'shoulder_surgery': 0.75,
    'hip_surgery': 0.70,
    'elbow_internal_brace': 0.80,
    'other_surgery': 0.75,
    'oblique_strain': 0.90,
    'hamstring_strain': 0.90,
    'shoulder_strain': 0.90,
    'back_strain': 0.90,
    'groin_strain': 0.90,
    'unknown': 0.80,
}

# Minimum playing time thresholds for a "qualifying" pre-injury season
MIN_PREINJURY_PA = 75
MIN_PREINJURY_IP = 20

# Full-season workload targets for rate-normalization
# WAR is a counting stat, so a half-season of 2 WAR quality work shows as ~1 WAR.
# We normalize to a full-season workload before applying recovery discounts.
FULL_SEASON_IP_SP = 170    # Typical full-season SP workload
FULL_SEASON_IP_RP = 60     # Typical full-season RP workload
FULL_SEASON_PA = 580       # Typical full-season hitter workload
GS_THRESHOLD_SP = 5        # >= 5 games started -> classify as SP for normalization


def _load_fg_stats_with_mlbamid(year: int, player_type: str) -> Optional[pd.DataFrame]:
    """
    Load FanGraphs hitter or pitcher stats for a year, returning key columns.

    Args:
        year: Season year.
        player_type: 'hitter' or 'pitcher'.

    Returns:
        DataFrame with MLBAMID, WAR, PA/IP, Name columns, or None if missing.
    """
    if player_type == 'hitter':
        path = FANGRAPHS_HITTER_DIR / f"fangraphs_hitters_{year}.csv"
        usage_col = 'PA'
    else:
        path = FANGRAPHS_PITCHER_DIR / f"fangraphs_pitchers_{year}.csv"
        usage_col = 'IP'

    if not path.exists():
        return None

    df = pd.read_csv(path, encoding='utf-8-sig')
    needed = ['MLBAMID', 'WAR', usage_col, 'Name']
    missing = [c for c in needed if c not in df.columns]
    if missing:
        return None

    result = df[needed].copy()
    # Include GS for pitchers to distinguish SP vs RP
    if player_type == 'pitcher' and 'GS' in df.columns:
        result['GS'] = df['GS']
    result = result.dropna(subset=['MLBAMID'])
    result['MLBAMID'] = result['MLBAMID'].astype(int)
    result['usage'] = result[usage_col]
    result['player_type'] = player_type
    return result


def build_injury_fallback_lookup(
    projection_year: int,
    roster_df: pd.DataFrame,
) -> Dict[int, dict]:
    """
    Build a lookup of injury-recovery WAR for rostered players missing projections.

    For players in the injury report who missed significant time, looks up their
    pre-injury WAR from FanGraphs historical data and applies injury-specific
    recovery discounts.

    Args:
        projection_year: The season being projected (e.g. 2026).
        roster_df: Roster DataFrame with 'playerid' (MLBAM ID) column.

    Returns:
        Dict mapping MLBAM ID -> {war, age, source_year, injury_type,
                                    recovery_factor, player_name}.
    """
    from new_pipeline.common.data_preparation.injury_data_loader import (
        load_injury_data,
        classify_injury_severity,
    )
    from new_pipeline.models.future_season.injury_recovery import (
        InjuryRecoveryAdjuster,
    )

    injury_year = projection_year - 1  # e.g. 2025 injuries for 2026 projection
    base_year = injury_year  # pre-injury data goes back from here

    # Load injury report
    try:
        injury_df = load_injury_data(injury_year)
    except FileNotFoundError:
        print(f"  No injury data for {injury_year} -- skipping injury fallback")
        return {}

    # Get set of MLBAM IDs on the roster
    roster_ids = set(
        roster_df['playerid'].dropna().astype(int).tolist()
    )

    # De-duplicate injury records: keep the most severe injury per player
    # (prefer 60-Day IL over 15-Day IL, prefer surgeries)
    severity_order = {'60-Day IL': 3, '10-Day IL': 1, '15-Day IL': 1}
    injury_df['_severity_rank'] = injury_df['status'].map(severity_order).fillna(2)
    injury_deduped = (
        injury_df.sort_values('_severity_rank', ascending=False)
        .drop_duplicates(subset='MLBAMID', keep='first')
    )

    # Filter to rostered players only
    injury_roster = injury_deduped[
        injury_deduped['MLBAMID'].isin(roster_ids)
    ].copy()

    if injury_roster.empty:
        print("  No rostered players found in injury report")
        return {}

    # Load historical FanGraphs data (search backwards from injury year - 1)
    # Build a combined lookup: MLBAMID -> best recent season {war, usage, year, player_type}
    preinjury_lookup = {}
    search_years = list(range(base_year - 1, base_year - 5, -1))  # e.g. 2024, 2023, 2022, 2021

    for year in search_years:
        for ptype in ['hitter', 'pitcher']:
            fg_df = _load_fg_stats_with_mlbamid(year, ptype)
            if fg_df is None:
                continue

            min_usage = MIN_PREINJURY_PA if ptype == 'hitter' else MIN_PREINJURY_IP

            for _, row in fg_df.iterrows():
                mid = int(row['MLBAMID'])
                if mid in preinjury_lookup:
                    continue  # already have a more recent season
                if row['usage'] < min_usage:
                    continue  # not enough playing time

                preinjury_lookup[mid] = {
                    'war': float(row['WAR']),
                    'usage': float(row['usage']),
                    'gs': int(row['GS']) if 'GS' in row.index and pd.notna(row.get('GS')) else 0,
                    'year': year,
                    'player_type': ptype,
                    'name': row.get('Name', ''),
                }

    # Build the final lookup with recovery-adjusted WAR
    adjuster = InjuryRecoveryAdjuster()
    result = {}

    for _, inj_row in injury_roster.iterrows():
        mid = int(inj_row['MLBAMID'])
        if mid not in preinjury_lookup:
            continue  # no historical data found

        pre = preinjury_lookup[mid]
        injury_desc = inj_row.get('injury_type', '')
        injury_class = classify_injury_severity(injury_desc)
        position = inj_row.get('Position', 'OF')
        il_status = inj_row.get('status', '')

        # Determine surgery year for TJ/ACL recovery factor calculation
        injury_date = inj_row.get('injury_date')
        if pd.notna(injury_date):
            surgery_year = pd.Timestamp(injury_date).year
        else:
            surgery_year = injury_year

        # Get recovery factor based on injury classification
        if injury_class == 'tommy_john':
            recovery_factor = adjuster.get_tommy_john_recovery_factors(
                age=28.0,  # default age, position-specific factor matters more
                position=position,
                surgery_year=surgery_year,
                projection_year=projection_year,
            )
        elif injury_class == 'acl_surgery':
            recovery_factor = adjuster.get_acl_recovery_factors(
                age=28.0,
                position=position,
                surgery_year=surgery_year,
                projection_year=projection_year,
            )
        elif injury_class in INJURY_RECOVERY_FACTORS:
            recovery_factor = INJURY_RECOVERY_FACTORS[injury_class]
        else:
            # Minor strain on short IL -> near full recovery
            if '15' in str(il_status) or '10' in str(il_status):
                recovery_factor = 0.90
            else:
                recovery_factor = INJURY_RECOVERY_FACTORS.get('unknown', 0.80)

        # Rate-normalize WAR to a full-season workload.
        # WAR is a counting stat, so injury-shortened seasons undercount ability.
        raw_war = pre['war']
        actual_usage = pre['usage']

        if pre['player_type'] == 'pitcher':
            is_sp = pre.get('gs', 0) >= GS_THRESHOLD_SP
            full_season_usage = FULL_SEASON_IP_SP if is_sp else FULL_SEASON_IP_RP
        else:
            full_season_usage = FULL_SEASON_PA

        if actual_usage > 0 and actual_usage < full_season_usage:
            rate_normalized_war = raw_war * (full_season_usage / actual_usage)
        else:
            rate_normalized_war = raw_war

        adjusted_war = rate_normalized_war * recovery_factor
        # Floor at 0 -- an injured player shouldn't project negative
        adjusted_war = max(0.0, adjusted_war)

        result[mid] = {
            'war': round(adjusted_war, 2),
            'age': np.nan,  # age not in FG base CSVs; pipeline will use roster age if available
            'source_year': pre['year'],
            'injury_type': injury_class,
            'injury_desc': str(injury_desc),
            'recovery_factor': round(recovery_factor, 3),
            'pre_injury_war': round(raw_war, 2),
            'rate_normalized_war': round(rate_normalized_war, 2),
            'usage': actual_usage,
            'player_name': pre['name'],
            'player_type': pre['player_type'],
        }

    # Report
    print(f"\n  Injury fallback lookup built: {len(result)} players")
    if result:
        sorted_players = sorted(result.items(), key=lambda x: x[1]['war'], reverse=True)
        for mid, info in sorted_players[:15]:
            normalized_note = ""
            if info['rate_normalized_war'] != info['pre_injury_war']:
                normalized_note = (f" -> {info['rate_normalized_war']:.1f} rate-norm "
                                   f"({info['usage']:.0f} IP/PA)")
            print(f"    {info['player_name']}: {info['pre_injury_war']:.1f} WAR "
                  f"({info['source_year']}){normalized_note} "
                  f"x {info['recovery_factor']:.2f} ({info['injury_type']}) "
                  f"= {info['war']:.2f} WAR")

    return result


def merge_roster_with_projections(
    roster_df: pd.DataFrame,
    hitter_projections: pd.DataFrame,
    pitcher_projections: pd.DataFrame,
    war_column: str,
    fv_lookup: Optional[Dict[int, dict]] = None,
    career_usage: Optional[Dict[int, dict]] = None,
    mle_lookup: Optional[Dict[int, dict]] = None,
    injury_lookup: Optional[Dict[int, dict]] = None,
) -> pd.DataFrame:
    """
    Map rostered players to their WAR projections.

    Joins on playerid. Players with no projection match get 0.0 WAR
    unless FV prospect data provides a fallback. Rookies with projections
    can have their WAR blended with FV-based estimates.

    Replacement-role players always get 0.0 WAR regardless of projection.

    Fallback chain: projection -> manual_war -> FV -> MLE -> injury_recovery -> 0.0

    Args:
        roster_df: Validated roster DataFrame (from load_roster).
        hitter_projections: Hitter projection CSV data.
        pitcher_projections: Pitcher projection CSV data.
        war_column: Which WAR column to use (e.g., 'war_2026').
        fv_lookup: Optional dict from match_prospects_to_roster().
            Maps MLBAM ID -> {fv, risk, fv_war, is_pitcher, match_method}.
        career_usage: Optional dict from build_career_usage_lookup().
            Maps MLBAM ID -> {career_pa, career_ip}.
        mle_lookup: Optional dict from match_mle_to_roster().
            Maps MLBAM ID -> {mle_war, translated_stat, age, player_type, ...}.
        injury_lookup: Optional dict from build_injury_fallback_lookup().
            Maps MLBAM ID -> {war, age, source_year, injury_type, recovery_factor, ...}.

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

        elif mle_lookup and pid in mle_lookup:
            # No projection or FV, but MLE data from AAA stats
            mle_info = mle_lookup[pid]
            rate_wars.append(mle_info['mle_war'])
            mle_age = mle_info.get('age')
            ages.append(float(mle_age) if mle_age is not None else np.nan)
            base_usages.append(0.0)
            sources.append('mle')
            fv_blend_details.append(None)

        elif injury_lookup and pid in injury_lookup:
            # No projection/FV/MLE, but player is in injury report with
            # historical WAR data and recovery discount applied
            inj_info = injury_lookup[pid]
            rate_wars.append(inj_info['war'])
            ages.append(inj_info.get('age', np.nan))
            base_usages.append(0.0)
            sources.append('injury_recovery')
            fv_blend_details.append(None)

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

    # Report MLE players
    mle_players = roster[roster['projection_source'] == 'mle']
    if len(mle_players) > 0:
        mle_wars = mle_players['rate_war']
        print(f"\nPlayers using MLE WAR ({len(mle_players)}):")
        print(f"  WAR range: [{mle_wars.min():.2f}, {mle_wars.max():.2f}], "
              f"median: {mle_wars.median():.2f}")
        for _, row in mle_players.sort_values('rate_war', ascending=False).head(10).iterrows():
            print(f"  {row['Team']}: {row['Name']} -> {row['rate_war']:.2f} WAR (mle)")

    # Report injury-recovery players
    injury_players = roster[roster['projection_source'] == 'injury_recovery']
    if len(injury_players) > 0:
        inj_wars = injury_players['rate_war']
        print(f"\nPlayers using injury-recovery WAR ({len(injury_players)}):")
        print(f"  WAR range: [{inj_wars.min():.2f}, {inj_wars.max():.2f}], "
              f"median: {inj_wars.median():.2f}")
        for _, row in injury_players.sort_values('rate_war', ascending=False).iterrows():
            pid = int(row['playerid'])
            inj_info = injury_lookup.get(pid, {}) if injury_lookup else {}
            inj_type = inj_info.get('injury_type', '?')
            recovery = inj_info.get('recovery_factor', '?')
            norm_war = inj_info.get('rate_normalized_war', '?')
            src_year = inj_info.get('source_year', '?')
            print(f"  {row['Team']}: {row['Name']} -> {row['rate_war']:.2f} WAR "
                  f"({inj_type}, {norm_war} norm WAR from {src_year} x {recovery})")

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
            'num_mle': len(team_data[team_data['projection_source'] == 'mle']),
            'num_injury_recovery': len(team_data[team_data['projection_source'] == 'injury_recovery']),
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
