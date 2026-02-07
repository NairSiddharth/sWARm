"""
Roster Loader - Load, validate, and generate roster CSVs.

Handles loading user-provided roster files, validating their structure,
and generating starter templates from existing projection CSVs.
"""

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from new_pipeline.models.future_season.team_wins.constants import (
    ALL_ROLES, HITTER_POSITIONS, HITTER_ROLES,
    MLB_TEAMS, REPLACEMENT_ROLES, ROLE_TO_PLAYER_TYPE
)


def load_roster(roster_path: str, projection_year: int) -> pd.DataFrame:
    """
    Load a roster CSV and validate its structure.

    Args:
        roster_path: Path to the roster CSV file.
        projection_year: Year this roster applies to (for labeling).

    Returns:
        Validated DataFrame with columns:
            Team, playerid, Name, role, position, player_type, projection_year

    Raises:
        FileNotFoundError: If roster file doesn't exist.
        ValueError: If required columns missing or critical validation fails.
    """
    path = Path(roster_path)
    if not path.exists():
        raise FileNotFoundError(f"Roster file not found: {roster_path}")

    roster_df = pd.read_csv(roster_path, encoding='utf-8-sig')

    # Check required columns
    required_cols = {'Team', 'Name', 'role'}
    missing_cols = required_cols - set(roster_df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Ensure playerid column exists (may be blank for replacement players)
    if 'playerid' not in roster_df.columns:
        roster_df['playerid'] = np.nan

    # Ensure position column exists
    if 'position' not in roster_df.columns:
        roster_df['position'] = ''

    # Clean data types
    roster_df['playerid'] = pd.to_numeric(roster_df['playerid'], errors='coerce')
    roster_df['Team'] = roster_df['Team'].astype(str).str.strip()
    roster_df['Name'] = roster_df['Name'].astype(str).str.strip()
    roster_df['role'] = roster_df['role'].astype(str).str.strip()
    roster_df['position'] = roster_df['position'].astype(str).str.strip()

    # Add derived columns
    roster_df['player_type'] = roster_df['role'].map(ROLE_TO_PLAYER_TYPE)
    roster_df['projection_year'] = projection_year

    # Run validation
    warnings = validate_roster(roster_df)
    if warnings:
        print(f"\nRoster validation warnings ({len(warnings)}):")
        for w in warnings:
            print(f"  - {w}")

    # Check for critical errors (missing teams)
    teams_present = set(roster_df['Team'].unique())
    missing_teams = MLB_TEAMS - teams_present
    if missing_teams:
        raise ValueError(
            f"Missing teams in roster ({len(missing_teams)}): {sorted(missing_teams)}"
        )

    print(f"\nLoaded roster: {len(roster_df)} players across {len(teams_present)} teams")
    return roster_df


def validate_roster(roster_df: pd.DataFrame) -> List[str]:
    """
    Validate roster data and return list of warnings.

    Checks team presence, valid roles, duplicate playerids,
    roster size reasonableness, and position validity.

    Args:
        roster_df: Roster DataFrame to validate.

    Returns:
        List of warning strings (empty if all clean).
    """
    warnings = []

    # Check all 30 teams present
    teams_present = set(roster_df['Team'].unique())
    missing_teams = MLB_TEAMS - teams_present
    if missing_teams:
        warnings.append(f"Missing teams: {sorted(missing_teams)}")

    unknown_teams = teams_present - MLB_TEAMS
    if unknown_teams:
        warnings.append(f"Unknown team abbreviations: {sorted(unknown_teams)}")

    # Check valid roles
    invalid_roles = set(roster_df['role'].unique()) - ALL_ROLES
    if invalid_roles:
        warnings.append(f"Invalid roles: {invalid_roles}. Valid: {sorted(ALL_ROLES)}")

    # Check for duplicate playerids within a team (excluding NaN/replacement)
    players_with_ids = roster_df.dropna(subset=['playerid'])
    dupes = players_with_ids.groupby(['Team', 'playerid']).size()
    dupes = dupes[dupes > 1]
    if len(dupes) > 0:
        for (team, pid), count in dupes.items():
            warnings.append(f"Duplicate playerid {int(pid)} on {team} ({count} entries)")

    # Check roster sizes and minimum starters per team
    warnings.extend(_validate_team_composition(roster_df))

    # Check positions for hitters
    hitter_rows = roster_df[roster_df['role'].isin(HITTER_ROLES - REPLACEMENT_ROLES)]
    invalid_hitter_pos = set(hitter_rows['position'].unique()) - HITTER_POSITIONS - {'', 'nan'}
    if invalid_hitter_pos:
        warnings.append(f"Invalid hitter positions: {invalid_hitter_pos}. Valid: {sorted(HITTER_POSITIONS)}")

    return warnings


def _validate_team_composition(roster_df: pd.DataFrame) -> List[str]:
    """
    Validate roster size and starter counts per team.

    Args:
        roster_df: Full roster DataFrame.

    Returns:
        List of warning strings for composition issues.
    """
    warnings = []
    for team in roster_df['Team'].unique():
        team_data = roster_df[roster_df['Team'] == team]
        team_size = len(team_data)

        if team_size < 20:
            warnings.append(
                f"{team}: Only {team_size} players (expected 20-40)"
            )
        elif team_size > 40:
            warnings.append(
                f"{team}: {team_size} players (expected 20-40)"
            )

        n_starter_h = len(
            team_data[team_data['role'] == 'starter_hitter']
        )
        n_starter_p = len(
            team_data[team_data['role'] == 'starter_pitcher']
        )

        if n_starter_h < 8:
            warnings.append(
                f"{team}: Only {n_starter_h} starter_hitters (expected 8+)"
            )
        if n_starter_p < 3:
            warnings.append(
                f"{team}: Only {n_starter_p} starter_pitchers (expected 3+)"
            )

    return warnings


def generate_roster_template(
    hitter_projections_path: str,
    pitcher_projections_path: str,
    output_path: str,
    projection_year: int = None
) -> str:
    """
    Generate a starter roster template CSV from existing projection files.

    Reads the projection CSVs, groups players by Team, and auto-assigns:
    - Roles: Top 9 hitters per team (by PA) -> starter_hitter, rest -> bench_hitter.
             Pitchers with GS >= 5 or total IP above threshold -> starter_pitcher,
             rest -> reliever.
    - Positions: From FanGraphs defensive data keyed by MLBAMID.

    Users then edit the template for offseason moves.

    Args:
        hitter_projections_path: Path to future_projections_hitter_YYYY.csv.
        pitcher_projections_path: Path to future_projections_pitcher_YYYY.csv.
        output_path: Path to write the template CSV.
        projection_year: Year label (auto-detected from filename if None).

    Returns:
        Path where template was saved.
    """
    hitters = pd.read_csv(hitter_projections_path, encoding='utf-8-sig')
    pitchers = pd.read_csv(pitcher_projections_path, encoding='utf-8-sig')

    # Detect projection year from war columns if not provided
    if projection_year is None:
        war_cols = [c for c in hitters.columns if c.startswith('war_') and c[4:].isdigit()]
        if war_cols:
            projection_year = int(sorted(war_cols)[0].split('_')[1])
        else:
            projection_year = 2026

    # Load position data from FanGraphs defensive files
    position_lookup = _load_position_lookup(projection_year)

    # --- Hitters: top 9 by PA per team = starter, rest = bench ---
    hitter_rows = []
    hitters_valid = hitters[~hitters['Team'].isin(['- - -', 'nan', '', 'None'])].copy()
    hitters_valid = hitters_valid[hitters_valid['Team'].notna()]
    hitters_valid['PA'] = pd.to_numeric(hitters_valid['PA'], errors='coerce').fillna(0)

    for team, group in hitters_valid.groupby('Team'):
        # Rank by PA within team (top 9 = starters)
        group = group.sort_values('PA', ascending=False)

        for i, (_, row) in enumerate(group.iterrows()):
            role = 'starter_hitter' if i < 9 else 'bench_hitter'

            pid = int(row['playerid']) if pd.notna(row['playerid']) else ''
            position = position_lookup.get(pid, 'DH') if pid != '' else 'DH'

            hitter_rows.append({
                'Team': team,
                'playerid': pid,
                'Name': row.get('Name', ''),
                'role': role,
                'position': position,
            })

    # --- Pitchers: use GS + total IP for role classification ---
    # Load GS data from FanGraphs pitcher CSV for reliable starter detection
    gs_lookup = _load_pitcher_gs_lookup(projection_year)

    # Build total IP lookup from combined '- - -' rows in projections
    # (handles traded players whose IP is split across teams)
    total_ip_lookup = {}
    combined_rows = pitchers[pitchers['Team'] == '- - -']
    for _, row in combined_rows.iterrows():
        if pd.notna(row['playerid']):
            pid = int(row['playerid'])
            ip = pd.to_numeric(row.get('IP', 0), errors='coerce')
            if pd.notna(ip):
                total_ip_lookup[pid] = ip

    pitcher_rows = []
    pitchers_valid = pitchers[~pitchers['Team'].isin(['- - -', 'nan', '', 'None'])].copy()
    pitchers_valid = pitchers_valid[pitchers_valid['Team'].notna()]
    pitchers_valid['IP'] = pd.to_numeric(pitchers_valid['IP'], errors='coerce').fillna(0)

    for team, group in pitchers_valid.groupby('Team'):
        group = group.sort_values('IP', ascending=False)

        # Per-team IP threshold as fallback
        median_ip = group['IP'].median()
        starter_threshold = max(median_ip * 1.5, 60)

        for _, row in group.iterrows():
            pid = int(row['playerid']) if pd.notna(row['playerid']) else ''
            ip = row['IP']

            # Primary: use GS from FanGraphs data (not split by team)
            gs = gs_lookup.get(pid, 0) if pid != '' else 0

            # Secondary: use total IP across all teams (from combined row)
            player_total_ip = total_ip_lookup.get(pid, ip) if pid != '' else ip

            if gs >= 5 or player_total_ip >= starter_threshold:
                role = 'starter_pitcher'
                position = 'SP'
            else:
                role = 'reliever'
                position = 'RP'

            pitcher_rows.append({
                'Team': team,
                'playerid': pid,
                'Name': row.get('Name', ''),
                'role': role,
                'position': position,
            })

    # Combine and sort
    all_rows = hitter_rows + pitcher_rows
    template_df = pd.DataFrame(all_rows)
    template_df = template_df.sort_values(['Team', 'role', 'Name']).reset_index(drop=True)

    # Reorder columns
    template_df = template_df[['Team', 'playerid', 'Name', 'role', 'position']]

    # Save
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    template_df.to_csv(output, index=False, encoding='utf-8-sig')

    # Report summary
    teams_count = template_df['Team'].nunique()
    n_starter_h = len(template_df[template_df['role'] == 'starter_hitter'])
    n_bench_h = len(template_df[template_df['role'] == 'bench_hitter'])
    n_starter_p = len(template_df[template_df['role'] == 'starter_pitcher'])
    n_reliever = len(template_df[template_df['role'] == 'reliever'])

    print(f"\nRoster template generated: {output}")
    print(f"  Total players: {len(template_df)}")
    print(f"  Teams: {teams_count}")
    print(f"  Starter hitters: {n_starter_h} ({n_starter_h / teams_count:.1f}/team)")
    print(f"  Bench hitters: {n_bench_h} ({n_bench_h / teams_count:.1f}/team)")
    print(f"  Starting pitchers: {n_starter_p} ({n_starter_p / teams_count:.1f}/team)")
    print(f"  Relievers: {n_reliever} ({n_reliever / teams_count:.1f}/team)")
    print("\nEdit this file to reflect current rosters (trades, signings, releases, role changes).")

    return str(output)


def _load_position_lookup(projection_year: int) -> dict:
    """
    Load player position data from FanGraphs defensive files.

    Uses load_positions_all_years() from the existing pipeline to get
    primary positions keyed by MLBAMID (matching 'playerid' in projection CSVs).

    Falls back to direct CSV loading if the import fails.

    Args:
        projection_year: Year to determine which historical data to load.

    Returns:
        Dict mapping MLBAMID (int) -> position string (e.g., 'SS', 'LF', 'DH').
    """
    try:
        from new_pipeline.common.loaders.hitter_loaders import load_positions_all_years
        # load_positions_all_years overwrites earlier entries with later ones,
        # so pass years oldest-first so the most recent year takes priority
        base_year = projection_year - 1
        years_oldest_first = [base_year - 2, base_year - 1, base_year]
        position_dict = load_positions_all_years(years_oldest_first)
        print(f"  Loaded position data for {len(position_dict)} players")
        return position_dict
    except Exception as e:
        print(f"  Could not load position data: {e}")
        print("  Positions will default to 'DH' for hitters -- edit manually")
        return {}


def _load_pitcher_gs_lookup(projection_year: int) -> dict:
    """
    Load Games Started (GS) data for pitchers from FanGraphs pitcher files.

    Uses base year data (projection_year - 1) to determine which pitchers
    are starters vs relievers. Keyed by MLBAMID to match 'playerid' in
    projection CSVs.

    Args:
        projection_year: Year of projections (loads data from year before).

    Returns:
        Dict mapping MLBAMID (int) -> GS count (int).
    """
    from new_pipeline.common.constants import FANGRAPHS_PITCHER_DIR

    gs_lookup = {}
    base_year = projection_year - 1
    years_to_try = [base_year, base_year - 1]

    for year in years_to_try:
        pitcher_path = FANGRAPHS_PITCHER_DIR / f"fangraphs_pitchers_{year}.csv"

        # Try partial season files if main doesn't exist
        if not pitcher_path.exists():
            import glob
            partial_files = glob.glob(
                str(FANGRAPHS_PITCHER_DIR / f"fangraphs_pitchers_{year}_*.csv")
            )
            if partial_files:
                base_files = [f for f in partial_files if not any(
                    suffix in f for suffix in [
                        '_advanced', '_standard', '_battedball',
                        '_stuff', '_winprobability'
                    ]
                )]
                if base_files:
                    pitcher_path = Path(base_files[0])

        if not pitcher_path.exists():
            continue

        try:
            df = pd.read_csv(pitcher_path, encoding='utf-8')
            if 'MLBAMID' not in df.columns or 'GS' not in df.columns:
                continue

            for _, row in df.iterrows():
                if pd.notna(row['MLBAMID']) and pd.notna(row['GS']):
                    mlbamid = int(row['MLBAMID'])
                    if mlbamid not in gs_lookup:
                        gs_lookup[mlbamid] = int(row['GS'])
        except Exception:
            continue

    print(f"  Loaded GS data for {len(gs_lookup)} pitchers")
    return gs_lookup
