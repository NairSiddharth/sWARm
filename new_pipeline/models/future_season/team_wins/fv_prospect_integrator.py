"""
FV Prospect Integrator -- Blend FanGraphs Future Value prospect grades
into the team wins pipeline.

Two modes:
1. Rookies WITH statistical projections: blend FV-based WAR with projection
2. Roster players WITHOUT projections: use FV-based WAR instead of 0.0

All FV-specific logic is encapsulated here.
"""

import unicodedata
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from new_pipeline.common.constants import FANGRAPHS_HITTER_DIR, FANGRAPHS_PITCHER_DIR
from new_pipeline.models.future_season.team_wins.constants import (
    FV_TO_FIRST_YEAR_WAR,
    FV_RISK_CONFIDENCE,
    ROOKIE_PA_THRESHOLD,
    ROOKIE_IP_THRESHOLD,
    FV_BLEND_ALPHA_MIN,
    FV_BLEND_ALPHA_MAX,
)

# Positions classified as pitcher on The Board / international prospects
PITCHER_POSITIONS = {'SP', 'MIRP', 'SIRP'}


def _normalize_name(name: str) -> str:
    """
    Normalize a player name for fuzzy matching.

    Strips accents, lowercases, removes periods/hyphens/apostrophes,
    and collapses whitespace.
    """
    if not isinstance(name, str):
        return ''
    # Decompose unicode, strip combining characters (accents)
    nfkd = unicodedata.normalize('NFKD', name)
    ascii_name = ''.join(c for c in nfkd if not unicodedata.combining(c))
    # Lowercase, strip punctuation that varies across sources
    ascii_name = ascii_name.lower()
    for ch in '.\'"-':
        ascii_name = ascii_name.replace(ch, '')
    # Collapse whitespace
    return ' '.join(ascii_name.split())


def _is_pitcher_position(pos: str) -> bool:
    """Check if a prospect board position is a pitcher role."""
    if not isinstance(pos, str):
        return False
    return pos.strip() in PITCHER_POSITIONS


def build_fg_to_mlbam_crosswalk(
    years: Optional[range] = None,
) -> Dict[str, int]:
    """
    Build a FanGraphs PlayerId -> MLBAM ID crosswalk from stat files.

    Scans hitter and pitcher base CSV files across the given years,
    extracting (PlayerId, MLBAMID) pairs.

    Args:
        years: Range of years to scan. Default: 2018-2025.

    Returns:
        Dict mapping FG PlayerId (str) to MLBAM ID (int).
    """
    if years is None:
        years = range(2018, 2026)

    crosswalk: Dict[str, int] = {}

    for year in years:
        for data_dir, pattern in [
            (FANGRAPHS_HITTER_DIR, f"fangraphs_hitters_{year}.csv"),
            (FANGRAPHS_PITCHER_DIR, f"fangraphs_pitchers_{year}.csv"),
        ]:
            path = data_dir / pattern
            if not path.exists():
                continue
            try:
                df = pd.read_csv(path, usecols=['PlayerId', 'MLBAMID'],
                                 encoding='utf-8-sig')
                for _, row in df.iterrows():
                    fg_id = row.get('PlayerId')
                    mlbam_id = row.get('MLBAMID')
                    if pd.notna(fg_id) and pd.notna(mlbam_id):
                        crosswalk[str(int(fg_id))] = int(mlbam_id)
            except Exception:
                continue

    return crosswalk


def build_career_usage_lookup(
    years: Optional[range] = None,
) -> Dict[int, dict]:
    """
    Build career PA/IP totals per MLBAM ID from FanGraphs stat files.

    Accumulates usage across all available years to determine
    how experienced a player is (for rookie detection).

    Args:
        years: Range of years to scan. Default: 2018-2025.

    Returns:
        Dict mapping MLBAM ID -> {'career_pa': float, 'career_ip': float}.
    """
    if years is None:
        years = range(2018, 2026)

    usage: Dict[int, dict] = {}

    for year in years:
        # Hitters: accumulate PA
        hitter_path = FANGRAPHS_HITTER_DIR / f"fangraphs_hitters_{year}.csv"
        if hitter_path.exists():
            try:
                df = pd.read_csv(hitter_path, usecols=['MLBAMID', 'PA'],
                                 encoding='utf-8-sig')
                for _, row in df.iterrows():
                    mlbam = row.get('MLBAMID')
                    pa = row.get('PA', 0)
                    if pd.notna(mlbam):
                        mlbam = int(mlbam)
                        if mlbam not in usage:
                            usage[mlbam] = {'career_pa': 0.0, 'career_ip': 0.0}
                        if pd.notna(pa):
                            usage[mlbam]['career_pa'] += float(pa)
            except Exception:
                continue

        # Pitchers: accumulate IP
        pitcher_path = FANGRAPHS_PITCHER_DIR / f"fangraphs_pitchers_{year}.csv"
        if pitcher_path.exists():
            try:
                df = pd.read_csv(pitcher_path, usecols=['MLBAMID', 'IP'],
                                 encoding='utf-8-sig')
                for _, row in df.iterrows():
                    mlbam = row.get('MLBAMID')
                    ip = row.get('IP', 0)
                    if pd.notna(mlbam):
                        mlbam = int(mlbam)
                        if mlbam not in usage:
                            usage[mlbam] = {'career_pa': 0.0, 'career_ip': 0.0}
                        if pd.notna(ip):
                            usage[mlbam]['career_ip'] += float(ip)
            except Exception:
                continue

    return usage


def build_fv_lookup(
    board_year: int,
    prospect_dir: Path,
    intl_dir: Path,
) -> pd.DataFrame:
    """
    Load and merge The Board + international prospect data for a given year.

    Domestic prospects take priority in deduplication.

    Args:
        board_year: Year of the prospect board (e.g. 2025).
        prospect_dir: Directory containing The Board CSVs.
        intl_dir: Directory containing international prospect CSVs.

    Returns:
        DataFrame with columns:
            fg_player_id, Name, Org, Pos, FV, Risk, name_normalized
    """
    rows = []

    # Load domestic board
    board_path = prospect_dir / f"fangraphs_the_board_{board_year}.csv"
    if board_path.exists():
        board = pd.read_csv(board_path, encoding='utf-8-sig')
        for _, row in board.iterrows():
            fv = str(row.get('FV', '')).strip()
            if fv not in FV_TO_FIRST_YEAR_WAR:
                continue
            rows.append({
                'fg_player_id': str(row.get('PlayerId', '')).strip(),
                'Name': str(row.get('Name', '')).strip(),
                'Org': str(row.get('Org', '')).strip(),
                'Pos': str(row.get('Pos', '')).strip(),
                'FV': fv,
                'Risk': str(row.get('Risk', 'High')).strip(),
                'source': 'domestic',
            })

    # Load international prospects
    intl_path = intl_dir / f"fangraphs_internationalprospects_{board_year}.csv"
    if intl_path.exists():
        intl = pd.read_csv(intl_path, encoding='utf-8-sig')
        for _, row in intl.iterrows():
            fv = str(row.get('FV', '')).strip()
            if fv not in FV_TO_FIRST_YEAR_WAR:
                continue
            # International uses 'Proj Team' instead of 'Org'
            org = str(row.get('Proj Team', row.get('Org', ''))).strip()
            rows.append({
                'fg_player_id': str(row.get('PlayerId', '')).strip(),
                'Name': str(row.get('Name', '')).strip(),
                'Org': org,
                'Pos': str(row.get('Pos', '')).strip(),
                'FV': fv,
                'Risk': str(row.get('Risk', 'High')).strip(),
                'source': 'international',
            })

    if not rows:
        return pd.DataFrame(columns=[
            'fg_player_id', 'Name', 'Org', 'Pos', 'FV', 'Risk', 'name_normalized'
        ])

    fv_df = pd.DataFrame(rows)

    # Deduplicate: domestic takes priority
    fv_df = fv_df.sort_values('source', ascending=True)  # domestic first
    fv_df = fv_df.drop_duplicates(subset='fg_player_id', keep='first')

    fv_df['name_normalized'] = fv_df['Name'].apply(_normalize_name)
    fv_df = fv_df.drop(columns=['source'])

    return fv_df


def calculate_fv_war(fv_grade: str, risk: str, is_pitcher: bool) -> float:
    """
    Convert an FV grade + risk level into an expected first-year WAR.

    Args:
        fv_grade: FV string (e.g. '55', '45+').
        risk: Risk level ('Low', 'Med', 'High').
        is_pitcher: Whether the prospect is a pitcher.

    Returns:
        Risk-adjusted first-year WAR estimate.
    """
    if fv_grade not in FV_TO_FIRST_YEAR_WAR:
        return 0.0

    hitter_war, pitcher_war = FV_TO_FIRST_YEAR_WAR[fv_grade]
    base_war = pitcher_war if is_pitcher else hitter_war

    confidence = FV_RISK_CONFIDENCE.get(risk, 0.70)
    return round(base_war * confidence, 3)


def calculate_blend_alpha(career_pa_ip: float, is_pitcher: bool) -> float:
    """
    Calculate the blending alpha (weight on statistical projection).

    Alpha interpolates linearly from ALPHA_MIN to ALPHA_MAX as
    career usage goes from 0 to the rookie threshold. Above the
    threshold, alpha = 1.0 (pure projection, no FV).

    Args:
        career_pa_ip: Career PA (hitters) or IP (pitchers).
        is_pitcher: Whether the player is a pitcher.

    Returns:
        Alpha in [ALPHA_MIN, 1.0].
    """
    threshold = ROOKIE_IP_THRESHOLD if is_pitcher else ROOKIE_PA_THRESHOLD

    if career_pa_ip >= threshold:
        return 1.0

    # Linear interpolation: 0 -> ALPHA_MIN, threshold -> ALPHA_MAX
    fraction = career_pa_ip / threshold
    alpha = FV_BLEND_ALPHA_MIN + fraction * (FV_BLEND_ALPHA_MAX - FV_BLEND_ALPHA_MIN)
    return round(alpha, 4)


def blend_war(projected_war: float, fv_war: float, alpha: float) -> float:
    """
    Blend statistical projection with FV-based WAR.

    Args:
        projected_war: WAR from statistical projection system.
        fv_war: WAR from FV grade calculation.
        alpha: Weight on statistical projection (0-1).

    Returns:
        Blended WAR value.
    """
    return round(alpha * projected_war + (1 - alpha) * fv_war, 3)


def match_prospects_to_roster(
    fv_df: pd.DataFrame,
    roster_df: pd.DataFrame,
    crosswalk: Dict[str, int],
) -> Dict[int, dict]:
    """
    Match prospect FV data to roster players via ID crosswalk and name matching.

    Three-tier matching:
    1. Crosswalk: FG PlayerId -> MLBAMID -> roster playerid
    2. Name match: normalized name, disambiguated by team
    3. Unresolved: silently skipped

    Args:
        fv_df: DataFrame from build_fv_lookup().
        roster_df: Roster DataFrame with playerid, Name, Team columns.
        crosswalk: Dict from build_fg_to_mlbam_crosswalk().

    Returns:
        Dict mapping MLBAM ID (int) -> {
            'fv': str, 'risk': str, 'fv_war': float,
            'is_pitcher': bool, 'match_method': str
        }
    """
    if fv_df.empty or roster_df is None or roster_df.empty:
        return {}

    result: Dict[int, dict] = {}

    # Build roster lookup sets
    roster_mlbam_ids = set()
    for _, row in roster_df.iterrows():
        pid = row.get('playerid')
        if pd.notna(pid):
            roster_mlbam_ids.add(int(pid))

    # Build name -> roster player(s) lookup for fallback matching
    roster_by_name: Dict[str, list] = {}
    for _, row in roster_df.iterrows():
        pid = row.get('playerid')
        if pd.isna(pid):
            continue
        name_norm = _normalize_name(str(row.get('Name', '')))
        team = str(row.get('Team', ''))
        if name_norm:
            if name_norm not in roster_by_name:
                roster_by_name[name_norm] = []
            roster_by_name[name_norm].append({
                'playerid': int(pid),
                'team': team,
            })

    # Match each prospect
    for _, prospect in fv_df.iterrows():
        fg_id = str(prospect['fg_player_id'])
        fv_grade = prospect['FV']
        risk = prospect['Risk']
        pos = prospect['Pos']
        is_pitcher = _is_pitcher_position(pos)

        fv_war = calculate_fv_war(fv_grade, risk, is_pitcher)

        matched_mlbam = None
        match_method = None

        # Tier 1: Crosswalk
        if fg_id in crosswalk:
            candidate = crosswalk[fg_id]
            if candidate in roster_mlbam_ids:
                matched_mlbam = candidate
                match_method = 'crosswalk'

        # Tier 2: Name match
        if matched_mlbam is None:
            name_norm = prospect.get('name_normalized', '')
            if name_norm and name_norm in roster_by_name:
                candidates = roster_by_name[name_norm]
                if len(candidates) == 1:
                    matched_mlbam = candidates[0]['playerid']
                    match_method = 'name'
                else:
                    # Disambiguate by team (Org from prospect board)
                    prospect_org = prospect.get('Org', '')
                    for c in candidates:
                        if c['team'] == prospect_org:
                            matched_mlbam = c['playerid']
                            match_method = 'name+team'
                            break

        if matched_mlbam is not None:
            result[matched_mlbam] = {
                'fv': fv_grade,
                'risk': risk,
                'fv_war': fv_war,
                'is_pitcher': is_pitcher,
                'match_method': match_method,
            }

    return result
