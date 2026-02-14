"""
Minor League Equivalency (MLE) Projector.

Translates AAA stats into MLB-equivalent WAR for players who lack
MLB projections and FV prospect grades. Uses historical AAA-to-MLB
translation factors computed from players who appeared at both levels.

Architecture:
    AAA wRC+ (hitters) / FIP (pitchers)
        -> apply translation factor (multi-year median ratio)
        -> MLB-equivalent wRC+ / FIP
        -> convert to WAR via linear fit from MLB data
        -> age adjustment
        -> single WAR estimate for team_war_aggregator
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from new_pipeline.common.constants import FANGRAPHS_HITTER_DIR, FANGRAPHS_PITCHER_DIR
from new_pipeline.models.future_season.team_wins.constants import (
    MILB_DATA_DIR,
    MLE_MIN_PA,
    MLE_MIN_IP,
    MLE_WAR_FLOOR,
    MLE_WAR_CAP,
    MLE_AGE_BONUS_THRESHOLD,
    MLE_AGE_PENALTY_THRESHOLD,
    MLE_AGE_ADJUSTMENT,
)


# WAR conversion coefficients derived from MLB data:
#   Hitters: WAR/600PA = 0.065 * wRC+ - 4.57  (R=0.80, 2634 MLB seasons 2018-2025)
#   Pitchers: WAR/180IP = -1.81 * FIP + 9.08  (R=0.59, MLB 2024)
HITTER_WAR_SLOPE = 0.065
HITTER_WAR_INTERCEPT = -4.57
HITTER_WAR_PA_BASIS = 600

PITCHER_WAR_SLOPE = -1.81
PITCHER_WAR_INTERCEPT = 9.08
PITCHER_WAR_IP_BASIS = 180


def _load_aaa_hitter_stats(year: int) -> Optional[pd.DataFrame]:
    """
    Load AAA hitter stats for a given year by merging standard and advanced CSVs.

    Returns DataFrame with columns: PlayerId, Name, Age, PA, wRC+
    or None if files don't exist.
    """
    aaa_dir = MILB_DATA_DIR / "AAA" / "hitters"
    standard_path = aaa_dir / f"fangraphs_hitters_{year}_standard_AAA.csv"
    advanced_path = aaa_dir / f"fangraphs_hitters_{year}_advanced_AAA.csv"

    if not standard_path.exists() or not advanced_path.exists():
        return None

    standard = pd.read_csv(standard_path, encoding='utf-8-sig')
    advanced = pd.read_csv(advanced_path, encoding='utf-8-sig')

    # Standard has PA; advanced has wRC+. Both have PlayerId.
    merged = standard[['PlayerId', 'Name', 'Age', 'PA']].merge(
        advanced[['PlayerId', 'wRC+']],
        on='PlayerId',
        how='inner',
    )
    return merged


def _load_aaa_pitcher_stats(year: int) -> Optional[pd.DataFrame]:
    """
    Load AAA pitcher stats for a given year by merging standard and advanced CSVs.

    Returns DataFrame with columns: PlayerId, Name, Age, IP, FIP
    or None if files don't exist.
    """
    aaa_dir = MILB_DATA_DIR / "AAA" / "pitchers"
    standard_path = aaa_dir / f"fangraphs_pitchers_{year}_standard_AAA.csv"
    advanced_path = aaa_dir / f"fangraphs_pitchers_{year}_advanced_AAA.csv"

    if not standard_path.exists() or not advanced_path.exists():
        return None

    standard = pd.read_csv(standard_path, encoding='utf-8-sig')
    advanced = pd.read_csv(advanced_path, encoding='utf-8-sig')

    merged = standard[['PlayerId', 'Name', 'Age', 'IP']].merge(
        advanced[['PlayerId', 'FIP']],
        on='PlayerId',
        how='inner',
    )
    return merged


def _load_mlb_hitter_stats(year: int) -> Optional[pd.DataFrame]:
    """
    Load MLB hitter stats for a given year.

    The base fangraphs_hitters_{year}.csv has WAR, PA, wRC+, PlayerId.

    Returns DataFrame with columns: PlayerId, PA, wRC+, WAR
    or None if file doesn't exist.
    """
    path = FANGRAPHS_HITTER_DIR / f"fangraphs_hitters_{year}.csv"
    if not path.exists():
        return None

    df = pd.read_csv(path, encoding='utf-8-sig')
    # PlayerId in MLB files is the FG PlayerId
    cols_needed = ['PlayerId', 'PA', 'wRC+', 'WAR']
    missing = [c for c in cols_needed if c not in df.columns]
    if missing:
        return None

    return df[cols_needed].copy()


def _load_mlb_pitcher_stats(year: int) -> Optional[pd.DataFrame]:
    """
    Load MLB pitcher stats for a given year.

    The base fangraphs_pitchers_{year}.csv has WAR, IP, FIP, PlayerId.

    Returns DataFrame with columns: PlayerId, IP, FIP, WAR
    or None if file doesn't exist.
    """
    path = FANGRAPHS_PITCHER_DIR / f"fangraphs_pitchers_{year}.csv"
    if not path.exists():
        return None

    df = pd.read_csv(path, encoding='utf-8-sig')
    cols_needed = ['PlayerId', 'IP', 'FIP', 'WAR']
    missing = [c for c in cols_needed if c not in df.columns]
    if missing:
        return None

    return df[cols_needed].copy()


def build_translation_model(
    years: Optional[List[int]] = None,
) -> Dict[str, float]:
    """
    Build AAA-to-MLB translation factors from historical paired data.

    For each year, finds players who appeared in both AAA and MLB with
    meaningful sample sizes. Computes per-player ratios of MLB stat to
    AAA stat, then takes the median across all player-years.

    Args:
        years: Years to include. Default: 2016-2025 excluding 2020.

    Returns:
        Dict with keys:
            'hitter_wrc_ratio': median(MLB_wRC+ / AAA_wRC+)
            'pitcher_fip_ratio': median(MLB_FIP / AAA_FIP)
            'hitter_pairs': total number of hitter player-years used
            'pitcher_pairs': total number of pitcher player-years used
    """
    if years is None:
        years = [y for y in range(2016, 2026) if y != 2020]

    hitter_ratios = []
    pitcher_ratios = []

    for year in years:
        # Hitters
        aaa_h = _load_aaa_hitter_stats(year)
        mlb_h = _load_mlb_hitter_stats(year)
        if aaa_h is not None and mlb_h is not None:
            # Ensure PlayerId types match for merge
            aaa_h['PlayerId'] = aaa_h['PlayerId'].astype(str)
            mlb_h['PlayerId'] = mlb_h['PlayerId'].astype(str)

            paired = aaa_h.merge(
                mlb_h, on='PlayerId', suffixes=('_aaa', '_mlb')
            )
            # Filter for meaningful sample
            paired = paired[
                (paired['PA_aaa'] >= MLE_MIN_PA) &
                (paired['PA_mlb'] >= MLE_MIN_PA)
            ]
            # Compute ratio, skip players with 0 or negative AAA wRC+
            for _, row in paired.iterrows():
                aaa_wrc = row['wRC+_aaa']
                mlb_wrc = row['wRC+_mlb']
                if pd.notna(aaa_wrc) and pd.notna(mlb_wrc) and aaa_wrc > 0:
                    hitter_ratios.append(mlb_wrc / aaa_wrc)

        # Pitchers
        aaa_p = _load_aaa_pitcher_stats(year)
        mlb_p = _load_mlb_pitcher_stats(year)
        if aaa_p is not None and mlb_p is not None:
            aaa_p['PlayerId'] = aaa_p['PlayerId'].astype(str)
            mlb_p['PlayerId'] = mlb_p['PlayerId'].astype(str)

            paired = aaa_p.merge(
                mlb_p, on='PlayerId', suffixes=('_aaa', '_mlb')
            )
            paired = paired[
                (paired['IP_aaa'] >= MLE_MIN_IP) &
                (paired['IP_mlb'] >= MLE_MIN_IP)
            ]
            for _, row in paired.iterrows():
                aaa_fip = row['FIP_aaa']
                mlb_fip = row['FIP_mlb']
                if pd.notna(aaa_fip) and pd.notna(mlb_fip) and aaa_fip > 0:
                    pitcher_ratios.append(mlb_fip / aaa_fip)

    hitter_wrc_ratio = float(np.median(hitter_ratios)) if hitter_ratios else 0.69
    pitcher_fip_ratio = float(np.median(pitcher_ratios)) if pitcher_ratios else 1.12

    print(f"\nMLE Translation Model:")
    print(f"  Hitter wRC+ ratio (AAA->MLB): {hitter_wrc_ratio:.3f} "
          f"({len(hitter_ratios)} player-years)")
    print(f"  Pitcher FIP ratio (AAA->MLB): {pitcher_fip_ratio:.3f} "
          f"({len(pitcher_ratios)} player-years)")

    return {
        'hitter_wrc_ratio': hitter_wrc_ratio,
        'pitcher_fip_ratio': pitcher_fip_ratio,
        'hitter_pairs': len(hitter_ratios),
        'pitcher_pairs': len(pitcher_ratios),
    }


def _wrc_to_war(mlb_equiv_wrc: float) -> float:
    """Convert MLB-equivalent wRC+ to rate WAR (per 600 PA)."""
    return HITTER_WAR_SLOPE * mlb_equiv_wrc + HITTER_WAR_INTERCEPT


def _fip_to_war(mlb_equiv_fip: float) -> float:
    """Convert MLB-equivalent FIP to rate WAR (per 180 IP)."""
    return PITCHER_WAR_SLOPE * mlb_equiv_fip + PITCHER_WAR_INTERCEPT


def _apply_age_adjustment(war: float, age: float) -> float:
    """
    Apply age-based WAR adjustment for MLE players.

    Young for AAA (under 25): +0.2 WAR (upside)
    Prime (25-28): no adjustment
    Old for AAA (29+): -0.2 WAR (AAAA ceiling)
    """
    if pd.isna(age):
        return war
    if age < MLE_AGE_BONUS_THRESHOLD:
        return war + MLE_AGE_ADJUSTMENT
    if age >= MLE_AGE_PENALTY_THRESHOLD:
        return war - MLE_AGE_ADJUSTMENT
    return war


def build_mle_lookup(
    projection_year: int,
    translation_factors: Dict[str, float],
) -> Dict[str, dict]:
    """
    Build MLE WAR lookup for all AAA players from the most recent season.

    Loads AAA stats from projection_year - 1, translates to MLB-equivalent
    stats using translation factors, converts to WAR, and applies age adjustment.

    Args:
        projection_year: The season being projected (e.g., 2026).
        translation_factors: Output of build_translation_model().

    Returns:
        Dict mapping FG PlayerId (str) -> {
            'mle_war': float,
            'translated_stat': float,  # MLB-equiv wRC+ or FIP
            'raw_stat': float,         # original AAA stat
            'age': float,
            'player_type': str,        # 'hitter' or 'pitcher'
            'name': str,
        }
    """
    stats_year = projection_year - 1
    hitter_wrc_ratio = translation_factors['hitter_wrc_ratio']
    pitcher_fip_ratio = translation_factors['pitcher_fip_ratio']

    lookup: Dict[str, dict] = {}

    # Hitters
    aaa_h = _load_aaa_hitter_stats(stats_year)
    if aaa_h is not None:
        for _, row in aaa_h.iterrows():
            fg_id = str(row['PlayerId']).strip()
            aaa_wrc = row.get('wRC+')
            pa = row.get('PA', 0)
            age = row.get('Age', np.nan)
            name = row.get('Name', '')

            if pd.isna(aaa_wrc) or pa < MLE_MIN_PA:
                continue

            mlb_equiv_wrc = aaa_wrc * hitter_wrc_ratio
            war = _wrc_to_war(mlb_equiv_wrc)
            war = _apply_age_adjustment(war, age)
            war = max(MLE_WAR_FLOOR, min(MLE_WAR_CAP, war))

            lookup[fg_id] = {
                'mle_war': round(war, 2),
                'translated_stat': round(mlb_equiv_wrc, 1),
                'raw_stat': round(float(aaa_wrc), 1),
                'age': float(age) if pd.notna(age) else None,
                'player_type': 'hitter',
                'name': str(name),
            }

    # Pitchers
    aaa_p = _load_aaa_pitcher_stats(stats_year)
    if aaa_p is not None:
        for _, row in aaa_p.iterrows():
            fg_id = str(row['PlayerId']).strip()
            aaa_fip = row.get('FIP')
            ip = row.get('IP', 0)
            age = row.get('Age', np.nan)
            name = row.get('Name', '')

            if pd.isna(aaa_fip) or ip < MLE_MIN_IP:
                continue

            # Don't overwrite a hitter entry if the same player hit and pitched
            if fg_id in lookup:
                continue

            mlb_equiv_fip = aaa_fip * pitcher_fip_ratio
            war = _fip_to_war(mlb_equiv_fip)
            war = _apply_age_adjustment(war, age)
            war = max(MLE_WAR_FLOOR, min(MLE_WAR_CAP, war))

            lookup[fg_id] = {
                'mle_war': round(war, 2),
                'translated_stat': round(mlb_equiv_fip, 2),
                'raw_stat': round(float(aaa_fip), 2),
                'age': float(age) if pd.notna(age) else None,
                'player_type': 'pitcher',
                'name': str(name),
            }

    print(f"\nMLE Lookup built from {stats_year} AAA stats:")
    print(f"  Hitters: {sum(1 for v in lookup.values() if v['player_type'] == 'hitter')}")
    print(f"  Pitchers: {sum(1 for v in lookup.values() if v['player_type'] == 'pitcher')}")

    # Show WAR distribution
    wars = [v['mle_war'] for v in lookup.values()]
    if wars:
        print(f"  WAR range: [{min(wars):.2f}, {max(wars):.2f}]")
        print(f"  WAR median: {np.median(wars):.2f}")

    return lookup


def match_mle_to_roster(
    mle_lookup: Dict[str, dict],
    roster_df: pd.DataFrame,
    crosswalk: Dict[str, int],
) -> Dict[int, dict]:
    """
    Convert FG PlayerId-keyed MLE lookup to MLBAM ID-keyed lookup,
    filtering to only players on the roster.

    Uses the same crosswalk and name-matching fallback as FV prospect integration.

    Args:
        mle_lookup: Dict from build_mle_lookup(), keyed by FG PlayerId str.
        roster_df: Roster DataFrame with playerid (MLBAM), Name columns.
        crosswalk: Dict from build_fg_to_mlbam_crosswalk(), FG ID str -> MLBAM int.

    Returns:
        Dict mapping MLBAM ID (int) -> mle_info dict.
    """
    from new_pipeline.models.future_season.team_wins.fv_prospect_integrator import (
        _normalize_name,
    )

    if not mle_lookup or roster_df is None or roster_df.empty:
        return {}

    # Build set of MLBAM IDs on roster
    roster_mlbam_ids = set()
    for _, row in roster_df.iterrows():
        pid = row.get('playerid')
        if pd.notna(pid):
            roster_mlbam_ids.add(int(pid))

    # Build name -> roster players lookup for fallback
    roster_by_name: Dict[str, list] = {}
    for _, row in roster_df.iterrows():
        pid = row.get('playerid')
        if pd.isna(pid):
            continue
        name_norm = _normalize_name(str(row.get('Name', '')))
        if name_norm:
            if name_norm not in roster_by_name:
                roster_by_name[name_norm] = []
            roster_by_name[name_norm].append(int(pid))

    result: Dict[int, dict] = {}
    matched_crosswalk = 0
    matched_name = 0

    for fg_id, mle_info in mle_lookup.items():
        matched_mlbam = None

        # Tier 1: Crosswalk
        if fg_id in crosswalk:
            candidate = crosswalk[fg_id]
            if candidate in roster_mlbam_ids:
                matched_mlbam = candidate
                matched_crosswalk += 1

        # Tier 2: Name match
        if matched_mlbam is None:
            name_norm = _normalize_name(mle_info.get('name', ''))
            if name_norm and name_norm in roster_by_name:
                candidates = roster_by_name[name_norm]
                if len(candidates) == 1:
                    matched_mlbam = candidates[0]
                    matched_name += 1

        if matched_mlbam is not None and matched_mlbam not in result:
            result[matched_mlbam] = mle_info

    print(f"\nMLE roster matching:")
    print(f"  Matched via crosswalk: {matched_crosswalk}")
    print(f"  Matched via name: {matched_name}")
    print(f"  Total MLE players on roster: {len(result)}")

    return result
