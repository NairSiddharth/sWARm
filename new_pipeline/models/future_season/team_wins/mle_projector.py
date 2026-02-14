"""
Minor League Equivalency (MLE) Projector.

Translates minor league stats (AA/AAA) into MLB-equivalent WAR for players
who lack MLB projections and FV prospect grades. Uses historical level-to-MLB
translation factors computed from players who appeared at both levels.

For players who split time across multiple levels in the same season,
stats are translated independently at each level then blended weighted
by PA/IP.

Architecture:
    MiLB wRC+ (hitters) / FIP (pitchers) at each level
        -> apply level-specific translation factor (multi-year median ratio)
        -> MLB-equivalent wRC+ / FIP per level
        -> PA/IP-weighted blend across levels
        -> convert blended MLB-equiv stat to WAR via linear fit
        -> age adjustment
        -> single WAR estimate for team_war_aggregator
"""

from typing import Dict, List, Optional

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

# Levels supported for MLE translation
MLE_LEVELS = ['AA', 'AAA']


def _load_milb_hitter_stats(year: int, level: str) -> Optional[pd.DataFrame]:
    """
    Load minor league hitter stats for a given year and level by merging
    standard and advanced CSVs.

    Args:
        year: Season year.
        level: Minor league level (e.g., 'AA', 'AAA').

    Returns:
        DataFrame with columns: PlayerId, Name, Age, PA, wRC+
        or None if files don't exist.
    """
    milb_dir = MILB_DATA_DIR / level / "hitters"
    standard_path = milb_dir / f"fangraphs_hitters_{year}_standard_{level}.csv"
    advanced_path = milb_dir / f"fangraphs_hitters_{year}_advanced_{level}.csv"

    if not standard_path.exists() or not advanced_path.exists():
        return None

    standard = pd.read_csv(standard_path, encoding='utf-8-sig')
    advanced = pd.read_csv(advanced_path, encoding='utf-8-sig')

    merged = standard[['PlayerId', 'Name', 'Age', 'PA']].merge(
        advanced[['PlayerId', 'wRC+']],
        on='PlayerId',
        how='inner',
    )
    return merged


def _load_milb_pitcher_stats(year: int, level: str) -> Optional[pd.DataFrame]:
    """
    Load minor league pitcher stats for a given year and level by merging
    standard and advanced CSVs.

    Args:
        year: Season year.
        level: Minor league level (e.g., 'AA', 'AAA').

    Returns:
        DataFrame with columns: PlayerId, Name, Age, IP, FIP
        or None if files don't exist.
    """
    milb_dir = MILB_DATA_DIR / level / "pitchers"
    standard_path = milb_dir / f"fangraphs_pitchers_{year}_standard_{level}.csv"
    advanced_path = milb_dir / f"fangraphs_pitchers_{year}_advanced_{level}.csv"

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

    Returns DataFrame with columns: PlayerId, PA, wRC+, WAR
    or None if file doesn't exist.
    """
    path = FANGRAPHS_HITTER_DIR / f"fangraphs_hitters_{year}.csv"
    if not path.exists():
        return None

    df = pd.read_csv(path, encoding='utf-8-sig')
    cols_needed = ['PlayerId', 'PA', 'wRC+', 'WAR']
    missing = [c for c in cols_needed if c not in df.columns]
    if missing:
        return None

    return df[cols_needed].copy()


def _load_mlb_pitcher_stats(year: int) -> Optional[pd.DataFrame]:
    """
    Load MLB pitcher stats for a given year.

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
    Build MiLB-to-MLB translation factors from historical paired data.

    For each level (AA, AAA) and each year, finds players who appeared at
    that level and in MLB with meaningful sample sizes. Computes per-player
    ratios of MLB stat to MiLB stat, then takes the median across all
    player-years for each level.

    Args:
        years: Years to include. Default: 2016-2025 excluding 2020.

    Returns:
        Dict with keys:
            'aaa_hitter_wrc_ratio': median(MLB_wRC+ / AAA_wRC+)
            'aaa_pitcher_fip_ratio': median(MLB_FIP / AAA_FIP)
            'aa_hitter_wrc_ratio': median(MLB_wRC+ / AA_wRC+)
            'aa_pitcher_fip_ratio': median(MLB_FIP / AA_FIP)
            'aaa_hitter_pairs': int
            'aaa_pitcher_pairs': int
            'aa_hitter_pairs': int
            'aa_pitcher_pairs': int
    """
    if years is None:
        years = [y for y in range(2016, 2026) if y != 2020]

    # Collect ratios per level
    hitter_ratios = {level: [] for level in MLE_LEVELS}
    pitcher_ratios = {level: [] for level in MLE_LEVELS}

    for year in years:
        mlb_h = _load_mlb_hitter_stats(year)
        mlb_p = _load_mlb_pitcher_stats(year)

        for level in MLE_LEVELS:
            # Hitters
            milb_h = _load_milb_hitter_stats(year, level)
            if milb_h is not None and mlb_h is not None:
                milb_h['PlayerId'] = milb_h['PlayerId'].astype(str)
                mlb_h_copy = mlb_h.copy()
                mlb_h_copy['PlayerId'] = mlb_h_copy['PlayerId'].astype(str)

                paired = milb_h.merge(
                    mlb_h_copy, on='PlayerId', suffixes=('_milb', '_mlb')
                )
                paired = paired[
                    (paired['PA_milb'] >= MLE_MIN_PA) &
                    (paired['PA_mlb'] >= MLE_MIN_PA)
                ]
                for _, row in paired.iterrows():
                    milb_wrc = row['wRC+_milb']
                    mlb_wrc = row['wRC+_mlb']
                    if pd.notna(milb_wrc) and pd.notna(mlb_wrc) and milb_wrc > 0:
                        hitter_ratios[level].append(mlb_wrc / milb_wrc)

            # Pitchers
            milb_p = _load_milb_pitcher_stats(year, level)
            if milb_p is not None and mlb_p is not None:
                milb_p['PlayerId'] = milb_p['PlayerId'].astype(str)
                mlb_p_copy = mlb_p.copy()
                mlb_p_copy['PlayerId'] = mlb_p_copy['PlayerId'].astype(str)

                paired = milb_p.merge(
                    mlb_p_copy, on='PlayerId', suffixes=('_milb', '_mlb')
                )
                paired = paired[
                    (paired['IP_milb'] >= MLE_MIN_IP) &
                    (paired['IP_mlb'] >= MLE_MIN_IP)
                ]
                for _, row in paired.iterrows():
                    milb_fip = row['FIP_milb']
                    mlb_fip = row['FIP_mlb']
                    if pd.notna(milb_fip) and pd.notna(mlb_fip) and milb_fip > 0:
                        pitcher_ratios[level].append(mlb_fip / milb_fip)

    # Compute median ratios per level with sensible fallbacks
    # AA ratio should be lower than AAA (further from MLB talent)
    aaa_hitter_wrc_ratio = float(np.median(hitter_ratios['AAA'])) if hitter_ratios['AAA'] else 0.69
    aaa_pitcher_fip_ratio = float(np.median(pitcher_ratios['AAA'])) if pitcher_ratios['AAA'] else 1.12
    aa_hitter_wrc_ratio = float(np.median(hitter_ratios['AA'])) if hitter_ratios['AA'] else 0.60
    aa_pitcher_fip_ratio = float(np.median(pitcher_ratios['AA'])) if pitcher_ratios['AA'] else 1.20

    print("\nMLE Translation Model:")
    print(f"  AAA hitter wRC+ ratio (AAA->MLB): {aaa_hitter_wrc_ratio:.3f} "
          f"({len(hitter_ratios['AAA'])} player-years)")
    print(f"  AAA pitcher FIP ratio (AAA->MLB): {aaa_pitcher_fip_ratio:.3f} "
          f"({len(pitcher_ratios['AAA'])} player-years)")
    print(f"  AA hitter wRC+ ratio  (AA->MLB):  {aa_hitter_wrc_ratio:.3f} "
          f"({len(hitter_ratios['AA'])} player-years)")
    print(f"  AA pitcher FIP ratio  (AA->MLB):  {aa_pitcher_fip_ratio:.3f} "
          f"({len(pitcher_ratios['AA'])} player-years)")

    return {
        'aaa_hitter_wrc_ratio': aaa_hitter_wrc_ratio,
        'aaa_pitcher_fip_ratio': aaa_pitcher_fip_ratio,
        'aa_hitter_wrc_ratio': aa_hitter_wrc_ratio,
        'aa_pitcher_fip_ratio': aa_pitcher_fip_ratio,
        'aaa_hitter_pairs': len(hitter_ratios['AAA']),
        'aaa_pitcher_pairs': len(pitcher_ratios['AAA']),
        'aa_hitter_pairs': len(hitter_ratios['AA']),
        'aa_pitcher_pairs': len(pitcher_ratios['AA']),
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

    Young for level (under 25): +0.2 WAR (upside)
    Prime (25-28): no adjustment
    Old for level (29+): -0.2 WAR (AAAA ceiling)
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
    Build MLE WAR lookup for minor league players from the most recent season.

    Loads AA and AAA stats from projection_year - 1, translates each level
    independently to MLB-equivalent stats, then blends across levels weighted
    by PA/IP for players who split time.

    Args:
        projection_year: The season being projected (e.g., 2026).
        translation_factors: Output of build_translation_model().

    Returns:
        Dict mapping FG PlayerId (str) -> {
            'mle_war': float,
            'translated_stat': float,  # MLB-equiv wRC+ or FIP (blended)
            'raw_stat': float,         # PA/IP-weighted raw stat across levels
            'age': float,
            'player_type': str,        # 'hitter' or 'pitcher'
            'name': str,
            'levels': str,             # e.g. 'AA', 'AAA', or 'AA+AAA'
        }
    """
    stats_year = projection_year - 1
    aaa_hitter_ratio = translation_factors['aaa_hitter_wrc_ratio']
    aa_hitter_ratio = translation_factors['aa_hitter_wrc_ratio']
    aaa_pitcher_ratio = translation_factors['aaa_pitcher_fip_ratio']
    aa_pitcher_ratio = translation_factors['aa_pitcher_fip_ratio']

    lookup: Dict[str, dict] = {}

    # --- Hitters ---
    # Load both levels
    aa_h = _load_milb_hitter_stats(stats_year, 'AA')
    aaa_h = _load_milb_hitter_stats(stats_year, 'AAA')

    # Build per-player data by level: {player_id: {'AA': row, 'AAA': row}}
    hitter_by_player: Dict[str, Dict[str, pd.Series]] = {}

    if aa_h is not None:
        aa_h['PlayerId'] = aa_h['PlayerId'].astype(str).str.strip()
        for _, row in aa_h.iterrows():
            fg_id = row['PlayerId']
            if fg_id not in hitter_by_player:
                hitter_by_player[fg_id] = {}
            hitter_by_player[fg_id]['AA'] = row

    if aaa_h is not None:
        aaa_h['PlayerId'] = aaa_h['PlayerId'].astype(str).str.strip()
        for _, row in aaa_h.iterrows():
            fg_id = row['PlayerId']
            if fg_id not in hitter_by_player:
                hitter_by_player[fg_id] = {}
            hitter_by_player[fg_id]['AAA'] = row

    multi_level_hitters = 0
    for fg_id, level_data in hitter_by_player.items():
        # Gather PA and wRC+ at each level
        entries = []
        for level, row in level_data.items():
            wrc = row.get('wRC+')
            pa = row.get('PA', 0)
            if pd.notna(wrc) and pa > 0:
                ratio = aaa_hitter_ratio if level == 'AAA' else aa_hitter_ratio
                entries.append({
                    'level': level,
                    'pa': pa,
                    'wrc': wrc,
                    'mlb_equiv_wrc': wrc * ratio,
                    'age': row.get('Age', np.nan),
                    'name': row.get('Name', ''),
                })

        if not entries:
            continue

        total_pa = sum(e['pa'] for e in entries)
        if total_pa < MLE_MIN_PA:
            continue

        # PA-weighted blend of MLB-equivalent wRC+
        blended_mlb_wrc = sum(e['pa'] * e['mlb_equiv_wrc'] for e in entries) / total_pa
        blended_raw_wrc = sum(e['pa'] * e['wrc'] for e in entries) / total_pa

        # Use age from highest level (AAA if available)
        if 'AAA' in level_data:
            age = level_data['AAA'].get('Age', np.nan)
            name = level_data['AAA'].get('Name', '')
        else:
            age = entries[0]['age']
            name = entries[0]['name']

        levels_str = '+'.join(sorted(level_data.keys()))
        if len(entries) > 1:
            multi_level_hitters += 1

        war = _wrc_to_war(blended_mlb_wrc)
        war = _apply_age_adjustment(war, age)
        war = max(MLE_WAR_FLOOR, min(MLE_WAR_CAP, war))

        lookup[fg_id] = {
            'mle_war': round(war, 2),
            'translated_stat': round(blended_mlb_wrc, 1),
            'raw_stat': round(float(blended_raw_wrc), 1),
            'age': float(age) if pd.notna(age) else None,
            'player_type': 'hitter',
            'name': str(name),
            'levels': levels_str,
        }

    # --- Pitchers ---
    aa_p = _load_milb_pitcher_stats(stats_year, 'AA')
    aaa_p = _load_milb_pitcher_stats(stats_year, 'AAA')

    pitcher_by_player: Dict[str, Dict[str, pd.Series]] = {}

    if aa_p is not None:
        aa_p['PlayerId'] = aa_p['PlayerId'].astype(str).str.strip()
        for _, row in aa_p.iterrows():
            fg_id = row['PlayerId']
            if fg_id not in pitcher_by_player:
                pitcher_by_player[fg_id] = {}
            pitcher_by_player[fg_id]['AA'] = row

    if aaa_p is not None:
        aaa_p['PlayerId'] = aaa_p['PlayerId'].astype(str).str.strip()
        for _, row in aaa_p.iterrows():
            fg_id = row['PlayerId']
            if fg_id not in pitcher_by_player:
                pitcher_by_player[fg_id] = {}
            pitcher_by_player[fg_id]['AAA'] = row

    multi_level_pitchers = 0
    for fg_id, level_data in pitcher_by_player.items():
        # Don't overwrite a hitter entry
        if fg_id in lookup:
            continue

        entries = []
        for level, row in level_data.items():
            fip = row.get('FIP')
            ip = row.get('IP', 0)
            if pd.notna(fip) and ip > 0:
                ratio = aaa_pitcher_ratio if level == 'AAA' else aa_pitcher_ratio
                entries.append({
                    'level': level,
                    'ip': ip,
                    'fip': fip,
                    'mlb_equiv_fip': fip * ratio,
                    'age': row.get('Age', np.nan),
                    'name': row.get('Name', ''),
                })

        if not entries:
            continue

        total_ip = sum(e['ip'] for e in entries)
        if total_ip < MLE_MIN_IP:
            continue

        # IP-weighted blend of MLB-equivalent FIP
        blended_mlb_fip = sum(e['ip'] * e['mlb_equiv_fip'] for e in entries) / total_ip
        blended_raw_fip = sum(e['ip'] * e['fip'] for e in entries) / total_ip

        if 'AAA' in level_data:
            age = level_data['AAA'].get('Age', np.nan)
            name = level_data['AAA'].get('Name', '')
        else:
            age = entries[0]['age']
            name = entries[0]['name']

        levels_str = '+'.join(sorted(level_data.keys()))
        if len(entries) > 1:
            multi_level_pitchers += 1

        war = _fip_to_war(blended_mlb_fip)
        war = _apply_age_adjustment(war, age)
        war = max(MLE_WAR_FLOOR, min(MLE_WAR_CAP, war))

        lookup[fg_id] = {
            'mle_war': round(war, 2),
            'translated_stat': round(blended_mlb_fip, 2),
            'raw_stat': round(float(blended_raw_fip), 2),
            'age': float(age) if pd.notna(age) else None,
            'player_type': 'pitcher',
            'name': str(name),
            'levels': levels_str,
        }

    n_hitters = sum(1 for v in lookup.values() if v['player_type'] == 'hitter')
    n_pitchers = sum(1 for v in lookup.values() if v['player_type'] == 'pitcher')

    print(f"\nMLE Lookup built from {stats_year} AA+AAA stats:")
    print(f"  Hitters: {n_hitters} ({multi_level_hitters} multi-level blends)")
    print(f"  Pitchers: {n_pitchers} ({multi_level_pitchers} multi-level blends)")

    # Show WAR distribution
    wars = [v['mle_war'] for v in lookup.values()]
    if wars:
        print(f"  WAR range: [{min(wars):.2f}, {max(wars):.2f}]")
        print(f"  WAR median: {np.median(wars):.2f}")

    # Print a few multi-level blend examples
    multi_examples = [
        (k, v) for k, v in lookup.items()
        if '+' in v.get('levels', '')
    ][:3]
    if multi_examples:
        print("\n  Multi-level blend examples:")
        for _fg_id, info in multi_examples:
            print(f"    {info['name']} ({info['levels']}): "
                  f"raw={info['raw_stat']}, mlb_equiv={info['translated_stat']}, "
                  f"MLE WAR={info['mle_war']}")

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

    print("\nMLE roster matching:")
    print(f"  Matched via crosswalk: {matched_crosswalk}")
    print(f"  Matched via name: {matched_name}")
    print(f"  Total MLE players on roster: {len(result)}")

    return result
