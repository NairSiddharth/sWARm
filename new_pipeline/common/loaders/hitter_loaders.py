"""
Hitter Feature Loaders - Clean Implementation

Each loader is declarative and focuses on:
1. What to load (file type, column name)
2. How to convert (decimal→percentage or keep raw)
3. What to validate (expected range)

All file handling, error handling, and repetitive logic is in helpers._load_fangraphs_feature()

Active Features:
1. K% - Strikeout percentage
2. BB% - Walk percentage
3. AVG - Batting average (park-adjusted, 3yr factor)
4. OBP - On-base percentage (park-adjusted, 3yr factor)
5. SLG - Slugging percentage (park-adjusted, 3yr factor)
6. PA - Plate appearances
7. Positional_WAR - Position adjustment per 600 PA
8. GDP - Ground into double play count (for GDP_rate calculation in transformer)
9. Enhanced_Baserunning - SB + XBT + sprint speed composite (3yr weighted)
10. Enhanced_Defense - Fielding + position-specific metrics composite (3yr weighted)

See: hitter_feature_pipeline_design.md for specifications
"""

from typing import Dict, List
from pathlib import Path
import pandas as pd
from .helpers import (
    _load_fangraphs_feature,
    _load_park_adjusted_fangraphs_feature,
    _convert_decimal_to_percentage,
    validate_percentage_scale
)
from ..constants import DEFENSIVE_DIR, BP_BASERUNNING_DIR, STATCAST_RUNNING_SPLITS_DIR

# Position adjustments (per 600 PA)
POSITION_WAR_ADJUSTMENTS = {
    'C': +1.25,
    'SS': +0.75,
    '2B': +0.30,
    '3B': +0.20,
    'CF': +0.25,
    'LF': -0.70,
    'RF': -0.75,
    '1B': -1.25,
    'DH': -1.50
}


def load_k_pct_all_years(years: List[int]) -> Dict[int, float]:
    """
    Load K% (strikeout percentage) for hitters.

    FanGraphs stores as decimal (0.232 = 23.2%), we convert to percentage.

    Args:
        years: Years to load

    Returns:
        dict: {MLBAMID: K% in percentage format}
    """
    # Load raw data (decimals) - K% is in advanced file for hitters
    k_raw = _load_fangraphs_feature(years, 'advanced', 'K%', player_type='hitters')

    # Convert decimal → percentage
    k_pct = {pid: _convert_decimal_to_percentage(val) for pid, val in k_raw.items()}

    # Validate
    if k_pct:
        validate_percentage_scale(k_pct, 'K% (Hitters)', expected_range=(0, 100))

    return k_pct


def load_bb_pct_all_years(years: List[int]) -> Dict[int, float]:
    """
    Load BB% (walk percentage) for hitters.

    FanGraphs stores as decimal (0.105 = 10.5%), we convert to percentage.

    Args:
        years: Years to load

    Returns:
        dict: {MLBAMID: BB% in percentage format}
    """
    # Load raw data (decimals) - BB% is in advanced file for hitters
    bb_raw = _load_fangraphs_feature(years, 'advanced', 'BB%', player_type='hitters')

    # Convert decimal → percentage
    bb_pct = {pid: _convert_decimal_to_percentage(val) for pid, val in bb_raw.items()}

    # Validate
    if bb_pct:
        validate_percentage_scale(bb_pct, 'BB% (Hitters)', expected_range=(0, 100))

    return bb_pct


def load_pa_all_years(years: List[int]) -> Dict[int, int]:
    """
    Load PA (plate appearances).

    This is a COUNT, not a percentage. No conversion needed.

    Args:
        years: Years to load

    Returns:
        dict: {MLBAMID: PA count}
    """
    # Load raw data (already correct scale - it's a count)
    pa_data = _load_fangraphs_feature(years, 'standard', 'PA', player_type='hitters')

    # Convert to int
    pa_int = {pid: int(val) for pid, val in pa_data.items()}

    return pa_int


def load_gdp_all_years(years: List[int]) -> Dict[int, int]:
    """
    Load GDP (ground into double play count).

    This is a COUNT, not a percentage. GDP_rate calculated in transformer.

    Args:
        years: Years to load

    Returns:
        dict: {MLBAMID: GDP count}
    """
    # Load raw data (already correct scale - it's a count)
    gdp_data = _load_fangraphs_feature(years, 'standard', 'GDP', player_type='hitters')

    # Convert to int
    gdp_int = {pid: int(val) for pid, val in gdp_data.items()}

    return gdp_int


# ============================================================================
# Park-Adjusted Loaders
# ============================================================================


def load_avg_park_adjusted(years: List[int]) -> Dict[int, float]:
    """
    Load AVG with 3-year park factor adjustment.

    FanGraphs stores AVG as decimal (0.285 = .285 batting average).
    We keep it in decimal format (not percentage).
    Park adjustment uses 3yr factor (already halved by FanGraphs).

    Args:
        years: Years to load

    Returns:
        dict: {MLBAMID: park-adjusted AVG in decimal format}

    Example:
        Coors hitter with raw AVG .300, park factor 108:
        - Adjusted AVG = .300 * (100/108) = .278
    """
    # Load AVG with park adjustment (no conversion - AVG is already decimal)
    # AVG is in advanced file for hitters
    avg_adjusted = _load_park_adjusted_fangraphs_feature(
        years, 'advanced', 'AVG', '3yr', player_type='hitters'
    )

    return avg_adjusted


def load_obp_park_adjusted(years: List[int]) -> Dict[int, float]:
    """
    Load OBP with 3-year park factor adjustment.

    FanGraphs stores OBP as decimal (0.355 = .355 OBP).
    We keep it in decimal format (not percentage).
    Park adjustment uses 3yr factor.

    Args:
        years: Years to load

    Returns:
        dict: {MLBAMID: park-adjusted OBP in decimal format}
    """
    # Load OBP with park adjustment (no conversion - OBP is already decimal)
    # OBP is in advanced file for hitters
    obp_adjusted = _load_park_adjusted_fangraphs_feature(
        years, 'advanced', 'OBP', '3yr', player_type='hitters'
    )

    return obp_adjusted


def load_slg_park_adjusted(years: List[int]) -> Dict[int, float]:
    """
    Load SLG with 3-year park factor adjustment.

    FanGraphs stores SLG as decimal (0.485 = .485 SLG).
    We keep it in decimal format (not percentage).
    Park adjustment uses 3yr factor.

    Args:
        years: Years to load

    Returns:
        dict: {MLBAMID: park-adjusted SLG in decimal format}
    """
    # Load SLG with park adjustment (no conversion - SLG is already decimal)
    # SLG is in advanced file for hitters
    slg_adjusted = _load_park_adjusted_fangraphs_feature(
        years, 'advanced', 'SLG', '3yr', player_type='hitters'
    )

    return slg_adjusted


# ============================================================================
# Positional Adjustment Loader
# ============================================================================


def load_positional_war(years: List[int]) -> Dict[int, float]:
    """
    Load positional WAR adjustments per 600 PA.

    Maps each player's primary position to WAR adjustment value.

    Source: FanGraphs_Data/defensive/fangraphs_defensive_advanced_{year}.csv
    Column: 'Pos' (primary position) - cross-referenced via MLBAMID

    Note: Position data is in defensive files, not offensive files

    Adjustments (per 600 PA):
    - C: +1.25 (hardest position)
    - SS: +0.75
    - 2B: +0.30
    - 3B: +0.20
    - CF: +0.25
    - LF: -0.70
    - RF: -0.75
    - 1B: -1.25
    - DH: -1.50 (easiest position)

    Args:
        years: Years to load

    Returns:
        dict: {MLBAMID: positional_war_adjustment}

    Example:
        Catcher with 500 PA:
        - Base adjustment: +1.25 WAR per 600 PA
        - Prorated: +1.25 * (500/600) = +1.04 WAR
    """
    pos_war_dict = {}

    for year in years:
        csv_path = DEFENSIVE_DIR / f"fangraphs_defensive_advanced_{year}.csv"

        if not csv_path.exists():
            continue

        try:
            df = pd.read_csv(csv_path)

            # Find player ID column
            id_col = None
            for col in ['MLBAMID', 'PlayerId', 'playerid']:
                if col in df.columns:
                    id_col = col
                    break

            if id_col is None or 'Pos' not in df.columns:
                continue

            # Extract position adjustments
            for _, row in df.iterrows():
                if pd.notna(row[id_col]) and pd.notna(row['Pos']):
                    mlbamid = int(row[id_col])
                    position = str(row['Pos']).strip()

                    # Handle multi-position players (e.g., "2B/SS" → use first)
                    primary_pos = position.split('/')[0]

                    # Map to adjustment value
                    adjustment = POSITION_WAR_ADJUSTMENTS.get(primary_pos, 0.0)
                    pos_war_dict[mlbamid] = adjustment

        except Exception as e:
            continue

    return pos_war_dict


# ============================================================================
# Enhanced Multi-Source Features
# ============================================================================


def load_enhanced_baserunning(years: List[int]) -> Dict[int, float]:
    """
    Calculate Enhanced_Baserunning composite from multi-source data.

    NEW IMPLEMENTATION (updated weights and yearly baselines):

    Sources:
    - BP baserunning: SB, CS, PO, XBT%
    - Statcast running splits: seconds_since_hit_090 (sprint speed)

    Formula:
        steal_runs = (SB × 0.25) - (CS × 0.50) - (PO × 0.50)
        xbt_runs = (player_xbt - yearly_median_xbt) × 10
        speed_value = (player_speed - yearly_median_speed) × 0.5
        total = steal_runs + xbt_runs + speed_value

    Key changes from old implementation:
    - SB: 0.20 → 0.25 (captures threat/strategic value)
    - CS: 0.40 → 0.50 (research-informed from run expectancy)
    - PO: NEW (-0.50, equal to CS)
    - XBT/Speed: Use yearly median baseline (not hardcoded)

    3-year weighted average: 50% recent, 30% year-1, 20% year-2
    Capped to [-7, 10] range

    Args:
        years: Years to load

    Returns:
        dict: {MLBAMID: Enhanced_Baserunning value}

    Example:
        Elite base stealer: 50 SB, 10 CS, 5 PO, 55% XBT, 29 ft/s
        Yearly medians: 40% XBT, 27 ft/s
        >>> steal_runs = (50×0.25) - (10×0.50) - (5×0.50) = 5.0
        >>> xbt_runs = (55 - 40) × 0.10 = 1.5
        >>> speed_value = (29 - 27) × 0.5 = 1.0
        >>> total = 7.5
    """
    from ..transformers.baserunning_helpers import (
        calculate_steal_runs,
        calculate_xbt_runs,
        calculate_speed_value,
        calculate_sprint_speed,
        apply_baserunning_cap,
        calculate_yearly_baselines
    )

    baserunning_dict = {}
    yearly_values = {}  # {mlbamid: {year: value}}

    for year in years:
        # Load BP baserunning data
        bp_path = BP_BASERUNNING_DIR / f"bp_baserunning_{year}.csv"

        if not bp_path.exists():
            continue

        try:
            bp_df = pd.read_csv(bp_path)

            # Load Statcast sprint speed (optional)
            statcast_df = None
            statcast_path = STATCAST_RUNNING_SPLITS_DIR / f"running_splits_statcast_{year}.csv"
            if statcast_path.exists():
                statcast_df = pd.read_csv(statcast_path)

            # Calculate yearly baselines (medians)
            baselines = calculate_yearly_baselines(bp_df, statcast_df)
            xbt_median = baselines['xbt_median']
            speed_median = baselines['speed_median']

            # Find ID column in BP data (BP uses 'mlbid' not 'MLBAMID')
            id_col = None
            for col in ['mlbid', 'MLBAMID', 'mlbam_id', 'player_id']:
                if col in bp_df.columns:
                    id_col = col
                    break

            if id_col is None:
                continue

            for _, row in bp_df.iterrows():
                if pd.notna(row[id_col]):
                    mlbamid = int(row[id_col])

                    # 1. Steal component (SB, CS, PO)
                    # Handle NaN values
                    sb = row.get('SB', 0)
                    cs = row.get('CS', 0)
                    po = row.get('PO', 0)

                    # Skip if all steal stats are missing (pitchers, etc.)
                    if pd.isna(sb) and pd.isna(cs) and pd.isna(po):
                        continue

                    sb = 0.0 if pd.isna(sb) else float(sb)
                    cs = 0.0 if pd.isna(cs) else float(cs)
                    po = 0.0 if pd.isna(po) else float(po)
                    steal_runs = calculate_steal_runs(sb, cs, po)

                    # 2. Extra base taking component (relative to yearly median)
                    xbt_pct = row.get('XBT%', xbt_median)
                    if pd.isna(xbt_pct):
                        xbt_pct = xbt_median  # Use median if missing
                    xbt_runs = calculate_xbt_runs(float(xbt_pct), xbt_median)

                    # 3. Sprint speed component (if available)
                    speed_value = 0.0
                    if statcast_df is not None and speed_median > 0:
                        player_statcast = statcast_df[statcast_df['player_id'] == mlbamid]
                        if not player_statcast.empty:
                            seconds = player_statcast.iloc[0].get('seconds_since_hit_090', 0)
                            if pd.notna(seconds) and float(seconds) > 0:
                                player_speed = calculate_sprint_speed(float(seconds))
                                speed_value = calculate_speed_value(player_speed, speed_median)

                    # Total baserunning value
                    total = steal_runs + xbt_runs + speed_value

                    # Apply cap
                    total = apply_baserunning_cap(total)

                    # Store yearly value
                    if mlbamid not in yearly_values:
                        yearly_values[mlbamid] = {}
                    yearly_values[mlbamid][year] = total

        except Exception as e:
            continue

    # Apply 3-year weighted average: 50% recent, 30% year-1, 20% year-2
    if len(years) >= 3:
        sorted_years = sorted(years, reverse=True)
        most_recent = sorted_years[0]

        for mlbamid, year_data in yearly_values.items():
            if most_recent in year_data:
                weighted_sum = year_data[most_recent] * 0.5

                # Add year-1 if exists
                if sorted_years[1] in year_data:
                    weighted_sum += year_data[sorted_years[1]] * 0.3

                # Add year-2 if exists
                if sorted_years[2] in year_data:
                    weighted_sum += year_data[sorted_years[2]] * 0.2

                baserunning_dict[mlbamid] = weighted_sum
            else:
                # Use most recent available
                baserunning_dict[mlbamid] = year_data[max(year_data.keys())]
    else:
        # Use most recent year if < 3 years
        for mlbamid, year_data in yearly_values.items():
            baserunning_dict[mlbamid] = year_data[max(year_data.keys())]

    return baserunning_dict


def load_enhanced_defense(years: List[int]) -> Dict[int, float]:
    """
    Load Enhanced_Defense composite using Range Factor baseline + position bonuses.

    NEW IMPLEMENTATION (redesigned from old fielding% approach):

    Methodology:
    1. Baseline: Position-relative Range Factor
       - RF = 9 × (A + PO) / Inn
       - Compare to position average
       - Scale to run value

    2. Position-specific bonuses:
       - 2B/SS: Weighted DP value (DPS×0.9 + DPT×1.0 + DPF×0.3)
       - 3B: DP starts (DPS×0.9)
       - 1B: Scoops (×0.6) + DP finishes (DPF×0.3)
       - C: Framing (60%) + Throwing (25%) + Blocking (15%)
       - OF: [Future: catch probability metrics]

    3. Position-specific caps (asymmetric):
       - SS/C: ±30/±25 (hardest positions, widest skill range)
       - 1B: +15/-8 (easiest position, elite can offset penalty to ~0)
       - Corner OF: +18/-10 (CF-caliber tools playing corners)

    4. 3-year weighted average: 50% recent, 30% year-1, 20% year-2

    Data Sources:
    - FanGraphs defensive_standard: PO, A, Inn, DPS, DPT, DPF, Scp
    - FanGraphs defensive_statcast: Framing, Throwing, Blocking (catchers)

    Args:
        years: Years to load

    Returns:
        dict: {MLBAMID: Enhanced_Defense runs}

    Example:
        Elite SS (Simmons-level): +30 runs (range + DPs)
        Elite 1B (Goldschmidt): +13 runs (scoops + DPF) → offsets -12.5 penalty to ~0
        Poor corner OF: -10 runs
    """
    from ..transformers.defensive_helpers import (
        calculate_range_factor,
        calculate_position_relative_rf,
        calculate_infielder_dp_value,
        calculate_first_base_scoop_value,
        calculate_catcher_metrics_value,
        apply_defensive_cap,
        POSITION_AVG_RF
    )

    defense_dict = {}
    yearly_values = {}

    for year in years:
        # Load defensive standard file (PO, A, Inn, DPS, DPT, DPF, Scp)
        def_standard_path = DEFENSIVE_DIR / f"fangraphs_defensive_standard_{year}.csv"

        # Load defensive statcast file (Framing, Throwing, Blocking for catchers)
        def_statcast_path = DEFENSIVE_DIR / f"fangraphs_defensive_statcast_{year}.csv"

        if not def_standard_path.exists():
            continue

        try:
            df_standard = pd.read_csv(def_standard_path)

            # Load statcast data for catchers (if available)
            catcher_statcast = {}
            if def_statcast_path.exists():
                df_statcast = pd.read_csv(def_statcast_path)
                for _, row in df_statcast.iterrows():
                    if pd.notna(row.get('MLBAMID')) and row.get('Pos') == 'C':
                        try:
                            mlbamid = int(float(row['MLBAMID']))
                            catcher_statcast[mlbamid] = {
                                'Framing': float(row.get('Framing', 0.0)),
                                'Throwing': float(row.get('Throwing', 0.0)),
                                'Blocking': float(row.get('Blocking', 0.0))
                            }
                        except (ValueError, TypeError):
                            continue

            # Process each player
            for _, row in df_standard.iterrows():
                if pd.notna(row.get('MLBAMID')):
                    try:
                        mlbamid = int(float(row['MLBAMID']))
                        position = row.get('Pos', '')

                        if not position or pd.isna(position) or position == 'DH':
                            continue

                        # Get counting stats
                        assists = float(row.get('A', 0.0))
                        putouts = float(row.get('PO', 0.0))
                        innings = float(row.get('Inn', 0.0))

                        if innings < 50:  # Minimum sample size
                            continue

                        # 1. Calculate Range Factor baseline
                        player_rf = calculate_range_factor(assists, putouts, innings)
                        position_avg = POSITION_AVG_RF.get(position, 3.0)
                        baseline_runs = calculate_position_relative_rf(player_rf, position_avg)

                        # 2. Calculate position-specific bonuses
                        position_bonus = 0.0

                        if position in ['2B', 'SS', '3B', '1B']:
                            # Infielders: DP value
                            dps = int(row.get('DPS', 0))
                            dpt = int(row.get('DPT', 0))
                            dpf = int(row.get('DPF', 0))
                            position_bonus = calculate_infielder_dp_value(dps, dpt, dpf, position)

                            # First basemen also get scoop value
                            if position == '1B':
                                scoops = int(row.get('Scp', 0))
                                position_bonus += calculate_first_base_scoop_value(scoops)

                        elif position == 'C':
                            # Catchers: Weighted framing/throwing/blocking
                            if mlbamid in catcher_statcast:
                                metrics = catcher_statcast[mlbamid]
                                position_bonus = calculate_catcher_metrics_value(
                                    metrics['Framing'],
                                    metrics['Throwing'],
                                    metrics['Blocking']
                                )

                        # elif position in ['LF', 'CF', 'RF']:
                        #     # Outfielders: Future implementation with catch probability
                        #     # For now, baseline RF is sufficient
                        #     pass

                        # 3. Combine baseline + bonus
                        total_runs = baseline_runs + position_bonus

                        # 4. Apply position-specific cap
                        capped_runs = apply_defensive_cap(total_runs, position)

                        # Store yearly value
                        if mlbamid not in yearly_values:
                            yearly_values[mlbamid] = {}
                        yearly_values[mlbamid][year] = capped_runs

                    except (ValueError, TypeError, KeyError):
                        continue

        except Exception as e:
            continue

    # Apply 3-year weighted average: 50% recent, 30% year-1, 20% year-2
    if len(years) >= 3:
        sorted_years = sorted(years, reverse=True)
        most_recent = sorted_years[0]

        for mlbamid, year_data in yearly_values.items():
            if most_recent in year_data:
                weighted_sum = year_data[most_recent] * 0.5

                # Add year-1 if exists
                if sorted_years[1] in year_data:
                    weighted_sum += year_data[sorted_years[1]] * 0.3

                # Add year-2 if exists
                if sorted_years[2] in year_data:
                    weighted_sum += year_data[sorted_years[2]] * 0.2

                defense_dict[mlbamid] = weighted_sum
            else:
                # Use most recent available
                defense_dict[mlbamid] = year_data[max(year_data.keys())]
    else:
        # Use most recent year if < 3 years
        for mlbamid, year_data in yearly_values.items():
            defense_dict[mlbamid] = year_data[max(year_data.keys())]

    return defense_dict
