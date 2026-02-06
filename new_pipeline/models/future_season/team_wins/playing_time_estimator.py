"""
Playing Time Estimator - Budget-based PA/IP allocation.

Distributes a fixed team-level PA/IP budget across rostered players
based on role, positional competition, projected quality (WAR), and age.

The core insight: WAR projections from the existing pipeline are
conditional on playing a full season. To estimate team wins, we need
to know how much of a full season each player will actually play.
This module converts rate-based WAR into playing-time-adjusted WAR
by allocating realistic PA/IP to each roster spot.
"""

import pandas as pd

from new_pipeline.models.future_season.team_wins.constants import (
    TEAM_PA_BUDGET, ROTATION_IP_BUDGET, BULLPEN_IP_BUDGET,
    STARTER_PA_HIGH, STARTER_PA_LOW, NUM_LINEUP_SPOTS,
    MAX_HITTER_PA, MAX_BENCH_PA,
    MIN_STARTER_IP, MAX_STARTER_IP, CLOSER_BASE_IP,
    MAX_RELIEVER_IP, MIN_RELIEVER_IP,
    AGE_PA_PENALTY_START, AGE_PA_PENALTY_PER_YEAR,
    AGE_IP_PENALTY_START, AGE_IP_PENALTY_PER_YEAR,
    REPLACEMENT_ROLES
)


def allocate_team_playing_time(roster_df: pd.DataFrame) -> pd.DataFrame:
    """
    Allocate playing time for all teams in a roster DataFrame.

    Processes each team independently, distributing PA/IP budgets
    across hitters and pitchers.

    Args:
        roster_df: Full roster DataFrame with columns:
            Team, playerid, Name, role, position, player_type,
            rate_war, age, base_pa_ip

    Returns:
        roster_df with new columns: allocated_pa_ip, playing_time_factor
    """
    result_frames = []

    for team in roster_df['Team'].unique():
        team_data = roster_df[roster_df['Team'] == team].copy()

        # Split hitters and pitchers
        hitters = team_data[team_data['player_type'] == 'hitter'].copy()
        pitchers = team_data[team_data['player_type'] == 'pitcher'].copy()

        # Allocate PA for hitters
        if len(hitters) > 0:
            hitters = allocate_hitter_pa(hitters)

        # Allocate IP for pitchers
        if len(pitchers) > 0:
            pitchers = allocate_pitcher_ip(pitchers)

        result_frames.append(hitters)
        result_frames.append(pitchers)

    result = pd.concat(result_frames, ignore_index=True)

    # Calculate playing time factors using base-year PA/IP as denominator
    # This compares allocated playing time to what the model implicitly
    # expected based on the player's historical usage pattern
    result['playing_time_factor'] = result.apply(
        lambda row: calculate_playing_time_factor(
            row['allocated_pa_ip'], row.get('base_pa_ip', 0), row['role']
        ),
        axis=1
    )

    return result


def allocate_hitter_pa(
    hitters_df: pd.DataFrame,
    team_pa_budget: float = TEAM_PA_BUDGET
) -> pd.DataFrame:
    """
    Distribute PA budget across a team's hitters.

    Allocation strategy:
    1. Starters at each position get priority PA
    2. Positional competition splits PA by WAR ratio
    3. Quality-weighted: better hitters (higher WAR) get more PA
    4. Age penalty reduces PA for older players
    5. Remaining PA goes to bench players by WAR rank

    Args:
        hitters_df: Single team's hitters with role, position, rate_war, age.
        team_pa_budget: Total PA to distribute (default: 5700).

    Returns:
        hitters_df with allocated_pa_ip column added.
    """
    df = hitters_df.copy()
    df['allocated_pa_ip'] = 0.0

    # Separate by role
    starters = df[df['role'] == 'starter_hitter'].copy()
    bench = df[df['role'] == 'bench_hitter'].copy()
    replacement = df[df['role'].isin(REPLACEMENT_ROLES)].copy()

    # Replacement players get 0 PA
    df.loc[replacement.index, 'allocated_pa_ip'] = 0.0

    if len(starters) == 0:
        # No starters defined -- distribute all PA to bench by WAR
        if len(bench) > 0:
            bench_pa = _distribute_pa_by_war(bench, team_pa_budget, MAX_BENCH_PA)
            df.loc[bench.index, 'allocated_pa_ip'] = bench_pa
        return df

    # Step 1: Handle positional competition among starters
    # Group starters by position
    position_groups = starters.groupby('position')
    starter_pa = pd.Series(0.0, index=starters.index)

    # Calculate per-position PA allocation
    num_positions_filled = min(len(position_groups), NUM_LINEUP_SPOTS)
    base_pa_per_position = team_pa_budget * 0.85 / max(num_positions_filled, 1)

    for _position, group in position_groups:
        if len(group) == 1:
            # Sole starter at this position
            starter_pa.loc[group.index[0]] = base_pa_per_position
        else:
            # Multiple starters at same position: split by WAR ratio
            wars = group['rate_war'].clip(lower=0.1)  # Floor to avoid zero-division
            war_shares = wars / wars.sum()
            for idx, share in war_shares.items():
                starter_pa.loc[idx] = base_pa_per_position * share

    # Step 2: Quality-weighted adjustment across all starters
    starter_pa = _apply_quality_adjustment(starters, starter_pa)

    # Step 3: Age penalty
    starter_pa = _apply_hitter_age_penalty(starters, starter_pa)

    # Step 4: Apply caps
    starter_pa = starter_pa.clip(upper=MAX_HITTER_PA)

    df.loc[starters.index, 'allocated_pa_ip'] = starter_pa

    # Step 5: Remaining PA goes to bench
    remaining_pa = team_pa_budget - starter_pa.sum()
    if remaining_pa > 0 and len(bench) > 0:
        bench_pa = _distribute_pa_by_war(bench, remaining_pa, MAX_BENCH_PA)
        df.loc[bench.index, 'allocated_pa_ip'] = bench_pa
    elif remaining_pa < 0:
        # Over-allocated; scale starters down proportionally
        scale = team_pa_budget / starter_pa.sum()
        df.loc[starters.index, 'allocated_pa_ip'] = starter_pa * scale

    return df


def _apply_quality_adjustment(
    starters: pd.DataFrame,
    starter_pa: pd.Series
) -> pd.Series:
    """
    Apply quality-weighted PA adjustment based on WAR rank.

    Better hitters bat higher in the order and get more PA.
    Blends positional allocation with quality-based allocation (60/40).

    Args:
        starters: Starter hitter DataFrame with rate_war column.
        starter_pa: Current PA allocation series.

    Returns:
        Adjusted PA allocation series.
    """
    if len(starters) > 1:
        war_rank = starters['rate_war'].rank(
            ascending=False, method='min'
        )
        num_starters = len(starters)
        for idx in starters.index:
            rank = war_rank.loc[idx]
            t = (rank - 1) / (num_starters - 1)
            quality_pa = (
                STARTER_PA_HIGH
                - t * (STARTER_PA_HIGH - STARTER_PA_LOW)
            )
            starter_pa.loc[idx] = (
                0.6 * starter_pa.loc[idx] + 0.4 * quality_pa
            )
    else:
        starter_pa.iloc[0] = (STARTER_PA_HIGH + STARTER_PA_LOW) / 2

    return starter_pa


def _apply_hitter_age_penalty(
    starters: pd.DataFrame,
    starter_pa: pd.Series
) -> pd.Series:
    """
    Reduce PA for older hitters based on age threshold.

    Players over AGE_PA_PENALTY_START lose PA proportional
    to years above the threshold, with a floor of 300 PA.

    Args:
        starters: Starter hitter DataFrame with age column.
        starter_pa: Current PA allocation series.

    Returns:
        Age-adjusted PA allocation series.
    """
    if 'age' not in starters.columns:
        return starter_pa

    for idx in starters.index:
        age = starters.loc[idx, 'age']
        if pd.notna(age) and age > AGE_PA_PENALTY_START:
            years_over = age - AGE_PA_PENALTY_START
            penalty = years_over * AGE_PA_PENALTY_PER_YEAR
            starter_pa.loc[idx] = max(
                300, starter_pa.loc[idx] - penalty
            )

    return starter_pa


def allocate_pitcher_ip(
    pitchers_df: pd.DataFrame,
    rotation_ip_budget: float = ROTATION_IP_BUDGET,
    bullpen_ip_budget: float = BULLPEN_IP_BUDGET
) -> pd.DataFrame:
    """
    Distribute IP budget across a team's pitchers.

    Allocation strategy:
    1. All starter_pitcher entries share the rotation IP budget,
       weighted by projected WAR (better starters get more IP)
    2. Closer gets a fixed base IP
    3. Remaining bullpen IP distributed to relievers by WAR
    4. Swing pitchers get IP from both pools
    5. Age penalty reduces IP for older pitchers

    Args:
        pitchers_df: Single team's pitchers with role, rate_war, age.
        rotation_ip_budget: Total rotation IP (default: 900).
        bullpen_ip_budget: Total bullpen IP (default: 550).

    Returns:
        pitchers_df with allocated_pa_ip column added.
    """
    df = pitchers_df.copy()
    df['allocated_pa_ip'] = 0.0

    starters = df[df['role'] == 'starter_pitcher'].copy()
    closers = df[df['role'] == 'closer'].copy()
    relievers = df[df['role'] == 'reliever'].copy()
    swing = df[df['role'] == 'swing_pitcher'].copy()
    replacement = df[df['role'].isin(REPLACEMENT_ROLES)].copy()

    # Replacement pitchers get 0 IP
    df.loc[replacement.index, 'allocated_pa_ip'] = 0.0

    # --- Starting Rotation ---
    if len(starters) > 0:
        starter_ip = _allocate_rotation_ip(starters, rotation_ip_budget)
        df.loc[starters.index, 'allocated_pa_ip'] = starter_ip

    # --- Swing pitchers get a share of rotation overflow ---
    swing_rotation_ip = 0.0
    if len(swing) > 0:
        # Swing pitchers get ~60% from rotation pool, ~40% from bullpen pool
        swing_rotation_share = 0.6
        swing_total_ip = min(len(swing) * 100, rotation_ip_budget * 0.15)  # Up to 15% of rotation
        swing_rotation_ip = swing_total_ip * swing_rotation_share

    # --- Closer ---
    closer_ip_total = 0.0
    if len(closers) > 0:
        closer_ip_each = CLOSER_BASE_IP / len(closers)  # Split if multiple closers
        df.loc[closers.index, 'allocated_pa_ip'] = closer_ip_each
        closer_ip_total = CLOSER_BASE_IP

    # --- Relievers ---
    remaining_bullpen_ip = bullpen_ip_budget - closer_ip_total
    swing_bullpen_ip = 0.0
    if len(swing) > 0:
        swing_bullpen_ip = swing_rotation_ip * (1 - 0.6) / 0.6 if swing_rotation_ip > 0 else 0
        remaining_bullpen_ip -= swing_bullpen_ip

    if remaining_bullpen_ip > 0 and len(relievers) > 0:
        reliever_ip = _distribute_ip_by_war(
            relievers, remaining_bullpen_ip, MAX_RELIEVER_IP, MIN_RELIEVER_IP
        )
        df.loc[relievers.index, 'allocated_pa_ip'] = reliever_ip

    # --- Swing pitchers ---
    if len(swing) > 0:
        swing_ip_each = (swing_rotation_ip + swing_bullpen_ip) / len(swing)
        df.loc[swing.index, 'allocated_pa_ip'] = swing_ip_each

    # --- Age penalty for all pitchers ---
    if 'age' in df.columns:
        for idx in df.index:
            if df.loc[idx, 'role'] in REPLACEMENT_ROLES:
                continue
            age = df.loc[idx, 'age']
            if pd.notna(age) and age > AGE_IP_PENALTY_START:
                years_over = age - AGE_IP_PENALTY_START
                penalty = years_over * AGE_IP_PENALTY_PER_YEAR
                current_ip = df.loc[idx, 'allocated_pa_ip']
                df.loc[idx, 'allocated_pa_ip'] = max(MIN_RELIEVER_IP, current_ip - penalty)

    return df


def _allocate_rotation_ip(
    starters_df: pd.DataFrame,
    rotation_budget: float
) -> pd.Series:
    """
    Distribute rotation IP across starting pitchers, weighted by WAR.

    Better starters get proportionally more innings. Each starter
    gets at least MIN_STARTER_IP and at most MAX_STARTER_IP.

    Args:
        starters_df: DataFrame of starting pitchers with rate_war column.
        rotation_budget: Total IP to distribute.

    Returns:
        Series of allocated IP per starter (indexed like starters_df).
    """
    n = len(starters_df)
    if n == 0:
        return pd.Series(dtype=float)

    # Use WAR as quality weight (floor at 0.1 to ensure all starters get some IP)
    wars = starters_df['rate_war'].clip(lower=0.1)
    war_shares = wars / wars.sum()

    # Allocate by WAR share
    allocated = war_shares * rotation_budget

    # Apply floor and cap
    allocated = allocated.clip(lower=MIN_STARTER_IP, upper=MAX_STARTER_IP)

    # Re-normalize if caps changed the total
    total = allocated.sum()
    if abs(total - rotation_budget) > 1.0:
        # Scale the non-capped starters to absorb the difference
        capped_mask = (allocated == MIN_STARTER_IP) | (allocated == MAX_STARTER_IP)
        capped_total = allocated[capped_mask].sum()
        uncapped_total = allocated[~capped_mask].sum()
        remaining = rotation_budget - capped_total

        if uncapped_total > 0 and remaining > 0:
            scale = remaining / uncapped_total
            allocated[~capped_mask] *= scale

    return allocated


def _distribute_pa_by_war(
    players_df: pd.DataFrame,
    total_pa: float,
    max_pa: float
) -> pd.Series:
    """
    Distribute PA among players weighted by WAR quality.

    Args:
        players_df: Players with rate_war column.
        total_pa: Total PA to distribute.
        max_pa: Per-player cap.

    Returns:
        Series of allocated PA.
    """
    n = len(players_df)
    if n == 0:
        return pd.Series(dtype=float)

    wars = players_df['rate_war'].clip(lower=0.05)
    war_shares = wars / wars.sum()
    allocated = war_shares * total_pa
    allocated = allocated.clip(upper=max_pa)

    # If capping reduced total, redistribute remainder
    remainder = total_pa - allocated.sum()
    if remainder > 5 and n > 1:
        uncapped = allocated[allocated < max_pa]
        if len(uncapped) > 0:
            extra_per = remainder / len(uncapped)
            allocated.loc[uncapped.index] += extra_per
            allocated = allocated.clip(upper=max_pa)

    return allocated


def _distribute_ip_by_war(
    players_df: pd.DataFrame,
    total_ip: float,
    max_ip: float,
    min_ip: float
) -> pd.Series:
    """
    Distribute IP among relievers weighted by WAR quality.

    Args:
        players_df: Relievers with rate_war column.
        total_ip: Total IP to distribute.
        max_ip: Per-player cap.
        min_ip: Per-player floor.

    Returns:
        Series of allocated IP.
    """
    n = len(players_df)
    if n == 0:
        return pd.Series(dtype=float)

    # Ensure minimum IP is feasible
    if n * min_ip > total_ip:
        # Not enough IP for everyone's minimum; distribute evenly
        return pd.Series(total_ip / n, index=players_df.index)

    wars = players_df['rate_war'].clip(lower=0.05)
    war_shares = wars / wars.sum()
    allocated = war_shares * total_ip
    allocated = allocated.clip(lower=min_ip, upper=max_ip)

    # Re-normalize
    total = allocated.sum()
    if abs(total - total_ip) > 1.0:
        capped_mask = (allocated == min_ip) | (allocated == max_ip)
        capped_total = allocated[capped_mask].sum()
        uncapped_total = allocated[~capped_mask].sum()
        remaining = total_ip - capped_total

        if uncapped_total > 0 and remaining > 0:
            scale = remaining / uncapped_total
            allocated[~capped_mask] *= scale
            allocated = allocated.clip(lower=min_ip, upper=max_ip)

    return allocated


def calculate_playing_time_factor(
    allocated_pa_ip: float,
    base_pa_ip: float,
    role: str
) -> float:
    """
    Calculate playing time adjustment factor.

    Compares allocated playing time to the typical full-season usage
    for the player's role. This captures roster construction effects:
    a bench hitter allocated 200 PA gets factor = 200/580 = 0.34,
    while a starter at 600 PA gets factor = 600/580 = 1.03.

    The WAR projections predict absolute season WAR (not rate WAR),
    but the constraint optimizer normalizes total to 1000. The factor
    redistributes each player's WAR share based on actual playing time
    in the team context.

    Args:
        allocated_pa_ip: Allocated PA (hitters) or IP (pitchers) from
            the budget allocation.
        base_pa_ip: Player's base-year actual PA or IP (currently unused
            in V1, reserved for future role-change detection).
        role: Player role string.

    Returns:
        Playing time factor (typically 0.3 to 1.3).
    """
    if allocated_pa_ip <= 0:
        return 0.0

    # Use role-based typical full-season denominator
    # These represent what a player in each role typically accumulates
    # over a full 162-game season
    denominator = _get_typical_full_season_usage(role)

    return allocated_pa_ip / denominator


def _get_typical_full_season_usage(role: str) -> float:
    """
    Get typical full-season PA/IP for a role.

    These represent what a typical everyday/full-time player in each
    role accumulates over a full 162-game season. Used as the
    denominator for playing time factor calculation.

    Args:
        role: Player role string.

    Returns:
        Typical full-season PA or IP.
    """
    defaults = {
        'starter_hitter': 580.0,   # Typical everyday hitter
        'bench_hitter': 580.0,     # Same denominator; PA allocation handles reduction
        'replacement_hitter': 580.0,
        'starter_pitcher': 170.0,  # Typical full-season starter
        'reliever': 55.0,          # Typical full-season reliever
        'closer': 60.0,            # Typical closer workload
        'swing_pitcher': 100.0,    # Swing/opener/long-relief
        'replacement_pitcher': 170.0,
    }
    return defaults.get(role, 580.0)


def adjust_war_for_playing_time(
    rate_war: float,
    playing_time_factor: float
) -> float:
    """
    Convert rate-based WAR to playing-time-adjusted WAR.

    Args:
        rate_war: Rate-based WAR projection from existing pipeline.
        playing_time_factor: From calculate_playing_time_factor().

    Returns:
        Playing-time-adjusted WAR contribution to team.
    """
    return rate_war * playing_time_factor
