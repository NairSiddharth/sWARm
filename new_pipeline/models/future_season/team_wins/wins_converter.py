"""
Wins Converter - Convert team WAR to projected wins with zero-sum constraint.

The standard WAR-to-wins formula:
    Team Wins = REPLACEMENT_WINS_PER_TEAM + Team WAR

A replacement-level team (0 WAR) wins ~47.67 games.
Each additional WAR above replacement adds ~1 win.

The zero-sum constraint ensures total league wins = 2430,
which is mathematically necessary (every game has a winner and loser).
"""

import numpy as np
import pandas as pd

from new_pipeline.models.future_season.team_wins.constants import (
    TOTAL_LEAGUE_WINS, REPLACEMENT_WINS_PER_TEAM, GAMES_PER_SEASON,
    DIVISION_MAP, DIVISION_ORDER
)


def war_to_wins_raw(team_war: float) -> float:
    """
    Convert team WAR to projected wins (before zero-sum constraint).

    Args:
        team_war: Total team WAR above replacement.

    Returns:
        Projected wins (unconstrained).
    """
    return REPLACEMENT_WINS_PER_TEAM + team_war


def apply_wins_constraint(
    team_wins_df: pd.DataFrame,
    target_total: int = TOTAL_LEAGUE_WINS,
    war_column: str = 'total_war',
    raw_wins_column: str = 'raw_wins'
) -> pd.DataFrame:
    """
    Enforce zero-sum constraint: total league wins must equal target_total.

    The correction is applied proportionally to each team's WAR share,
    preserving relative rankings. Only the WAR component is scaled,
    not the replacement-level baseline, to prevent impossible results.

    Args:
        team_wins_df: DataFrame with Team, total_war, and raw_wins columns.
        target_total: Target total wins (default: 2430).
        war_column: Column containing team WAR totals.
        raw_wins_column: Column containing unconstrained wins.

    Returns:
        DataFrame with additional columns:
            constrained_wins (int), projected_losses (int),
            win_pct (float), wins_adjustment (float)
    """
    df = team_wins_df.copy()

    # Calculate raw wins if not present
    if raw_wins_column not in df.columns:
        df[raw_wins_column] = df[war_column].apply(war_to_wins_raw)

    total_raw_wins = df[raw_wins_column].sum()
    correction = target_total - total_raw_wins

    print("\nZero-sum constraint:")
    print(f"  Raw total wins: {total_raw_wins:.1f}")
    print(f"  Target total:   {target_total}")
    print(f"  Correction:     {correction:+.1f}")

    # Distribute correction proportionally to WAR share
    total_war = df[war_column].sum()

    if total_war > 0:
        war_shares = df[war_column] / total_war
        df['wins_adjustment'] = correction * war_shares
    else:
        # Edge case: all teams at replacement level
        df['wins_adjustment'] = correction / len(df)

    df['constrained_wins_raw'] = df[raw_wins_column] + df['wins_adjustment']

    # Round to integers (maintaining total)
    df['constrained_wins'] = _round_preserving_total(
        df['constrained_wins_raw'], target_total
    )

    df['projected_losses'] = GAMES_PER_SEASON - df['constrained_wins']
    df['win_pct'] = (df['constrained_wins'] / GAMES_PER_SEASON).round(3)
    df['wins_adjustment'] = df['wins_adjustment'].round(1)

    # Drop intermediate column
    df = df.drop(columns=['constrained_wins_raw'])

    # Validate bounds
    min_wins = df['constrained_wins'].min()
    max_wins = df['constrained_wins'].max()
    if min_wins < 30:
        print(f"  Warning: Minimum projected wins ({min_wins}) below 30")
    if max_wins > 130:
        print(f"  Warning: Maximum projected wins ({max_wins}) above 130")

    actual_total = df['constrained_wins'].sum()
    print(f"  Constrained total: {actual_total} (target: {target_total})")

    return df


def _round_preserving_total(values: pd.Series, target_total: int) -> pd.Series:
    """
    Round a series of floats to integers while preserving their sum.

    Uses the largest-remainder method to distribute rounding residuals.

    Args:
        values: Float values to round.
        target_total: Desired integer sum.

    Returns:
        Integer series summing to target_total.
    """
    floored = np.floor(values).astype(int)
    remainders = values - floored
    deficit = target_total - floored.sum()

    # Give the extra wins to teams with the largest fractional parts
    if deficit > 0:
        top_indices = remainders.nlargest(int(deficit)).index
        floored.loc[top_indices] += 1
    elif deficit < 0:
        # Need to remove wins from teams with smallest fractional parts
        bottom_indices = remainders.nsmallest(int(abs(deficit))).index
        floored.loc[bottom_indices] -= 1

    return floored


def generate_standings(
    team_wins_df: pd.DataFrame,
    projection_year: int
) -> pd.DataFrame:
    """
    Format team wins into standings with divisions, leagues, and rankings.

    Args:
        team_wins_df: DataFrame with Team and constrained_wins columns.
        projection_year: Year for labeling.

    Returns:
        Formatted standings DataFrame sorted by division then wins (desc).
    """
    df = team_wins_df.copy()

    # Add division and league info
    df['League'] = df['Team'].map(lambda t: DIVISION_MAP.get(t, ('', ''))[0])
    df['Division'] = df['Team'].map(lambda t: DIVISION_MAP.get(t, ('', ''))[1])

    # Overall rank
    df['rank'] = df['constrained_wins'].rank(ascending=False, method='min').astype(int)

    # Division rank
    df['div_rank'] = df.groupby('Division')['constrained_wins'].rank(
        ascending=False, method='min'
    ).astype(int)

    df['projection_year'] = projection_year

    # Sort by division order, then within division by wins
    div_order_map = {d: i for i, d in enumerate(DIVISION_ORDER)}
    df['_div_sort'] = df['Division'].map(div_order_map)
    df = df.sort_values(['_div_sort', 'constrained_wins'], ascending=[True, False])
    df = df.drop(columns=['_div_sort'])

    return df
