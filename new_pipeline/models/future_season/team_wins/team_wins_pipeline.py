"""
Team Wins Projection Pipeline - End-to-End Orchestration.

Consumes existing WAR projections and roster data to generate
team win totals for a single upcoming season.

Designed to run AFTER generate_league_projections() has produced WAR CSVs.

Usage:
    >>> from new_pipeline.models.future_season.team_wins import generate_team_wins
    >>> results = generate_team_wins(base_year=2025)
    >>> print(results.sort_values('constrained_wins', ascending=False))
"""

from pathlib import Path
from typing import Optional

import pandas as pd

from new_pipeline.models.future_season.team_wins.roster_loader import (
    load_roster,
)
from new_pipeline.models.future_season.team_wins.team_war_aggregator import (
    merge_roster_with_projections, allocate_and_adjust, aggregate_team_war
)
from new_pipeline.models.future_season.team_wins.wins_converter import (
    war_to_wins_raw, apply_wins_constraint, generate_standings
)


# Resolve project root
PROJECT_ROOT = Path(__file__).resolve().parents[4]


class TeamWinsPipeline:
    """
    Orchestrates the full flow from roster CSV to team win projections.

    Steps:
    1. Load roster CSV and existing WAR projection CSVs
    2. Merge rosters with projections (map players to WAR)
    3. Allocate playing time (budget-based PA/IP per team)
    4. Adjust WAR for playing time
    5. Aggregate to team-level WAR
    6. Convert to team wins with zero-sum constraint
    7. Generate standings and save output
    """

    def __init__(
        self,
        base_year: int,
        roster_path: Optional[str] = None,
        projections_dir: Optional[str] = None
    ):
        """
        Initialize team wins pipeline.

        Args:
            base_year: Base year of projections (e.g., 2025 means
                       projections are for 2026 season).
            roster_path: Path to roster CSV. Default: data/rosters/rosters_{year}.csv
            projections_dir: Directory with WAR CSVs. Default: predictions/
        """
        self.base_year = base_year
        self.projection_year = base_year + 1

        # Default paths
        if roster_path is None:
            self.roster_path = str(
                PROJECT_ROOT / f"data/rosters/rosters_{self.projection_year}.csv"
            )
        else:
            self.roster_path = roster_path

        if projections_dir is None:
            self.projections_dir = str(PROJECT_ROOT / "predictions")
        else:
            self.projections_dir = projections_dir

        # Data storage
        self.roster_df = None
        self.hitter_projections = None
        self.pitcher_projections = None
        self.enriched_roster = None
        self.team_war_df = None
        self.team_wins_df = None
        self.standings_df = None

        print("Initialized TeamWinsPipeline:")
        print(f"  Base year: {base_year}")
        print(f"  Projection year: {self.projection_year}")
        print(f"  Roster: {self.roster_path}")
        print(f"  Projections: {self.projections_dir}")

    def load_data(self) -> None:
        """Load roster and WAR projection CSVs."""
        print("\n--- Step 1: Loading Data ---")

        # Load roster
        self.roster_df = load_roster(self.roster_path, self.projection_year)

        # Load projections
        hitter_path = Path(self.projections_dir) / f"future_projections_hitter_{self.projection_year}.csv"
        pitcher_path = Path(self.projections_dir) / f"future_projections_pitcher_{self.projection_year}.csv"

        if not hitter_path.exists():
            raise FileNotFoundError(f"Hitter projections not found: {hitter_path}")
        if not pitcher_path.exists():
            raise FileNotFoundError(f"Pitcher projections not found: {pitcher_path}")

        self.hitter_projections = pd.read_csv(hitter_path, encoding='utf-8-sig')
        self.pitcher_projections = pd.read_csv(pitcher_path, encoding='utf-8-sig')

        print(f"  Hitter projections: {len(self.hitter_projections)} players")
        print(f"  Pitcher projections: {len(self.pitcher_projections)} players")

    def build_team_projections(self) -> pd.DataFrame:
        """
        Build team win projections: merge, allocate, adjust, aggregate, convert.

        Returns:
            Standings DataFrame with constrained wins.
        """
        if self.roster_df is None:
            raise RuntimeError("No data loaded. Call load_data() first.")

        war_column = f'war_{self.projection_year}'

        # Step 2: Merge roster with projections
        print("\n--- Step 2: Merging Roster with Projections ---")
        self.enriched_roster = merge_roster_with_projections(
            self.roster_df,
            self.hitter_projections,
            self.pitcher_projections,
            war_column=war_column
        )

        # Step 3: Allocate playing time and adjust WAR
        print("\n--- Step 3: Allocating Playing Time ---")
        self.enriched_roster = allocate_and_adjust(self.enriched_roster)

        # Step 4: Aggregate to team level
        print("\n--- Step 4: Aggregating Team WAR ---")
        self.team_war_df = aggregate_team_war(self.enriched_roster)

        # Step 5: Convert to wins
        print("\n--- Step 5: Converting to Wins ---")
        self.team_war_df['raw_wins'] = self.team_war_df['total_war'].apply(war_to_wins_raw)

        self.team_wins_df = apply_wins_constraint(self.team_war_df)

        # Step 6: Generate standings
        print("\n--- Step 6: Generating Standings ---")
        self.standings_df = generate_standings(self.team_wins_df, self.projection_year)

        return self.standings_df

    def save_results(self, output_dir: Optional[str] = None) -> tuple:
        """
        Save team wins CSV and detailed player breakdown CSV.

        Args:
            output_dir: Directory for output files. Default: predictions/

        Returns:
            Tuple of (standings_path, detail_path).
        """
        if self.standings_df is None:
            raise RuntimeError("No results to save. Call build_team_projections() first.")

        if output_dir is None:
            output_dir = str(PROJECT_ROOT / "predictions")

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Save standings
        standings_path = out_dir / f"team_wins_{self.projection_year}.csv"
        standings_cols = [
            'Team', 'Division', 'League', 'total_war', 'hitter_war', 'pitcher_war',
            'constrained_wins', 'projected_losses', 'win_pct',
            'rank', 'div_rank', 'wins_adjustment',
            'num_hitters', 'num_pitchers', 'num_replacement', 'num_not_found',
            'projection_year'
        ]
        available_cols = [c for c in standings_cols if c in self.standings_df.columns]
        self.standings_df[available_cols].to_csv(standings_path, index=False, encoding='utf-8-sig')

        # Save detail file
        detail_path = out_dir / f"team_wins_detail_{self.projection_year}.csv"
        detail_cols = [
            'Team', 'playerid', 'Name', 'role', 'position', 'player_type',
            'rate_war', 'age', 'allocated_pa_ip', 'playing_time_factor',
            'adjusted_war', 'projection_source', 'projection_year'
        ]
        if self.enriched_roster is not None:
            avail_detail = [c for c in detail_cols if c in self.enriched_roster.columns]
            detail_df = self.enriched_roster[avail_detail].copy()
            detail_df = detail_df.round({
                'rate_war': 2, 'allocated_pa_ip': 1,
                'playing_time_factor': 3, 'adjusted_war': 2
            })
            detail_df = detail_df.sort_values(['Team', 'adjusted_war'], ascending=[True, False])
            detail_df.to_csv(detail_path, index=False, encoding='utf-8-sig')

        print(f"\nResults saved:")
        print(f"  Standings: {standings_path}")
        print(f"  Detail:    {detail_path}")

        return str(standings_path), str(detail_path)

    def run_full_pipeline(self, save_output: bool = True) -> pd.DataFrame:
        """
        Run complete team wins pipeline end-to-end.

        Args:
            save_output: Save results to CSV files.

        Returns:
            Standings DataFrame.
        """
        print("=" * 70)
        print(f"TEAM WINS PROJECTION PIPELINE - {self.projection_year} SEASON")
        print("=" * 70)

        self.load_data()
        self.build_team_projections()

        if save_output:
            self.save_results()

        # Print final standings summary
        self._print_standings_summary()

        print("\n" + "=" * 70)
        print("PIPELINE COMPLETE")
        print("=" * 70)

        return self.standings_df

    def _print_standings_summary(self) -> None:
        """Print a formatted standings summary."""
        if self.standings_df is None:
            return

        print(f"\n{'=' * 50}")
        print(f"  {self.projection_year} PROJECTED STANDINGS")
        print(f"{'=' * 50}")

        for division in self.standings_df['Division'].unique():
            div_teams = self.standings_df[self.standings_df['Division'] == division]
            div_teams = div_teams.sort_values('constrained_wins', ascending=False)
            print(f"\n  {division}")
            print(f"  {'Team':<6} {'W':>4} {'L':>4} {'Pct':>6} {'WAR':>6}")
            print(f"  {'-'*28}")
            for _, row in div_teams.iterrows():
                print(f"  {row['Team']:<6} {row['constrained_wins']:>4} "
                      f"{row['projected_losses']:>4} "
                      f"{row['win_pct']:>6.3f} "
                      f"{row['total_war']:>6.1f}")


def generate_team_wins(
    base_year: int,
    roster_path: Optional[str] = None,
    projections_dir: Optional[str] = None,
    save_output: bool = True
) -> pd.DataFrame:
    """
    Convenience function to generate team wins projections.

    Follows the pattern of generate_league_projections() in the
    existing future projection pipeline.

    Args:
        base_year: Base year of WAR projections (e.g., 2025).
        roster_path: Path to roster CSV (default: data/rosters/rosters_{year}.csv).
        projections_dir: Directory with projection CSVs (default: predictions/).
        save_output: Save results to CSV.

    Returns:
        Standings DataFrame with team wins.

    Example:
        >>> standings = generate_team_wins(base_year=2025)
        >>> print(standings.sort_values('constrained_wins', ascending=False).head(10))
    """
    pipeline = TeamWinsPipeline(
        base_year=base_year,
        roster_path=roster_path,
        projections_dir=projections_dir
    )
    return pipeline.run_full_pipeline(save_output=save_output)
