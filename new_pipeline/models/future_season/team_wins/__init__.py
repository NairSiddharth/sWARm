"""
Team Wins Projection System.

Extends the future WAR projection pipeline to predict team wins
for an upcoming MLB season. Consumes individual player WAR projections
and user-provided rosters to produce team-level win totals with
zero-sum constraint enforcement.

Public API:
    generate_team_wins(base_year) - Convenience function for full pipeline
    TeamWinsPipeline - Full pipeline class with step-by-step control
    generate_roster_template() - Generate starter roster CSV from projections
    load_roster() - Load and validate a roster CSV
"""

from new_pipeline.models.future_season.team_wins.team_wins_pipeline import (
    TeamWinsPipeline,
    generate_team_wins
)
from new_pipeline.models.future_season.team_wins.roster_loader import (
    load_roster,
    generate_roster_template
)

__all__ = [
    'TeamWinsPipeline',
    'generate_team_wins',
    'load_roster',
    'generate_roster_template',
]
