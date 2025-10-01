"""
Player comparison charts for current season analysis

Contains visualization functions for comparing multiple players'
projections and performance metrics.
"""

# Third-party imports
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict

# Local application imports
from common_modules.logging import get_logger

# Module logger
logger = get_logger(__name__)


def create_player_comparison_dashboard(players_data: Dict[str, Dict]) -> go.Figure:
    """
    Create dashboard comparing multiple players' projections

    Args:
        players_data: Dictionary with player names as keys and their projection data

    Returns:
        Plotly figure with multi-player comparison
    """
    logger.debug(f"Creating player comparison dashboard for {len(players_data)} players")

    player_names = list(players_data.keys())

    # Extract comparison data
    war_100 = [players_data[p]['scenarios']['100%']['war'] for p in player_names]
    warp_100 = [players_data[p]['scenarios']['100%']['warp'] for p in player_names]
    war_career = [players_data[p]['scenarios']['career_avg']['war'] for p in player_names]
    warp_career = [players_data[p]['scenarios']['career_avg']['warp'] for p in player_names]

    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            'WAR Projections (100% Pace)',
            'WARP Projections (100% Pace)',
            'WAR Projections (Career Regression)',
            'WARP Projections (Career Regression)'
        ]
    )

    # 100% pace WAR
    fig.add_trace(
        go.Bar(x=player_names, y=war_100, name='100% WAR', marker_color='lightblue'),
        row=1, col=1
    )

    # 100% pace WARP
    fig.add_trace(
        go.Bar(x=player_names, y=warp_100, name='100% WARP', marker_color='lightcoral'),
        row=1, col=2
    )

    # Career regression WAR
    fig.add_trace(
        go.Bar(x=player_names, y=war_career, name='Career WAR', marker_color='darkblue'),
        row=2, col=1
    )

    # Career regression WARP
    fig.add_trace(
        go.Bar(x=player_names, y=warp_career, name='Career WARP', marker_color='darkred'),
        row=2, col=2
    )

    fig.update_layout(
        title='Multi-Player Projection Comparison',
        showlegend=False,
        height=800,
        template='plotly_white'
    )

    return fig
