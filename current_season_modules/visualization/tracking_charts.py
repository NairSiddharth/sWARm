"""
Tracking charts for WAR/WARP progression analysis

Contains visualization functions for tracking player performance
over time with current season and projections.
"""

# Third-party imports
import plotly.graph_objects as go
from typing import Dict, List

# Local application imports
from common_modules.config import CURRENT_SEASON
from common_modules.logging import get_logger

# Module logger
logger = get_logger(__name__)


def create_war_warp_tracking_chart(historical_data: List[Dict],
                                   current_war: float,
                                   current_warp: float,
                                   projected_war: float,
                                   projected_warp: float,
                                   player_name: str) -> go.Figure:
    """
    Create tracking chart showing WAR/WARP progression over time with projections

    Args:
        historical_data: List of historical season data
        current_war: Current season WAR
        current_warp: Current season WARP
        projected_war: Projected final WAR
        projected_warp: Projected final WARP
        player_name: Name of the player

    Returns:
        Plotly figure with WAR/WARP tracking
    """
    logger.debug(f"Creating WAR/WARP tracking chart for {player_name}")

    # Extract historical years and values
    years = [d.get('year', 0) for d in historical_data]
    historical_war = [d.get('war', 0) for d in historical_data]
    historical_warp = [d.get('warp', 0) for d in historical_data]

    # Add current and projected points
    current_year = max(years) + 1 if years else CURRENT_SEASON
    years.extend([current_year, current_year])
    historical_war.extend([current_war, projected_war])
    historical_warp.extend([current_warp, projected_warp])

    fig = go.Figure()

    # Historical WAR
    fig.add_trace(go.Scatter(
        x=years[:-2],
        y=historical_war[:-2],
        mode='lines+markers',
        name='Historical WAR',
        line=dict(color='blue', width=2),
        marker=dict(size=8)
    ))

    # Historical WARP
    fig.add_trace(go.Scatter(
        x=years[:-2],
        y=historical_warp[:-2],
        mode='lines+markers',
        name='Historical WARP',
        line=dict(color='red', width=2),
        marker=dict(size=8)
    ))

    # Current season points
    fig.add_trace(go.Scatter(
        x=[current_year],
        y=[current_war],
        mode='markers',
        name='Current WAR',
        marker=dict(size=12, color='blue', symbol='star')
    ))

    fig.add_trace(go.Scatter(
        x=[current_year],
        y=[current_warp],
        mode='markers',
        name='Current WARP',
        marker=dict(size=12, color='red', symbol='star')
    ))

    # Projected points
    fig.add_trace(go.Scatter(
        x=[current_year],
        y=[projected_war],
        mode='markers',
        name='Projected WAR',
        marker=dict(size=12, color='lightblue', symbol='diamond')
    ))

    fig.add_trace(go.Scatter(
        x=[current_year],
        y=[projected_warp],
        mode='markers',
        name='Projected WARP',
        marker=dict(size=12, color='lightcoral', symbol='diamond')
    ))

    fig.update_layout(
        title=f'{player_name} - WAR/WARP Progression',
        xaxis_title='Season',
        yaxis_title='WAR/WARP Value',
        template='plotly_white',
        hovermode='x unified'
    )

    return fig
