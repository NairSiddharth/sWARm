"""
Scenario projection charts for current season analysis

Contains visualization functions for displaying scenario-based projections
and comparisons between current and projected statistics.
"""

# Third-party imports
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict

# Local application imports
from common_modules.logging import get_logger

# Module logger
logger = get_logger(__name__)


def create_scenario_projection_chart(scenarios_data: Dict[str, Dict],
                                     player_name: str,
                                     player_type: str = 'hitter') -> go.Figure:
    """
    Create interactive chart showing all 5 projection scenarios for a player

    Args:
        scenarios_data: Dictionary with scenario projections
        player_name: Name of the player
        player_type: 'hitter' or 'pitcher'

    Returns:
        Plotly figure with scenario visualizations
    """
    logger.debug(f"Creating scenario projection chart for {player_name}")

    # Prepare data for visualization
    scenario_names = ['100%', '75%', '50%', '25%', 'career_avg']
    war_values = [scenarios_data.get(s, {}).get('war', 0) for s in scenario_names]
    warp_values = [scenarios_data.get(s, {}).get('warp', 0) for s in scenario_names]

    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            f'{player_name} - WAR Projections',
            f'{player_name} - WARP Projections',
            'Scenario Comparison',
            'Performance Stats by Scenario'
        ],
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "scatter"}, {"type": "bar"}]]
    )

    # WAR projection bar chart
    fig.add_trace(
        go.Bar(
            x=scenario_names,
            y=war_values,
            name='WAR',
            marker=dict(color='lightblue'),
            text=[f'{v:.2f}' for v in war_values],
            textposition='outside'
        ),
        row=1, col=1
    )

    # WARP projection bar chart
    fig.add_trace(
        go.Bar(
            x=scenario_names,
            y=warp_values,
            name='WARP',
            marker=dict(color='lightcoral'),
            text=[f'{v:.2f}' for v in warp_values],
            textposition='outside'
        ),
        row=1, col=2
    )

    # Scenario comparison scatter plot
    fig.add_trace(
        go.Scatter(
            x=war_values,
            y=warp_values,
            mode='markers+text',
            text=scenario_names,
            textposition='top center',
            marker=dict(size=12, color=['red', 'orange', 'yellow', 'lightgreen', 'blue']),
            name='WAR vs WARP'
        ),
        row=2, col=1
    )

    # Key stats by scenario (example with HR for hitters)
    if player_type == 'hitter':
        stat_key = 'HR'
        stat_label = 'Home Runs'
    else:
        stat_key = 'SO'
        stat_label = 'Strikeouts'

    stat_values = []
    for scenario in scenario_names:
        stats = scenarios_data.get(scenario, {}).get('projected_stats', {})
        stat_values.append(stats.get(stat_key, 0))

    fig.add_trace(
        go.Bar(
            x=scenario_names,
            y=stat_values,
            name=stat_label,
            marker=dict(color='lightgreen'),
            text=[f'{v:.0f}' for v in stat_values],
            textposition='outside'
        ),
        row=2, col=2
    )

    # Update layout
    fig.update_layout(
        title=f'{player_name} - End of Season Projections ({player_type.title()})',
        showlegend=False,
        height=800,
        template='plotly_white'
    )

    # Update axes labels
    fig.update_xaxes(title_text="Scenario", row=1, col=1)
    fig.update_xaxes(title_text="Scenario", row=1, col=2)
    fig.update_xaxes(title_text="WAR", row=2, col=1)
    fig.update_xaxes(title_text="Scenario", row=2, col=2)

    fig.update_yaxes(title_text="WAR", row=1, col=1)
    fig.update_yaxes(title_text="WARP", row=1, col=2)
    fig.update_yaxes(title_text="WARP", row=2, col=1)
    fig.update_yaxes(title_text=stat_label, row=2, col=2)

    return fig


def create_current_vs_projected_comparison(current_stats: Dict,
                                           projected_stats: Dict,
                                           player_name: str,
                                           player_type: str = 'hitter') -> go.Figure:
    """
    Create comparison chart showing current vs projected season-end stats

    Args:
        current_stats: Current season statistics
        projected_stats: Projected end-of-season statistics
        player_name: Name of the player
        player_type: 'hitter' or 'pitcher'

    Returns:
        Plotly figure comparing current and projected stats
    """
    logger.debug(f"Creating current vs projected comparison for {player_name}")

    # Select key stats to display
    if player_type == 'hitter':
        key_stats = ['HR', 'RBI', 'R', 'SB', 'AVG', 'OBP', 'SLG']
        stat_labels = ['Home Runs', 'RBI', 'Runs', 'Stolen Bases', 'Avg', 'OBP', 'SLG']
    else:
        key_stats = ['W', 'SO', 'ERA', 'WHIP', 'K/9', 'BB/9']
        stat_labels = ['Wins', 'Strikeouts', 'ERA', 'WHIP', 'K/9', 'BB/9']

    current_values = [current_stats.get(stat, 0) for stat in key_stats]
    projected_values = [projected_stats.get(stat, 0) for stat in key_stats]

    fig = go.Figure()

    # Current stats
    fig.add_trace(go.Bar(
        name='Current',
        x=stat_labels,
        y=current_values,
        marker=dict(color='lightblue'),
        text=[f'{v:.3f}' if v < 1 else f'{v:.0f}' for v in current_values],
        textposition='outside'
    ))

    # Projected stats
    fig.add_trace(go.Bar(
        name='Projected (100%)',
        x=stat_labels,
        y=projected_values,
        marker=dict(color='darkblue'),
        text=[f'{v:.3f}' if v < 1 else f'{v:.0f}' for v in projected_values],
        textposition='outside'
    ))

    fig.update_layout(
        title=f'{player_name} - Current vs Projected Stats',
        xaxis_title='Statistics',
        yaxis_title='Value',
        barmode='group',
        template='plotly_white'
    )

    return fig
