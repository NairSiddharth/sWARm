"""
Player Projection Visualizations for oWAR System

This module provides interactive visualizations for player projections including:
- Enhanced boxplots with scenario points
- Comparison charts (current vs projected)
- Range charts with key scenarios
- Scenario heatmaps

Refactored from notebook to maintain clean separation of concerns.

Author: oWAR Development Team
"""

# Standard library imports
from typing import Dict, List, Optional
import logging

# Third-party imports
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

# Project imports
from common_modules.logging import get_logger

# Module logger
logger = get_logger(__name__)

# Public API
__all__ = [
    'create_projection_dashboard',
    'create_comparison_chart',
    'create_enhanced_boxplot',
    'create_range_chart',
    'create_scenario_heatmap',
    'format_war_display',
    'SCENARIO_STYLES'
]

# Default scenario colors and symbols
SCENARIO_STYLES = {
    '150% (Hot Streak)': {'color': '#2ca02c', 'symbol': 'triangle-up', 'size': 12},
    '125% (Above Pace)': {'color': '#98df8a', 'symbol': 'diamond', 'size': 10},
    '100% (Maintain Pace)': {'color': '#1f77b4', 'symbol': 'circle', 'size': 12},
    '75% (Slight Regression)': {'color': '#ffbb78', 'symbol': 'square', 'size': 10},
    '50% (Major Regression)': {'color': '#ff9896', 'symbol': 'diamond', 'size': 10},
    '25% (Horrible Regression)': {'color': '#d62728', 'symbol': 'triangle-down', 'size': 12},
    'Career Average': {'color': '#9467bd', 'symbol': 'star', 'size': 14}
}


def create_projection_dashboard(
    projections: List[Dict],
    config: Optional[Dict] = None
) -> Dict[str, go.Figure]:
    """
    Create all projection visualizations.

    Args:
        projections: List of player projection dictionaries
        config: Optional configuration dictionary with scenario settings

    Returns:
        Dictionary with visualization figures:
        - 'comparison': Current vs Projected bar chart
        - 'boxplot': Enhanced boxplot with scenario points
        - 'range': Range chart with key scenarios
        - 'heatmap': Scenario heatmap (optional)
    """
    if not projections:
        logger.warning("No projections provided for visualization")
        return {}

    logger.info(f"Creating projection dashboard for {len(projections)} players")

    figures = {}

    # Extract scenarios from config or use defaults
    scenarios = config.get('SCENARIOS', {}) if config else {}

    # Create each visualization
    try:
        figures['comparison'] = create_comparison_chart(projections)
        figures['boxplot'] = create_enhanced_boxplot(projections, scenarios)
        figures['range'] = create_range_chart(projections)

        # Optional: Add heatmap if multiple players and scenarios
        if len(projections) > 1 and scenarios:
            figures['heatmap'] = create_scenario_heatmap(projections, scenarios)

    except Exception as e:
        logger.error(f"Error creating projection visualizations: {e}")
        raise

    logger.info(f"Created {len(figures)} projection visualizations")
    return figures


def create_comparison_chart(projections: List[Dict]) -> go.Figure:
    """
    Create bar chart comparing current vs projected WAR.

    Args:
        projections: List of player projection dictionaries

    Returns:
        Plotly figure with grouped bar chart
    """
    # Prepare data
    players = [p['player_name'] for p in projections]
    current = [p.get('current_war', 0) for p in projections]
    projected = [
        p['projections'].get('100% (Maintain Pace)', {}).get('full_season_war', 0)
        for p in projections
    ]

    # Create figure
    fig = go.Figure()

    # Add current WAR bars
    fig.add_trace(go.Bar(
        x=players,
        y=current,
        name='Current WAR',
        marker_color='steelblue',
        text=[f'{val:.2f}' for val in current],
        textposition='auto',
        hovertemplate='<b>%{x}</b><br>Current WAR: %{y:.3f}<extra></extra>'
    ))

    # Add projected WAR bars
    fig.add_trace(go.Bar(
        x=players,
        y=projected,
        name='Projected WAR (100% Pace)',
        marker_color='coral',
        text=[f'{val:.2f}' for val in projected],
        textposition='auto',
        hovertemplate='<b>%{x}</b><br>Projected WAR: %{y:.3f}<extra></extra>'
    ))

    # Update layout
    fig.update_layout(
        title={
            'text': 'Current vs Projected WAR',
            'font': {'size': 18, 'color': '#333'},
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis_title="Player",
        yaxis_title="WAR",
        barmode='group',
        height=400,
        template='plotly_white',
        hovermode='x unified',
        xaxis={'tickangle': 45}
    )

    return fig


def create_enhanced_boxplot(
    projections: List[Dict],
    scenarios: Optional[Dict] = None
) -> go.Figure:
    """
    Create enhanced boxplot with overlaid scenario points.

    This visualization shows:
    - Box: Quartiles and median of all scenario projections
    - Whiskers: Min/max scenarios
    - Colored points: Individual scenarios with hover details

    Args:
        projections: List of player projection dictionaries
        scenarios: Dictionary of scenario names and multipliers

    Returns:
        Plotly figure with boxplot and overlaid points
    """
    fig = go.Figure()

    # Use default scenario styles or customize based on provided scenarios
    scenario_styles = SCENARIO_STYLES.copy()
    if scenarios:
        # Update styles if custom scenarios provided
        for scenario in scenarios.keys():
            if scenario not in scenario_styles:
                # Assign a default style for unknown scenarios
                scenario_styles[scenario] = {
                    'color': '#808080',
                    'symbol': 'circle',
                    'size': 10
                }

    # Process each player
    for i, player_data in enumerate(projections):
        player_name = player_data['player_name']
        player_projections = player_data.get('projections', {})

        # Extract all scenario values
        scenario_values = []
        scenario_names = []
        for scenario_name in player_projections.keys():
            value = player_projections[scenario_name].get('full_season_war', 0)
            scenario_values.append(value)
            scenario_names.append(scenario_name)

        if not scenario_values:
            logger.warning(f"No scenario values for player {player_name}")
            continue

        # Add boxplot for this player
        fig.add_trace(go.Box(
            y=scenario_values,
            x=[player_name] * len(scenario_values),
            name=player_name,
            boxmean='sd',  # Show mean and standard deviation
            marker_color='lightblue',
            line=dict(color='darkblue'),
            fillcolor='rgba(135, 206, 250, 0.5)',
            whiskerwidth=0.8,
            notched=False,
            showlegend=False,
            hoverinfo='skip'  # We'll use the points for hover info
        ))

        # Overlay individual scenario points
        for scenario, value in zip(scenario_names, scenario_values):
            style = scenario_styles.get(scenario, {
                'color': '#808080',
                'symbol': 'circle',
                'size': 10
            })

            # Determine if this is the first trace for this scenario (for legend)
            show_legend = i == 0

            fig.add_trace(go.Scatter(
                x=[player_name],
                y=[value],
                mode='markers',
                name=scenario.split(' (')[0] if show_legend else None,
                legendgroup=scenario,  # Group all same scenarios together
                showlegend=show_legend,
                marker=dict(
                    color=style['color'],
                    size=style['size'],
                    symbol=style.get('symbol', 'circle'),
                    line=dict(width=1, color='white')
                ),
                hovertemplate=(
                    f'<b>{player_name}</b><br>'
                    f'{scenario}<br>'
                    f'WAR: {value:.3f}<br>'
                    f'<extra></extra>'
                )
            ))

    # Update layout
    fig.update_layout(
        title={
            'text': 'WAR Projection Scenarios - Distribution & Individual Outcomes',
            'font': {'size': 18, 'color': '#333'},
            'x': 0.5,
            'xanchor': 'center'
        },
        yaxis_title="Projected WAR",
        xaxis_title="Player",
        height=600,
        template='plotly_white',
        boxmode='group',
        hovermode='closest',
        legend=dict(
            title="Scenarios",
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor="lightgray",
            borderwidth=1
        ),
        margin=dict(r=200)  # Make room for legend
    )

    # Add annotations explaining the visualization
    fig.add_annotation(
        text=(
            "Box shows quartiles and median | "
            "Points show specific scenarios | "
            "Hover for details"
        ),
        xref="paper", yref="paper",
        x=0.5, y=-0.12,
        showarrow=False,
        font=dict(size=10, color='gray'),
        align='center'
    )

    # Add annotation explaining symbols
    fig.add_annotation(
        text="▲ Hot Streak | ◆ Above/Below | ● Maintain | ■ Slight | ▼ Horrible | ★ Career",
        xref="paper", yref="paper",
        x=0.5, y=-0.16,
        showarrow=False,
        font=dict(size=9, color='gray'),
        align='center'
    )

    return fig


def create_range_chart(projections: List[Dict]) -> go.Figure:
    """
    Create a range chart showing min/max/median scenarios.

    This provides a cleaner view of the projection ranges.

    Args:
        projections: List of player projection dictionaries

    Returns:
        Plotly figure with range visualization
    """
    fig = go.Figure()

    players = []
    medians = []
    ranges = []
    maintain_pace = []
    career_avg = []

    for player_data in projections:
        player_name = player_data['player_name']
        players.append(player_name)
        player_projections = player_data.get('projections', {})

        # Get all scenario values
        scenario_values = [
            proj.get('full_season_war', 0)
            for proj in player_projections.values()
        ]

        if scenario_values:
            # Calculate statistics
            medians.append(np.median(scenario_values))
            ranges.append([min(scenario_values), max(scenario_values)])

            # Get specific scenarios
            maintain = player_projections.get('100% (Maintain Pace)', {})
            maintain_pace.append(maintain.get('full_season_war', 0))

            career = player_projections.get('Career Average', {})
            career_avg.append(career.get('full_season_war', 0))
        else:
            # Handle missing data
            medians.append(0)
            ranges.append([0, 0])
            maintain_pace.append(0)
            career_avg.append(0)

    # Add range bars (min to max)
    for i, player in enumerate(players):
        fig.add_trace(go.Scatter(
            x=[player, player],
            y=ranges[i],
            mode='lines',
            line=dict(color='lightgray', width=20),
            showlegend=(i == 0),
            name='Range (Min-Max)',
            hovertemplate=f'<b>{player}</b><br>Range: {ranges[i][0]:.2f} - {ranges[i][1]:.2f}<extra></extra>'
        ))

    # Add median line
    fig.add_trace(go.Scatter(
        x=players,
        y=medians,
        mode='markers+lines',
        name='Median Projection',
        marker=dict(color='black', size=10, symbol='line-ew-open', line=dict(width=3)),
        line=dict(color='black', width=1, dash='dot'),
        hovertemplate='<b>%{x}</b><br>Median: %{y:.3f}<extra></extra>'
    ))

    # Add maintain pace points
    fig.add_trace(go.Scatter(
        x=players,
        y=maintain_pace,
        mode='markers',
        name='Maintain Pace (100%)',
        marker=dict(color='blue', size=12, symbol='circle'),
        hovertemplate='<b>%{x}</b><br>Maintain Pace: %{y:.3f}<extra></extra>'
    ))

    # Add career average points if available
    if any(career_avg):
        fig.add_trace(go.Scatter(
            x=players,
            y=career_avg,
            mode='markers',
            name='Career Average',
            marker=dict(color='purple', size=12, symbol='star'),
            hovertemplate='<b>%{x}</b><br>Career Average: %{y:.3f}<extra></extra>'
        ))

    # Update layout
    fig.update_layout(
        title={
            'text': 'Projection Ranges with Key Scenarios',
            'font': {'size': 18, 'color': '#333'},
            'x': 0.5,
            'xanchor': 'center'
        },
        yaxis_title="Projected WAR",
        xaxis_title="Player",
        height=450,
        template='plotly_white',
        hovermode='x unified',
        xaxis={'tickangle': 45},
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.25,
            xanchor="center",
            x=0.5
        )
    )

    return fig


def create_scenario_heatmap(
    projections: List[Dict],
    scenarios: Dict
) -> go.Figure:
    """
    Create heatmap showing all scenarios for all players.

    Args:
        projections: List of player projection dictionaries
        scenarios: Dictionary of scenario names

    Returns:
        Plotly figure with heatmap
    """
    # Prepare data matrix
    players = [p['player_name'] for p in projections]
    scenario_names = list(scenarios.keys())

    # Create matrix of WAR values
    war_matrix = []
    for scenario in scenario_names:
        row = []
        for player_data in projections:
            player_projections = player_data.get('projections', {})
            scenario_data = player_projections.get(scenario, {})
            value = scenario_data.get('full_season_war', 0)
            row.append(value)
        war_matrix.append(row)

    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=war_matrix,
        x=players,
        y=scenario_names,
        colorscale='RdYlGn',
        text=[[f'{val:.2f}' for val in row] for row in war_matrix],
        texttemplate='%{text}',
        textfont={"size": 10},
        colorbar=dict(title="WAR"),
        hovertemplate='<b>%{x}</b><br>%{y}<br>WAR: %{z:.3f}<extra></extra>'
    ))

    # Update layout
    fig.update_layout(
        title={
            'text': 'Projection Scenarios Heatmap - All Outcomes',
            'font': {'size': 18, 'color': '#333'},
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis_title="Player",
        yaxis_title="Scenario",
        height=400,
        template='plotly_white',
        xaxis={'tickangle': 45},
        margin=dict(l=150, r=50, t=80, b=80)
    )

    return fig


def format_war_display(war_value: float) -> str:
    """
    Format WAR value for display.

    Args:
        war_value: WAR value to format

    Returns:
        Formatted string
    """
    return f"{war_value:.3f}"