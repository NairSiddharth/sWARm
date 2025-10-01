"""
Shared utilities for oWAR visualizations.

This module provides common utilities, color schemes, and helper functions
used across all visualization modules to ensure consistency and reduce duplication.

Author: oWAR Development Team
"""

# Standard library imports
from typing import Dict, Tuple, Optional, List
import logging

# Third-party imports
import plotly.graph_objects as go
import numpy as np

# Project imports
from common_modules.logging import get_logger

# Module logger
logger = get_logger(__name__)

# Public API
__all__ = [
    # Functions
    'calculate_tier',
    'get_tier_color',
    'get_error_magnitude_color',
    'apply_standard_layout',
    'add_statistical_annotations',
    'create_perfect_prediction_line',
    'validate_projection_data',
    'validate_validation_data',
    'export_figure',
    'create_custom_legend',
    'create_error_figure',
    # Constants
    'SCENARIO_COLORS',
    'TIER_COLORS',
    'ERROR_MAGNITUDE_COLORS',
    'MODEL_TYPE_COLORS',
    'DEFAULT_TIER_BOUNDARIES'
]

# ========================
# Color Schemes
# ========================

SCENARIO_COLORS = {
    '150% (Hot Streak)': '#2ca02c',      # Green
    '125% (Above Pace)': '#98df8a',      # Light green
    '100% (Maintain Pace)': '#1f77b4',   # Blue
    '75% (Slight Regression)': '#ffbb78', # Light orange
    '50% (Major Regression)': '#ff9896',  # Light red
    '25% (Horrible Regression)': '#d62728', # Red
    'Career Average': '#9467bd'           # Purple
}

TIER_COLORS = {
    'elite': '#2ca02c',           # Green (>5 WAR)
    'all_star': '#1f77b4',        # Blue (3-5 WAR)
    'above_average': '#87ceeb',   # Light Blue (2-3 WAR)
    'average': '#808080',         # Gray (1-2 WAR)
    'below_average': '#ff8c00',   # Orange (0-1 WAR)
    'replacement': '#dc143c'      # Red (<0 WAR)
}

ERROR_MAGNITUDE_COLORS = {
    'extreme_high': '#8b0000',   # Dark red (>3 WAR error)
    'high': '#dc143c',           # Red (2-3 WAR error)
    'medium': '#ff8c00',         # Orange (1-2 WAR error)
    'low': '#ffd700',            # Gold (0.5-1 WAR error)
    'very_low': '#90ee90',       # Light green (0.25-0.5 WAR error)
    'minimal': '#228b22'         # Green (<0.25 WAR error)
}

MODEL_TYPE_COLORS = {
    'hitter_war': '#1f77b4',     # Blue
    'pitcher_war': '#ff7f0e',    # Orange
    'hitter_warp': '#2ca02c',    # Green
    'pitcher_warp': '#d62728'    # Red
}

# ========================
# Tier Classification
# ========================

DEFAULT_TIER_BOUNDARIES = {
    'elite': 5.0,
    'all_star': 3.0,
    'above_average': 2.0,
    'average': 1.0,
    'below_average': 0.0,
    'replacement': -1.0
}


def calculate_tier(
    war_value: float,
    tier_boundaries: Optional[Dict[str, float]] = None
) -> str:
    """
    Categorize player by WAR tier.

    Args:
        war_value: WAR value to categorize
        tier_boundaries: Optional custom tier boundaries

    Returns:
        Tier name as string
    """
    if tier_boundaries is None:
        tier_boundaries = DEFAULT_TIER_BOUNDARIES

    if war_value >= tier_boundaries['elite']:
        return 'elite'
    elif war_value >= tier_boundaries['all_star']:
        return 'all_star'
    elif war_value >= tier_boundaries['above_average']:
        return 'above_average'
    elif war_value >= tier_boundaries['average']:
        return 'average'
    elif war_value >= tier_boundaries['below_average']:
        return 'below_average'
    else:
        return 'replacement'


def get_tier_color(war_value: float) -> str:
    """
    Get color for a given WAR value based on tier.

    Args:
        war_value: WAR value

    Returns:
        Hex color string
    """
    tier = calculate_tier(war_value)
    return TIER_COLORS.get(tier, '#808080')


def get_error_magnitude_color(error: float) -> str:
    """
    Get color based on prediction error magnitude.

    Args:
        error: Absolute error value

    Returns:
        Hex color string
    """
    if error > 3:
        return ERROR_MAGNITUDE_COLORS['extreme_high']
    elif error > 2:
        return ERROR_MAGNITUDE_COLORS['high']
    elif error > 1:
        return ERROR_MAGNITUDE_COLORS['medium']
    elif error > 0.5:
        return ERROR_MAGNITUDE_COLORS['low']
    elif error > 0.25:
        return ERROR_MAGNITUDE_COLORS['very_low']
    else:
        return ERROR_MAGNITUDE_COLORS['minimal']


# ========================
# Layout and Styling
# ========================

def apply_standard_layout(
    fig: go.Figure,
    title: str,
    height: int = 500,
    template: str = 'plotly_white',
    showlegend: bool = True
) -> go.Figure:
    """
    Apply consistent styling to all charts.

    Args:
        fig: Plotly figure to style
        title: Chart title
        height: Chart height in pixels
        template: Plotly template name
        showlegend: Whether to show legend

    Returns:
        Styled figure
    """
    fig.update_layout(
        title={
            'text': title,
            'font': {'size': 18, 'color': '#333'},
            'x': 0.5,
            'xanchor': 'center'
        },
        height=height,
        template=template,
        showlegend=showlegend,
        font={'family': 'Arial, sans-serif', 'size': 12},
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_family="Arial"
        ),
        margin=dict(l=60, r=60, t=80, b=60)
    )

    # Update grid styling
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='lightgray',
        zeroline=True,
        zerolinewidth=2,
        zerolinecolor='gray'
    )

    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='lightgray',
        zeroline=True,
        zerolinewidth=2,
        zerolinecolor='gray'
    )

    return fig


# ========================
# Statistical Annotations
# ========================

def add_statistical_annotations(
    fig: go.Figure,
    actual: np.ndarray,
    predicted: np.ndarray,
    position: Tuple[float, float],
    xref: str = 'paper',
    yref: str = 'paper'
) -> None:
    """
    Add R², RMSE, MAE annotations to plot.

    Args:
        fig: Plotly figure
        actual: Actual values array
        predicted: Predicted values array
        position: (x, y) position for annotation
        xref: X reference system
        yref: Y reference system
    """
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

    # Calculate metrics
    r2 = r2_score(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae = mean_absolute_error(actual, predicted)

    # Create annotation text
    annotation_text = (
        f'R² = {r2:.3f}<br>'
        f'RMSE = {rmse:.2f}<br>'
        f'MAE = {mae:.2f}'
    )

    # Add annotation
    fig.add_annotation(
        text=annotation_text,
        xref=xref,
        yref=yref,
        x=position[0],
        y=position[1],
        showarrow=False,
        bgcolor='rgba(255, 255, 255, 0.8)',
        bordercolor='gray',
        borderwidth=1,
        font=dict(size=10, color='black')
    )


def create_perfect_prediction_line(
    min_val: float,
    max_val: float,
    num_points: int = 100
) -> Dict:
    """
    Create y=x reference line for scatter plots.

    Args:
        min_val: Minimum value for line
        max_val: Maximum value for line
        num_points: Number of points to generate

    Returns:
        Dictionary with x and y values for perfect prediction line
    """
    values = np.linspace(min_val, max_val, num_points)
    return {'x': values, 'y': values}


# ========================
# Data Validation
# ========================

def validate_projection_data(projections: List[Dict]) -> Tuple[bool, List[str]]:
    """
    Validate projection data structure.

    Args:
        projections: List of projection dictionaries

    Returns:
        Tuple of (is_valid, list of issues)
    """
    issues = []

    if not projections:
        issues.append("No projections provided")
        return False, issues

    required_fields = ['player_name', 'current_war', 'projections']

    for i, proj in enumerate(projections):
        for field in required_fields:
            if field not in proj:
                issues.append(f"Player {i}: missing field '{field}'")

        # Check projections structure
        if 'projections' in proj:
            if not isinstance(proj['projections'], dict):
                issues.append(f"Player {i}: 'projections' must be a dictionary")
            elif not proj['projections']:
                issues.append(f"Player {i}: 'projections' is empty")

    return len(issues) == 0, issues


def validate_validation_data(validation_data: Dict) -> Tuple[bool, List[str]]:
    """
    Validate validation data structure.

    Args:
        validation_data: Dictionary with validation results

    Returns:
        Tuple of (is_valid, list of issues)
    """
    issues = []

    if not validation_data:
        issues.append("No validation data provided")
        return False, issues

    expected_keys = ['hitter_war', 'pitcher_war', 'hitter_warp', 'pitcher_warp']
    required_fields = ['actual', 'predicted']

    for key in expected_keys:
        if key not in validation_data:
            logger.info(f"Optional validation data missing: {key}")
            continue

        data = validation_data[key]
        for field in required_fields:
            if field not in data:
                issues.append(f"{key}: missing field '{field}'")
            elif not isinstance(data[field], (list, np.ndarray)):
                issues.append(f"{key}: '{field}' must be a list or array")
            elif len(data[field]) == 0:
                issues.append(f"{key}: '{field}' is empty")

    return len(issues) == 0, issues


# ========================
# Export Utilities
# ========================

def export_figure(
    fig: go.Figure,
    filename: str,
    format: str = 'html',
    width: Optional[int] = None,
    height: Optional[int] = None
) -> bool:
    """
    Export figure to file.

    Args:
        fig: Plotly figure to export
        filename: Output filename
        format: Export format ('html', 'png', 'svg', 'pdf')
        width: Width for image exports
        height: Height for image exports

    Returns:
        True if successful, False otherwise
    """
    try:
        if format == 'html':
            fig.write_html(filename)
        elif format in ['png', 'svg', 'pdf', 'jpeg']:
            fig.write_image(
                filename,
                format=format,
                width=width,
                height=height
            )
        else:
            logger.error(f"Unsupported export format: {format}")
            return False

        logger.info(f"Exported figure to {filename}")
        return True

    except Exception as e:
        logger.error(f"Failed to export figure: {e}")
        return False


# ========================
# Legend Utilities
# ========================

def create_custom_legend(
    fig: go.Figure,
    legend_items: Dict[str, str],
    position: str = 'top-right'
) -> None:
    """
    Add custom legend to figure.

    Args:
        fig: Plotly figure
        legend_items: Dictionary of {label: color}
        position: Legend position
    """
    # Define position mappings
    positions = {
        'top-right': dict(x=1, y=1, xanchor='right', yanchor='top'),
        'top-left': dict(x=0, y=1, xanchor='left', yanchor='top'),
        'bottom-right': dict(x=1, y=0, xanchor='right', yanchor='bottom'),
        'bottom-left': dict(x=0, y=0, xanchor='left', yanchor='bottom'),
        'center': dict(x=0.5, y=0.5, xanchor='center', yanchor='middle')
    }

    legend_config = positions.get(position, positions['top-right'])

    # Add invisible traces for legend
    for label, color in legend_items.items():
        fig.add_trace(go.Scatter(
            x=[None],
            y=[None],
            mode='markers',
            name=label,
            marker=dict(size=10, color=color),
            showlegend=True
        ))

    # Update legend layout
    fig.update_layout(
        legend=dict(
            **legend_config,
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='gray',
            borderwidth=1
        )
    )


# ========================
# Error Handling
# ========================

def create_error_figure(
    error_message: str,
    height: int = 400
) -> go.Figure:
    """
    Create a figure displaying an error message.

    Args:
        error_message: Error message to display
        height: Figure height

    Returns:
        Plotly figure with error message
    """
    fig = go.Figure()

    fig.add_annotation(
        text=f"❌ Error: {error_message}",
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=16, color="red"),
        bgcolor='rgba(255, 200, 200, 0.8)',
        bordercolor='red',
        borderwidth=2
    )

    fig.update_layout(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        template='plotly_white',
        height=height,
        title={
            'text': 'Visualization Error',
            'font': {'size': 18, 'color': 'red'},
            'x': 0.5,
            'xanchor': 'center'
        }
    )

    return fig