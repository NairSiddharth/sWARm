"""
Model Validation Visualizations for oWAR System

This module provides comprehensive validation visualizations including:
- Actual vs Predicted scatter plots with facets
- Residual analysis plots
- Cross-validation performance metrics
- Feature importance comparisons
- Tier-based performance analysis

Author: oWAR Development Team
"""

# Standard library imports
from typing import Dict, List, Tuple, Optional, Union
import logging

# Third-party imports
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy import stats
from scipy.interpolate import interp1d

# Project imports
from common_modules.logging import get_logger

# Module logger
logger = get_logger(__name__)

# Public API
__all__ = [
    'create_validation_dashboard',
    'create_actual_vs_predicted_facet',
    'create_residual_analysis_facet',
    'create_cv_performance_plot',
    'create_tier_based_performance',
    'get_error_color',
    'get_tier_color',
    'create_error_figure'
]

# Color schemes
ERROR_MAGNITUDE_COLORS = {
    'extreme_high': '#8b0000',  # > 3 WAR error
    'high': '#dc143c',          # 2-3 WAR error
    'medium': '#ff8c00',         # 1-2 WAR error
    'low': '#ffd700',            # 0.5-1 WAR error
    'very_low': '#90ee90',       # 0.25-0.5 WAR error
    'minimal': '#228b22'         # < 0.25 WAR error
}

TIER_COLORS = {
    'elite': '#2ca02c',          # Green (>5 WAR)
    'all_star': '#1f77b4',       # Blue (3-5 WAR)
    'above_average': '#87ceeb',   # Light Blue (2-3 WAR)
    'average': '#808080',         # Gray (1-2 WAR)
    'below_average': '#ff8c00',   # Orange (0-1 WAR)
    'replacement': '#dc143c'      # Red (<0 WAR)
}


def create_validation_dashboard(
    validation_results: Dict,
    model_name: str = "Ensemble",
    include_residuals: bool = True,
    include_cv: bool = True,
    include_tiers: bool = True
) -> Dict[str, go.Figure]:
    """
    Create complete validation dashboard with all visualization types.

    Args:
        validation_results: Dictionary containing validation data for each model type
        model_name: Name of model for titles
        include_residuals: Whether to include residual analysis
        include_cv: Whether to include cross-validation plots
        include_tiers: Whether to include tier-based analysis

    Returns:
        Dictionary of Plotly figures with keys:
        - 'actual_vs_predicted': Main scatter plot with facets
        - 'residuals': Residual analysis (if included)
        - 'cv_performance': Cross-validation metrics (if included)
        - 'tier_analysis': Performance by player tier (if included)
    """
    logger.info(f"Creating validation dashboard for {model_name}")

    figures = {}

    # Create actual vs predicted plot (always included)
    figures['actual_vs_predicted'] = create_actual_vs_predicted_facet(
        validation_results, model_name
    )

    # Add optional plots
    if include_residuals:
        figures['residuals'] = create_residual_analysis_facet(
            validation_results, model_name
        )

    if include_cv and 'cv_scores' in validation_results.get('hitter_war', {}):
        figures['cv_performance'] = create_cv_performance_plot(
            validation_results, model_name
        )

    if include_tiers:
        figures['tier_analysis'] = create_tier_based_performance(
            validation_results, model_name
        )

    logger.info(f"Created {len(figures)} validation figures")
    return figures


def create_actual_vs_predicted_facet(
    validation_data: Dict,
    model_name: str = "Ensemble"
) -> go.Figure:
    """
    Create 2x2 faceted scatter plot of actual vs predicted values.

    Features:
    - Perfect prediction line (y=x)
    - Points colored by error magnitude
    - R² and RMSE annotations
    - Interactive hover with details
    """
    # Create subplot titles
    subplot_titles = (
        'Hitter WAR', 'Pitcher WAR',
        'Hitter WARP', 'Pitcher WARP'
    )

    # Create figure with 2x2 subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.15,
        vertical_spacing=0.15
    )

    # Define subplot positions
    positions = [
        (1, 1, 'hitter_war'),
        (1, 2, 'pitcher_war'),
        (2, 1, 'hitter_warp'),
        (2, 2, 'pitcher_warp')
    ]

    # Track min/max for perfect prediction line
    overall_min = float('inf')
    overall_max = float('-inf')

    # Add data to each subplot
    for row, col, data_key in positions:
        if data_key not in validation_data:
            logger.warning(f"Missing validation data for {data_key}")
            continue

        data = validation_data[data_key]
        actual = np.array(data.get('actual', []))
        predicted = np.array(data.get('predicted', []))

        if len(actual) == 0 or len(predicted) == 0:
            continue

        # Calculate error metrics
        errors = np.abs(actual - predicted)
        r2 = r2_score(actual, predicted)
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        mae = mean_absolute_error(actual, predicted)

        # Update min/max
        overall_min = min(overall_min, actual.min(), predicted.min())
        overall_max = max(overall_max, actual.max(), predicted.max())

        # Color points by error magnitude
        colors = [get_error_color(e) for e in errors]

        # Add scatter plot
        fig.add_trace(
            go.Scatter(
                x=predicted,
                y=actual,
                mode='markers',
                marker=dict(
                    color=colors,
                    size=8,
                    line=dict(width=0.5, color='white')
                ),
                text=[f"Error: {e:.2f}" for e in errors],
                hovertemplate=(
                    'Predicted: %{x:.2f}<br>'
                    'Actual: %{y:.2f}<br>'
                    '%{text}<br>'
                    '<extra></extra>'
                ),
                showlegend=False
            ),
            row=row, col=col
        )

        # Add metrics annotation
        fig.add_annotation(
            text=(
                f'R² = {r2:.3f}<br>'
                f'RMSE = {rmse:.2f}<br>'
                f'MAE = {mae:.2f}'
            ),
            xref=f'x{(row-1)*2 + col}',
            yref=f'y{(row-1)*2 + col}',
            x=predicted.max() * 0.95,
            y=actual.min() + (actual.max() - actual.min()) * 0.1,
            showarrow=False,
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='gray',
            borderwidth=1,
            font=dict(size=10)
        )

    # Add perfect prediction line to all subplots
    perfect_line = np.linspace(overall_min, overall_max, 100)
    for row, col, _ in positions:
        fig.add_trace(
            go.Scatter(
                x=perfect_line,
                y=perfect_line,
                mode='lines',
                line=dict(color='red', width=2, dash='dash'),
                showlegend=False,
                hoverinfo='skip'
            ),
            row=row, col=col
        )

    # Update layout
    fig.update_layout(
        title={
            'text': f'{model_name} Model: Actual vs Predicted Values',
            'font': {'size': 20},
            'x': 0.5,
            'xanchor': 'center'
        },
        height=700,
        template='plotly_white',
        showlegend=False
    )

    # Update axes labels
    for i in range(1, 5):
        fig.update_xaxes(title_text="Predicted", row=(i-1)//2 + 1, col=(i-1)%2 + 1)
        fig.update_yaxes(title_text="Actual", row=(i-1)//2 + 1, col=(i-1)%2 + 1)

    return fig


def create_residual_analysis_facet(
    validation_data: Dict,
    model_name: str = "Ensemble"
) -> go.Figure:
    """
    Create 2x2 faceted residual plots.

    Features:
    - Residuals vs fitted values
    - LOESS smoothing to identify trends
    - Zero reference line
    - Heteroscedasticity detection
    """
    # Create subplot titles
    subplot_titles = (
        'Hitter WAR Residuals', 'Pitcher WAR Residuals',
        'Hitter WARP Residuals', 'Pitcher WARP Residuals'
    )

    # Create figure with 2x2 subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.15,
        vertical_spacing=0.15
    )

    # Define subplot positions
    positions = [
        (1, 1, 'hitter_war'),
        (1, 2, 'pitcher_war'),
        (2, 1, 'hitter_warp'),
        (2, 2, 'pitcher_warp')
    ]

    # Add data to each subplot
    for row, col, data_key in positions:
        if data_key not in validation_data:
            continue

        data = validation_data[data_key]
        actual = np.array(data.get('actual', []))
        predicted = np.array(data.get('predicted', []))

        if len(actual) == 0 or len(predicted) == 0:
            continue

        # Calculate residuals
        residuals = actual - predicted

        # Color by player tier (based on actual value)
        colors = [get_tier_color(a) for a in actual]

        # Add scatter plot
        fig.add_trace(
            go.Scatter(
                x=predicted,
                y=residuals,
                mode='markers',
                marker=dict(
                    color=colors,
                    size=6,
                    opacity=0.6
                ),
                hovertemplate=(
                    'Fitted: %{x:.2f}<br>'
                    'Residual: %{y:.2f}<br>'
                    '<extra></extra>'
                ),
                showlegend=False
            ),
            row=row, col=col
        )

        # Add zero line
        fig.add_trace(
            go.Scatter(
                x=[predicted.min(), predicted.max()],
                y=[0, 0],
                mode='lines',
                line=dict(color='black', width=1, dash='dash'),
                showlegend=False,
                hoverinfo='skip'
            ),
            row=row, col=col
        )

        # Add LOESS smoothing if enough points
        if len(predicted) > 20:
            try:
                sorted_idx = np.argsort(predicted)
                sorted_pred = predicted[sorted_idx]
                sorted_res = residuals[sorted_idx]

                # Simple moving average as LOESS approximation
                window = max(5, len(predicted) // 10)
                smoothed = pd.Series(sorted_res).rolling(window, center=True).mean()

                fig.add_trace(
                    go.Scatter(
                        x=sorted_pred,
                        y=smoothed,
                        mode='lines',
                        line=dict(color='red', width=2),
                        name='Trend',
                        showlegend=False,
                        hoverinfo='skip'
                    ),
                    row=row, col=col
                )
            except Exception as e:
                logger.warning(f"Could not add LOESS smoothing: {e}")

    # Update layout
    fig.update_layout(
        title={
            'text': f'{model_name} Model: Residual Analysis',
            'font': {'size': 20},
            'x': 0.5,
            'xanchor': 'center'
        },
        height=700,
        template='plotly_white'
    )

    # Update axes labels
    for i in range(1, 5):
        fig.update_xaxes(title_text="Fitted Values", row=(i-1)//2 + 1, col=(i-1)%2 + 1)
        fig.update_yaxes(title_text="Residuals", row=(i-1)//2 + 1, col=(i-1)%2 + 1)

    return fig


def create_cv_performance_plot(
    validation_data: Dict,
    model_name: str = "Ensemble"
) -> go.Figure:
    """
    Create cross-validation stability visualization.

    Shows consistency of model performance across different folds.
    """
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('R² Scores', 'RMSE', 'MAE', 'Fold Comparison'),
        specs=[[{"type": "box"}, {"type": "box"}],
               [{"type": "box"}, {"type": "scatter"}]]
    )

    model_types = ['hitter_war', 'pitcher_war', 'hitter_warp', 'pitcher_warp']
    colors = ['blue', 'red', 'green', 'orange']

    # Collect CV scores for each model type
    r2_scores = []
    rmse_scores = []
    mae_scores = []

    for model_type, color in zip(model_types, colors):
        if model_type in validation_data:
            cv_data = validation_data[model_type].get('cv_scores', {})
            if cv_data:
                # Add R² scores
                if 'r2' in cv_data:
                    fig.add_trace(
                        go.Box(
                            y=cv_data['r2'],
                            name=model_type.replace('_', ' ').title(),
                            marker_color=color,
                            boxmean='sd'
                        ),
                        row=1, col=1
                    )

                # Add RMSE scores
                if 'rmse' in cv_data:
                    fig.add_trace(
                        go.Box(
                            y=cv_data['rmse'],
                            name=model_type.replace('_', ' ').title(),
                            marker_color=color,
                            boxmean='sd'
                        ),
                        row=1, col=2
                    )

                # Add MAE scores
                if 'mae' in cv_data:
                    fig.add_trace(
                        go.Box(
                            y=cv_data['mae'],
                            name=model_type.replace('_', ' ').title(),
                            marker_color=color,
                            boxmean='sd'
                        ),
                        row=2, col=1
                    )

                # Add fold-by-fold comparison
                if 'r2' in cv_data:
                    fold_numbers = list(range(1, len(cv_data['r2']) + 1))
                    fig.add_trace(
                        go.Scatter(
                            x=fold_numbers,
                            y=cv_data['r2'],
                            mode='lines+markers',
                            name=model_type.replace('_', ' ').title(),
                            line=dict(color=color)
                        ),
                        row=2, col=2
                    )

    # Update layout
    fig.update_layout(
        title={
            'text': f'{model_name} Model: Cross-Validation Performance',
            'font': {'size': 20},
            'x': 0.5,
            'xanchor': 'center'
        },
        height=700,
        template='plotly_white',
        showlegend=True
    )

    # Update axes labels
    fig.update_yaxes(title_text="R² Score", row=1, col=1)
    fig.update_yaxes(title_text="RMSE", row=1, col=2)
    fig.update_yaxes(title_text="MAE", row=2, col=1)
    fig.update_xaxes(title_text="Fold", row=2, col=2)
    fig.update_yaxes(title_text="R² Score", row=2, col=2)

    return fig


def create_tier_based_performance(
    validation_data: Dict,
    model_name: str = "Ensemble",
    tier_boundaries: Optional[Dict] = None
) -> go.Figure:
    """
    Analyze performance by player tier.

    Shows how well the model performs for different quality levels of players.
    """
    if tier_boundaries is None:
        tier_boundaries = {
            'elite': 5.0,
            'all_star': 3.0,
            'above_average': 2.0,
            'average': 1.0,
            'below_average': 0.0,
            'replacement': -1.0
        }

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Hitter WAR', 'Pitcher WAR', 'Hitter WARP', 'Pitcher WARP')
    )

    positions = [
        (1, 1, 'hitter_war'),
        (1, 2, 'pitcher_war'),
        (2, 1, 'hitter_warp'),
        (2, 2, 'pitcher_warp')
    ]

    for row, col, data_key in positions:
        if data_key not in validation_data:
            continue

        data = validation_data[data_key]
        actual = np.array(data.get('actual', []))
        predicted = np.array(data.get('predicted', []))

        if len(actual) == 0:
            continue

        # Categorize by tier
        tier_metrics = {}
        for tier_name, min_val in tier_boundaries.items():
            if tier_name == 'replacement':
                mask = actual < tier_boundaries['below_average']
            else:
                max_val = next((v for k, v in tier_boundaries.items()
                              if v > min_val), float('inf'))
                mask = (actual >= min_val) & (actual < max_val)

            if mask.sum() > 0:
                tier_actual = actual[mask]
                tier_predicted = predicted[mask]

                tier_metrics[tier_name] = {
                    'r2': r2_score(tier_actual, tier_predicted) if len(tier_actual) > 1 else 0,
                    'mae': mean_absolute_error(tier_actual, tier_predicted),
                    'count': len(tier_actual)
                }

        # Create bar chart for this model type
        tiers = list(tier_metrics.keys())
        r2_values = [tier_metrics[t]['r2'] for t in tiers]
        counts = [tier_metrics[t]['count'] for t in tiers]

        fig.add_trace(
            go.Bar(
                x=tiers,
                y=r2_values,
                text=[f'n={c}' for c in counts],
                textposition='auto',
                marker_color=[TIER_COLORS.get(t, 'gray') for t in tiers],
                showlegend=False,
                hovertemplate=(
                    'Tier: %{x}<br>'
                    'R²: %{y:.3f}<br>'
                    'Count: %{text}<br>'
                    '<extra></extra>'
                )
            ),
            row=row, col=col
        )

    # Update layout
    fig.update_layout(
        title={
            'text': f'{model_name} Model: Performance by Player Tier',
            'font': {'size': 20},
            'x': 0.5,
            'xanchor': 'center'
        },
        height=700,
        template='plotly_white'
    )

    # Update axes
    for i in range(1, 5):
        fig.update_yaxes(title_text="R² Score", row=(i-1)//2 + 1, col=(i-1)%2 + 1)
        fig.update_xaxes(title_text="Player Tier", row=(i-1)//2 + 1, col=(i-1)%2 + 1)

    return fig


# Utility functions

def get_error_color(error: float) -> str:
    """Get color based on error magnitude."""
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


def get_tier_color(war_value: float) -> str:
    """Get color based on WAR tier."""
    if war_value > 5:
        return TIER_COLORS['elite']
    elif war_value > 3:
        return TIER_COLORS['all_star']
    elif war_value > 2:
        return TIER_COLORS['above_average']
    elif war_value > 1:
        return TIER_COLORS['average']
    elif war_value > 0:
        return TIER_COLORS['below_average']
    else:
        return TIER_COLORS['replacement']


def create_error_figure(error_message: str) -> go.Figure:
    """Create a figure displaying an error message."""
    fig = go.Figure()
    fig.add_annotation(
        text=f"Error: {error_message}",
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=16, color="red")
    )
    fig.update_layout(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        template='plotly_white',
        height=400
    )
    return fig