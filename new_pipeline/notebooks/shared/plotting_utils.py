"""
Plotting utilities for oWAR notebooks.

Provides interactive Plotly visualizations:
- WAR scatter plots (cumulative WAR vs usage)
- Actual vs predicted validation plots
- Residual analysis plots
- Feature importance charts
- Correlation heatmaps
- Partial dependence plots
"""

from typing import List, Dict, Optional
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.inspection import PartialDependenceDisplay


def create_war_scatter(
    df: pd.DataFrame,
    player_type: str,
    title: str,
    color_by: str = 'Type',
    hover_data: Optional[List[str]] = None
) -> go.Figure:
    """
    Create interactive cumulative WAR scatter plot.

    Args:
        df: Data with predictions
        player_type: 'pitcher' or 'hitter'
        title: Plot title
        color_by: Column to color by ('Type', 'Pos', etc.)
        hover_data: Columns to show on hover (default: ['Name', 'Team'])

    Returns:
        plotly.graph_objects.Figure: Interactive scatter plot

    Features:
        - X-axis: IP (pitchers) or PA (hitters)
        - Y-axis: Cumulative WAR
        - Reference lines: 0, 3, 6 WAR
        - Color by type/position
        - Scattergl for performance

    Example:
        >>> fig = create_war_scatter(
        ...     df=pitcher_predictions,
        ...     player_type='pitcher',
        ...     title="2025 Pitcher Projections",
        ...     color_by='Type',
        ...     hover_data=['Name', 'Team', 'ERA', 'K%']
        ... )
        >>> fig.show()
    """
    if player_type not in ['pitcher', 'hitter']:
        raise ValueError(f"player_type must be 'pitcher' or 'hitter', got: {player_type}")

    # Determine X-axis column (try projection columns first, fallback to raw data)
    if player_type == 'pitcher':
        if 'Total_Projected_IP' in df.columns:
            x_col = 'Total_Projected_IP'
        elif 'IP' in df.columns:
            x_col = 'IP'
        else:
            raise ValueError("No IP column found (tried 'Total_Projected_IP', 'IP')")
        x_label = 'Innings Pitched'
    else:
        if 'Total_Projected_PA' in df.columns:
            x_col = 'Total_Projected_PA'
        elif 'PA' in df.columns:
            x_col = 'PA'
        else:
            raise ValueError("No PA column found (tried 'Total_Projected_PA', 'PA')")
        x_label = 'Plate Appearances'

    # Determine WAR column
    if 'Total_Projected_WAR' in df.columns:
        war_col = 'Total_Projected_WAR'
    elif 'WAR' in df.columns:
        war_col = 'WAR'
    else:
        raise ValueError("No WAR column found (tried 'Total_Projected_WAR', 'WAR')")

    # Default hover data
    if hover_data is None:
        hover_data = ['Name', 'Team'] if 'Name' in df.columns and 'Team' in df.columns else []

    # Create figure using scattergl for performance
    fig = px.scatter(
        df,
        x=x_col,
        y=war_col,
        color=color_by if color_by in df.columns else None,
        hover_data=hover_data,
        title=title,
        labels={
            x_col: x_label,
            war_col: 'WAR',
            color_by: color_by
        },
        render_mode='webgl'  # Use scattergl for performance
    )

    # Add reference lines (0, 3, 6 WAR)
    for war_threshold in [0, 3, 6]:
        fig.add_hline(
            y=war_threshold,
            line_dash='dash',
            line_color='gray',
            opacity=0.5,
            annotation_text=f'{war_threshold} WAR',
            annotation_position='right'
        )

    # Update layout
    fig.update_layout(
        hovermode='closest',
        template='plotly_white',
        height=600,
        showlegend=True
    )

    return fig


def create_actual_vs_predicted(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    color_by: Optional[np.ndarray] = None,
    title: str = 'Actual vs Predicted WAR'
) -> go.Figure:
    """
    Create actual vs predicted scatter plot.

    Args:
        y_true: Actual WAR values
        y_pred: Predicted WAR values
        color_by: Group labels for coloring (optional)
        title: Plot title

    Returns:
        plotly.graph_objects.Figure: Scatter plot with diagonal line

    Features:
        - Diagonal line (perfect predictions)
        - Color by role/position
        - Point size proportional to magnitude

    Example:
        >>> fig = create_actual_vs_predicted(
        ...     y_true=y_val,
        ...     y_pred=predictions,
        ...     color_by=roles,
        ...     title="2024 Validation"
        ... )
    """
    # Create DataFrame for plotting
    plot_data = pd.DataFrame({
        'Actual': y_true,
        'Predicted': y_pred
    })

    if color_by is not None:
        plot_data['Group'] = color_by

    # Create scatter plot
    if color_by is not None:
        fig = px.scatter(
            plot_data,
            x='Actual',
            y='Predicted',
            color='Group',
            title=title,
            labels={'Actual': 'Actual WAR', 'Predicted': 'Predicted WAR'},
            render_mode='webgl'
        )
    else:
        fig = px.scatter(
            plot_data,
            x='Actual',
            y='Predicted',
            title=title,
            labels={'Actual': 'Actual WAR', 'Predicted': 'Predicted WAR'},
            render_mode='webgl'
        )

    # Add diagonal reference line (perfect predictions)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())

    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        line=dict(color='red', dash='dash'),
        name='Perfect Prediction',
        showlegend=True
    ))

    # Update layout
    fig.update_layout(
        template='plotly_white',
        height=600,
        hovermode='closest'
    )

    # Equal aspect ratio
    fig.update_xaxes(scaleanchor='y', scaleratio=1)

    return fig


def create_residual_plot(
    residuals: np.ndarray,
    color_by: Optional[np.ndarray] = None,
    title: str = 'Residual Plot'
) -> go.Figure:
    """
    Create residual plot with marginals.

    Args:
        residuals: Prediction residuals (actual - predicted)
        color_by: Group labels (optional)
        title: Plot title

    Returns:
        plotly.graph_objects.Figure: Residual plot with marginal histograms

    Features:
        - Main: Residuals vs predicted
        - Right marginal: Residual distribution
        - Top marginal: Predicted distribution (if provided)
        - Reference line at 0

    Example:
        >>> fig = create_residual_plot(
        ...     residuals=y_true - y_pred,
        ...     color_by=positions,
        ...     title="Residuals by Position"
        ... )
    """
    # Create DataFrame
    plot_data = pd.DataFrame({
        'Index': np.arange(len(residuals)),
        'Residual': residuals
    })

    if color_by is not None:
        plot_data['Group'] = color_by

    # Create figure with marginal histogram
    if color_by is not None:
        fig = px.scatter(
            plot_data,
            x='Index',
            y='Residual',
            color='Group',
            marginal_y='histogram',
            title=title,
            labels={'Index': 'Sample Index', 'Residual': 'Residual (Actual - Predicted)'},
            render_mode='webgl'
        )
    else:
        fig = px.scatter(
            plot_data,
            x='Index',
            y='Residual',
            marginal_y='histogram',
            title=title,
            labels={'Index': 'Sample Index', 'Residual': 'Residual (Actual - Predicted)'},
            render_mode='webgl'
        )

    # Add reference line at 0
    fig.add_hline(
        y=0,
        line_dash='dash',
        line_color='red',
        opacity=0.7,
        annotation_text='Zero Error',
        annotation_position='right'
    )

    # Update layout
    fig.update_layout(
        template='plotly_white',
        height=600,
        hovermode='closest'
    )

    return fig


def create_feature_importance(
    importance_dict: Dict[str, float],
    top_n: int = 10,
    title: str = 'Feature Importance'
) -> go.Figure:
    """
    Create feature importance bar chart.

    Args:
        importance_dict: {feature: importance} mapping
        top_n: Number of top features to show
        title: Plot title

    Returns:
        plotly.graph_objects.Figure: Horizontal bar chart

    Example:
        >>> importance = {'K%': 0.25, 'ERA': 0.18, 'BB%': 0.12, ...}
        >>> fig = create_feature_importance(importance, top_n=10, title="Top 10 Features")
    """
    # Sort by importance and get top N
    sorted_items = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]

    features = [item[0] for item in sorted_items]
    importances = [item[1] for item in sorted_items]

    # Reverse for horizontal bar chart (highest at top)
    features = features[::-1]
    importances = importances[::-1]

    # Create horizontal bar chart
    fig = go.Figure(data=[
        go.Bar(
            x=importances,
            y=features,
            orientation='h',
            marker=dict(
                color=importances,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title='Importance')
            )
        )
    ])

    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title='Importance',
        yaxis_title='Feature',
        template='plotly_white',
        height=400 + (top_n * 20),  # Scale height with number of features
        showlegend=False
    )

    return fig


def create_correlation_heatmap(
    df: pd.DataFrame,
    features: List[str]
) -> go.Figure:
    """
    Create correlation matrix heatmap.

    Args:
        df: Data with features
        features: Feature columns to correlate

    Returns:
        plotly.graph_objects.Figure: Heatmap

    Example:
        >>> fig = create_correlation_heatmap(df, features=['K%', 'BB%', 'ERA', 'GB%'])
    """
    # Validate features exist
    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        raise ValueError(f"Features not found in DataFrame: {missing_features}")

    # Calculate correlation matrix
    corr_matrix = df[features].corr()

    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=corr_matrix.values.round(2),
        texttemplate='%{text}',
        textfont={"size": 10},
        colorbar=dict(title='Correlation')
    ))

    # Update layout
    fig.update_layout(
        title='Feature Correlation Matrix',
        xaxis_title='',
        yaxis_title='',
        template='plotly_white',
        height=600,
        width=700
    )

    return fig


def create_partial_dependence(
    model,
    X,
    feature: str,
    feature_names: Optional[List[str]] = None
) -> go.Figure:
    """
    Create partial dependence plot for a feature.

    Args:
        model: Trained model (sklearn estimator)
        X: Feature data (DataFrame or numpy array)
        feature: Feature to analyze (name or index)
        feature_names: Feature names (required if X is numpy array)

    Returns:
        plotly.graph_objects.Figure: Partial dependence plot

    Example:
        >>> # With DataFrame
        >>> fig = create_partial_dependence(model, X_train_df, feature='K%')
        >>>
        >>> # With numpy array
        >>> fig = create_partial_dependence(model, X_train, feature='K%',
        ...                                 feature_names=['K%', 'BB%', 'ERA'])
    """
    # Determine feature index
    if isinstance(X, pd.DataFrame):
        if feature not in X.columns:
            raise ValueError(f"Feature '{feature}' not found in DataFrame columns")
        feature_idx = list(X.columns).index(feature)
        X_array = X.values
    elif isinstance(X, np.ndarray):
        if feature_names is None:
            raise ValueError("feature_names must be provided when X is numpy array")
        if feature not in feature_names:
            raise ValueError(f"Feature '{feature}' not found in feature_names")
        feature_idx = feature_names.index(feature)
        X_array = X
    else:
        raise TypeError(f"X must be DataFrame or numpy array, got: {type(X)}")

    # Calculate partial dependence using sklearn
    try:
        from sklearn.inspection import partial_dependence

        # Calculate PD
        pd_result = partial_dependence(
            model,
            X_array,
            features=[feature_idx],
            grid_resolution=50
        )

        # Extract values
        pd_values = pd_result['average'][0]
        grid_values = pd_result['grid_values'][0]

    except Exception as e:
        raise RuntimeError(f"Failed to calculate partial dependence: {e}")

    # Create line plot
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=grid_values,
        y=pd_values,
        mode='lines',
        line=dict(color='blue', width=2),
        name='Partial Dependence'
    ))

    # Update layout
    fig.update_layout(
        title=f'Partial Dependence: {feature}',
        xaxis_title=feature,
        yaxis_title='Partial Dependence (WAR)',
        template='plotly_white',
        height=500,
        hovermode='x unified'
    )

    return fig
