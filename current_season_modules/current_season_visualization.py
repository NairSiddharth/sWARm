"""
Current Season Visualization Module - Real-time WAR/WARP Analysis Visualizations
Evolved from temp_data_visualization.py to support current season analysis and scenario projections
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.metrics import r2_score, mean_squared_error
from typing import Dict, List, Optional, Tuple
import sys
import os

# Import scenario projections for visualization
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


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

    # Extract historical years and values
    years = [d.get('year', 0) for d in historical_data]
    historical_war = [d.get('war', 0) for d in historical_data]
    historical_warp = [d.get('warp', 0) for d in historical_data]

    # Add current and projected points
    current_year = max(years) + 1 if years else 2025
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


def create_player_comparison_dashboard(players_data: Dict[str, Dict]) -> go.Figure:
    """
    Create dashboard comparing multiple players' projections

    Args:
        players_data: Dictionary with player names as keys and their projection data

    Returns:
        Plotly figure with multi-player comparison
    """

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


# Backward compatibility functions for existing K-fold visualization

def plot_year_specific_analysis(cv_results, year, models_to_show=None):
    """
    Create year-specific 4-subplot graph showing predicted vs actual for a single year

    Maintained for backward compatibility with existing sWARm_CS.ipynb
    """
    print(f"Creating year-specific analysis for {year}...")

    # Get data for the specific year
    year_data = cv_results.get_year_data(year)

    if not year_data:
        print(f"No data available for year {year}")
        return None

    # Determine which models and categories have data
    available_categories = set()
    available_models = set()

    for key in year_data.keys():
        if len(year_data[key]['y_true']) > 0:
            model_name, player_type, metric_type = key.split('_')
            available_categories.add(f"{player_type.title()} {metric_type.upper()}")
            available_models.add(model_name.title())

    if models_to_show:
        available_models = available_models.intersection(set([m.lower() for m in models_to_show]))

    print(f"  Available categories for {year}: {sorted(available_categories)}")
    print(f"  Available models for {year}: {sorted(available_models)}")

    # Create subplot structure (2x2 grid)
    category_order = ['Hitter WAR', 'Hitter WARP', 'Pitcher WAR', 'Pitcher WARP']
    available_categories_ordered = [cat for cat in category_order if cat in available_categories]

    if len(available_categories_ordered) == 0:
        print(f"No valid categories found for {year}")
        return None

    # Create subplots
    n_categories = len(available_categories_ordered)
    cols = 2
    rows = (n_categories + 1) // 2

    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=available_categories_ordered,
        vertical_spacing=0.15,
        horizontal_spacing=0.15
    )

    colors = ['blue', 'red', 'green', 'purple']

    for i, category in enumerate(available_categories_ordered):
        row = (i // cols) + 1
        col = (i % cols) + 1

        player_type = category.split()[0].lower()
        metric_type = category.split()[1].lower()

        # Plot each model for this category
        for j, model in enumerate(sorted(available_models)):
            key = f"{model.lower()}_{player_type}_{metric_type}"

            if key in year_data and len(year_data[key]['y_true']) > 0:
                y_true = year_data[key]['y_true']
                y_pred = year_data[key]['y_pred']

                # Calculate metrics
                r2 = r2_score(y_true, y_pred)
                rmse = np.sqrt(mean_squared_error(y_true, y_pred))

                # Create scatter plot
                fig.add_trace(
                    go.Scatter(
                        x=y_true,
                        y=y_pred,
                        mode='markers',
                        name=f'{model} (R²={r2:.3f})',
                        marker=dict(color=colors[j % len(colors)], size=6, opacity=0.7),
                        showlegend=(i == 0)  # Only show legend for first subplot
                    ),
                    row=row, col=col
                )

                print(f"    {category} - {model}: {len(y_true)} predictions, R²={r2:.3f}")

        # Add perfect prediction line
        if key in year_data and len(year_data[key]['y_true']) > 0:
            min_val = min(min(year_data[key]['y_true']), min(year_data[key]['y_pred']))
            max_val = max(max(year_data[key]['y_true']), max(year_data[key]['y_pred']))

            fig.add_trace(
                go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode='lines',
                    line=dict(color='black', dash='dash', width=1),
                    name='Perfect Prediction',
                    showlegend=(i == 0)
                ),
                row=row, col=col
            )

        # Update axes
        fig.update_xaxes(title_text="Actual", row=row, col=col)
        fig.update_yaxes(title_text="Predicted", row=row, col=col)

    fig.update_layout(
        title=f'Year {year} - Predicted vs Actual Performance',
        height=600 if rows == 1 else 800,
        template='plotly_white'
    )

    return fig


def create_all_year_graphs(cv_results, output_format='show'):
    """
    Create graphs for all available years

    Maintained for backward compatibility
    """
    available_years = cv_results.get_available_years()
    year_figures = {}

    for year in available_years:
        fig = plot_year_specific_analysis(cv_results, year)
        if fig:
            year_figures[year] = fig
            if output_format == 'show':
                fig.show()

    return year_figures


def plot_year_comparison_summary(cv_results):
    """
    Create summary plots comparing performance across years

    Maintained for backward compatibility
    """
    print("Creating year comparison summary...")

    available_years = cv_results.get_available_years()
    if len(available_years) == 0:
        return None

    # Collect R² scores by year and model
    r2_data = {}
    count_data = {}

    for key, data in cv_results.results.items():
        model_name, player_type, metric_type = key.split('_')
        category = f"{model_name}_{player_type}_{metric_type}"

        if category not in r2_data:
            r2_data[category] = {}
            count_data[category] = {}

        # Calculate year-by-year metrics
        year_breakdown = {}
        for i, year in enumerate(data['years']):
            year_str = str(year)
            if year_str not in year_breakdown:
                year_breakdown[year_str] = {'y_true': [], 'y_pred': []}
            year_breakdown[year_str]['y_true'].append(data['y_true'][i])
            year_breakdown[year_str]['y_pred'].append(data['y_pred'][i])

        for year, year_data in year_breakdown.items():
            if len(year_data['y_true']) > 0:
                r2 = r2_score(year_data['y_true'], year_data['y_pred'])
                r2_data[category][year] = r2
                count_data[category][year] = len(year_data['y_true'])

    # Create R² trends plot
    fig_r2 = go.Figure()

    for category in r2_data.keys():
        years = sorted(r2_data[category].keys())
        r2_values = [r2_data[category][year] for year in years]

        fig_r2.add_trace(go.Scatter(
            x=years,
            y=r2_values,
            mode='lines+markers',
            name=category,
            line=dict(width=2),
            marker=dict(size=6)
        ))

    fig_r2.update_layout(
        title='R² Performance Trends by Year',
        xaxis_title='Year',
        yaxis_title='R² Score',
        template='plotly_white'
    )

    # Create count summary plot
    fig_count = go.Figure()

    years = sorted(available_years)
    total_counts = []
    for year in years:
        total = sum(count_data[cat].get(year, 0) for cat in count_data.keys())
        total_counts.append(total)

    fig_count.add_trace(go.Bar(
        x=years,
        y=total_counts,
        marker_color='lightblue',
        text=total_counts,
        textposition='outside'
    ))

    fig_count.update_layout(
        title='Total Predictions by Year',
        xaxis_title='Year',
        yaxis_title='Number of Predictions',
        template='plotly_white'
    )

    return {
        'r2_trends': fig_r2,
        'count_summary': fig_count
    }


def print_year_analysis_summary(cv_results):
    """
    Print comprehensive analysis summary

    Maintained for backward compatibility
    """
    available_years = cv_results.get_available_years()

    print("\nYEAR-BY-YEAR ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"Total years analyzed: {len(available_years)}")

    # Calculate total predictions
    total_predictions = sum(len(data['y_true']) for data in cv_results.results.values())
    print(f"Total predictions: {total_predictions}")

    # Get unique models and categories
    models = set()
    categories = set()
    for key in cv_results.results.keys():
        model, player_type, metric = key.split('_')
        models.add(model)
        categories.add(f"{player_type}_{metric}")

    print(f"Models evaluated: {sorted(models)}")
    print(f"Categories analyzed: {len(categories)}")
    print()

    # Year-by-year breakdown
    print("Year-by-year breakdown:")
    for year in sorted(available_years):
        year_data = cv_results.get_year_data(year)
        year_total = sum(len(data['y_true']) for data in year_data.values())
        year_categories = len([k for k, v in year_data.items() if len(v['y_true']) > 0])

        # Find best performance for the year
        best_r2 = -999
        best_category = ""
        for key, data in year_data.items():
            if len(data['y_true']) > 0:
                r2 = r2_score(data['y_true'], data['y_pred'])
                if r2 > best_r2:
                    best_r2 = r2
                    model, player_type, metric = key.split('_')
                    best_category = f"{model.title()} {player_type} {metric.upper()}"

        print(f"  {year}: {year_total} predictions across {year_categories} categories")
        if best_category:
            print(f"    Best performance: {best_category} (R²={best_r2:.3f})")

    print("\nAnalysis complete - individual year graphs generated above")