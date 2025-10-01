"""
Backward compatibility visualization functions

Maintained for compatibility with existing sWARm_CS.ipynb and other notebooks.
These functions support K-fold cross-validation visualization and year-by-year analysis.
"""

# Third-party imports
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import r2_score, mean_squared_error
from typing import Dict, Optional

# Local application imports
from common_modules.logging import get_logger

# Module logger
logger = get_logger(__name__)


def plot_year_specific_analysis(cv_results, year, models_to_show=None):
    """
    Create year-specific 4-subplot graph showing predicted vs actual for a single year

    Maintained for backward compatibility with existing sWARm_CS.ipynb

    Args:
        cv_results: Cross-validation results object
        year: Year to analyze
        models_to_show: Optional list of models to display

    Returns:
        Plotly figure with year-specific analysis
    """
    logger.info(f"Creating year-specific analysis for {year}")

    # Get data for the specific year
    year_data = cv_results.get_year_data(year)

    if not year_data:
        logger.warning(f"No data available for year {year}")
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

    logger.debug(f"  Available categories for {year}: {sorted(available_categories)}")
    logger.debug(f"  Available models for {year}: {sorted(available_models)}")

    # Create subplot structure (2x2 grid)
    category_order = ['Hitter WAR', 'Hitter WARP', 'Pitcher WAR', 'Pitcher WARP']
    available_categories_ordered = [cat for cat in category_order if cat in available_categories]

    if len(available_categories_ordered) == 0:
        logger.warning(f"No valid categories found for {year}")
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

                logger.debug(f"    {category} - {model}: {len(y_true)} predictions, R²={r2:.3f}")

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

    Args:
        cv_results: Cross-validation results object
        output_format: 'show' to display graphs, other values to return only

    Returns:
        Dictionary of year figures
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

    Args:
        cv_results: Cross-validation results object

    Returns:
        Dictionary with summary figures
    """
    logger.info("Creating year comparison summary")

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

    Args:
        cv_results: Cross-validation results object
    """
    available_years = cv_results.get_available_years()

    logger.info("YEAR-BY-YEAR ANALYSIS SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total years analyzed: {len(available_years)}")

    # Calculate total predictions
    total_predictions = sum(len(data['y_true']) for data in cv_results.results.values())
    logger.info(f"Total predictions: {total_predictions}")

    # Get unique models and categories
    models = set()
    categories = set()
    for key in cv_results.results.keys():
        model, player_type, metric = key.split('_')
        models.add(model)
        categories.add(f"{player_type}_{metric}")

    logger.info(f"Models evaluated: {sorted(models)}")
    logger.info(f"Categories analyzed: {len(categories)}")

    # Year-by-year breakdown
    logger.info("Year-by-year breakdown:")
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

        logger.info(f"  {year}: {year_total} predictions across {year_categories} categories")
        if best_category:
            logger.info(f"    Best performance: {best_category} (R²={best_r2:.3f})")

    logger.info("Analysis complete - individual year graphs generated above")
