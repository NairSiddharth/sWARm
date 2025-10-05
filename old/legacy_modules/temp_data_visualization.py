"""
Temporary Data Visualization Module for Year-Specific Analysis
Modified from original data_visualization.py to create year-by-year graphs
for K-fold cross-validation results analysis
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.metrics import r2_score, mean_squared_error

def plot_year_specific_analysis(cv_results, year, models_to_show=None):
    """
    Create year-specific 4-subplot graph showing predicted vs actual for a single year

    Args:
        cv_results: CrossValidationResults object with K-fold CV predictions
        year: Specific year to analyze (e.g., '2020')
        models_to_show: List of models to include (default: all available)
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
    subplot_titles = [f"{cat}" for cat in category_order]

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.1,
        vertical_spacing=0.15
    )

    # Colors for different models
    model_colors = {
        'Ridge': '#1f77b4',
        'Randomforest': '#ff7f0e',
        'Svr': '#2ca02c',
        'Keras': '#d62728'
    }

    # Track if any data was plotted
    data_plotted = False

    # Plot data for each category
    for i, category in enumerate(category_order):
        row = (i // 2) + 1
        col = (i % 2) + 1

        category_data_found = False

        # Look for data matching this category
        for key in year_data.keys():
            model_name, player_type, metric_type = key.split('_')
            key_category = f"{player_type.title()} {metric_type.upper()}"

            if key_category == category and len(year_data[key]['y_true']) > 0:
                data = year_data[key]

                # Get model color
                model_title = model_name.title()
                color = model_colors.get(model_title, '#666666')

                # Add scatter plot
                fig.add_trace(
                    go.Scatter(
                        x=data['y_pred'],
                        y=data['y_true'],
                        mode='markers',
                        marker=dict(
                            color=color,
                            size=8,
                            opacity=0.7,
                            line=dict(width=1, color='white')
                        ),
                        name=f"{model_title}",
                        text=data['player_names'],
                        customdata=np.column_stack((
                            data['y_pred'],
                            data['y_true'],
                            np.abs(data['y_pred'] - data['y_true'])
                        )),
                        hovertemplate="<b>%{text}</b><br>" +
                                    "Predicted: %{customdata[0]:.2f}<br>" +
                                    "Actual: %{customdata[1]:.2f}<br>" +
                                    "Error: %{customdata[2]:.2f}<br>" +
                                    f"Model: {model_title}<br>" +
                                    "<extra></extra>",
                        legendgroup=model_title,
                        showlegend=(i == 0)  # Only show legend for first subplot
                    ),
                    row=row, col=col
                )

                # Add perfect prediction line
                if not category_data_found:  # Only add once per category
                    min_val = min(min(data['y_pred']), min(data['y_true']))
                    max_val = max(max(data['y_pred']), max(data['y_true']))

                    # Add some padding
                    padding = (max_val - min_val) * 0.1
                    min_val -= padding
                    max_val += padding

                    fig.add_trace(
                        go.Scatter(
                            x=[min_val, max_val],
                            y=[min_val, max_val],
                            mode='lines',
                            line=dict(dash='dash', color='rgba(255,0,0,0.7)', width=2),
                            name='Perfect Prediction',
                            showlegend=False,
                            hoverinfo='skip'
                        ),
                        row=row, col=col
                    )

                category_data_found = True
                data_plotted = True

                # Calculate and display R² for this model/category
                r2 = r2_score(data['y_true'], data['y_pred'])
                print(f"    {category} - {model_title}: {len(data['y_true'])} predictions, R²={r2:.3f}")

    if not data_plotted:
        print(f"No prediction data found for year {year}")
        return None

    # Update layout
    fig.update_layout(
        title=dict(
            text=f"Model Performance Analysis - {year}",
            x=0.5,
            font=dict(size=20, color='black')
        ),
        template='plotly_white',
        width=1000,
        height=800,
        legend=dict(
            x=1.02,
            y=1,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='rgba(0,0,0,0.2)',
            borderwidth=1
        ),
        margin=dict(r=150)  # Extra margin for legend
    )

    # Update axes labels
    fig.update_xaxes(title_text="Predicted Value")
    fig.update_yaxes(title_text="Actual Value")

    # Make axes have equal scaling for better comparison
    for i in range(1, 5):
        row = ((i-1) // 2) + 1
        col = ((i-1) % 2) + 1
        fig.update_xaxes(scaleanchor=f"y{i}", scaleratio=1, row=row, col=col)

    return fig

def create_all_year_graphs(cv_results, output_format='show'):
    """
    Create graphs for all available years in the cross-validation results

    Args:
        cv_results: CrossValidationResults object
        output_format: 'show' to display, 'save' to save as files, 'both' for both
    """
    years = cv_results.get_available_years()
    print(f"Creating graphs for {len(years)} years: {years}")

    figures = {}

    for year in years:
        print(f"\nProcessing year {year}...")
        fig = plot_year_specific_analysis(cv_results, year)

        if fig is not None:
            figures[year] = fig

            if output_format in ['show', 'both']:
                fig.show()

            if output_format in ['save', 'both']:
                filename = f"model_performance_{year}.html"
                fig.write_html(filename)
                print(f"  Saved graph to {filename}")
        else:
            print(f"  Skipped {year} - no data available")

    print(f"\nCompleted analysis for {len(figures)} years with data")
    return figures

def plot_year_comparison_summary(cv_results):
    """
    Create summary plots comparing performance across all years
    """
    print("Creating year comparison summary...")

    years = cv_results.get_available_years()

    # Collect year-by-year statistics
    summary_data = []

    for year in years:
        year_data = cv_results.get_year_data(year)

        for key, data in year_data.items():
            if len(data['y_true']) > 0:
                model_name, player_type, metric_type = key.split('_')

                r2 = r2_score(data['y_true'], data['y_pred'])
                rmse = np.sqrt(mean_squared_error(data['y_true'], data['y_pred']))
                count = len(data['y_true'])

                summary_data.append({
                    'Year': int(year),
                    'Model': model_name.title(),
                    'Category': f"{player_type.title()} {metric_type.upper()}",
                    'R2': r2,
                    'RMSE': rmse,
                    'Count': count
                })

    if not summary_data:
        print("No summary data available")
        return None

    df_summary = pd.DataFrame(summary_data)

    # Create R² trend plot
    fig_r2 = px.line(
        df_summary,
        x='Year',
        y='R2',
        color='Model',
        facet_col='Category',
        facet_col_wrap=2,
        title="Model R² Performance Trends Across Years",
        markers=True
    )

    fig_r2.update_layout(
        template='plotly_white',
        width=1200,
        height=600
    )

    # Create sample count plot
    fig_count = px.bar(
        df_summary,
        x='Year',
        y='Count',
        color='Model',
        facet_col='Category',
        facet_col_wrap=2,
        title="Prediction Counts by Year and Category",
        barmode='group'
    )

    fig_count.update_layout(
        template='plotly_white',
        width=1200,
        height=600
    )

    return {'r2_trends': fig_r2, 'count_summary': fig_count}

def print_year_analysis_summary(cv_results):
    """Print comprehensive summary of year-by-year analysis"""
    print("\nYEAR-BY-YEAR ANALYSIS SUMMARY")
    print("="*60)

    years = cv_results.get_available_years()

    # Overall statistics
    total_predictions = 0
    total_models = set()
    total_categories = set()

    for key, data in cv_results.results.items():
        total_predictions += len(data['y_true'])
        model_name, player_type, metric_type = key.split('_')
        total_models.add(model_name)
        total_categories.add(f"{player_type}_{metric_type}")

    print(f"Total years analyzed: {len(years)}")
    print(f"Total predictions: {total_predictions}")
    print(f"Models evaluated: {sorted(total_models)}")
    print(f"Categories analyzed: {len(total_categories)}")

    # Year-by-year breakdown
    print(f"\nYear-by-year breakdown:")
    for year in years:
        year_data = cv_results.get_year_data(year)
        year_predictions = sum(len(data['y_true']) for data in year_data.values())
        year_categories = len([k for k, v in year_data.items() if len(v['y_true']) > 0])

        print(f"  {year}: {year_predictions} predictions across {year_categories} categories")

        # Best performing model for this year
        best_r2 = -999
        best_model_category = ""

        for key, data in year_data.items():
            if len(data['y_true']) > 0:
                r2 = r2_score(data['y_true'], data['y_pred'])
                if r2 > best_r2:
                    best_r2 = r2
                    model_name, player_type, metric_type = key.split('_')
                    best_model_category = f"{model_name.title()} {player_type} {metric_type.upper()}"

        if best_r2 > -999:
            print(f"    Best performance: {best_model_category} (R²={best_r2:.3f})")

    print("\nAnalysis complete - individual year graphs generated above")