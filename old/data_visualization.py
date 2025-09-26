"""
Data Visualization Module for sWARm

This module contains all visualization and plotting functions including:
- Enhanced interactive scatter plots with player hover details
- Consolidated model comparison visualizations
- Enhanced quadrant analysis with dual accuracy zones
- Comprehensive residual analysis plotting
- Animated temporal visualizations
- Training history plots for neural networks
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.metrics import r2_score, mean_squared_error

def plot_results(title, y_true, y_pred, player_names=None):
    """Enhanced plot with player names in hover tooltips"""
    if player_names is None:
        player_names = [f"Player_{i}" for i in range(len(y_true))]

    # Calculate errors for additional hover info
    errors = np.array(y_pred) - np.array(y_true)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=y_true,
        y=y_pred,
        mode='markers',
        marker=dict(size=8, opacity=0.7),
        text=player_names,
        customdata=np.column_stack((errors, y_true, y_pred)),
        hovertemplate="<b>%{text}</b><br>" +
                      "Actual WAR: %{customdata[1]:.3f}<br>" +
                      "Predicted WAR: %{customdata[2]:.3f}<br>" +
                      "Error: %{customdata[0]:.3f}<br>" +
                      "<extra></extra>",
        name='Predictions'
    ))

    # Add perfect prediction line
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        line=dict(dash='dash', color='red'),
        name='Perfect Prediction'
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Actual WAR",
        yaxis_title="Predicted WAR",
        template='plotly_white',
        width=600,
        height=600
    )

    fig.show()

def plot_training_history(history):
    """Plot training and validation loss over epochs"""
    if hasattr(history, 'history'):
        # Keras history object
        loss = history.history.get('loss', [])
        val_loss = history.history.get('val_loss', [])

        fig = go.Figure()

        epochs = list(range(1, len(loss) + 1))

        fig.add_trace(go.Scatter(
            x=epochs,
            y=loss,
            mode='lines',
            name='Training Loss',
            line=dict(color='blue')
        ))

        if val_loss:
            fig.add_trace(go.Scatter(
                x=epochs,
                y=val_loss,
                mode='lines',
                name='Validation Loss',
                line=dict(color='red')
            ))

        fig.update_layout(
            title='Training History',
            xaxis_title='Epoch',
            yaxis_title='Loss',
            template='plotly_white'
        )

        fig.show()
    else:
        print("No training history available")

def plot_consolidated_model_comparison(model_results, model_names=None, show_residuals=True, show_metrics=True):
    """
    Consolidated model comparison system that replaces individual print_metrics graphs.
    Creates unified visualizations with selectable traces for easy comparison.

    Args:
        model_results: Your ModelResults object
        model_names: List of models to compare (None = auto-select best)
        show_residuals: Whether to include residual analysis plots
        show_metrics: Whether to include scatter plot comparisons

    Returns:
        Dictionary with analysis results for each model
    """
    if model_names is None:
        from legacy_modules.modeling import select_best_models_by_category
        model_names = select_best_models_by_category(model_results)
        print(f"TARGET Auto-selected models for comparison: {[m.upper() for m in model_names]}")

    print("\nCHART CONSOLIDATED MODEL COMPARISON SYSTEM")
    print("="*70)
    print("  Replacing individual graphs with unified selectable trace visualizations...")

    # Collect all data for consolidated visualizations
    all_data = []
    model_stats = {}

    for model_name in model_names:
        model_stats[model_name] = {}

        for player_type in ['hitter', 'pitcher']:
            for metric in ['war', 'warp']:
                key = f"{model_name}_{player_type}_{metric}"
                if key in model_results.results:
                    data = model_results.results[key]

                    if len(data['y_true']) > 0:
                        y_true = np.array(data['y_true'])
                        y_pred = np.array(data['y_pred'])
                        residuals = y_true - y_pred

                        # Calculate comprehensive statistics
                        rmse = np.sqrt(np.mean(residuals**2))
                        mae = np.mean(np.abs(residuals))
                        r2 = 1 - (np.sum(residuals**2) / np.sum((y_true - np.mean(y_true))**2))

                        # Store statistics
                        model_stats[model_name][f"{player_type}_{metric}"] = {
                            'rmse': rmse,
                            'mae': mae,
                            'r2': r2,
                            'count': len(y_true)
                        }

                        # Add to plotting data
                        for i in range(len(residuals)):
                            all_data.append({
                                'Model': model_name.title(),
                                'PlayerType': player_type.title(),
                                'Metric': metric.upper(),
                                'Category': f"{player_type.title()} {metric.upper()}",
                                'Actual': y_true[i],
                                'Predicted': y_pred[i],
                                'Residual': residuals[i],
                                'Player': data['player_names'][i] if 'player_names' in data else f"Player_{i}"
                            })

    if not all_data:
        print("ERROR No data available for consolidated comparison")
        return {}

    df = pd.DataFrame(all_data)

    # Utility: add group toggle buttons
    def add_group_buttons(fig, group_labels):
        n_traces = len(fig.data)
        buttons = []

        for g in group_labels:
            buttons.append(
                dict(
                    label=f"Toggle {g}",
                    method="restyle",
                    args=[
                        {"visible": "toggle"},
                        [i for i, tr in enumerate(fig.data) if tr.legendgroup == g]
                    ]
                )
            )

        # Show all
        buttons.append(
            dict(
                label="Show All",
                method="restyle",
                args=[{"visible": True}, list(range(n_traces))]
            )
        )

        fig.update_layout(
            updatemenus=[dict(
                type="buttons",
                direction="right",
                x=0.5, xanchor="center",
                y=1.15, yanchor="top",
                buttons=buttons
            )]
        )

    # 1. CONSOLIDATED SCATTER PLOTS WITH SELECTABLE TRACES
    if show_metrics:
        print("\nTREND Creating consolidated prediction accuracy plots...")

        fig_scatter = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Hitter WAR', 'Hitter WARP', 'Pitcher WAR', 'Pitcher WARP'],
            vertical_spacing=0.1,
            horizontal_spacing=0.1
        )

        colors = px.colors.qualitative.Set1

        for i, category in enumerate(['Hitter WAR', 'Hitter WARP', 'Pitcher WAR', 'Pitcher WARP']):
            cat_data = df[df['Category'] == category]

            row = (i // 2) + 1
            col = (i % 2) + 1

            for j, model in enumerate(cat_data['Model'].unique()):
                model_data = cat_data[cat_data['Model'] == model]

                fig_scatter.add_trace(
                    go.Scatter(
                        x=model_data['Actual'],
                        y=model_data['Predicted'],
                        mode='markers',
                        name=f"{model} {category}",
                        legendgroup=model,
                        marker=dict(color=colors[j % len(colors)], size=6, opacity=0.7),
                        text=model_data['Player'],
                        hovertemplate="<b>%{text}</b><br>" +
                                      "Actual: %{x:.3f}<br>" +
                                      "Predicted: %{y:.3f}<br>" +
                                      f"Model: {model}<br>" +
                                      "<extra></extra>",
                        showlegend=(i == 0)
                    ),
                    row=row, col=col
                )

                # Add perfect prediction line
                if j == 0:  # Only add once per subplot
                    min_val = min(model_data['Actual'].min(), model_data['Predicted'].min())
                    max_val = max(model_data['Actual'].max(), model_data['Predicted'].max())

                    fig_scatter.add_trace(
                        go.Scatter(
                            x=[min_val, max_val],
                            y=[min_val, max_val],
                            mode='lines',
                            line=dict(dash='dash', color='red', width=2),
                            name='Perfect Prediction',
                            legendgroup='Perfect Prediction',
                            showlegend=(i == 0)
                        ),
                        row=row, col=col
                    )

        fig_scatter.update_layout(
            title="Consolidated Model Comparison: Prediction Accuracy (Click Legend or Use Buttons to Toggle)",
            height=800,
            width=1200,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
        )

        add_group_buttons(fig_scatter, df['Model'].unique())
        fig_scatter.show()

    # 2. CONSOLIDATED RESIDUAL ANALYSIS
    if show_residuals:
        print("\n  Creating consolidated residual analysis...")

        fig_residuals = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Residuals vs Fitted', 'Residual Distributions', 'Q-Q Plot', 'Model Performance'],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )

        colors = px.colors.qualitative.Set1

        # Residuals vs Fitted
        for j, model in enumerate(df['Model'].unique()):
            model_data = df[df['Model'] == model]

            fig_residuals.add_trace(
                go.Scatter(
                    x=model_data['Predicted'],
                    y=model_data['Residual'],
                    mode='markers',
                    name=f"{model} Residuals",
                    legendgroup=model,
                    marker=dict(color=colors[j % len(colors)], size=4, opacity=0.6),
                    showlegend=True
                ),
                row=1, col=1
            )

        # Add horizontal line at y=0
        fig_residuals.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=1)

        # Residual Distributions
        for j, model in enumerate(df['Model'].unique()):
            model_data = df[df['Model'] == model]

            fig_residuals.add_trace(
                go.Histogram(
                    x=model_data['Residual'],
                    name=f"{model} Distribution",
                    legendgroup=model,
                    opacity=0.7,
                    nbinsx=30,
                    showlegend=False
                ),
                row=1, col=2
            )

        # Q-Q Plot (simplified - one model for clarity)
        if len(df['Model'].unique()) > 0:
            best_model = df['Model'].unique()[0]
            best_data = df[df['Model'] == best_model]
            residuals = best_data['Residual'].values

            sorted_residuals = np.sort(residuals)
            n = len(sorted_residuals)
            theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, n))

            fig_residuals.add_trace(
                go.Scatter(
                    x=theoretical_quantiles,
                    y=sorted_residuals,
                    mode='markers',
                    name=f"{best_model} Q-Q",
                    legendgroup=best_model,
                    marker=dict(size=4),
                    showlegend=False
                ),
                row=2, col=1
            )

            # Theoretical line
            slope = np.std(residuals)
            intercept = np.mean(residuals)
            line_min, line_max = min(theoretical_quantiles), max(theoretical_quantiles)

            fig_residuals.add_trace(
                go.Scatter(
                    x=[line_min, line_max],
                    y=[intercept + slope * line_min, intercept + slope * line_max],
                    mode='lines',
                    line=dict(color='red', dash='dash'),
                    name='Normal Line',
                    legendgroup='Normal Line',
                    showlegend=False
                ),
                row=2, col=1
            )

        # Model Performance Comparison
        models = list(model_stats.keys())
        metrics = ['R^2', 'RMSE', 'MAE']

        for metric_name in metrics:
            metric_values = []
            for model in models:
                # Average across all model variants
                values = []
                for key, model_stat_data in model_stats[model].items():
                    if metric_name == 'R^2':
                        values.append(model_stat_data['r2'])
                    elif metric_name == 'RMSE':
                        values.append(model_stat_data['rmse'])
                    elif metric_name == 'MAE':
                        values.append(model_stat_data['mae'])

                metric_values.append(np.mean(values) if values else 0)

            fig_residuals.add_trace(
                go.Bar(
                    x=models,
                    y=metric_values,
                    name=metric_name,
                    legendgroup=metric_name,
                    showlegend=False
                ),
                row=2, col=2
            )

        fig_residuals.update_layout(
            title="Consolidated Residual Analysis (Click Legend or Use Buttons to Toggle Models)",
            height=800,
            width=1200,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
        )

        add_group_buttons(fig_residuals, df['Model'].unique())
        fig_residuals.show()

    # 3. COMPREHENSIVE STATISTICAL SUMMARY
    print("\n  CONSOLIDATED MODEL PERFORMANCE SUMMARY")
    print("="*70)

    for model_name in model_names:
        if model_name in model_stats:
            print(f"\n  {model_name.upper()} MODEL:")

            total_predictions = sum(model_stat_data['count'] for model_stat_data in model_stats[model_name].values())
            avg_r2 = np.mean([model_stat_data['r2'] for model_stat_data in model_stats[model_name].values()])
            avg_rmse = np.mean([model_stat_data['rmse'] for model_stat_data in model_stats[model_name].values()])
            avg_mae = np.mean([model_stat_data['mae'] for model_stat_data in model_stats[model_name].values()])

            print(f"   CHART Overall Performance:")
            print(f"      - Total Predictions: {total_predictions}")
            print(f"      - Average R^2: {avg_r2:.4f}")
            print(f"      - Average RMSE: {avg_rmse:.4f}")
            print(f"      - Average MAE: {avg_mae:.4f}")

            print(f"   TREND By Category:")
            for key, model_stat_data in model_stats[model_name].items():
                category = key.replace('_', ' ').title()
                print(f"      - {category}: R^2={model_stat_data['r2']:.4f}, RMSE={model_stat_data['rmse']:.4f}, Count={model_stat_data['count']}")

    print(f"\nSUCCESS CONSOLIDATED COMPARISON COMPLETE")
    print(f"   TREND Unified scatter plots: All models on same plots with toggleable traces")
    print(f"     Integrated residual analysis: Comprehensive diagnostic plots")
    print(f"   CHART Statistical summary: Complete performance metrics")
    print(f"   CLICK  Interactive legends: Click to show/hide individual models, use buttons for group control")

    return model_stats

def plot_comprehensive_residual_analysis(model_results, model_names=None, player_type="both", metric="both"):
    """
    Comprehensive residual plot comparison system for ML model diagnostics.

    This function creates multiple residual visualizations to diagnose model performance:
    1. Residuals vs Fitted Values (heteroscedasticity detection)
    2. Q-Q plots (normality assessment)
    3. Residual distribution histograms
    4. Scale-Location plots (variance homogeneity)
    5. Model comparison summary statistics

    Args:
        model_results: Your ModelResults object
        model_names: List of models to compare (None = auto-select best)
        player_type: 'hitter', 'pitcher', or 'both'
        metric: 'war', 'warp', or 'both'

    Returns:
        Dictionary with residual analysis results for each model
    """
    # Use the new consolidated system
    return plot_consolidated_model_comparison(model_results, model_names, show_residuals=True, show_metrics=False)

def plot_quadrant_analysis_px_toggle(model_results, season_col="Season", model_names=None, show_hitters=True, show_pitchers=True):
    """
    Enhanced quadrant analysis using Plotly Express facets with year-over-year animation.
    Dual accuracy zone visualization (orange cross + green intersection) with clickable legend toggles.
    ENHANCED: Comprehensive accuracy analysis with error percentage calculations
    FIXED: Data display issues, chronological year sorting, improved legend positioning
    """
    if model_names is None:
        from legacy_modules.modeling import select_best_models_by_category
        model_names = select_best_models_by_category(model_results)
        print(f"TARGET Auto-selected models: {[m.upper() for m in model_names]}")

    # FIXED: More robust data collection with debugging
    data = []
    data_found = False

    print("  Collecting data from model results...")
    for model in model_names:
        for player_type in ["hitter", "pitcher"]:
            for metric in ["war", "warp"]:
                key = f"{model}_{player_type}_{metric}"
                if key in model_results.results:
                    res = model_results.results[key]
                    if len(res["player_names"]) > 0:
                        data_found = True
                        print(f"   Found {len(res['player_names'])} entries for {key}")

                        for i, player in enumerate(res["player_names"]):
                            # FIXED: Better season handling with debugging
                            if season_col in res and len(res[season_col]) > i and res[season_col][i] is not None:
                                season_value = res[season_col][i]
                                # Convert to int first for proper sorting, then back to string for consistency
                                try:
                                    season_int = int(season_value)
                                    season_value = str(season_int)  # Normalized string format
                                except (ValueError, TypeError):
                                    season_value = str(season_value)
                            else:
                                season_value = "2021"  # Default

                            data.append({
                                "Player": player,
                                "Model": model.title(),
                                "PlayerType": player_type.title(),
                                season_col: season_value,
                                f"Actual {metric.upper()}": res["y_true"][i],
                                f"Predicted {metric.upper()}": res["y_pred"][i]
                            })
                    else:
                        print(f"   No data for {key}")
                else:
                    print(f"   Key {key} not found in results")

    if not data_found or not data:
        print("ERROR No data available for quadrant analysis.")
        print("Available keys in model_results:", list(model_results.results.keys())[:10])
        return

    df = pd.DataFrame(data)
    print(f"SUCCESS Collected {len(df)} data points for analysis")

    # Enhanced delta and error calculations
    df["WAR_Delta"] = df["Actual WAR"] - df["Predicted WAR"]
    df["WARP_Delta"] = df["Actual WARP"] - df["Predicted WARP"]

    # Error percentage calculations for 10% accuracy zone
    df["WAR_Error_%"] = abs(df["WAR_Delta"]) / df["Actual WAR"].replace(0, float("nan")).abs() * 100
    df["WARP_Error_%"] = abs(df["WARP_Delta"]) / df["Actual WARP"].replace(0, float("nan")).abs() * 100

    # Multiple accuracy zone definitions
    df["In_Accuracy_Zone"] = (df["WAR_Error_%"] <= 10) & (df["WARP_Error_%"] <= 10)
    df["WAR_Delta_1"] = abs(df["WAR_Delta"]) <= 1.0
    df["WARP_Delta_1"] = abs(df["WARP_Delta"]) <= 1.0
    df["Both_Delta_1"] = df["WAR_Delta_1"] & df["WARP_Delta_1"]  # Green intersection
    df["Either_Delta_1"] = df["WAR_Delta_1"] | df["WARP_Delta_1"]  # Orange cross

    df["AccuracyZone"] = df["In_Accuracy_Zone"].map({True: "<=10% Error Both", False: "Outside Zone"})
    df["Delta1Zone"] = df["Both_Delta_1"].map({True: "Both <=1", False: "Outside +-1"})

    # FIXED: Proper chronological sorting for animation frames
    unique_seasons = df[season_col].unique()
    try:
        # Convert to int for proper chronological sorting
        sorted_seasons = sorted([int(s) for s in unique_seasons if s is not None])
        # Convert back to strings for consistency
        sorted_season_strings = [str(s) for s in sorted_seasons]
        # Create categorical with proper order
        df[season_col] = pd.Categorical(df[season_col], categories=sorted_season_strings, ordered=True)
        print(f"DATE Sorted seasons chronologically: {sorted_season_strings}")
    except (ValueError, TypeError):
        # Fallback to string sorting
        sorted_season_strings = sorted([str(s) for s in unique_seasons if s is not None])
        df[season_col] = pd.Categorical(df[season_col], categories=sorted_season_strings, ordered=True)
        print(f"DATE Sorted seasons as strings: {sorted_season_strings}")

    min_val = min(df["WAR_Delta"].min(), df["WARP_Delta"].min())
    max_val = max(df["WAR_Delta"].max(), df["WARP_Delta"].max())
    buffer = (max_val - min_val) * 0.05

    # Create the enhanced faceted figure
    fig = px.scatter(
        df,
        x="WAR_Delta",
        y="WARP_Delta",
        color="PlayerType",
        symbol="AccuracyZone",
        hover_name="Player",
        facet_col="Model",
        facet_row="PlayerType",
        animation_frame=season_col,
        animation_group="Player",
        title="Enhanced Quadrant Analysis: WAR vs WARP Deltas (Chronological Animation)",
        range_x=[min_val - buffer, max_val + buffer],
        range_y=[min_val - buffer, max_val + buffer],
        width=1200,
        height=800,
        template="seaborn"
    )

    # Convert to go.Figure for advanced customization
    fig = go.Figure(fig)

    # Add quadrant reference lines
    fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1)
    fig.add_vline(x=0, line_dash="dash", line_color="gray", line_width=1)

    # Add dual accuracy zone visualization
    accuracy_shapes = []

    # Orange cross lines (+-1 margins)
    accuracy_shapes.extend([
        dict(type="line", x0=-1, y0=min_val-buffer, x1=-1, y1=max_val+buffer,
             line=dict(color="orange", width=2, dash="dot")),
        dict(type="line", x0=1, y0=min_val-buffer, x1=1, y1=max_val+buffer,
             line=dict(color="orange", width=2, dash="dot")),
        dict(type="line", x0=min_val-buffer, y0=-1, x1=max_val+buffer, y1=-1,
             line=dict(color="orange", width=2, dash="dot")),
        dict(type="line", x0=min_val-buffer, y0=1, x1=max_val+buffer, y1=1,
             line=dict(color="orange", width=2, dash="dot"))
    ])

    # Green intersection rectangle
    accuracy_shapes.append(
        dict(type="rect", x0=-1, y0=-1, x1=1, y1=1,
             line=dict(color="green", width=2, dash="dash"),
             fillcolor="green", opacity=0.1)
    )

    fig.update_layout(shapes=accuracy_shapes)

    # FIXED: Improved legend positioning (above traces, not covering y-axis)
    fig.update_layout(
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,  # Above the plot
            xanchor="center",
            x=0.5,   # Centered horizontally
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="gray",
            borderwidth=1
        )
    )

    # FIXED: Reposition animation controls to be usable
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                x=0.15,  # Moved right so not covered by legend
                y=1.15,  # Above legend
                showactive=True,
                buttons=[
                    dict(label="All Zones", method="relayout",
                         args=[{"shapes": [{**s, "visible": True} for s in accuracy_shapes]}]),
                    dict(label="Cross Only", method="relayout",
                         args=[{"shapes": [{**s, "visible": True if "line" in s.get("type", "") else False}
                                          for s in accuracy_shapes]}]),
                    dict(label="Intersection", method="relayout",
                         args=[{"shapes": [{**s, "visible": True if s.get("type") == "rect" else False}
                                          for s in accuracy_shapes]}]),
                    dict(label="No Zones", method="relayout",
                         args=[{"shapes": [{**s, "visible": False} for s in accuracy_shapes]}])
                ]
            )
        ],
        # FIXED: Position animation controls to the right
        sliders=[dict(
            currentvalue={"prefix": "Year: "},
            x=0.15,  # Moved right
            len=0.7   # Adjusted length
        )]
    )

    fig.show()

    # Enhanced statistical summary with better formatting
    print("\n" + "="*60)
    print("CHART INTERACTIVE QUADRANT ANALYSIS SUMMARY")
    print("="*60)

    for model in df["Model"].unique():
        mdf = df[df["Model"] == model]
        total = len(mdf)

        acc_10pct = mdf["In_Accuracy_Zone"].sum()
        both_delta1 = mdf["Both_Delta_1"].sum()
        either_delta1 = mdf["Either_Delta_1"].sum()

        print(f"\n  {model.upper()} MODEL ({total} predictions):")
        print(f"   TREND 10% Accuracy Zone (both WAR & WARP): {acc_10pct}/{total} ({acc_10pct/total*100:.1f}%)")
        print(f"   TARGET Delta 1 Cross (either <=1): {either_delta1}/{total} ({either_delta1/total*100:.1f}%)")
        print(f"   SUCCESS Delta 1 Intersection (both <=1): {both_delta1}/{total} ({both_delta1/total*100:.1f}%)")

        # Sample accurate players
        accurate_players = mdf[mdf["In_Accuracy_Zone"]]["Player"].unique()
        if len(accurate_players) > 0:
            sample = ", ".join(list(accurate_players[:3]))
            print(f"     Sample accurate: {sample}{'...' if len(accurate_players) > 3 else ''}")

    print(f"\n  INTERACTIVE FEATURES:")
    print(f"   CLICK  Legend: Click PlayerType/AccuracyZone to show/hide")
    print(f"   ANIMATION Animation: Chronologically ordered year progression")
    print(f"     Accuracy Zones: Toggle orange cross vs green intersection")
    print(f"   TARGET Hover: Detailed player performance information")

def plot_war_warp_animated(model_results, season_col="Season", model_names=None, show_hitters=True, show_pitchers=True):
    """
    ANIMATION SOPHISTICATED ANIMATED WAR vs WARP PREDICTION ANALYSIS

    Creates aesthetically pleasing animated visualizations showing predicted vs actual values:
    - 4 subplots: Predicted Hitter WAR, Predicted Pitcher WAR, Predicted Hitter WARP, Predicted Pitcher WARP
    - X-axis: Predicted values, Y-axis: Actual values
    - Smooth temporal transitions with custom easing
    - Dynamic color schemes and visual themes
    - Perfect prediction diagonal lines for reference
    - Reduced dot size to prevent overlap

    Args:
        model_results: ModelResults object with prediction data
        season_col: Column name for temporal animation (default: "Season")
        model_names: List of models to include (auto-selected if None)
        show_hitters: Include hitter data in animation
        show_pitchers: Include pitcher data in animation
    """
    if model_names is None:
        from legacy_modules.modeling import select_best_models_by_category
        model_names = select_best_models_by_category(model_results)
        print(f"TARGET Auto-selected models for cinematic animation: {[m.upper() for m in model_names]}")

    print("ANIMATION Creating sophisticated predicted vs actual animated analysis...")
    print("DEBUG Available model result keys:", list(model_results.results.keys())[:10])

    # FIXED: Use exact same data collection approach as consolidated model comparison
    all_data = []
    performance_stats = {}

    print(f"DEBUG Selected models: {model_names}")
    print(f"DEBUG First 10 available keys: {list(model_results.results.keys())[:10]}")

    for model_name in model_names:
        performance_stats[model_name] = {"accuracy_count": 0, "total_count": 0}
        print(f"DEBUG Processing model: {model_name}")

        for player_type in ['hitter', 'pitcher']:
            if (player_type == "hitter" and not show_hitters) or (player_type == "pitcher" and not show_pitchers):
                continue

            for metric in ['war', 'warp']:
                key = f"{model_name}_{player_type}_{metric}"
                print(f"DEBUG Checking key: {key}")

                if key in model_results.results:
                    data = model_results.results[key]
                    print(f"DEBUG Found {len(data['y_true'])} data points for {key}")

                    if len(data['y_true']) > 0:
                        y_true = np.array(data['y_true'])
                        y_pred = np.array(data['y_pred'])

                        # Add to plotting data - using exact same approach as consolidated function
                        for i in range(len(y_true)):
                            # Get season data if available
                            if season_col in data and i < len(data[season_col]):
                                season = data[season_col][i]
                                try:
                                    season_str = str(int(float(str(season))))
                                except (ValueError, TypeError):
                                    season_str = str(season) if season is not None else "2021"
                            else:
                                season_str = "2021"

                            error = abs(y_true[i] - y_pred[i])
                            is_accurate = error <= 1.0

                            performance_stats[model_name]["total_count"] += 1
                            if is_accurate:
                                performance_stats[model_name]["accuracy_count"] += 1

                            all_data.append({
                                'Model': model_name.title(),
                                'PlayerType': player_type.title(),
                                'Metric': metric.upper(),
                                'Category': f"{player_type.title()} {metric.upper()}",
                                'Actual': y_true[i],
                                'Predicted': y_pred[i],
                                'Error': error,
                                'Player': data['player_names'][i] if 'player_names' in data and i < len(data['player_names']) else f"Player_{i}",
                                season_col: season_str,
                                "Accuracy_Status": "High Accuracy" if is_accurate else "Lower Accuracy",
                                "Performance_Score": max(0.5, 3 - error),
                                "Elite_Player": y_true[i] > 3.0
                            })
                else:
                    print(f"WARNING: Key {key} not found in model_results.results")

    # Rename for consistency with rest of function
    data = all_data

    if not data:
        print("ERROR No matching WAR/WARP data available for animation")
        print("Available keys:", list(model_results.results.keys()))
        return

    df = pd.DataFrame(data)
    print(f"SUCCESS Collected {len(df)} total data points")
    print("DEBUG Data breakdown by category:")
    for category in df["Category"].unique():
        count = len(df[df["Category"] == category])
        print(f"  {category}: {count} points")

    # FIXED: Enhanced chronological sorting for smooth animation
    unique_seasons = df[season_col].unique()
    print(f"DEBUG Raw seasons found: {sorted(unique_seasons)}")

    # Better season cleaning and sorting
    clean_seasons = []
    for s in unique_seasons:
        if s is not None and str(s).strip():
            try:
                season_int = int(float(str(s)))  # Handle potential float strings
                clean_seasons.append(season_int)
            except (ValueError, TypeError):
                print(f"WARNING: Could not convert season '{s}' to integer")

    if clean_seasons:
        sorted_seasons = sorted(set(clean_seasons))  # Remove duplicates and sort
        sorted_season_strings = [str(s) for s in sorted_seasons]
        df[season_col] = pd.Categorical(df[season_col], categories=sorted_season_strings, ordered=True)
        print(f"DATE FIXED Chronological sequence: {sorted_season_strings}")
    else:
        print("ERROR: No valid seasons found")
        return

    # Create the main predicted vs actual animation with 4 subplots
    print("   STYLE Creating predicted vs actual comparison animation...")

    fig_bubble = px.scatter(
        df,
        x="Predicted",
        y="Actual",
        size="Performance_Score",
        color="Error",
        hover_name="Player",
        animation_frame=season_col,
        animation_group="Player",
        facet_col="Category",
        facet_col_wrap=2,  # 2x2 grid layout
        title="ANIMATION Cinematic WAR vs WARP Performance Analysis",
        color_continuous_scale="viridis_r",
        size_max=12,  # Restored per user request
        template="plotly_dark",
        width=1400,
        height=800,
        labels={
            "Predicted": "Predicted Value",
            "Actual": "Actual Value",
            "Error": "Prediction Error"
        }
    )

    # FIXED: Enhanced styling with visible axes
    fig_bubble.update_layout(
        font=dict(family="Arial Black", size=12, color="white"),
        title=dict(font=dict(size=20), x=0.5, xanchor="center"),
        plot_bgcolor="rgba(0,0,0,0.9)",
        paper_bgcolor="rgba(0,0,0,0.95)",
        coloraxis_colorbar=dict(
            title="Prediction Error",
            title_font=dict(color="white"),
            tickfont=dict(color="white")
        )
    )

    # FIXED: Make axes numbers visible and granular
    fig_bubble.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(255,255,255,0.2)',
        tickfont=dict(color="white", size=10),
        dtick=1,  # Show tick every 1 unit
        tickcolor="white",
        showticklabels=True
    )

    fig_bubble.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(255,255,255,0.2)',
        tickfont=dict(color="white", size=10),
        dtick=1,  # Show tick every 1 unit
        tickcolor="white",
        showticklabels=True
    )

    # FIXED: Add perfect prediction diagonal lines and regression trendlines
    from scipy.stats import linregress

    categories = df['Category'].unique()
    subplot_positions = [(1, 1), (1, 2), (2, 1), (2, 2)]  # row, col for 2x2 grid

    for i, category in enumerate(categories):
        category_data = df[df['Category'] == category]
        if len(category_data) > 0:
            predicted_vals = category_data["Predicted"].values
            actual_vals = category_data["Actual"].values

            min_val = min(predicted_vals.min(), actual_vals.min())
            max_val = max(predicted_vals.max(), actual_vals.max())

            # Add buffer for better visualization
            buffer = (max_val - min_val) * 0.1
            min_val -= buffer
            max_val += buffer

            if i < len(subplot_positions):
                row, col = subplot_positions[i]

                # Perfect prediction diagonal line (y=x)
                fig_bubble.add_shape(
                    type="line",
                    x0=min_val, y0=min_val,
                    x1=max_val, y1=max_val,
                    line=dict(color="rgba(255,255,255,0.8)", width=2, dash="dash"),
                    row=row, col=col
                )

                # Add regression trendline
                if len(predicted_vals) > 1:
                    slope, intercept, r_value, p_value, std_err = linregress(predicted_vals, actual_vals)

                    # Calculate trendline points
                    x_trend = np.array([min_val, max_val])
                    y_trend = slope * x_trend + intercept

                    fig_bubble.add_shape(
                        type="line",
                        x0=x_trend[0], y0=y_trend[0],
                        x1=x_trend[1], y1=y_trend[1],
                        line=dict(color="rgba(255,165,0,0.9)", width=3, dash="solid"),
                        row=row, col=col
                    )

    # Enhanced animation controls with smooth transitions
    if fig_bubble.layout.updatemenus and len(fig_bubble.layout.updatemenus) > 0:
        fig_bubble.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 1500
        fig_bubble.layout.updatemenus[0].buttons[0].args[1]["frame"]["redraw"] = True
        fig_bubble.layout.updatemenus[0].buttons[0].args[1]["transition"]["duration"] = 800
        fig_bubble.layout.updatemenus[0].buttons[0].args[1]["transition"]["easing"] = "cubic-in-out"

    fig_bubble.show()

    # Enhanced statistical summary with aesthetic formatting
    print("\n" + "="*70)
    print("ANIMATION PREDICTED VS ACTUAL ANALYSIS SUMMARY")
    print("="*70)

    # Group by category to show performance for each subplot
    for category in df["Category"].unique():
        category_data = df[df["Category"] == category]
        total_points = len(category_data)
        accurate_points = len(category_data[category_data["Accuracy_Status"] == "High Accuracy"])
        accuracy_rate = (accurate_points / total_points * 100) if total_points > 0 else 0
        avg_error = category_data["Error"].mean()

        print(f"\n{category.upper()}:")
        print(f"   CHART Total Predictions: {total_points}")
        print(f"   SUCCESS High Accuracy Rate: {accuracy_rate:.1f}% ({accurate_points}/{total_points})")
        print(f"   TREND Average Error: {avg_error:.3f}")

    for model in model_names:
        if model in performance_stats:
            stats = performance_stats[model]
            accuracy_rate = (stats["accuracy_count"] / stats["total_count"] * 100) if stats["total_count"] > 0 else 0

            print(f"\nTARGET {model.upper()} MODEL OVERALL:")
            print(f"   CHART Total Predictions: {stats['total_count']}")
            print(f"   SUCCESS High Accuracy Rate: {accuracy_rate:.1f}% ({stats['accuracy_count']}/{stats['total_count']})")

    print(f"\nSTYLE UPDATED FEATURES IMPLEMENTED:")
    print(f"   ANIMATION Predicted vs Actual comparison with 4 subplots")
    print(f"   CHART X-axis: Predicted values, Y-axis: Actual values")
    print(f"   TARGET Perfect prediction diagonal lines for reference")
    print(f"   ? Reduced dot size (12 max) to prevent overlap")
    print(f"   DATE Proper chronological year ordering")
    print(f"   CLICK  Interactive legends and hover details")

    print(f"\n  SUBPLOT BREAKDOWN:")
    print(f"   - Hitter WAR: Predicted Hitter WAR vs Actual Hitter WAR")
    print(f"   - Hitter WARP: Predicted Hitter WARP vs Actual Hitter WARP")
    print(f"   - Pitcher WAR: Predicted Pitcher WAR vs Actual Pitcher WAR")
    print(f"   - Pitcher WARP: Predicted Pitcher WARP vs Actual Pitcher WARP")

    return {
        'performance_stats': performance_stats,
        'total_observations': len(df),
        'temporal_range': sorted_season_strings,
        'aesthetic_features': ['predicted_vs_actual', 'four_subplots', 'chronological_animation', 'reduced_overlap']
    }

# Export all functions for easy import
__all__ = [
    'plot_results',
    'plot_training_history',
    'plot_consolidated_model_comparison',
    'plot_comprehensive_residual_analysis',
    'plot_quadrant_analysis_px_toggle',
    'plot_war_warp_animated'
]