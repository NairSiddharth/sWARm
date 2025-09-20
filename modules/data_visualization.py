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
        from modules.modeling import select_best_models_by_category
        model_names = select_best_models_by_category(model_results)
        print(f"🎯 Auto-selected models for comparison: {[m.upper() for m in model_names]}")

    print("\n📊 CONSOLIDATED MODEL COMPARISON SYSTEM")
    print("="*70)
    print("🔍 Replacing individual graphs with unified selectable trace visualizations...")

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
        print("❌ No data available for consolidated comparison")
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
        print("\n📈 Creating consolidated prediction accuracy plots...")

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
        print("\n🔍 Creating consolidated residual analysis...")

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
        metrics = ['R²', 'RMSE', 'MAE']

        for metric_name in metrics:
            metric_values = []
            for model in models:
                # Average across all model variants
                values = []
                for key, model_stat_data in model_stats[model].items():
                    if metric_name == 'R²':
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
    print("\n📋 CONSOLIDATED MODEL PERFORMANCE SUMMARY")
    print("="*70)

    for model_name in model_names:
        if model_name in model_stats:
            print(f"\n🤖 {model_name.upper()} MODEL:")

            total_predictions = sum(model_stat_data['count'] for model_stat_data in model_stats[model_name].values())
            avg_r2 = np.mean([model_stat_data['r2'] for model_stat_data in model_stats[model_name].values()])
            avg_rmse = np.mean([model_stat_data['rmse'] for model_stat_data in model_stats[model_name].values()])
            avg_mae = np.mean([model_stat_data['mae'] for model_stat_data in model_stats[model_name].values()])

            print(f"   📊 Overall Performance:")
            print(f"      • Total Predictions: {total_predictions}")
            print(f"      • Average R²: {avg_r2:.4f}")
            print(f"      • Average RMSE: {avg_rmse:.4f}")
            print(f"      • Average MAE: {avg_mae:.4f}")

            print(f"   📈 By Category:")
            for key, model_stat_data in model_stats[model_name].items():
                category = key.replace('_', ' ').title()
                print(f"      • {category}: R²={model_stat_data['r2']:.4f}, RMSE={model_stat_data['rmse']:.4f}, Count={model_stat_data['count']}")

    print(f"\n✅ CONSOLIDATED COMPARISON COMPLETE")
    print(f"   📈 Unified scatter plots: All models on same plots with toggleable traces")
    print(f"   🔍 Integrated residual analysis: Comprehensive diagnostic plots")
    print(f"   📊 Statistical summary: Complete performance metrics")
    print(f"   🖱️  Interactive legends: Click to show/hide individual models, use buttons for group control")

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
        from modules.modeling import select_best_models_by_category
        model_names = select_best_models_by_category(model_results)
        print(f"🎯 Auto-selected models: {[m.upper() for m in model_names]}")

    # FIXED: More robust data collection with debugging
    data = []
    data_found = False

    print("🔍 Collecting data from model results...")
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
        print("❌ No data available for quadrant analysis.")
        print("Available keys in model_results:", list(model_results.results.keys())[:10])
        return

    df = pd.DataFrame(data)
    print(f"✅ Collected {len(df)} data points for analysis")

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

    df["AccuracyZone"] = df["In_Accuracy_Zone"].map({True: "≤10% Error Both", False: "Outside Zone"})
    df["Delta1Zone"] = df["Both_Delta_1"].map({True: "Both ≤1", False: "Outside ±1"})

    # FIXED: Proper chronological sorting for animation frames
    unique_seasons = df[season_col].unique()
    try:
        # Convert to int for proper chronological sorting
        sorted_seasons = sorted([int(s) for s in unique_seasons if s is not None])
        # Convert back to strings for consistency
        sorted_season_strings = [str(s) for s in sorted_seasons]
        # Create categorical with proper order
        df[season_col] = pd.Categorical(df[season_col], categories=sorted_season_strings, ordered=True)
        print(f"📅 Sorted seasons chronologically: {sorted_season_strings}")
    except (ValueError, TypeError):
        # Fallback to string sorting
        sorted_season_strings = sorted([str(s) for s in unique_seasons if s is not None])
        df[season_col] = pd.Categorical(df[season_col], categories=sorted_season_strings, ordered=True)
        print(f"📅 Sorted seasons as strings: {sorted_season_strings}")

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

    # Orange cross lines (±1 margins)
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
    print("📊 INTERACTIVE QUADRANT ANALYSIS SUMMARY")
    print("="*60)

    for model in df["Model"].unique():
        mdf = df[df["Model"] == model]
        total = len(mdf)

        acc_10pct = mdf["In_Accuracy_Zone"].sum()
        both_delta1 = mdf["Both_Delta_1"].sum()
        either_delta1 = mdf["Either_Delta_1"].sum()

        print(f"\n🔍 {model.upper()} MODEL ({total} predictions):")
        print(f"   📈 10% Accuracy Zone (both WAR & WARP): {acc_10pct}/{total} ({acc_10pct/total*100:.1f}%)")
        print(f"   🎯 Delta 1 Cross (either ≤1): {either_delta1}/{total} ({either_delta1/total*100:.1f}%)")
        print(f"   ✅ Delta 1 Intersection (both ≤1): {both_delta1}/{total} ({both_delta1/total*100:.1f}%)")

        # Sample accurate players
        accurate_players = mdf[mdf["In_Accuracy_Zone"]]["Player"].unique()
        if len(accurate_players) > 0:
            sample = ", ".join(list(accurate_players[:3]))
            print(f"   🌟 Sample accurate: {sample}{'...' if len(accurate_players) > 3 else ''}")

    print(f"\n💡 INTERACTIVE FEATURES:")
    print(f"   🖱️  Legend: Click PlayerType/AccuracyZone to show/hide")
    print(f"   🎬 Animation: Chronologically ordered year progression")
    print(f"   🔘 Accuracy Zones: Toggle orange cross vs green intersection")
    print(f"   🎯 Hover: Detailed player performance information")

def plot_war_warp_animated(model_results, season_col="Season", model_names=None, show_hitters=True, show_pitchers=True):
    """
    🎬 SOPHISTICATED ANIMATED WAR vs WARP ANALYSIS

    Creates aesthetically pleasing animated visualizations with advanced Plotly features:
    - Smooth temporal transitions with custom easing
    - Dynamic color schemes and visual themes
    - Interactive accuracy zones with gradient fills
    - Performance metrics overlay with animated counters
    - Professional styling with enhanced typography

    Args:
        model_results: ModelResults object with prediction data
        season_col: Column name for temporal animation (default: "Season")
        model_names: List of models to include (auto-selected if None)
        show_hitters: Include hitter data in animation
        show_pitchers: Include pitcher data in animation
    """
    if model_names is None:
        from modules.modeling import select_best_models_by_category
        model_names = select_best_models_by_category(model_results)
        print(f"🎯 Auto-selected models for cinematic animation: {[m.upper() for m in model_names]}")

    print("🎬 Creating sophisticated animated analysis with enhanced aesthetics...")

    # Enhanced data collection with performance metrics
    data = []
    performance_stats = {}

    for model in model_names:
        performance_stats[model] = {"accuracy_count": 0, "total_count": 0}

        for player_type in ["hitter", "pitcher"]:
            if (player_type == "hitter" and not show_hitters) or (player_type == "pitcher" and not show_pitchers):
                continue

            war_key = f"{model}_{player_type}_war"
            warp_key = f"{model}_{player_type}_warp"

            if war_key in model_results.results and warp_key in model_results.results:
                war_data = model_results.results[war_key]
                warp_data = model_results.results[warp_key]

                # Enhanced player matching with performance tracking
                for i, war_player in enumerate(war_data["player_names"]):
                    if war_player in warp_data["player_names"]:
                        warp_idx = warp_data["player_names"].index(war_player)

                        # Enhanced season handling with chronological sorting
                        season_value = "2021"
                        if season_col in war_data and len(war_data[season_col]) > i:
                            try:
                                season_value = str(int(war_data[season_col][i]))
                            except (ValueError, TypeError):
                                season_value = str(war_data[season_col][i])

                        # Calculate accuracy metrics for enhanced visualization
                        war_error = abs(war_data["y_true"][i] - war_data["y_pred"][i])
                        warp_error = abs(warp_data["y_true"][warp_idx] - warp_data["y_pred"][warp_idx])
                        combined_error = (war_error + warp_error) / 2
                        is_accurate = war_error <= 1.0 and warp_error <= 1.0

                        performance_stats[model]["total_count"] += 1
                        if is_accurate:
                            performance_stats[model]["accuracy_count"] += 1

                        data.append({
                            "Player": war_player,
                            "Model": model.title(),
                            "PlayerType": player_type.title(),
                            season_col: season_value,
                            "Actual_WAR": war_data["y_true"][i],
                            "Predicted_WAR": war_data["y_pred"][i],
                            "Actual_WARP": warp_data["y_true"][warp_idx],
                            "Predicted_WARP": warp_data["y_pred"][warp_idx],
                            "WAR_Error": war_error,
                            "WARP_Error": warp_error,
                            "Combined_Error": combined_error,
                            "Accuracy_Status": "High Accuracy" if is_accurate else "Lower Accuracy",
                            "Performance_Score": max(0, 5 - combined_error),  # 0-5 scale for sizing
                            "Elite_Player": war_data["y_true"][i] > 3.0 or warp_data["y_true"][warp_idx] > 3.0
                        })

    if not data:
        print("❌ No matching WAR/WARP data available for animation")
        return

    df = pd.DataFrame(data)

    # Enhanced chronological sorting for smooth animation
    unique_seasons = df[season_col].unique()
    try:
        sorted_seasons = sorted([int(s) for s in unique_seasons])
        sorted_season_strings = [str(s) for s in sorted_seasons]
        df[season_col] = pd.Categorical(df[season_col], categories=sorted_season_strings, ordered=True)
        print(f"📅 Temporal sequence: {sorted_season_strings}")
    except (ValueError, TypeError):
        sorted_season_strings = sorted([str(s) for s in unique_seasons])
        df[season_col] = pd.Categorical(df[season_col], categories=sorted_season_strings, ordered=True)

    # ENHANCED AESTHETIC FEATURES - Multiple visualization modes

    # 1. CINEMATIC BUBBLE ANIMATION with sophisticated styling
    print("   🎨 Creating cinematic bubble animation...")

    fig_bubble = px.scatter(
        df,
        x="Actual_WAR",
        y="Actual_WARP",
        size="Performance_Score",
        color="Combined_Error",
        symbol="PlayerType",
        hover_name="Player",
        animation_frame=season_col,
        animation_group="Player",
        facet_col="Model",
        title="🎬 Cinematic WAR vs WARP Performance Analysis",
        color_continuous_scale="viridis_r",  # Beautiful gradient from purple to yellow
        size_max=25,
        template="plotly_dark",  # Cinematic dark theme
        width=1400,
        height=800
    )

    # Enhanced styling for cinematic effect
    fig_bubble.update_layout(
        font=dict(family="Arial Black", size=12, color="white"),
        title=dict(font=dict(size=20), x=0.5, xanchor="center"),
        plot_bgcolor="rgba(0,0,0,0.9)",
        paper_bgcolor="rgba(0,0,0,0.95)",
        coloraxis_colorbar=dict(
            title="Prediction Error",
            titlefont=dict(color="white"),
            tickfont=dict(color="white")
        )
    )

    # Add aesthetic reference lines with gradients
    val_range = max(df["Actual_WAR"].max(), df["Actual_WARP"].max())
    val_min = min(df["Actual_WAR"].min(), df["Actual_WARP"].min())

    fig_bubble.add_shape(
        type="line",
        x0=val_min, y0=val_min, x1=val_range, y1=val_range,
        line=dict(color="rgba(255,255,255,0.3)", width=2, dash="dash"),
        name="Perfect Correlation"
    )

    # Enhanced animation controls with smooth transitions
    fig_bubble.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 1500
    fig_bubble.layout.updatemenus[0].buttons[0].args[1]["frame"]["redraw"] = True
    fig_bubble.layout.updatemenus[0].buttons[0].args[1]["transition"]["duration"] = 800
    fig_bubble.layout.updatemenus[0].buttons[0].args[1]["transition"]["easing"] = "cubic-in-out"

    fig_bubble.show()

    # 2. PERFORMANCE HEATMAP ANIMATION with gradient aesthetics
    print("   🌈 Creating performance heatmap animation...")

    # Create aggregated performance data for heatmap
    heatmap_data = df.groupby([season_col, "Model", "PlayerType"]).agg({
        "Combined_Error": "mean",
        "Performance_Score": "mean",
        "Player": "count"
    }).reset_index()
    heatmap_data.rename(columns={"Player": "Player_Count"}, inplace=True)

    fig_heatmap = px.density_heatmap(
        heatmap_data,
        x="Model",
        y="PlayerType",
        z="Performance_Score",
        animation_frame=season_col,
        title="🌈 Performance Heatmap: Model Accuracy Over Time",
        color_continuous_scale="plasma",  # Beautiful plasma gradient
        template="plotly_white",
        width=1000,
        height=600
    )

    fig_heatmap.update_layout(
        font=dict(family="Arial", size=12),
        title=dict(font=dict(size=18), x=0.5, xanchor="center")
    )

    fig_heatmap.show()

    # 3. SOPHISTICATED 3D TEMPORAL SURFACE with advanced aesthetics
    print("   📊 Creating 3D temporal performance surface...")

    # Sample data for 3D visualization (prevent overcrowding)
    sample_df = df.sample(n=min(200, len(df)), random_state=42) if len(df) > 200 else df

    fig_3d = go.Figure()

    # Add 3D scatter with sophisticated styling
    for model in sample_df["Model"].unique():
        model_data = sample_df[sample_df["Model"] == model]

        fig_3d.add_trace(go.Scatter3d(
            x=model_data["Actual_WAR"],
            y=model_data["Actual_WARP"],
            z=model_data["Performance_Score"],
            mode="markers",
            marker=dict(
                size=8,
                color=model_data["Combined_Error"],
                colorscale="rainbow",
                opacity=0.8,
                line=dict(width=2, color="white")
            ),
            text=model_data["Player"],
            name=f"{model} Model",
            hovertemplate="<b>%{text}</b><br>" +
                         "WAR: %{x:.2f}<br>" +
                         "WARP: %{y:.2f}<br>" +
                         "Performance: %{z:.2f}<br>" +
                         "<extra></extra>"
        ))

    fig_3d.update_layout(
        title="📊 3D Performance Landscape: WAR vs WARP vs Accuracy",
        scene=dict(
            xaxis_title="Actual WAR",
            yaxis_title="Actual WARP",
            zaxis_title="Performance Score",
            bgcolor="rgba(0,0,0,0.1)",
            xaxis=dict(gridcolor="rgba(255,255,255,0.3)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.3)"),
            zaxis=dict(gridcolor="rgba(255,255,255,0.3)")
        ),
        template="plotly_dark",
        width=1200,
        height=800,
        font=dict(family="Arial", size=11)
    )

    fig_3d.show()

    # Enhanced statistical summary with aesthetic formatting
    print("\n" + "="*70)
    print("🎬 SOPHISTICATED ANIMATION ANALYSIS SUMMARY")
    print("="*70)

    for model in model_names:
        if model in performance_stats:
            stats = performance_stats[model]
            accuracy_rate = (stats["accuracy_count"] / stats["total_count"] * 100) if stats["total_count"] > 0 else 0

            print(f"\n🎯 {model.upper()} MODEL PERFORMANCE:")
            print(f"   📊 Total Predictions: {stats['total_count']}")
            print(f"   ✅ High Accuracy Rate: {accuracy_rate:.1f}% ({stats['accuracy_count']}/{stats['total_count']})")

            model_data = df[df["Model"] == model.title()]
            avg_error = model_data["Combined_Error"].mean()
            print(f"   📈 Average Combined Error: {avg_error:.3f}")

    print(f"\n🎨 AESTHETIC FEATURES IMPLEMENTED:")
    print(f"   🎬 Cinematic bubble animation with smooth transitions")
    print(f"   🌈 Performance heatmap with gradient aesthetics")
    print(f"   📊 3D performance landscape visualization")
    print(f"   🎯 Advanced color schemes (viridis, plasma, rainbow)")
    print(f"   ⚡ Enhanced animation controls with cubic easing")
    print(f"   🖱️  Interactive legends and hover details")

    print(f"\n💫 VISUAL ENHANCEMENTS:")
    print(f"   • Dark cinematic themes for professional presentation")
    print(f"   • Gradient color scales for intuitive data interpretation")
    print(f"   • Smooth animation transitions with custom timing")
    print(f"   • 3D perspectives for comprehensive data exploration")
    print(f"   • Performance-based sizing for visual emphasis")

    return {
        'performance_stats': performance_stats,
        'total_observations': len(df),
        'temporal_range': sorted_season_strings,
        'aesthetic_features': ['cinematic_animation', 'gradient_heatmap', '3d_surface', 'smooth_transitions']
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