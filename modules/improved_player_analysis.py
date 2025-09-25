# FUTURE REPLACEMENT FOR: sWARm_AgeCurve.ipynb player analysis functions
# This module will replace the analyze_player_historical_performance and
# create_player_visualization functions in the notebook once validated

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def validate_player_data(player_data, player_name):
    """
    Validate player data for integrity and detect potential issues.

    Returns:
        dict: Validation results with warnings and data quality metrics
    """
    validation = {
        'warnings': [],
        'data_quality_score': 1.0,
        'season_duplicates': False,
        'total_records': len(player_data)
    }

    if len(player_data) == 0:
        validation['warnings'].append(f"No data found for {player_name}")
        validation['data_quality_score'] = 0.0
        return validation

    # Check for duplicate season/datasource combinations
    duplicates = player_data.groupby(['Season', 'DataSource']).size()
    if any(duplicates > 1):
        validation['season_duplicates'] = True
        validation['warnings'].append("Found duplicate season/datasource combinations")
        validation['data_quality_score'] -= 0.3

        # Log the duplicates for debugging
        duplicate_seasons = duplicates[duplicates > 1].index.tolist()
        for season, datasource in duplicate_seasons:
            validation['warnings'].append(f"  Duplicate: {season} {datasource} ({duplicates[(season, datasource)]} records)")

    # Check for reasonable WAR ranges
    war_values = player_data['WAR'].dropna()
    if len(war_values) > 0:
        if war_values.min() < -5 or war_values.max() > 15:
            validation['warnings'].append(f"Extreme WAR values detected: {war_values.min():.1f} to {war_values.max():.1f}")
            validation['data_quality_score'] -= 0.2

    # Check age consistency
    if 'Age' in player_data.columns:
        age_data = player_data[player_data['Age'].notna()].copy()
        if len(age_data) > 1:
            age_data = age_data.sort_values('Season')
            age_jumps = age_data['Age'].diff().dropna()
            # Age should increase by ~1 per season, allow some tolerance
            unusual_jumps = age_jumps[(age_jumps < 0.5) | (age_jumps > 2.0)]
            if len(unusual_jumps) > 0:
                validation['warnings'].append("Unusual age progression detected")
                validation['data_quality_score'] -= 0.1

    return validation

def analyze_player_historical_performance_improved(player_name, data):
    """
    Improved version of analyze_player_historical_performance with data validation
    and cleaner aggregation logic.
    """
    player_data = data[data['Name'] == player_name].copy()

    if len(player_data) == 0:
        return None

    # Validate data first
    validation = validate_player_data(player_data, player_name)

    # Sort by season for consistent processing
    player_data = player_data.sort_values('Season')

    # Handle duplicates by taking the first occurrence (or could average/sum based on context)
    if validation['season_duplicates']:
        print(f"⚠ Removing duplicates for {player_name}")
        player_data = player_data.drop_duplicates(subset=['Season', 'DataSource'], keep='first')

    # Separate WARP and WAR data cleanly
    warp_data = player_data[player_data['DataSource'] == 'WARP'].copy()
    war_data = player_data[player_data['DataSource'] == 'WAR'].copy()

    # Debug: Show what we're summing
    print(f"Debug - {player_name}:")
    print(f"  WARP seasons: {len(warp_data)} records, values: {warp_data['WAR'].tolist()}")
    print(f"  WAR seasons: {len(war_data)} records, values: {war_data['WAR'].tolist()}")

    # Calculate career totals with validation
    career_warp = warp_data['WAR'].sum() if len(warp_data) > 0 else 0
    career_war = war_data['WAR'].sum() if len(war_data) > 0 else 0

    # Sanity check: warn if career totals seem unreasonable
    expected_career_range = (0, 120)  # Most players won't exceed 120 career WAR
    if career_war > expected_career_range[1]:
        validation['warnings'].append(f"Career WAR ({career_war:.1f}) exceeds typical range")
    if career_warp > expected_career_range[1]:
        validation['warnings'].append(f"Career WARP ({career_warp:.1f}) exceeds typical range")

    # Peak seasons for each metric
    peak_warp = warp_data['WAR'].max() if len(warp_data) > 0 else 0
    peak_war = war_data['WAR'].max() if len(war_data) > 0 else 0

    peak_warp_season = warp_data.loc[warp_data['WAR'].idxmax(), 'Season'] if len(warp_data) > 0 else None
    peak_war_season = war_data.loc[war_data['WAR'].idxmax(), 'Season'] if len(war_data) > 0 else None

    # Build analysis dictionary
    analysis = {
        # Raw data for plotting
        'seasons': player_data['Season'].tolist(),
        'ages': player_data['Age'].tolist(),
        'war_values': player_data['WAR'].tolist(),
        'positions': player_data['Primary_Position'].tolist(),
        'data_sources': player_data['DataSource'].tolist(),

        # Separate career totals
        'career_warp': career_warp,
        'career_war': career_war,
        'peak_warp': peak_warp,
        'peak_war': peak_war,
        'peak_warp_season': peak_warp_season,
        'peak_war_season': peak_war_season,

        # Additional info
        'age_range': f"{player_data['Age'].min():.0f}-{player_data['Age'].max():.0f}" if player_data['Age'].notna().any() else "Unknown",
        'primary_position': player_data['Primary_Position'].mode().iloc[0] if len(player_data['Primary_Position'].mode()) > 0 else 'Unknown',
        'warp_seasons': len(warp_data),
        'war_seasons': len(war_data),
        'total_records': len(player_data),

        # Data quality information
        'validation': validation
    }

    # Print warnings if any
    if validation['warnings']:
        print(f"⚠ Data quality warnings for {player_name}:")
        for warning in validation['warnings']:
            print(f"    {warning}")
        print(f"  Data quality score: {validation['data_quality_score']:.2f}")

    return analysis

def create_improved_player_visualization(player_analysis, player_name):
    """
    Create improved visualization with streamlined 3-plot layout:
    1. WAR by Season (with age in hover)
    2. WARP by Season (with age in hover)
    3. Combined Age Curve Analysis
    """
    if player_analysis is None:
        return None

    # Create subplot layout
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            f'{player_name} - WAR by Season',
            f'{player_name} - WARP by Season',
            'Age Curve Analysis',
            'Performance Summary'
        ),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"type": "table"}]]
    )

    # Separate data by metric for cleaner plotting
    war_indices = [i for i, ds in enumerate(player_analysis['data_sources']) if ds == 'WAR']
    warp_indices = [i for i, ds in enumerate(player_analysis['data_sources']) if ds == 'WARP']

    # Plot 1: WAR by Season (blue)
    if war_indices:
        war_seasons = [player_analysis['seasons'][i] for i in war_indices]
        war_values = [player_analysis['war_values'][i] for i in war_indices]
        war_ages = [player_analysis['ages'][i] for i in war_indices]

        hover_text = [f"Season: {s}<br>Age: {a:.1f}<br>WAR: {w:.1f}"
                     for s, a, w in zip(war_seasons, war_ages, war_values)]

        fig.add_trace(
            go.Scatter(
                x=war_seasons,
                y=war_values,
                mode='lines+markers',
                name='WAR',
                line=dict(color='blue', width=2),
                marker=dict(size=8, color='blue'),
                hovertext=hover_text,
                hoverinfo='text',
                showlegend=False
            ),
            row=1, col=1
        )

        # Highlight peak WAR season
        if player_analysis['peak_war_season']:
            peak_idx = war_seasons.index(player_analysis['peak_war_season'])
            fig.add_trace(
                go.Scatter(
                    x=[war_seasons[peak_idx]],
                    y=[war_values[peak_idx]],
                    mode='markers',
                    marker=dict(size=12, color='gold', line=dict(color='blue', width=2)),
                    hovertext=f"Peak WAR: {war_values[peak_idx]:.1f}",
                    hoverinfo='text',
                    showlegend=False
                ),
                row=1, col=1
            )

    # Plot 2: WARP by Season (red)
    if warp_indices:
        warp_seasons = [player_analysis['seasons'][i] for i in warp_indices]
        warp_values = [player_analysis['war_values'][i] for i in warp_indices]
        warp_ages = [player_analysis['ages'][i] for i in warp_indices]

        hover_text = [f"Season: {s}<br>Age: {a:.1f}<br>WARP: {w:.1f}"
                     for s, a, w in zip(warp_seasons, warp_ages, warp_values)]

        fig.add_trace(
            go.Scatter(
                x=warp_seasons,
                y=warp_values,
                mode='lines+markers',
                name='WARP',
                line=dict(color='red', width=2),
                marker=dict(size=8, color='red'),
                hovertext=hover_text,
                hoverinfo='text',
                showlegend=False
            ),
            row=1, col=2
        )

        # Highlight peak WARP season
        if player_analysis['peak_warp_season']:
            peak_idx = warp_seasons.index(player_analysis['peak_warp_season'])
            fig.add_trace(
                go.Scatter(
                    x=[warp_seasons[peak_idx]],
                    y=[warp_values[peak_idx]],
                    mode='markers',
                    marker=dict(size=12, color='gold', line=dict(color='red', width=2)),
                    hovertext=f"Peak WARP: {warp_values[peak_idx]:.1f}",
                    hoverinfo='text',
                    showlegend=False
                ),
                row=1, col=2
            )

    # Plot 3: Age Curve Analysis (if age_curve_model available)
    # This would need to be passed in or accessed from global scope
    try:
        # Assuming age_curve_model is available in the calling context
        ages_range = np.arange(20, 40, 0.5)
        position = player_analysis['primary_position']

        # This line will need to be adapted based on how age_curve_model is accessed
        # age_factors = [age_curve_model.calculate_age_curve_factor(age, position) for age in ages_range]

        # For now, create a placeholder curve
        age_factors = [1.0 - abs(age - 27) * 0.02 for age in ages_range]  # Simple peak at 27

        fig.add_trace(
            go.Scatter(
                x=ages_range,
                y=age_factors,
                mode='lines',
                name='Expected Age Curve',
                line=dict(color='gray', width=2, dash='dash'),
                showlegend=False
            ),
            row=2, col=1
        )

        # Add player's actual performance by age
        if player_analysis['ages']:
            player_ages = [age for age in player_analysis['ages'] if pd.notna(age)]
            if player_ages:
                # Normalize player performance for comparison
                max_war = max([abs(w) for w in player_analysis['war_values']])
                normalized_performance = [w/max_war for w in player_analysis['war_values']] if max_war > 0 else [0] * len(player_analysis['war_values'])

                colors = ['red' if ds == 'WARP' else 'blue' for ds in player_analysis['data_sources']]
                fig.add_trace(
                    go.Scatter(
                        x=player_analysis['ages'],
                        y=normalized_performance,
                        mode='markers',
                        marker=dict(size=8, color=colors),
                        name='Actual Performance (normalized)',
                        showlegend=False
                    ),
                    row=2, col=1
                )
    except:
        # If age curve analysis fails, show a simple message
        fig.add_annotation(
            text="Age curve analysis unavailable",
            xref="x3", yref="y3",
            x=0.5, y=0.5,
            showarrow=False,
            row=2, col=1
        )

    # Performance Summary Table
    table_data = [
        ['Metric', 'Career Total', 'Peak Season', 'Peak Value'],
        ['WAR', f"{player_analysis['career_war']:.1f}" if player_analysis['career_war'] > 0 else 'N/A',
         f"{player_analysis['peak_war_season']}" if player_analysis['peak_war_season'] else 'N/A',
         f"{player_analysis['peak_war']:.1f}" if player_analysis['peak_war'] > 0 else 'N/A'],
        ['WARP', f"{player_analysis['career_warp']:.1f}" if player_analysis['career_warp'] > 0 else 'N/A',
         f"{player_analysis['peak_warp_season']}" if player_analysis['peak_warp_season'] else 'N/A',
         f"{player_analysis['peak_warp']:.1f}" if player_analysis['peak_warp'] > 0 else 'N/A'],
        ['Age Range', player_analysis['age_range'], '', ''],
        ['Position', player_analysis['primary_position'], '', ''],
        ['Data Quality', f"{player_analysis['validation']['data_quality_score']:.2f}", '', '']
    ]

    fig.add_trace(
        go.Table(
            header=dict(values=table_data[0], fill_color='lightblue', align='left'),
            cells=dict(values=list(zip(*table_data[1:])), fill_color='white', align='left')
        ),
        row=2, col=2
    )

    # Update layout
    fig.update_layout(
        title=f"Player Analysis: {player_name}",
        height=800,
        showlegend=False
    )

    # Update axis labels
    fig.update_xaxes(title_text="Season", row=1, col=1)
    fig.update_yaxes(title_text="WAR", row=1, col=1)
    fig.update_xaxes(title_text="Season", row=1, col=2)
    fig.update_yaxes(title_text="WARP", row=1, col=2)
    fig.update_xaxes(title_text="Age", row=2, col=1)
    fig.update_yaxes(title_text="Performance Factor", row=2, col=1)

    return fig

def quick_player_comparison(players, data):
    """
    Quick comparison function to validate career totals against expected values.
    Useful for debugging data integrity issues.
    """
    comparison_results = []

    for player in players:
        analysis = analyze_player_historical_performance_improved(player, data)
        if analysis:
            comparison_results.append({
                'player': player,
                'career_war': analysis['career_war'],
                'career_warp': analysis['career_warp'],
                'war_seasons': analysis['war_seasons'],
                'warp_seasons': analysis['warp_seasons'],
                'data_quality': analysis['validation']['data_quality_score'],
                'warnings': len(analysis['validation']['warnings'])
            })

    return pd.DataFrame(comparison_results)