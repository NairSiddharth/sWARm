"""
Shared utilities for oWAR notebooks.

Provides high-level functions for:
- Pipeline execution (pipeline_runner)
- Interactive plotting (plotting_utils)
- Table formatting (table_utils)
- Advanced analysis (analysis_utils)
"""

# Pipeline execution
from .pipeline_runner import (
    load_current_season_data,
    load_historical_data,
    run_data_pipeline,
    generate_predictions,
    split_by_role,
    split_by_position,
    calculate_metrics,
    create_combined_leaderboard
)

# Plotting
from .plotting_utils import (
    create_war_scatter,
    create_actual_vs_predicted,
    create_residual_plot,
    create_feature_importance,
    create_correlation_heatmap,
    create_partial_dependence
)

# Table formatting
from .table_utils import (
    create_featured_table,
    create_metrics_table,
    get_pitcher_type,
    get_hitter_position,
    get_rank_within_type,
    handle_two_way_player
)

# Analysis
from .analysis_utils import (
    calculate_elite_performance,
    calculate_replacement_performance,
    analyze_errors_by_group,
    calculate_shap_values,
    find_outliers,
    compare_models
)

__all__ = [
    # Pipeline execution
    'load_current_season_data',
    'load_historical_data',
    'run_data_pipeline',
    'generate_predictions',
    'split_by_role',
    'split_by_position',
    'calculate_metrics',
    'create_combined_leaderboard',
    # Plotting
    'create_war_scatter',
    'create_actual_vs_predicted',
    'create_residual_plot',
    'create_feature_importance',
    'create_correlation_heatmap',
    'create_partial_dependence',
    # Table formatting
    'create_featured_table',
    'create_metrics_table',
    'get_pitcher_type',
    'get_hitter_position',
    'get_rank_within_type',
    'handle_two_way_player',
    # Analysis
    'calculate_elite_performance',
    'calculate_replacement_performance',
    'analyze_errors_by_group',
    'calculate_shap_values',
    'find_outliers',
    'compare_models'
]
