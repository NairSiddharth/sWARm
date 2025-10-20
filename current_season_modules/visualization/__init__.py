"""
Visualization package for current season analysis

This package contains visualization functions for WAR/WARP analysis,
scenario projections, model validation, and player comparisons.
"""

# Existing imports (maintain backward compatibility)
try:
    from .scenario_charts import (
        create_scenario_projection_chart,
        create_current_vs_projected_comparison
    )
except ImportError:
    pass

try:
    from .tracking_charts import (
        create_war_warp_tracking_chart
    )
except ImportError:
    pass

try:
    from .comparison_charts import (
        create_player_comparison_dashboard
    )
except ImportError:
    pass

try:
    from .compatibility import (
        plot_year_specific_analysis,
        create_all_year_graphs,
        plot_year_comparison_summary,
        print_year_analysis_summary
    )
except ImportError:
    pass

# New visualization modules
from .validation_charts import (
    create_validation_dashboard,
    create_actual_vs_predicted_facet,
    create_residual_analysis_facet,
    create_cv_performance_plot,
    create_tier_based_performance
)

from .projection_charts import (
    create_projection_dashboard,
    create_comparison_chart,
    create_enhanced_boxplot,
    create_range_chart,
    create_scenario_heatmap
)

from .chart_utils import (
    apply_standard_layout,
    calculate_tier,
    validate_projection_data,
    validate_validation_data,
    export_figure,
    create_error_figure,
    SCENARIO_COLORS,
    TIER_COLORS,
    ERROR_MAGNITUDE_COLORS,
    MODEL_TYPE_COLORS
)

__all__ = [
    # Scenario visualizations
    'create_scenario_projection_chart',
    'create_current_vs_projected_comparison',

    # Tracking visualizations
    'create_war_warp_tracking_chart',

    # Comparison visualizations
    'create_player_comparison_dashboard',

    # Backward compatibility functions
    'plot_year_specific_analysis',
    'create_all_year_graphs',
    'plot_year_comparison_summary',
    'print_year_analysis_summary',

    # New validation visualizations
    'create_validation_dashboard',
    'create_actual_vs_predicted_facet',
    'create_residual_analysis_facet',
    'create_cv_performance_plot',
    'create_tier_based_performance',

    # New projection visualizations
    'create_projection_dashboard',
    'create_comparison_chart',
    'create_enhanced_boxplot',
    'create_range_chart',
    'create_scenario_heatmap',

    # Utilities
    'apply_standard_layout',
    'calculate_tier',
    'validate_projection_data',
    'validate_validation_data',
    'export_figure',
    'create_error_figure',
    'SCENARIO_COLORS',
    'TIER_COLORS',
    'ERROR_MAGNITUDE_COLORS',
    'MODEL_TYPE_COLORS'
]
