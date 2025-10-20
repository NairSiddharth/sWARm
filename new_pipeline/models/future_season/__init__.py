"""
Future Season Projection Models

Multi-year WAR projections (1-3 years ahead) using longitudinal and survival modeling.
"""

# Data preparation (Week 1)
from .data_preparation import (
    load_historical_player_data,
    build_longitudinal_sequences,
    add_age_context_features,
    create_temporal_splits
)

# Core models (Week 2)
from .age_curves import AgeCurveAdjuster
from .longitudinal_model import LongitudinalModel
from .survival_model import SurvivalModel

# Joint projection and adjustments (Week 3)
from .joint_projection import JointProjectionModel
from .expected_stats import ExpectedStatsCalculator

# Phase 4: Adjustments (Week 4)
from .elite_adjustments import (
    apply_elite_adjustments,
    apply_elite_adjustment_to_war,
    FutureEliteProtector
)
from .injury_recovery import (
    InjuryRecoveryAdjuster,
    apply_injury_recovery
)
from .constraint_optimizer import ConstraintOptimizer

# Phase 5: Validation & Pipeline (Week 5)
from .temporal_validation import (
    TemporalValidator,
    validate_model_temporal_cv
)
from .future_projection_pipeline import (
    FutureProjectionPipeline,
    generate_league_projections
)

__all__ = [
    # Data preparation
    'load_historical_player_data',
    'build_longitudinal_sequences',
    'add_age_context_features',
    'create_temporal_splits',

    # Core models
    'AgeCurveAdjuster',
    'LongitudinalModel',
    'SurvivalModel',

    # Joint projection and adjustments
    'JointProjectionModel',
    'ExpectedStatsCalculator',

    # Phase 4: Adjustments
    'apply_elite_adjustments',
    'apply_elite_adjustment_to_war',
    'FutureEliteProtector',
    'InjuryRecoveryAdjuster',
    'apply_injury_recovery',
    'ConstraintOptimizer',

    # Phase 5: Validation & Pipeline
    'TemporalValidator',
    'validate_model_temporal_cv',
    'FutureProjectionPipeline',
    'generate_league_projections',
]
