"""
SYSTEM 2: Future Performance Projections
========================================

This package implements ZIPS-style future projections using joint longitudinal-survival modeling
for 1-3 year performance forecasting with temporal GroupKFold cross-validation.

Components:
- data_loader: Age data extraction and MLBID integration
- expected_stats: 3-year weighted averages with expected metrics blending
- future_projections: Joint longitudinal-survival modeling for multi-year forecasts
- validation: Temporal cross-validation with survival considerations
- integration: Complete SYSTEM 2 pipeline orchestration
"""

from .expected_stats import ExpectedStatsCalculator
from .future_projections import FutureProjectionAgeCurve
from .validation import AgeCurveValidator
from .integration import System2Pipeline

__all__ = [
    'ExpectedStatsCalculator',
    'FutureProjectionAgeCurve',
    'AgeCurveValidator',
    'System2Pipeline'
]