"""
Data preparation utilities for oWAR system.

Provides functions for creating training data with flexible season splits.
"""

from .season_splitter import create_multipoint_splits, calculate_season_completion

__all__ = [
    'create_multipoint_splits',
    'calculate_season_completion'
]
