"""
Pipeline assembly helpers for the oWAR pipeline.

Convenience functions to build complete sklearn pipelines.
"""
from typing import List

from sklearn.pipeline import Pipeline

from ..constants import (
    COL_MLBAMID,
    COL_BB_PCT,
    COL_K_PCT,
    COL_SWSTR_PCT,
    COL_ERA,
    COL_GB_PCT,
    COL_WAR,
    COL_AVG,
    COL_OBP,
    COL_SLG,
    COL_PA,
    MIN_IP_DEFAULT,
    MIN_PA_DEFAULT,
    VALID_RANGE_BB_PCT,
    VALID_RANGE_K_PCT,
    VALID_RANGE_SWSTR_PCT,
    VALID_RANGE_ERA,
    VALID_RANGE_GB_PCT,
    VALID_RANGE_WAR,
    VALID_RANGE_HITTER_BB_PCT,
    VALID_RANGE_HITTER_K_PCT,
    VALID_RANGE_AVG,
    VALID_RANGE_OBP,
    VALID_RANGE_SLG,
    PITCHER_CRITICAL_FEATURES,
    HITTER_CRITICAL_FEATURES,
    PITCHER_MODEL_FEATURES,
    HITTER_MODEL_FEATURES
)
from .filters import NoIDFilter, IPFilter, PAFilter, TwoWayPlayerFilter
from .normalizers import WARNormalizer
from .validators import FeatureValidator, MissingValueImputer
from .pitcher_features import PitcherFeatureTransformer
from .pitcher_composite_transformer import PitcherCompositeTransformer
from .hitter_features import HitterFeatureTransformer
from .feature_selector import FeatureSelector


def build_pitcher_pipeline(
    years: List[int],
    min_ip: float = MIN_IP_DEFAULT,
    include_validation: bool = True
) -> Pipeline:
    """
    Build a complete pitcher feature pipeline.

    Pipeline steps:
    1. NoIDFilter - Remove rows without player IDs
    2. TwoWayPlayerFilter - Identify and mark two-way players
    3. IPFilter - Multi-criteria pitcher qualification (excludes position players)
    4. PitcherFeatureTransformer - Load all 11 base features
    5. PitcherCompositeTransformer - Calculate 6 composite features (total 13 features)
    6. MissingValueImputer - Impute missing values with replacement level (25th percentile)
    7. FeatureValidator - Validate ranges (optional)
    8. FeatureSelector - Select 13 modeling features + metadata
    9. WARNormalizer - Add WAR_per_162 column

    Args:
        years: Years to load features for
        min_ip: Minimum IP threshold (DEPRECATED - filter uses multi-criteria now)
        include_validation: Include feature validation step (default True)

    Returns:
        Complete pitcher pipeline

    Example:
        >>> pipeline = build_pitcher_pipeline([2022, 2023, 2024])
        >>> processed_data = pipeline.fit_transform(raw_pitcher_df)
    """
    steps = [
        ('no_id_filter', NoIDFilter(id_column=COL_MLBAMID)),
        ('two_way_filter', TwoWayPlayerFilter()),
        ('ip_filter', IPFilter()),
        ('feature_loader', PitcherFeatureTransformer(years=years)),
        ('composite_calculator', PitcherCompositeTransformer()),
        ('imputer', MissingValueImputer()),
    ]

    if include_validation:
        # Define expected feature ranges for pitchers
        range_checks = {
            COL_BB_PCT: VALID_RANGE_BB_PCT,
            COL_K_PCT: VALID_RANGE_K_PCT,
            COL_SWSTR_PCT: VALID_RANGE_SWSTR_PCT,
            COL_ERA: VALID_RANGE_ERA,
            COL_GB_PCT: VALID_RANGE_GB_PCT,
            COL_WAR: VALID_RANGE_WAR
        }
        steps.append(('validator', FeatureValidator(
            critical_features=PITCHER_CRITICAL_FEATURES,
            range_checks=range_checks,
            strict_mode=False  # Warn, don't fail
        )))

    # Select only modeling features + metadata (filters out intermediate features)
    steps.append(('feature_selector', FeatureSelector(
        feature_columns=PITCHER_MODEL_FEATURES,
        keep_metadata=True
    )))

    steps.append(('war_normalizer', WARNormalizer(player_type='pitcher')))

    return Pipeline(steps)


def build_hitter_pipeline(
    years: List[int],
    min_pa: int = MIN_PA_DEFAULT,
    include_validation: bool = True
) -> Pipeline:
    """
    Build a complete hitter feature pipeline.

    Pipeline steps:
    1. NoIDFilter - Remove rows without player IDs
    2. TwoWayPlayerFilter - Identify and mark two-way players
    3. PAFilter - Remove hitters with < min_pa (auto-adjusts for partial seasons)
    4. HitterFeatureTransformer - Load all 10 features
    5. MissingValueImputer - Impute missing values with replacement level (25th percentile)
    6. FeatureValidator - Validate ranges (optional)
    7. FeatureSelector - Select 10 modeling features + metadata
    8. WARNormalizer - Add WAR_per_600 column

    Args:
        years: Years to load features for
        min_pa: Minimum PA threshold (default from constants, auto-adjusts for partial seasons)
        include_validation: Include feature validation step (default True)

    Returns:
        Complete hitter pipeline

    Example:
        >>> pipeline = build_hitter_pipeline([2022, 2023, 2024], min_pa=100)
        >>> processed_data = pipeline.fit_transform(raw_hitter_df)
    """
    steps = [
        ('no_id_filter', NoIDFilter(id_column=COL_MLBAMID)),
        ('two_way_filter', TwoWayPlayerFilter()),
        ('pa_filter', PAFilter(min_pa=min_pa)),
        ('feature_loader', HitterFeatureTransformer(years=years)),
        ('imputer', MissingValueImputer()),
    ]

    if include_validation:
        # Define expected feature ranges for hitters
        range_checks = {
            COL_BB_PCT: VALID_RANGE_HITTER_BB_PCT,
            COL_K_PCT: VALID_RANGE_HITTER_K_PCT,
            COL_AVG: VALID_RANGE_AVG,
            COL_OBP: VALID_RANGE_OBP,
            COL_SLG: VALID_RANGE_SLG,
            COL_WAR: VALID_RANGE_WAR
        }
        steps.append(('validator', FeatureValidator(
            critical_features=HITTER_CRITICAL_FEATURES,
            range_checks=range_checks,
            strict_mode=False  # Warn, don't fail
        )))

    # Select only modeling features + metadata
    steps.append(('feature_selector', FeatureSelector(
        feature_columns=HITTER_MODEL_FEATURES,
        keep_metadata=True
    )))

    steps.append(('war_normalizer', WARNormalizer(player_type='hitter')))

    return Pipeline(steps)
