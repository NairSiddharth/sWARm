"""
Validation transformers for the oWAR pipeline.

All validators follow sklearn's transformer pattern (BaseEstimator, TransformerMixin).
"""
from typing import Dict, List, Optional, Tuple

from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

from ..constants import REPLACEMENT_LEVEL_PERCENTILE
from ..exceptions import InvalidDataTypeError, ImputerNotFittedError
from ..logging_config import get_logger

logger = get_logger(__name__)


class FeatureValidator(BaseEstimator, TransformerMixin):
    """
    Validate feature ranges and data quality.

    Checks:
    - No infinite values
    - No NaN in critical features
    - Features within expected ranges

    Can either raise errors or log warnings based on strict_mode.
    """

    def __init__(
        self,
        critical_features: Optional[List[str]] = None,
        range_checks: Optional[Dict[str, Tuple[float, float]]] = None,
        strict_mode: bool = False
    ):
        """
        Args:
            critical_features: Features that must not have NaN
            range_checks: {feature: (min, max)} expected ranges
            strict_mode: If True, raise errors. If False, warn only.
        """
        self.critical_features = critical_features or []
        self.range_checks = range_checks or {}
        self.strict_mode = strict_mode

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'FeatureValidator':
        """Nothing to fit."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Validate features.

        Args:
            X: Input data

        Returns:
            Same data (possibly with warnings)

        Raises:
            InvalidDataTypeError: If X is not a DataFrame
            ValueError: If strict_mode=True and issues found
        """
        if not isinstance(X, pd.DataFrame):
            raise InvalidDataTypeError('X', 'pd.DataFrame', type(X))

        issues = []

        # Check for infinite values
        inf_cols = []
        for col in X.select_dtypes(include=[np.number]).columns:
            if np.isinf(X[col]).any():
                inf_cols.append(col)

        if inf_cols:
            msg = f"Infinite values found in: {inf_cols}"
            issues.append(msg)

        # Check critical features for NaN
        for feature in self.critical_features:
            if feature in X.columns:
                nan_count = X[feature].isna().sum()
                if nan_count > 0:
                    msg = f"Critical feature '{feature}' has {nan_count} NaN values"
                    issues.append(msg)

        # Check ranges
        for feature, (min_val, max_val) in self.range_checks.items():
            if feature in X.columns:
                actual_min = X[feature].min()
                actual_max = X[feature].max()

                if actual_min < min_val or actual_max > max_val:
                    msg = (
                        f"Feature '{feature}' range [{actual_min:.2f}, {actual_max:.2f}] "
                        f"outside expected [{min_val}, {max_val}]"
                    )
                    issues.append(msg)

        # Handle issues
        if issues:
            full_msg = "FeatureValidator found issues:\n  - " + "\n  - ".join(issues)

            if self.strict_mode:
                raise ValueError(full_msg)
            else:
                logger.warning(full_msg)
        else:
            logger.info("FeatureValidator: All checks passed")

        return X


class MissingValueImputer(BaseEstimator, TransformerMixin):
    """
    Impute missing values using replacement level (25th percentile).

    For each feature, calculate the 25th percentile from training data
    and use it to fill missing values.

    This represents "replacement level" performance - below-average but not terrible.
    """

    def __init__(self, features_to_impute: Optional[List[str]] = None):
        """
        Args:
            features_to_impute: Features to impute. If None, impute all numeric.
        """
        self.features_to_impute = features_to_impute
        self.replacement_values_: Dict[str, float] = {}
        self._is_fitted = False

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'MissingValueImputer':
        """
        Calculate replacement level (25th percentile) for each feature.

        Args:
            X: Training data

        Returns:
            self

        Raises:
            InvalidDataTypeError: If X is not a DataFrame
        """
        if not isinstance(X, pd.DataFrame):
            raise InvalidDataTypeError('X', 'pd.DataFrame', type(X))

        # Determine which features to impute
        if self.features_to_impute is None:
            features = X.select_dtypes(include=[np.number]).columns.tolist()
        else:
            features = self.features_to_impute

        # Calculate 25th percentile for each feature
        self.replacement_values_ = {}
        for feature in features:
            if feature in X.columns:
                self.replacement_values_[feature] = X[feature].quantile(REPLACEMENT_LEVEL_PERCENTILE)

        self._is_fitted = True
        logger.info(f"MissingValueImputer: Learned replacement values for {len(self.replacement_values_)} features")

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Fill missing values with learned replacement levels.

        Args:
            X: Data to impute

        Returns:
            Imputed data

        Raises:
            InvalidDataTypeError: If X is not a DataFrame
            ImputerNotFittedError: If transform called before fit
        """
        if not isinstance(X, pd.DataFrame):
            raise InvalidDataTypeError('X', 'pd.DataFrame', type(X))

        if not self._is_fitted:
            raise ImputerNotFittedError()

        X = X.copy()
        imputed_count = 0

        for feature, replacement_val in self.replacement_values_.items():
            if feature in X.columns:
                nan_count = X[feature].isna().sum()
                if nan_count > 0:
                    X[feature] = X[feature].fillna(replacement_val)
                    imputed_count += nan_count

        if imputed_count > 0:
            logger.info(f"MissingValueImputer: Imputed {imputed_count} missing values")

        return X
