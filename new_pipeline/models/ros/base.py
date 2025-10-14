"""
Base classes for ROS prediction models.

Defines interfaces and common utilities for all ROS models.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator, RegressorMixin


class BaseROSModel(ABC, BaseEstimator, RegressorMixin):
    """
    Abstract base class for ROS prediction models.

    All ROS models must implement fit() and predict() methods.

    Inherits from sklearn.base.BaseEstimator to provide get_params()/set_params()
    and from RegressorMixin to be recognized as a regressor by sklearn/sktime.
    """

    def __init__(self, player_type: str = 'hitter'):
        """
        Initialize base model.

        Args:
            player_type: 'hitter' or 'pitcher'
        """
        self.player_type = player_type
        self.is_fitted = False

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BaseROSModel':
        """
        Fit model to training data.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target values (n_samples,)

        Returns:
            self (fitted model)
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate predictions.

        Args:
            X: Feature matrix (n_samples, n_features)

        Returns:
            Predictions (n_samples,)
        """
        pass

    def predict_quantiles(
        self,
        X: np.ndarray,
        quantiles: List[float] = [0.1, 0.25, 0.5, 0.75, 0.9]
    ) -> Dict[float, np.ndarray]:
        """
        Generate quantile predictions (if supported).

        Args:
            X: Feature matrix
            quantiles: Target quantiles

        Returns:
            Dictionary mapping quantile -> predictions

        Raises:
            NotImplementedError: If model doesn't support quantile prediction
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} doesn't support quantile prediction"
        )


class BaseEnsemble(BaseROSModel):
    """
    Base class for ensemble models.

    Provides common ensemble utilities (weighted averaging, etc).
    """

    def __init__(self, player_type: str = 'hitter', weights: Optional[List[float]] = None):
        """
        Initialize ensemble.

        Args:
            player_type: 'hitter' or 'pitcher'
            weights: Optional weights for component models (must sum to 1.0)
        """
        super().__init__(player_type)
        self.weights = weights
        self.component_models = []

    def weighted_average(
        self,
        predictions: List[np.ndarray],
        weights: Optional[List[float]] = None
    ) -> np.ndarray:
        """
        Compute weighted average of predictions.

        Args:
            predictions: List of prediction arrays
            weights: Optional weights (default: self.weights or equal)

        Returns:
            Weighted average predictions

        Example:
            >>> preds = [np.array([4.0, 5.0]), np.array([4.5, 5.5])]
            >>> ensemble.weighted_average(preds, weights=[0.6, 0.4])
            array([4.2, 5.2])
        """
        if weights is None:
            weights = self.weights if self.weights is not None else [1.0 / len(predictions)] * len(predictions)

        if len(predictions) != len(weights):
            raise ValueError(f"Predictions ({len(predictions)}) and weights ({len(weights)}) must have same length")

        # Stack predictions and apply weights
        pred_stack = np.column_stack(predictions)  # (n_samples, n_models)
        weights_array = np.array(weights)  # (n_models,)

        return pred_stack @ weights_array  # (n_samples,)

    def calculate_prediction_std(self, predictions: List[np.ndarray]) -> np.ndarray:
        """
        Calculate standard deviation across component predictions.

        Args:
            predictions: List of prediction arrays

        Returns:
            Standard deviation for each sample

        Example:
            >>> preds = [np.array([4.0, 5.0]), np.array([4.8, 5.2])]
            >>> ensemble.calculate_prediction_std(preds)
            array([0.4, 0.1])
        """
        pred_stack = np.column_stack(predictions)
        return pred_stack.std(axis=1)
