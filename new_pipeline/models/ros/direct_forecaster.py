"""
DirectTimeSeriesRegressionForecaster for ROS Prediction

Uses sktime's DirectTimeSeriesRegressionForecaster to convert time series
forecasting to supervised learning with lagged features + exogenous variables.
"""

import numpy as np
import pandas as pd
from typing import Optional
from sktime.forecasting.compose import make_reduction
from sktime.forecasting.base import ForecastingHorizon
from .quantile_model import MultiQuantileHistGB
from .base import BaseROSModel


class DirectROSForecaster(BaseROSModel):
    """
    ROS forecaster using DirectTimeSeriesRegressionForecaster.

    Combines:
    - Lagged features (previous WAR values)
    - Exogenous features (all ROS features)
    - Multi-quantile regressor as backend
    """

    def __init__(
        self,
        player_type: str = 'hitter',
        lags: int = 3,
        window_length: Optional[int] = None
    ):
        """
        Initialize direct forecaster.

        Args:
            player_type: 'hitter' or 'pitcher'
            lags: Number of lagged values to use (default: 3 years)
            window_length: Optional window for lag features
        """
        super().__init__(player_type)
        self.lags = lags
        self.window_length = window_length

        # Create base regressor (multi-quantile)
        self.base_regressor = MultiQuantileHistGB(player_type=player_type)

        # Create forecaster
        # Note: window_length parameter creates lagged features for time series reduction
        self.forecaster = make_reduction(
            self.base_regressor,
            window_length=lags,
            strategy="direct"
        )

    def fit(
        self,
        y: pd.DataFrame,
        X: Optional[pd.DataFrame] = None,
        fh: Optional[ForecastingHorizon] = None
    ) -> 'DirectROSForecaster':
        """
        Fit forecaster to time series data.

        Args:
            y: Target time series DataFrame with MultiIndex for panel data
                For panel data: pd.DataFrame with MultiIndex (player_id, datetime)
                For single series: pd.Series or DataFrame with datetime index
                Example (panel): columns=['WAR_per_600'], MultiIndex
            X: Exogenous features (same index as y)
                Example: pd.DataFrame({'age': [27, 28, 29, 30], 'wOBA': [...]})
            fh: Forecasting horizon (default: 1 step ahead)

        Returns:
            self (fitted model)

        Example:
            >>> # Panel data (multiple players)
            >>> y = pd.DataFrame(
            ...     {'WAR_per_600': [5.8, 4.1, 5.3, 5.5]},
            ...     index=pd.MultiIndex.from_tuples([
            ...         (12345, '2019'), (12345, '2020'),
            ...         (67890, '2019'), (67890, '2020')
            ...     ])
            ... )
            >>> forecaster.fit(y, X)
        """
        if fh is None:
            fh = ForecastingHorizon([1], is_relative=True)  # 1 step ahead (ROS)

        self.forecaster.fit(y, X=X, fh=fh)
        self.is_fitted = True
        return self

    def predict(
        self,
        fh: Optional[ForecastingHorizon] = None,
        X: Optional[pd.DataFrame] = None
    ) -> np.ndarray:
        """
        Generate ROS predictions.

        Args:
            fh: Forecasting horizon (default: 1)
            X: Exogenous features for prediction period

        Returns:
            ROS predictions (n_samples,)

        Example:
            >>> # Predict Soto 2025 ROS given firsthalf data
            >>> X_ros = pd.DataFrame({
            ...     'age': [26],
            ...     'wOBA': [0.411],
            ...     'elite_tier_level': [5],
            ...     # ... other features
            ... }, index=pd.date_range('2025', periods=1, freq='Y'))
            >>> ros_pred = forecaster.predict(X=X_ros)
            >>> ros_pred
            array([4.2])  # Predicted secondhalf WAR
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        if fh is None:
            fh = ForecastingHorizon([1], is_relative=True)

        predictions = self.forecaster.predict(fh=fh, X=X)
        return predictions.values.ravel()  # Flatten to 1D array for consistent indexing
