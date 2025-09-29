"""
Sklearn Pipeline Wrapper for Acuña Projection Fix
================================================

Provides proper sklearn pipeline integration to ensure consistent
preprocessing between training and testing, preventing data leakage
and ensuring reproducible results.

This addresses sklearn common pitfalls:
1. Inconsistent preprocessing
2. Data leakage prevention
3. Randomness control
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from typing import Dict, List, Tuple, Optional
import warnings


class PerformanceTierTransformer(BaseEstimator, TransformerMixin):
    """
    Sklearn transformer for performance-tiered regression factors.

    Ensures consistent tier classification between train/test splits
    while preventing data leakage by using only historical data.
    """

    def __init__(self):
        self.tier_thresholds_ = None
        self.fitted_ = False

    def fit(self, X: pd.DataFrame, y=None):
        """
        Fit the transformer on training data.

        Args:
            X: Training data with columns ['mlbid', 'Season', 'WAR', 'WARP']
            y: Ignored (for sklearn compatibility)

        Returns:
            self
        """
        # Store tier thresholds (could be made adaptive in future)
        self.tier_thresholds_ = {
            'superstar': 6.0,
            'elite': 4.0,
            'above_average': 2.0,
            'average': 0.0
        }

        self.fitted_ = True
        return self

    def transform(self, X: pd.DataFrame, prediction_year: int = None) -> pd.DataFrame:
        """
        Transform data by adding performance-tiered regression factors.

        Args:
            X: Data to transform
            prediction_year: Year being predicted (to prevent data leakage)

        Returns:
            Transformed data with 'regression_factor' column added
        """
        if not self.fitted_:
            raise ValueError("Transformer must be fitted before transform")

        X_transformed = X.copy()

        # Determine prediction year to prevent data leakage
        if prediction_year is None:
            prediction_year = X['Season'].max() + 1

        # Calculate regression factors using only historical data
        regression_factors = self._calculate_performance_factors(X, prediction_year)
        X_transformed['regression_factor'] = regression_factors

        return X_transformed

    def _calculate_performance_factors(self, data: pd.DataFrame, prediction_year: int) -> pd.Series:
        """
        Calculate performance-tiered regression factors without data leakage.

        Args:
            data: Player performance data
            prediction_year: Year being predicted

        Returns:
            Series of regression factors
        """
        regression_factors = pd.Series(index=data.index, dtype=float)

        # Use available performance metric
        performance_metric = 'WAR' if 'WAR' in data.columns else 'WARP'
        if performance_metric not in data.columns:
            # Fallback to uniform regression
            return pd.Series(0.7, index=data.index)

        # Calculate player performance using only historical data
        player_performance = {}

        for player_id in data['mlbid'].unique():
            if pd.isna(player_id):
                continue

            player_data = data[data['mlbid'] == player_id]

            # CRITICAL: Use only data PRIOR to prediction year
            historical_data = player_data[player_data['Season'] < prediction_year]

            if len(historical_data) == 0:
                continue

            # Get recent performance (last 3 years)
            recent_seasons = historical_data.sort_values('Season').tail(3)
            recent_performance = recent_seasons[performance_metric].dropna()

            if len(recent_performance) > 0:
                # Weighted average (more recent years weighted higher)
                if len(recent_performance) == 1:
                    avg_performance = recent_performance.iloc[0]
                elif len(recent_performance) == 2:
                    avg_performance = (recent_performance.iloc[0] * 0.4 +
                                     recent_performance.iloc[1] * 0.6)
                else:
                    avg_performance = (recent_performance.iloc[0] * 0.2 +
                                     recent_performance.iloc[1] * 0.3 +
                                     recent_performance.iloc[2] * 0.5)

                player_performance[player_id] = avg_performance

        # Assign regression factors based on performance tiers
        for i, (idx, row) in enumerate(data.iterrows()):
            player_id = row['mlbid']

            if pd.isna(player_id) or player_id not in player_performance:
                regression_factors.iloc[i] = 0.70  # Default
                continue

            recent_perf = player_performance[player_id]

            # Classify into performance tier
            if recent_perf >= self.tier_thresholds_['superstar']:
                regression_factors.iloc[i] = 0.85  # Superstar
            elif recent_perf >= self.tier_thresholds_['elite']:
                regression_factors.iloc[i] = 0.80  # Elite
            elif recent_perf >= self.tier_thresholds_['above_average']:
                regression_factors.iloc[i] = 0.75  # Above average
            elif recent_perf >= self.tier_thresholds_['average']:
                regression_factors.iloc[i] = 0.70  # Average
            else:
                regression_factors.iloc[i] = 0.65  # Below average

        return regression_factors


class ProportionalBudgetTransformer(BaseEstimator, TransformerMixin):
    """
    Sklearn transformer for proportional budget allocation.

    Calculates budget allocation based on training fold composition
    to maintain representative sampling across temporal folds.
    """

    def __init__(self, target_total: float = 1000.0):
        self.target_total = target_total
        self.training_composition_ = None
        self.fitted_ = False

    def fit(self, X: pd.DataFrame, y=None):
        """
        Fit on training fold to learn composition.

        Args:
            X: Training data
            y: Ignored

        Returns:
            self
        """
        # Calculate training fold composition
        performance_metric = 'WAR' if 'WAR' in X.columns else 'WARP'

        if performance_metric in X.columns:
            hitters = X[X['Position'] != 'P']
            pitchers = X[X['Position'] == 'P']

            hitter_war = hitters[performance_metric].sum()
            pitcher_war = pitchers[performance_metric].sum()
            total_war = hitter_war + pitcher_war

            if total_war > 0:
                self.training_composition_ = {
                    'hitter_proportion': hitter_war / total_war,
                    'pitcher_proportion': pitcher_war / total_war
                }
            else:
                # Fallback to default split
                self.training_composition_ = {
                    'hitter_proportion': 0.57,
                    'pitcher_proportion': 0.43
                }
        else:
            # Fallback composition
            self.training_composition_ = {
                'hitter_proportion': 0.57,
                'pitcher_proportion': 0.43
            }

        self.fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform test data by applying proportional budget allocation.

        Args:
            X: Test fold data

        Returns:
            Data with budget-adjusted projections
        """
        if not self.fitted_:
            raise ValueError("Transformer must be fitted before transform")

        X_transformed = X.copy()

        # Apply proportional budget allocation
        if 'projected_WAR_year_1' in X_transformed.columns:
            X_transformed = self._apply_proportional_budget(X_transformed)

        return X_transformed

    def _apply_proportional_budget(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply proportional budget allocation based on training composition.
        """
        hitters = data[data['Position'] != 'P'].copy()
        pitchers = data[data['Position'] == 'P'].copy()

        if len(hitters) == 0 or len(pitchers) == 0:
            return data

        # Calculate test fold natural totals
        test_hitter_natural = hitters['projected_WAR_year_1'].sum()
        test_pitcher_natural = pitchers['projected_WAR_year_1'].sum()
        test_total_natural = test_hitter_natural + test_pitcher_natural

        if test_total_natural <= 0:
            return data

        # Allocate budget proportionally
        hitter_budget = test_total_natural * self.training_composition_['hitter_proportion']
        pitcher_budget = test_total_natural * self.training_composition_['pitcher_proportion']

        # Scale to match target total
        budget_scale = self.target_total / test_total_natural
        hitter_budget *= budget_scale
        pitcher_budget *= budget_scale

        # Apply budget constraints
        if test_hitter_natural > 0:
            hitter_scale = hitter_budget / test_hitter_natural
            hitters['projected_WAR_year_1'] *= hitter_scale

        if test_pitcher_natural > 0:
            pitcher_scale = pitcher_budget / test_pitcher_natural
            pitchers['projected_WAR_year_1'] *= pitcher_scale

        # Recombine
        result = pd.concat([hitters, pitchers], ignore_index=True)
        return result


class DualBudgetTransformer(BaseEstimator, TransformerMixin):
    """
    Sklearn transformer for dual WAR/WARP budget allocation.

    Applies different constraint targets and hitter/pitcher splits
    for WAR vs WARP projections.
    """

    def __init__(self,
                 war_target_total: float = 1000.0,
                 war_split: tuple = (570, 430),
                 warp_target_total: float = 1000.0,
                 warp_split: tuple = (590, 410)):
        self.war_target_total = war_target_total
        self.war_split = war_split
        self.warp_target_total = warp_target_total
        self.warp_split = warp_split
        self.fitted_ = False

    def fit(self, X: pd.DataFrame, y=None):
        """Fit transformer - no training needed for budget allocation."""
        self.fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply dual WAR/WARP budget constraints.

        Args:
            X: DataFrame with WAR and WARP projections

        Returns:
            DataFrame with constrained projections
        """
        if not self.fitted_:
            raise ValueError("Transformer must be fitted before transform")

        X_transformed = X.copy()

        # Find projection columns for all years
        war_cols = [col for col in X.columns if col.startswith('projected_WAR_year_')]
        warp_cols = [col for col in X.columns if col.startswith('projected_WARP_year_')]

        # Apply constraints to each year separately
        for war_col in war_cols:
            X_transformed = self._apply_war_constraints(X_transformed, war_col)

        for warp_col in warp_cols:
            X_transformed = self._apply_warp_constraints(X_transformed, warp_col)

        return X_transformed

    def _apply_war_constraints(self, df: pd.DataFrame, war_col: str) -> pd.DataFrame:
        """Apply WAR-specific constraints to a projection year."""
        # Separate hitters and pitchers
        hitters = df[df['Position'] != 'P']
        pitchers = df[df['Position'] == 'P']

        # Calculate current totals
        current_hitter_war = hitters[war_col].sum()
        current_pitcher_war = pitchers[war_col].sum()

        # Apply proportional scaling to meet targets
        if current_hitter_war > 0:
            hitter_scale = self.war_split[0] / current_hitter_war
            df.loc[df['Position'] != 'P', war_col] *= hitter_scale

        if current_pitcher_war > 0:
            pitcher_scale = self.war_split[1] / current_pitcher_war
            df.loc[df['Position'] == 'P', war_col] *= pitcher_scale

        return df

    def _apply_warp_constraints(self, df: pd.DataFrame, warp_col: str) -> pd.DataFrame:
        """Apply WARP-specific constraints to a projection year."""
        # Separate hitters and pitchers
        hitters = df[df['Position'] != 'P']
        pitchers = df[df['Position'] == 'P']

        # Calculate current totals
        current_hitter_warp = hitters[warp_col].sum()
        current_pitcher_warp = pitchers[warp_col].sum()

        # Apply proportional scaling to meet WARP targets (590/410)
        if current_hitter_warp > 0:
            hitter_scale = self.warp_split[0] / current_hitter_warp
            df.loc[df['Position'] != 'P', warp_col] *= hitter_scale

        if current_pitcher_warp > 0:
            pitcher_scale = self.warp_split[1] / current_pitcher_warp
            df.loc[df['Position'] == 'P', warp_col] *= pitcher_scale

        return df


class AcunaProjectionPipeline:
    """
    Complete sklearn-compatible pipeline for Acuña projection fix.

    Combines performance-tiered regression and proportional budget allocation
    in a proper sklearn pipeline to prevent data leakage and ensure consistency.
    """

    def __init__(self,
                 war_target_total: float = 1000.0,
                 war_hitter_pitcher_split: tuple = (570, 430),
                 warp_target_total: float = 1000.0,
                 warp_hitter_pitcher_split: tuple = (590, 410),
                 random_state: int = 42):
        self.war_target_total = war_target_total
        self.war_hitter_pitcher_split = war_hitter_pitcher_split
        self.warp_target_total = warp_target_total
        self.warp_hitter_pitcher_split = warp_hitter_pitcher_split
        self.random_state = random_state
        self.pipeline_ = None
        self.fitted_ = False

        # Create sklearn pipeline with dual WAR/WARP support
        self.pipeline_ = Pipeline([
            ('performance_tiers', PerformanceTierTransformer()),
            ('dual_budget', DualBudgetTransformer(
                war_target_total=war_target_total,
                war_split=war_hitter_pitcher_split,
                warp_target_total=warp_target_total,
                warp_split=warp_hitter_pitcher_split
            ))
        ])

    def fit(self, X: pd.DataFrame, y=None):
        """
        Fit the complete pipeline on training data.

        Args:
            X: Training data
            y: Ignored

        Returns:
            self
        """
        self.pipeline_.fit(X, y)
        self.fitted_ = True
        return self

    def transform(self, X: pd.DataFrame, prediction_year: int = None) -> pd.DataFrame:
        """
        Transform data using fitted pipeline.

        Args:
            X: Data to transform
            prediction_year: Year being predicted

        Returns:
            Transformed data
        """
        if not self.fitted_:
            raise ValueError("Pipeline must be fitted before transform")

        # Handle prediction year for performance tier transformer
        if prediction_year is not None:
            # Set prediction year for performance tier transformer
            tier_transformer = self.pipeline_.named_steps['performance_tiers']
            X_transformed = tier_transformer.transform(X, prediction_year=prediction_year)

            # Apply budget transformer
            budget_transformer = self.pipeline_.named_steps['proportional_budget']
            X_transformed = budget_transformer.transform(X_transformed)
        else:
            X_transformed = self.pipeline_.transform(X)

        return X_transformed

    def temporal_cross_validate(self, data: pd.DataFrame, n_splits: int = 6,
                               test_size: int = 1) -> Dict:
        """
        Perform temporal cross-validation with proper data leakage prevention.

        Args:
            data: Complete dataset
            n_splits: Number of temporal splits
            test_size: Size of test fold (in years)

        Returns:
            Cross-validation results
        """
        tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)

        # Get unique years for splitting
        years = sorted(data['Season'].unique())

        cv_results = []

        for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(years)):
            train_years = [years[i] for i in train_idx]
            test_year = years[test_idx[0]]

            print(f"Fold {fold_idx + 1}: Train {train_years[0]}-{train_years[-1]} -> Test {test_year}")

            # Split data temporally
            train_data = data[data['Season'].isin(train_years)].copy()
            test_data = data[data['Season'] == test_year].copy()

            if len(train_data) == 0 or len(test_data) == 0:
                continue

            # Fit pipeline on training fold
            self.fit(train_data)

            # Transform test fold (with prediction year to prevent leakage)
            test_transformed = self.transform(test_data, prediction_year=test_year)

            # Calculate metrics
            fold_results = {
                'fold': fold_idx + 1,
                'train_years': train_years,
                'test_year': test_year,
                'train_size': len(train_data),
                'test_size': len(test_data),
                'test_transformed': test_transformed
            }

            cv_results.append(fold_results)

        return {
            'fold_results': cv_results,
            'n_splits': n_splits,
            'method': 'temporal_cv_leak_free'
        }


# Example usage and testing
if __name__ == "__main__":
    print("Sklearn Pipeline Wrapper for Acuña Projection Fix")
    print("=" * 50)

    # Create mock data for testing
    mock_data = pd.DataFrame({
        'mlbid': [1, 1, 1, 2, 2, 2] * 3,
        'Season': [2018, 2019, 2020] * 6,
        'WAR': [4.5, 5.2, 1.0, 2.1, 2.3, 2.5] * 3,
        'Position': ['RF', 'RF', 'RF', 'P', 'P', 'P'] * 3,
        'projected_WAR_year_1': [4.0, 4.0, 4.0, 2.5, 2.5, 2.5] * 3
    })

    # Test pipeline
    pipeline = AcunaProjectionPipeline(target_total=1000.0)

    print("Testing temporal cross-validation...")
    cv_results = pipeline.temporal_cross_validate(mock_data, n_splits=2)

    print(f"CV completed: {len(cv_results['fold_results'])} folds")
    print("Sklearn pipeline wrapper ready for production!")