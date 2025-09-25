"""
Future Projection Age Curve Module
=================================

Implements joint longitudinal-survival modeling for SYSTEM 2 future projections.
ZIPS-style prediction system with 1-3 year forecasting capabilities.

Classes:
    FutureProjectionAgeCurve: Main class for joint modeling and multi-year projections
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import warnings
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib

# Survival analysis imports (install with: pip install lifelines scikit-survival)
try:
    from lifelines import CoxPHFitter, WeibullAFTFitter
    from lifelines.utils import concordance_index
    SURVIVAL_AVAILABLE = True
except ImportError:
    SURVIVAL_AVAILABLE = False
    warnings.warn("Survival analysis packages not available. Install lifelines for full functionality.")


class FutureProjectionAgeCurve:
    """
    Joint longitudinal-survival model for future performance projections.

    Implements ZIPS-style methodology with position-specific aging curves
    and retirement risk modeling for 1-3 year forecasts.
    """

    def __init__(self, max_projection_years: int = 3):
        """
        Initialize the future projection system.

        Args:
            max_projection_years: Maximum years to project (default 3)
        """
        self.max_projection_years = max_projection_years
        self.longitudinal_model = None
        self.survival_model = None
        self.feature_scaler = StandardScaler()
        self.position_curves = self._initialize_position_curves()
        self.is_fitted = False

    def _initialize_position_curves(self) -> Dict[str, Dict]:
        """
        Initialize position-specific aging parameters from research.

        Returns:
            Dictionary of position -> aging parameters
        """
        return {
            'C': {'peak': 26, 'decline_rate': 0.035, 'career_length_median': 8},
            'SS': {'peak': 27, 'decline_rate': 0.025, 'career_length_median': 10},
            '2B': {'peak': 27, 'decline_rate': 0.025, 'career_length_median': 9},
            '3B': {'peak': 28, 'decline_rate': 0.020, 'career_length_median': 10},
            '1B': {'peak': 29, 'decline_rate': 0.015, 'career_length_median': 11},
            'LF': {'peak': 28, 'decline_rate': 0.020, 'career_length_median': 10},
            'CF': {'peak': 27, 'decline_rate': 0.025, 'career_length_median': 9},
            'RF': {'peak': 28, 'decline_rate': 0.020, 'career_length_median': 10},
            'DH': {'peak': 30, 'decline_rate': 0.015, 'career_length_median': 8},
            'P': {'peak': 27, 'decline_rate': 0.030, 'career_length_median': 7}
        }

    def prepare_longitudinal_training_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training data for year-to-year WAR prediction (iterative approach).

        Args:
            data: Historical player data with features

        Returns:
            Tuple of (features, next_year_targets)
        """
        print("Preparing longitudinal data for year-to-year WAR prediction...")

        # Group by player to create year-to-year transitions
        training_examples = []
        targets = []

        valid_transitions = 0
        nan_features = 0
        nan_targets = 0

        for player_id, player_data in data.groupby('mlbid'):
            player_data = player_data.sort_values('Season')

            # Create year-to-year transitions
            for i in range(len(player_data) - 1):
                current_row = player_data.iloc[i]
                next_row = player_data.iloc[i+1]

                # Only use consecutive seasons (no gaps)
                if next_row['Season'] - current_row['Season'] == 1:
                    valid_transitions += 1

                    # Extract features for current state
                    features = self._extract_longitudinal_features(current_row, player_data.iloc[:i+1])

                    # Target is next year's performance
                    target = next_row['TARGET_METRIC']

                    if np.any(np.isnan(features)):
                        nan_features += 1
                    elif np.isnan(target):
                        nan_targets += 1
                    else:
                        training_examples.append(features)
                        targets.append(target)

        print(f"  Valid year-to-year transitions: {valid_transitions}")
        print(f"  Transitions with NaN features: {nan_features}")
        print(f"  Transitions with NaN targets: {nan_targets}")
        print(f"  Final training examples: {len(training_examples)}")

        if not training_examples:
            raise ValueError("No valid year-to-year training examples found")

        X = np.array(training_examples)
        y = np.array(targets)

        print(f"Longitudinal data prepared: {len(X)} training examples, {X.shape[1]} features")
        return X, y

    def _extract_longitudinal_features(self, current_row: pd.Series, player_history: pd.DataFrame) -> np.ndarray:
        """
        Extract features for longitudinal modeling from current player state.

        Args:
            current_row: Current season data
            player_history: Historical data up to current season

        Returns:
            Feature vector for modeling
        """
        features = []

        # Age and age-squared (non-linear aging)
        age = current_row.get('Age', 27)  # Default to typical age
        features.extend([age, age**2])

        # 3-season weighted average for injury robustness (75%, 20%, 5% weights)
        if len(player_history) >= 3:
            recent_3_seasons = player_history['TARGET_METRIC'].tail(3).values
            # Weighted average: most recent 75%, previous 20%, earliest 5%
            weights = np.array([0.05, 0.20, 0.75])
            weighted_performance = np.average(recent_3_seasons, weights=weights)
        elif len(player_history) >= 2:
            recent_2_seasons = player_history['TARGET_METRIC'].tail(2).values
            # Use 30%/70% weighting for 2 seasons
            weights = np.array([0.30, 0.70])
            weighted_performance = np.average(recent_2_seasons, weights=weights)
        else:
            # Single season - use current performance
            weighted_performance = current_row.get('TARGET_METRIC', current_row.get('WAR', 0))

        features.append(weighted_performance)

        # Performance trend (last 2-3 seasons)
        if len(player_history) >= 2:
            # Use TARGET_METRIC if available, fallback to WAR for backward compatibility
            metric_col = 'TARGET_METRIC' if 'TARGET_METRIC' in player_history.columns else 'WAR'
            recent_metrics = player_history[metric_col].tail(2).values

            # Calculate trend with validated data
            if len(recent_metrics) == 2:
                metric_trend = recent_metrics[-1] - recent_metrics[0]
            else:
                metric_trend = 0.0
        else:
            metric_trend = 0.0
        features.append(metric_trend)

        # Career stage (years of experience)
        career_length = len(player_history)
        features.append(career_length)

        # Expected stats gap if available
        expected_gap = current_row.get('regression_factor', 1.0)
        features.append(expected_gap)

        # Position-specific age interaction
        position = current_row.get('Primary_Position', 'OF')
        position_curve = self.position_curves.get(position, self.position_curves['CF'])
        age_vs_peak = age - position_curve['peak']
        features.append(age_vs_peak)

        # Convert to array and handle any remaining NaN values
        feature_array = np.array(features)

        # Sanity check: no NaN values should remain with validated data
        if np.any(np.isnan(feature_array)):
            print(f"   Warning: Unexpected NaN values found in features, replacing with defaults")
            feature_array = np.nan_to_num(feature_array, nan=0.0, posinf=10.0, neginf=-10.0)

        return feature_array

    def prepare_survival_training_data(self, data: pd.DataFrame, cutoff_season: Optional[int] = None) -> pd.DataFrame:
        """
        Prepare training data for retirement hazard modeling WITHOUT data leakage.

        Args:
            data: Complete career data for all players
            cutoff_season: Last season available for training (prevents future info leakage)
                          If None, uses max season in data (for final model training only)

        Returns:
            DataFrame formatted for survival analysis without future information
        """
        if not SURVIVAL_AVAILABLE:
            raise ImportError("Survival analysis packages required. Install lifelines.")

        # Use cutoff season or max available season
        if cutoff_season is None:
            cutoff_season = data['Season'].max()
            print(f"Preparing survival data for retirement risk modeling (full data through {cutoff_season})...")
        else:
            print(f"Preparing survival data for temporal validation (cutoff: {cutoff_season})...")

        # CRITICAL FIX: Only use data up to cutoff season for ALL calculations
        available_data = data[data['Season'] <= cutoff_season].copy()

        survival_data = []

        for player_id, player_data in available_data.groupby('mlbid'):
            player_data = player_data.sort_values('Season')

            # Calculate career duration and retirement status using only available data
            career_start = player_data['Season'].min()
            career_end = player_data['Season'].max()
            career_duration = career_end - career_start + 1

            # CRITICAL FIX: Retirement status without future information
            # A player is "retired" if their career ended BEFORE the cutoff season
            # Players active in cutoff season are "censored" (unknown future status)
            retired = 1 if career_end < cutoff_season else 0

            # Get final season stats using only available data
            final_season = player_data.iloc[-1]

            # CRITICAL FIX: Career statistics using ONLY data up to cutoff
            career_war = player_data['TARGET_METRIC'].sum() if 'TARGET_METRIC' in player_data.columns else player_data['WAR'].sum()
            peak_war = player_data['TARGET_METRIC'].max() if 'TARGET_METRIC' in player_data.columns else player_data['WAR'].max()
            war_decline = self._calculate_war_decline(player_data)  # Now uses only available data

            survival_record = {
                'mlbid': player_id,
                'duration': career_duration,
                'event': retired,  # 1 = retired before cutoff, 0 = active in cutoff (censored)
                'age_at_end': final_season.get('Age', 30),
                'final_war': final_season.get('TARGET_METRIC', final_season.get('WAR', 0)),
                'position': final_season.get('Primary_Position', 'OF'),
                'career_war': career_war,
                'peak_war': peak_war,
                'war_decline': war_decline,
                'cutoff_season': cutoff_season,  # Track which cutoff was used
                'career_start': career_start,
                'career_end': career_end
            }

            survival_data.append(survival_record)

        survival_df = pd.DataFrame(survival_data)
        print(f"Survival data prepared: {len(survival_df)} observations, {survival_df['event'].sum()} retirement events")
        print(f"  Event rate: {survival_df['event'].mean():.3f}")
        print(f"  Censored (active): {(~survival_df['event'].astype(bool)).sum()}")

        return survival_df

    def _calculate_war_decline(self, player_data: pd.DataFrame) -> float:
        """
        Calculate WAR decline rate in final seasons.

        Args:
            player_data: Player's complete career data

        Returns:
            WAR decline rate (negative = declining, positive = improving)
        """
        if len(player_data) < 3:
            return 0.0

        # Compare last 2 seasons to peak performance
        metric_col = 'TARGET_METRIC' if 'TARGET_METRIC' in player_data.columns else 'WAR'
        recent_wars = player_data[metric_col].tail(2).values
        peak_war = player_data[metric_col].max()

        if peak_war <= 0:
            return 0.0

        recent_avg = np.mean(recent_wars)
        decline_rate = (recent_avg - peak_war) / peak_war

        return decline_rate

    def fit_performance_trajectory_model(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Fit the longitudinal model for year-to-year performance prediction.

        Args:
            X: Feature matrix
            y: Target vector (next year performance)

        Returns:
            Dictionary with performance metrics
        """
        print("Fitting year-to-year performance trajectory model...")

        # Use RandomForest for better outlier handling and non-linear relationships
        self.longitudinal_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )

        # Scale features
        X_scaled = self.feature_scaler.fit_transform(X)

        print(f"   Training samples: {X_scaled.shape[0]}")
        print(f"   Features: {X_scaled.shape[1]}")
        print(f"   Target shape: {y.shape}")

        # Fit the model
        self.longitudinal_model.fit(X_scaled, y)

        # Calculate performance metrics
        y_pred = self.longitudinal_model.predict(X_scaled)

        r2 = r2_score(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        mae = mean_absolute_error(y, y_pred)

        metrics = {
            'r2': r2,
            'rmse': rmse,
            'mae': mae
        }

        print(f"   Year-to-year prediction performance:")
        print(f"     R² = {r2:.3f}")
        print(f"     RMSE = {rmse:.3f}")
        print(f"     MAE = {mae:.3f}")

        return metrics

    def fit_retirement_hazard_model(self, survival_df: pd.DataFrame) -> Dict[str, float]:
        """
        Fit the survival model for retirement risk.

        Args:
            survival_df: Survival analysis formatted data

        Returns:
            Dictionary with model performance metrics
        """
        if not SURVIVAL_AVAILABLE:
            raise ImportError("Survival analysis packages required")

        print("Fitting retirement hazard model...")

        # Validate survival data
        print(f"   Validating survival data: {len(survival_df)} observations, {survival_df['event'].sum()} events")
        print(f"   Duration range: {survival_df['duration'].min()} - {survival_df['duration'].max()}")
        print(f"   Event rate: {survival_df['event'].mean():.3f}")

        # Select features for survival model
        survival_features = [
            'age_at_end', 'final_war', 'career_war', 'peak_war', 'war_decline'
        ]

        # Prepare data for Cox PH model
        model_data = survival_df[['duration', 'event'] + survival_features].copy()
        model_data = model_data.dropna()

        if len(model_data) < 10:
            raise ValueError("Insufficient data for survival modeling")

        # Clean data for convergence stability
        # Remove extreme outliers and handle multicollinearity
        numeric_features = [col for col in survival_features if col not in model_data.columns or pd.api.types.is_numeric_dtype(model_data[col])]

        # Standardize features to improve convergence
        for feature in numeric_features:
            if feature in model_data.columns:
                q1, q99 = model_data[feature].quantile([0.01, 0.99])
                model_data[feature] = model_data[feature].clip(q1, q99)
                # Standardize
                mean_val = model_data[feature].mean()
                std_val = model_data[feature].std()
                if std_val > 0:
                    model_data[feature] = (model_data[feature] - mean_val) / std_val

        # Check for zero variance features
        for feature in survival_features:
            if feature in model_data.columns and model_data[feature].std() == 0:
                print(f"   Warning: Removing zero-variance feature: {feature}")
                model_data = model_data.drop(columns=[feature])
                survival_features.remove(feature)

        # Ensure minimum duration > 0
        model_data['duration'] = model_data['duration'].clip(lower=0.1)

        # Fit Cox Proportional Hazards model with enhanced convergence parameters
        self.survival_model = CoxPHFitter(penalizer=0.01)  # Add regularization
        try:
            self.survival_model.fit(
                model_data,
                duration_col='duration',
                event_col='event',
                fit_options={'step_size': 0.1, 'max_steps': 100}  # Updated syntax
            )
        except Exception as e:
            print(f"   Warning: Cox model fitting failed ({str(e)}). Using simplified approach.")
            # Fallback to a simpler model with fewer features
            simple_features = ['age_at_end', 'final_war', 'career_war']
            available_features = [f for f in simple_features if f in model_data.columns]

            if available_features:
                simple_data = model_data[['duration', 'event'] + available_features].copy()
                self.survival_model = CoxPHFitter(penalizer=0.1)
                self.survival_model.fit(simple_data, duration_col='duration', event_col='event')
            else:
                # Ultimate fallback - no features, just intercept model
                print("   Warning: Using intercept-only survival model")
                self.survival_model = CoxPHFitter()
                intercept_data = model_data[['duration', 'event']].copy()
                intercept_data['intercept'] = 1.0
                self.survival_model.fit(intercept_data, duration_col='duration', event_col='event')

        # Calculate concordance index
        try:
            # Use the actual features that were used in the final model
            final_features = [col for col in self.survival_model.params_.index
                            if col != 'intercept']  # Exclude intercept if present
            if final_features:
                test_data = model_data[final_features]
                concordance = concordance_index(
                    model_data['duration'],
                    -self.survival_model.predict_partial_hazard(test_data),
                    model_data['event']
                )
            else:
                # For intercept-only models, use a default value
                concordance = 0.5
        except Exception as e:
            print(f"   Warning: Could not calculate concordance index: {str(e)}")
            concordance = 0.5

        print(f"   Cox PH model fitted successfully")
        print(f"   Concordance index: {concordance:.3f}")

        return {
            'concordance_index': concordance,
            'n_observations': len(model_data),
            'n_events': model_data['event'].sum()
        }

    def fit_joint_model(self, data: pd.DataFrame) -> Dict[str, Union[float, Dict]]:
        """
        Fit the complete joint longitudinal-survival model.

        Args:
            data: Complete training dataset

        Returns:
            Dictionary with all model performance metrics
        """
        print("Fitting joint longitudinal-survival model for future projections...")

        # Prepare longitudinal data
        X, y = self.prepare_longitudinal_training_data(data)

        # Prepare survival data
        survival_df = self.prepare_survival_training_data(data)

        # Fit longitudinal component
        longitudinal_metrics = self.fit_performance_trajectory_model(X, y)

        # Fit survival component
        survival_metrics = self.fit_retirement_hazard_model(survival_df)

        self.is_fitted = True

        print("Joint model fitting complete!")

        return {
            'longitudinal_performance': longitudinal_metrics,
            'survival_performance': survival_metrics,
            'training_samples': len(X),
            'survival_observations': len(survival_df)
        }

    def predict_performance_path(self,
                                age: float,
                                position: str,
                                current_war: float,
                                player_history: pd.DataFrame,
                                years_ahead: int = 3) -> np.ndarray:
        """
        Predict multi-year performance trajectory using iterative approach.

        Args:
            age: Current age
            position: Primary position
            current_war: Current season WAR
            player_history: Player's historical data
            years_ahead: Number of years to project

        Returns:
            Array of projected WAR values
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        years_ahead = min(years_ahead, self.max_projection_years)
        predictions = []

        # Start with current state
        current_age = age
        current_performance = current_war
        extended_history = player_history.copy()

        # Iteratively predict each year
        for year in range(years_ahead):
            # Create current state representation
            current_state = pd.Series({
                'Age': current_age,
                'Primary_Position': position,
                'TARGET_METRIC': current_performance,  # Use TARGET_METRIC instead of WAR
                'regression_factor': 1.0  # Default
            })

            # Extract features for current state
            features = self._extract_longitudinal_features(current_state, extended_history)
            features_scaled = self.feature_scaler.transform(features.reshape(1, -1))

            # Predict next year
            next_year_prediction = self.longitudinal_model.predict(features_scaled)[0]
            predictions.append(next_year_prediction)

            # Update state for next iteration
            current_age += 1
            current_performance = next_year_prediction

            # Add the predicted year to history for next prediction
            # (This simulates having the predicted season as part of the player's record)
            new_season = current_state.copy()
            new_season['TARGET_METRIC'] = next_year_prediction
            new_season['Age'] = current_age

            # Add to extended history as a DataFrame row
            new_row_df = pd.DataFrame([new_season])
            extended_history = pd.concat([extended_history, new_row_df], ignore_index=True)

        return np.array(predictions)

    def calculate_survival_probabilities(self,
                                       age: float,
                                       position: str,
                                       recent_performance: float,
                                       career_stats: Dict,
                                       recent_pa: float = 600) -> np.ndarray:
        """
        Calculate probability of remaining active for each future year.

        Args:
            age: Current age
            position: Primary position
            recent_performance: Recent WAR performance
            career_stats: Dictionary with career statistics

        Returns:
            Array of survival probabilities for each future year
        """
        if not self.is_fitted or self.survival_model is None:
            # Default survival probabilities if model not available
            base_prob = 0.9  # 90% base survival probability
            age_penalty = max(0, (age - 30) * 0.02)  # Penalty for age > 30
            performance_bonus = min(0.05, recent_performance * 0.01)  # Bonus for good performance

            annual_prob = max(0.5, base_prob - age_penalty + performance_bonus)
            return np.array([annual_prob ** year for year in range(1, self.max_projection_years + 1)])

        # Create survival features
        survival_features = {
            'age_at_end': age,
            'final_war': recent_performance,
            'career_war': career_stats.get('career_war', 0),
            'peak_war': career_stats.get('peak_war', 0),
            'war_decline': career_stats.get('war_decline', 0)
        }

        # Note: Positional adjustments already included in WAR/WARP values

        # Create DataFrame for prediction
        pred_df = pd.DataFrame([survival_features])

        # Predict survival function
        survival_function = self.survival_model.predict_survival_function(pred_df)

        # Extract probabilities for each future year
        survival_probs = []
        for year in range(1, self.max_projection_years + 1):
            if year in survival_function.index:
                prob = survival_function.iloc[:, 0].loc[year]
            else:
                # Interpolate or use closest value
                closest_time = min(survival_function.index, key=lambda x: abs(x - year))
                prob = survival_function.iloc[:, 0].loc[closest_time]

            survival_probs.append(max(0.1, min(1.0, prob)))  # Reasonable bounds

        return np.array(survival_probs)

    def generate_future_projections(self,
                                  current_state: Dict,
                                  player_history: pd.DataFrame,
                                  years_ahead: int = 3) -> Dict[str, float]:
        """
        Generate comprehensive future projections combining performance and survival.

        Args:
            current_state: Dictionary with current player state
            player_history: Player's historical data
            years_ahead: Number of years to project

        Returns:
            Dictionary with projected WAR for each future year
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before generating projections")

        # Extract current state information
        age = current_state.get('age', 27)
        position = current_state.get('position', 'OF')
        current_war = current_state.get('war', 0)
        recent_pa = current_state.get('pa', 600)  # Default to full season

        # Predict performance trajectory
        performance_path = self.predict_performance_path(
            age, position, current_war, player_history, years_ahead
        )

        # Calculate survival probabilities
        career_stats = {
            'career_war': player_history['TARGET_METRIC'].sum() if 'TARGET_METRIC' in player_history.columns else player_history['WAR'].sum(),
            'peak_war': player_history['TARGET_METRIC'].max() if 'TARGET_METRIC' in player_history.columns else player_history['WAR'].max(),
            'war_decline': self._calculate_war_decline(player_history)
        }

        survival_probs = self.calculate_survival_probabilities(
            age, position, current_war, career_stats, recent_pa
        )

        # Combine performance and survival (expected value)
        projections = {}
        for year in range(years_ahead):
            expected_war = performance_path[year] * survival_probs[year]
            projections[f'year_{year + 1}'] = expected_war

        return projections

    def save_model(self, filepath: str) -> None:
        """
        Save the fitted joint model to disk.

        Args:
            filepath: Path to save the model
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")

        model_data = {
            'longitudinal_model': self.longitudinal_model,
            'survival_model': self.survival_model,
            'feature_scaler': self.feature_scaler,
            'position_curves': self.position_curves,
            'max_projection_years': self.max_projection_years,
            'is_fitted': self.is_fitted
        }

        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath: str) -> None:
        """
        Load a fitted joint model from disk.

        Args:
            filepath: Path to the saved model
        """
        model_data = joblib.load(filepath)

        self.longitudinal_model = model_data['longitudinal_model']
        self.survival_model = model_data['survival_model']
        self.feature_scaler = model_data['feature_scaler']
        self.position_curves = model_data['position_curves']
        self.max_projection_years = model_data['max_projection_years']
        self.is_fitted = model_data['is_fitted']

        print(f"Model loaded from {filepath}")