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
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib

# Survival analysis imports (install with: pip install lifelines scikit-survival)
try:
    from lifelines import CoxPHFitter
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

    def __init__(self, max_projection_years: int = 3, use_dynasty_guru: bool = False):
        """
        Initialize the future projection system.

        Args:
            max_projection_years: Maximum years to project (default 3)
            use_dynasty_guru: Whether to use Dynasty Guru enhanced aging curves
        """
        self.max_projection_years = max_projection_years
        self.use_dynasty_guru = use_dynasty_guru
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

    def _calculate_enhanced_age_factor(self, age: float, position: str, use_log_transform: bool = True) -> float:
        """
        Enhanced age curve calculation based on Dynasty Guru research.

        Key improvements:
        - Peak range 26-29 instead of single age
        - Logarithmic growth for young players (accelerating improvement)
        - Continued development 24-26
        - More realistic decline patterns

        Args:
            age: Player's age
            position: Player's primary position
            use_log_transform: Whether to use logarithmic growth for young players

        Returns:
            Age factor multiplier (1.0 = peak performance)
        """
        # Ages < 20: Conservative baseline (limited data)
        if age < 20:
            return 0.70

        # Ages 20-24: Logarithmic growth implementation
        elif age < 24:
            if use_log_transform:
                # Dynasty Guru insight: accelerating improvement before age 24
                age_progress = (age - 20) / 4.0  # 0-1 scale for ages 20-24
                log_factor = np.log1p(age_progress) / np.log1p(1.0)  # Normalized log transform
                return 0.70 + (0.25 * log_factor)  # 70% → 95% performance by age 24
            else:
                # Linear fallback
                return 0.70 + (0.25 * (age - 20) / 4.0)

        # Ages 24-25: Continued improvement phase
        elif age < 26:
            base_factor = 0.95  # From age 24 peak
            improvement = (age - 24) * 0.025  # 2.5% per year improvement
            return base_factor + improvement

        # Ages 26-29: Peak performance range (Dynasty Guru finding)
        elif 26 <= age <= 29:
            # Gentle variation within range using inverted parabola
            range_position = (age - 26) / 3.0  # 0-1 scale within peak range
            # Slight variation around 1.0, peaking in middle of range
            peak_variation = 0.03 * (1 - 4 * (range_position - 0.5)**2)
            return 1.0 + peak_variation

        # Ages 30+: Gradual decline with early-stage protection
        else:
            position_curve = self.position_curves.get(position, self.position_curves['CF'])

            if age <= 31:
                # Ages 30-31: Gentle decline (1.5% per year)
                years_past_peak = age - 29
                return 1.0 - (years_past_peak * 0.015)
            else:
                # Ages 32+: Standard position-based decline
                base_decline = 1.0 - 2 * 0.015  # From ages 30-31
                years_past_31 = age - 31
                position_decline_rate = position_curve['decline_rate']
                return max(0.3, base_decline - (years_past_31 * position_decline_rate))

    def create_age_features(self, age: float, position: str, use_dynasty_guru: bool = False) -> List[float]:
        """
        Create age-related features for model learning instead of predetermined coefficients.

        Args:
            age: Player age
            position: Player position
            use_dynasty_guru: Whether to use Dynasty Guru enhancements

        Returns:
            List of age-related features
        """
        if use_dynasty_guru:
            # Dynasty Guru enhanced age features
            features = []

            # Enhanced age factor
            age_factor = self._calculate_enhanced_age_factor(age, position)
            features.append(age_factor)

            # Age-related indicators
            features.append(age)  # Linear age term
            features.append(age**2)  # Quadratic aging pattern

            # Logarithmic growth for young players
            if age < 24:
                log_young_age = np.log1p(max(0, age - 19))  # log(age - 19 + 1)
                features.append(log_young_age)
            else:
                features.append(0.0)

            # Peak range indicator
            peak_range_indicator = 1.0 if 26 <= age <= 29 else 0.0
            features.append(peak_range_indicator)

            # Post-peak terms
            if age > 29:
                post_peak_linear = age - 29
                features.append(post_peak_linear)
                # Accelerated decline after 32
                post_peak_accelerated = max(0, age - 32)**2
                features.append(post_peak_accelerated)
            else:
                features.append(0.0)
                features.append(0.0)

            return features
        else:
            # Original simple features for backward compatibility
            return [age, age**2]

    def prepare_longitudinal_training_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training data with confidence features for year-to-year WAR prediction.

        Args:
            data: Historical player data with features

        Returns:
            Tuple of (features, next_year_targets)
        """
        print("Preparing longitudinal data with confidence features for year-to-year WAR prediction...")

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

                    # Extract features with confidence data (OPTION B ENHANCEMENT)
                    features = self._extract_longitudinal_features(
                        current_row,
                        player_data.iloc[:i+1],
                        data  # Pass full dataset for confidence calculation
                    )

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

    def _extract_longitudinal_features(self, current_row: pd.Series, player_history: pd.DataFrame, training_data: pd.DataFrame = None) -> np.ndarray:
        """
        Extract features for longitudinal modeling from current player state, including confidence scores.

        Args:
            current_row: Current season data
            player_history: Historical data up to current season
            training_data: Full training dataset for confidence calculation (Option B enhancement)

        Returns:
            Enhanced feature vector including confidence features
        """
        features = []

        # Age features (Dynasty Guru enhanced or original)
        age = current_row.get('Age', 27)  # Default to typical age
        position = current_row.get('Primary_Position', 'OF')
        age_features = self.create_age_features(age, position, self.use_dynasty_guru)
        features.extend(age_features)

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

        # OPTION B ENHANCEMENT: Add confidence-based features (with validation safeguards)
        if training_data is not None and len(player_history) >= 3:
            try:
                player_id = current_row.get('mlbid')

                # Check if we have sufficient data for reliable confidence calculation
                player_training_data = training_data[training_data['mlbid'] == player_id]
                if len(player_training_data) >= 3:  # Require at least 3 seasons in training data
                    # Calculate confidence score for this player
                    confidence = self.calculate_player_confidence_score(player_id, training_data, age, position)

                    # Ensure confidence is within reasonable bounds
                    confidence = max(0.5, min(8.0, confidence))
                    features.append(confidence)

                    # Confidence-weighted recent performance
                    confidence_weighted_perf = confidence * weighted_performance
                    features.append(confidence_weighted_perf)

                    # Elite tier indicators based on player history (not training data)
                    wars = player_history['TARGET_METRIC'].dropna().values
                    elite_bonus = self.calculate_elite_performance_bonus(wars)
                    features.append(1.0 if elite_bonus >= 2.0 else 0.0)  # Generational talent
                    features.append(1.0 if elite_bonus >= 1.5 else 0.0)  # Elite consistent
                    features.append(1.0 if elite_bonus >= 1.0 else 0.0)  # Emerging elite

                    # Confidence-age interaction (normalized)
                    features.append(confidence * age / 30.0)
                else:
                    # Insufficient training data - use fallback
                    features.extend([1.0, weighted_performance, 0.0, 0.0, 0.0, 1.0])

            except Exception as e:
                # Fallback if confidence calculation fails - use silent fallback to avoid noise
                features.extend([1.0, weighted_performance, 0.0, 0.0, 0.0, 1.0])
        else:
            # Fallback for cases without training data or insufficient history
            features.extend([1.0, weighted_performance, 0.0, 0.0, 0.0, 1.0])

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
        Fit the confidence-aware longitudinal model for year-to-year performance prediction.

        Args:
            X: Feature matrix (now includes confidence features)
            y: Target vector (next year performance)

        Returns:
            Dictionary with performance metrics
        """
        print("Fitting confidence-aware year-to-year performance trajectory model...")

        # Use RandomForest for better outlier handling and non-linear relationships
        self.longitudinal_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )

        # Scale features (handles increased dimensionality from confidence features)
        X_scaled = self.feature_scaler.fit_transform(X)

        print(f"   Training samples: {X_scaled.shape[0]}")
        print(f"   Features: {X_scaled.shape[1]} (includes confidence features)")
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

        print(f"   Confidence-aware year-to-year prediction performance:")
        print(f"     R² = {r2:.3f}")
        print(f"     RMSE = {rmse:.3f}")
        print(f"     MAE = {mae:.3f}")

        # Log feature importance for confidence features validation
        if hasattr(self.longitudinal_model, 'feature_importances_'):
            feature_importance = self.longitudinal_model.feature_importances_
            if len(feature_importance) >= 8:  # Basic features + confidence features
                print(f"   Confidence feature importance: {feature_importance[7]:.3f}")
                if len(feature_importance) >= 12:  # All confidence features present
                    print(f"   Elite tier indicators importance: {feature_importance[9:12].mean():.3f}")

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

    def fit_joint_model(self, data: pd.DataFrame, training_data: pd.DataFrame = None) -> Dict[str, Union[float, Dict]]:
        """
        Fit the complete confidence-aware joint longitudinal-survival model.

        Args:
            data: Complete training dataset
            training_data: Full dataset for confidence calculations (OPTION B)

        Returns:
            Dictionary with all model performance metrics
        """
        print("Fitting confidence-aware joint longitudinal-survival model for future projections...")

        # Store training data reference for confidence calculations
        if training_data is not None:
            self.training_data = training_data
        else:
            self.training_data = data

        # Prepare longitudinal data with confidence features
        X, y = self.prepare_longitudinal_training_data(data)

        # Prepare survival data
        survival_df = self.prepare_survival_training_data(data)

        # Fit longitudinal component
        longitudinal_metrics = self.fit_performance_trajectory_model(X, y)

        # Fit survival component
        survival_metrics = self.fit_retirement_hazard_model(survival_df)

        self.is_fitted = True

        print("Confidence-aware joint model fitting complete!")

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
                                years_ahead: int = 3,
                                training_data: pd.DataFrame = None) -> np.ndarray:
        """
        Predict multi-year performance trajectory using confidence-aware iterative approach.

        Args:
            age: Current age
            position: Primary position
            current_war: Current season WAR
            player_history: Player's historical data
            years_ahead: Number of years to project
            training_data: Full dataset for confidence calculations (OPTION B)

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

        # Iteratively predict each year with confidence features
        for year in range(years_ahead):
            # Create current state representation
            current_state = pd.Series({
                'Age': current_age,
                'Primary_Position': position,
                'TARGET_METRIC': current_performance,  # Use TARGET_METRIC instead of WAR
                'regression_factor': 1.0,  # Default
                'mlbid': player_history.iloc[0].get('mlbid') if len(player_history) > 0 else None
            })

            # Extract features WITH confidence (OPTION B ENHANCEMENT)
            features = self._extract_longitudinal_features(
                current_state,
                extended_history,
                training_data  # Pass training data for confidence calculation
            )
            features_scaled = self.feature_scaler.transform(features.reshape(1, -1))

            # Predict next year using confidence-aware model
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

    def calculate_health_pattern_factor(self, games_played: List[int], position: str, seasons: int = 3) -> float:
        """
        Distinguish between freak injuries and recurring injury patterns.

        Args:
            games_played: List of games played in recent seasons
            position: Player position for expected games calculation
            seasons: Number of seasons to analyze

        Returns:
            Health pattern factor (0.5-1.0 range)
        """
        if not games_played:
            return 1.0

        recent_games = games_played[-seasons:] if len(games_played) >= seasons else games_played
        expected_games = 30 if position == 'P' else 140  # Pitchers vs position players

        # Calculate availability rates
        availability_rates = [games / expected_games for games in recent_games]

        # Pattern analysis
        healthy_seasons = sum(rate >= 0.85 for rate in availability_rates)  # 85%+ games
        injury_seasons = sum(rate < 0.6 for rate in availability_rates)     # <60% games

        if injury_seasons == 0:
            return 1.0  # No significant injuries
        elif injury_seasons == 1 and healthy_seasons >= 2:
            return 0.9  # Likely freak injury - minimal penalty
        elif injury_seasons == 2 and len(recent_games) >= 3:
            return 0.7  # Concerning pattern emerging
        elif injury_seasons >= 2:
            return 0.5  # Clear recurring injury pattern
        else:
            return 0.8  # Mixed/moderate concerns

    def calculate_elite_performance_bonus(self, wars: np.ndarray) -> float:
        """
        Additional confidence bonus for sustained elite performance.

        Returns:
            Elite performance bonus (0.0-2.0 range)
        """
        elite_seasons = sum(wars >= 6.0)  # 6+ WAR seasons
        superstar_seasons = sum(wars >= 8.0)  # 8+ WAR seasons
        generational_seasons = sum(wars >= 7.0)  # 7+ WAR seasons

        if superstar_seasons >= 2:
            return 2.0  # Judge, Trout tier - multiple 8+ WAR seasons
        elif generational_seasons >= 3:
            return 1.8  # Sustained 7+ WAR excellence
        elif elite_seasons >= 3:
            return 1.5  # Consistent elite performer
        elif elite_seasons >= 1 and wars.max() >= 6.5:
            return 1.0  # Emerging elite with high peak
        return 0.0

    def apply_age_decay(self, base_confidence: float, age: float) -> float:
        """
        Apply age-based confidence decay - even elites face aging.

        Returns:
            Age-adjusted confidence multiplier
        """
        if age <= 30:
            return 1.0  # No decay in prime years
        elif age <= 33:
            return 0.95  # Minimal decay in early 30s
        elif age <= 36:
            return 0.85  # Moderate decay mid-30s
        else:
            return 0.70  # Significant decay late 30s

    def calculate_recent_performance_factor(self, wars: np.ndarray, seasons: int = 3) -> float:
        """
        Weight recent performance vs career peak to catch decline.

        Returns:
            Recent performance factor (0.6-1.0 range)
        """
        if len(wars) < 2:
            return 1.0

        # Weight recent seasons more heavily
        if len(wars) >= seasons:
            weights = [0.2] * (len(wars) - 2) + [0.3, 0.5]  # 50% last, 30% prev, 20% earlier
            recent_avg = np.average(wars, weights=weights)
        else:
            recent_avg = wars[-min(2, len(wars)):].mean()  # Last 1-2 seasons

        career_peak = wars.max()

        # Performance ratio vs peak
        performance_ratio = recent_avg / max(career_peak, 1.0)

        if performance_ratio < 0.4:  # Recent performance <40% of peak
            return 0.6
        elif performance_ratio < 0.6:  # Recent performance <60% of peak
            return 0.8
        else:
            return 1.0

    def calculate_player_confidence_score(self, player_id: int, training_data: pd.DataFrame,
                                        current_age: float, position: str) -> float:
        """
        Enhanced confidence score calculation with injury patterns, age decay, and elite detection.

        Factors:
        1. Base consistency (career variance)
        2. Elite performance bonus (sustained excellence)
        3. Health pattern factor (injury vs freak accidents)
        4. Age decay factor (aging affects everyone)
        5. Recent performance factor (catch declining players)

        Returns:
            Confidence multiplier (0.5 to 6.0 range)
        """
        # Get player's historical performance
        player_history = training_data[training_data['mlbid'] == player_id].copy()

        if len(player_history) < 3:
            return 0.8  # Low confidence for limited data

        # Extract performance and games data
        wars = player_history['TARGET_METRIC'].values
        wars_clean = wars[~pd.isna(wars)]

        if len(wars_clean) < 3:
            return 0.8

        # Get games played data if available
        games_played = []
        if 'Games' in player_history.columns:
            games_played = player_history['Games'].dropna().tolist()

        # 1. Base consistency score with elite adjustment
        raw_consistency = min(3.0, 1.0 / (np.std(wars_clean) + 0.25))

        # 2. Elite performance bonus
        elite_bonus = self.calculate_elite_performance_bonus(wars_clean)

        # 3. Elite consistency adjustment - elite players get consistency bonus
        if elite_bonus >= 2.0:  # Generational talent
            consistency_score = max(raw_consistency, 1.5)  # Minimum consistency for elites
        elif elite_bonus >= 1.5:  # Elite consistent
            consistency_score = max(raw_consistency, 1.2)  # Good floor for elites
        elif elite_bonus >= 1.0:  # Emerging elite
            consistency_score = max(raw_consistency, 1.0)  # Moderate floor
        else:
            consistency_score = raw_consistency

        # 3. Health pattern factor
        health_factor = self.calculate_health_pattern_factor(games_played, position)

        # 4. Age decay factor
        age_decay = self.apply_age_decay(1.0, current_age)

        # 5. Recent performance factor
        performance_factor = self.calculate_recent_performance_factor(wars_clean)

        # 6. Extended peak bonus for proven elites
        peak_extension_bonus = 1.0
        if elite_bonus >= 1.5 and 26 <= current_age <= 32:
            peak_extension_bonus = 1.15  # Extend peak for elite players

        # Combined calculation - more aggressive scaling for elite players
        base_confidence = consistency_score * peak_extension_bonus

        # Elite multiplier - additive approach for true elites
        if elite_bonus >= 2.0:  # Generational talent
            elite_multiplier = 2.5
        elif elite_bonus >= 1.5:  # Elite consistent
            elite_multiplier = 2.0
        elif elite_bonus >= 1.0:  # Emerging elite
            elite_multiplier = 1.5
        else:
            elite_multiplier = 1.0

        # Apply elite multiplier to base, then apply penalties
        enhanced_confidence = (base_confidence * elite_multiplier + elite_bonus) * health_factor * age_decay * performance_factor

        # Higher ceiling for true elites, lower floor for declining players
        return np.clip(enhanced_confidence, 0.5, 8.0)

    def classify_player_archetype(self, player_history: pd.DataFrame, current_projection: float) -> str:
        """
        Enhanced player archetype classification with refined elite detection.

        Returns:
            'generational_talent': Highest confidence, minimal adjustment
            'elite_consistent': High confidence, minimal adjustment
            'emerging_elite': Medium-high confidence, moderate adjustment
            'breakout_candidate': Medium confidence, moderate adjustment
            'decline_expected': Lower confidence, more adjustment allowed
            'statistical_outlier': Very low confidence, heavy regression
        """
        wars = player_history['TARGET_METRIC'].dropna()

        if len(wars) < 3:
            return 'insufficient_data'

        # Enhanced elite detection
        elite_seasons = sum(wars >= 6.0)  # 6+ WAR seasons
        superstar_seasons = sum(wars >= 8.0)  # 8+ WAR seasons
        generational_seasons = sum(wars >= 7.0)  # 7+ WAR seasons
        recent_performance = wars.tail(2).mean()  # More recent focus
        career_peak = wars.max()

        # Generational talent (Judge, prime Trout, peak Bonds)
        if superstar_seasons >= 2 and recent_performance >= 6.0:
            return 'generational_talent'
        elif generational_seasons >= 3 and recent_performance >= 5.0:
            return 'generational_talent'

        # Elite consistent (Soto, Betts, Freeman)
        elif elite_seasons >= 3 and recent_performance >= 4.0:
            return 'elite_consistent'
        elif career_peak >= 7.0 and recent_performance >= 4.5:
            return 'elite_consistent'

        # Emerging elite (young players with elite peaks)
        elif career_peak >= 6.5 and recent_performance >= 4.0 and wars.iloc[-1] >= 5.0:
            return 'emerging_elite'

        # Statistical outlier (projection much higher than recent performance)
        elif current_projection > recent_performance * 1.4:
            return 'statistical_outlier'

        # Breakout candidate (improving trajectory)
        elif recent_performance > wars.head(3).mean() * 1.3:
            return 'breakout_candidate'

        # Standard case
        else:
            return 'decline_expected'

    def apply_tommy_john_recovery(self, projections: pd.DataFrame,
                                injury_data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply evidence-based Tommy John recovery adjustments to projections.

        Based on statistical analysis of 244 Tommy John cases (2020-2024):
        - Surgery to return: Position-specific recovery times (8-14 months average)
        - Year 1 post-return: 78-90% of pre-injury baseline (position/age dependent)
        - Year 2 post-return: 85-97% of baseline
        - Year 3+ post-return: 93-99% recovery (near full recovery)

        Args:
            projections: DataFrame with player projections
            injury_data: Injury data including Tommy John cases

        Returns:
            DataFrame with Tommy John recovery adjustments applied
        """
        if injury_data is None or len(injury_data) == 0:
            return projections

        # Filter for Tommy John cases
        tj_cases = injury_data[injury_data['injury_type'] == 'tommy_john'].copy()
        if len(tj_cases) == 0:
            print("  No Tommy John cases found in injury data")
            return projections

        adjusted_projections = projections.copy()
        adjustments_made = 0

        print(f"  Processing {len(tj_cases)} Tommy John cases for recovery modeling...")

        # Statistical coefficients from subagent analysis
        RECOVERY_COEFFICIENTS = {
            'SP': {'baseline_days': 399.5, 'age_effect': 7.12, 'year1_base': 0.833, 'year1_age': -0.008},
            'RP': {'baseline_days': 383.2, 'age_effect': 9.75, 'year1_base': 0.854, 'year1_age': -0.006},
            'INF': {'baseline_days': 349.4, 'age_effect': 7.78, 'year1_base': 0.891, 'year1_age': -0.009},
            'C': {'baseline_days': 441.0, 'age_effect': 12.07, 'year1_base': 0.785, 'year1_age': -0.012},
            'OF': {'baseline_days': 335.1, 'age_effect': 6.04, 'year1_base': 0.904, 'year1_age': -0.005}
        }

        for _, tj_case in tj_cases.iterrows():
            player_id = tj_case.get('mlbid')
            surgery_date = tj_case.get('injury_date')
            return_date = tj_case.get('return_date')

            # Find corresponding player in projections
            player_proj = adjusted_projections[adjusted_projections['mlbid'] == player_id]
            if len(player_proj) == 0:
                continue

            player_idx = player_proj.index[0]
            player_age = player_proj.iloc[0].get('Age', 28)
            player_position = player_proj.iloc[0].get('Position', 'OF')

            # Map position to coefficient key
            coeff_key = self._map_position_to_coefficient_key(player_position)
            if coeff_key not in RECOVERY_COEFFICIENTS:
                coeff_key = 'OF'  # Default to outfielder coefficients

            coeffs = RECOVERY_COEFFICIENTS[coeff_key]

            # Calculate recovery factors
            recovery_factors = self._calculate_tommy_john_recovery_factors(
                player_age, coeffs, surgery_date, return_date
            )

            # Apply recovery adjustments to projection years
            projection_cols = [col for col in adjusted_projections.columns
                             if col.startswith('projected_') and 'year_' in col]

            for col in projection_cols:
                if pd.isna(adjusted_projections.at[player_idx, col]):
                    continue

                year_num = int(col.split('_')[-1].replace('year_', ''))
                recovery_factor = recovery_factors.get(f'year_{year_num}', 1.0)

                # Apply recovery adjustment
                original_proj = adjusted_projections.at[player_idx, col]
                adjusted_proj = original_proj * recovery_factor

                adjusted_projections.at[player_idx, col] = adjusted_proj

            adjustments_made += 1

        print(f"  Tommy John recovery adjustments applied to {adjustments_made} players")
        return adjusted_projections

    def _map_position_to_coefficient_key(self, position: str) -> str:
        """Map player position to recovery coefficient key."""
        position_mapping = {
            'SP': 'SP', 'RP': 'RP', 'P': 'SP',
            'C': 'C',
            '1B': 'INF', '2B': 'INF', '3B': 'INF', 'SS': 'INF', 'INF': 'INF',
            'LF': 'OF', 'CF': 'OF', 'RF': 'OF', 'OF': 'OF'
        }
        return position_mapping.get(position, 'OF')

    def _calculate_tommy_john_recovery_factors(self, age: float, coefficients: Dict,
                                             surgery_date: pd.Timestamp,
                                             return_date: pd.Timestamp) -> Dict[str, float]:
        """
        Calculate position and age-specific Tommy John recovery factors.

        Args:
            age: Player age at time of surgery
            coefficients: Position-specific recovery coefficients
            surgery_date: Date of Tommy John surgery
            return_date: Date of return (if available)

        Returns:
            Dictionary with recovery factors for each projection year
        """
        # Calculate baseline recovery factors
        age_adjustment = age - 28  # Centered at age 28

        # Year 1 recovery factor (most impacted)
        year1_factor = coefficients['year1_base'] + (coefficients['year1_age'] * age_adjustment)
        year1_factor = max(0.5, min(1.0, year1_factor))  # Bound between 50-100%

        # Year 2 recovery factor (improved recovery)
        year2_improvement = 0.15  # Average 15% improvement from Year 1 to Year 2
        year2_factor = min(1.0, year1_factor + year2_improvement)

        # Year 3+ recovery factor (near full recovery)
        year3_improvement = 0.08  # Average 8% improvement from Year 2 to Year 3+
        year3_factor = min(1.0, year2_factor + year3_improvement)

        return {
            'year_1': year1_factor,
            'year_2': year2_factor,
            'year_3': year3_factor
        }

    def get_tommy_john_recovery_timeline(self, age: float, position: str) -> Dict[str, Union[int, float]]:
        """
        Calculate expected Tommy John recovery timeline for a player.

        Args:
            age: Player age
            position: Player position

        Returns:
            Dictionary with recovery timeline and success probability
        """
        coeff_key = self._map_position_to_coefficient_key(position)

        # Default coefficients if position not found
        default_coeffs = {'baseline_days': 360, 'age_effect': 8.0}

        if hasattr(self, 'RECOVERY_COEFFICIENTS'):
            coeffs = self.RECOVERY_COEFFICIENTS.get(coeff_key, default_coeffs)
        else:
            # Fallback coefficients based on statistical analysis
            recovery_coeffs = {
                'SP': {'baseline_days': 399.5, 'age_effect': 7.12},
                'RP': {'baseline_days': 383.2, 'age_effect': 9.75},
                'INF': {'baseline_days': 349.4, 'age_effect': 7.78},
                'C': {'baseline_days': 441.0, 'age_effect': 12.07},
                'OF': {'baseline_days': 335.1, 'age_effect': 6.04}
            }
            coeffs = recovery_coeffs.get(coeff_key, default_coeffs)

        # Calculate expected recovery time
        age_adjustment = age - 28
        expected_days = coeffs['baseline_days'] + (coeffs['age_effect'] * age_adjustment)
        expected_months = expected_days / 30.44

        # Success probability (age-dependent)
        base_success_rates = {'SP': 0.896, 'RP': 0.912, 'INF': 0.963, 'C': 0.667, 'OF': 0.900}
        base_success = base_success_rates.get(coeff_key, 0.850)

        # Age penalty for success rate
        age_penalty = max(0, (age - 28) * 0.02)  # 2% penalty per year over 28
        success_probability = max(0.3, base_success - age_penalty)

        return {
            'expected_recovery_days': int(expected_days),
            'expected_recovery_months': round(expected_months, 1),
            'success_probability': round(success_probability, 3),
            'position_category': coeff_key
        }

    def apply_comprehensive_injury_recovery(self, projections: pd.DataFrame,
                                          injury_data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply comprehensive injury recovery adjustments beyond Tommy John surgery.

        Covers top 5 high-priority injuries identified from FanGraphs analysis:
        1. Shoulder Surgery (65+ cases) - Critical priority
        2. Elbow Surgery Non-TJ (30+ cases) - High priority
        3. Hip Surgery (24+ cases) - Medium-high priority
        4. Back Surgery (24+ cases) - Medium-high priority
        5. Oblique Strain (48+ cases) - Medium-high priority

        Args:
            projections: DataFrame with player projections
            injury_data: Comprehensive injury database

        Returns:
            DataFrame with comprehensive injury recovery adjustments applied
        """
        if injury_data is None or len(injury_data) == 0:
            return projections

        adjusted_projections = projections.copy()

        # Process each injury type in priority order
        injury_processors = [
            ('Shoulder Surgery', self._apply_shoulder_surgery_recovery),
            ('Elbow Surgery', self._apply_elbow_surgery_recovery),
            ('Hip Surgery', self._apply_hip_surgery_recovery),
            ('Back Surgery', self._apply_back_surgery_recovery),
            ('Oblique Strain', self._apply_oblique_strain_recovery)
        ]

        total_adjustments = 0
        for injury_type, processor_func in injury_processors:
            injury_cases = injury_data[injury_data['injury_type'] == injury_type].copy()
            if len(injury_cases) > 0:
                print(f"  Processing {len(injury_cases)} {injury_type} cases...")
                adjusted_projections, adjustments = processor_func(adjusted_projections, injury_cases)
                total_adjustments += adjustments

        print(f"  Total comprehensive injury adjustments applied: {total_adjustments} players")
        return adjusted_projections

    def _apply_shoulder_surgery_recovery(self, projections: pd.DataFrame,
                                       injury_cases: pd.DataFrame) -> tuple[pd.DataFrame, int]:
        """
        Apply shoulder surgery recovery adjustments.

        Based on analysis: 65+ cases, 6-12 month recovery, severe impact on pitchers
        Recovery pattern: Year 1: 60-80%, Year 2: 85-95%, Year 3: 95-100%
        """
        # Position-specific shoulder surgery coefficients
        SHOULDER_COEFFICIENTS = {
            'SP': {'year1_base': 0.65, 'year1_age': -0.010, 'year2_base': 0.88, 'year3_base': 0.96},
            'RP': {'year1_base': 0.72, 'year1_age': -0.008, 'year2_base': 0.91, 'year3_base': 0.98},
            'INF': {'year1_base': 0.78, 'year1_age': -0.006, 'year2_base': 0.92, 'year3_base': 0.99},
            'C': {'year1_base': 0.70, 'year1_age': -0.012, 'year2_base': 0.87, 'year3_base': 0.96},
            'OF': {'year1_base': 0.80, 'year1_age': -0.005, 'year2_base': 0.94, 'year3_base': 0.99}
        }

        return self._apply_surgery_recovery_pattern(
            projections, injury_cases, SHOULDER_COEFFICIENTS, 'Shoulder Surgery'
        )

    def _apply_elbow_surgery_recovery(self, projections: pd.DataFrame,
                                    injury_cases: pd.DataFrame) -> tuple[pd.DataFrame, int]:
        """
        Apply elbow surgery (non-Tommy John) recovery adjustments.

        Based on analysis: 30+ cases, 3-8 month recovery, includes internal brace procedures
        Recovery pattern: Year 1: 70-85%, Year 2: 90-100%, Year 3: 100%
        """
        ELBOW_COEFFICIENTS = {
            'SP': {'year1_base': 0.72, 'year1_age': -0.008, 'year2_base': 0.92, 'year3_base': 1.00},
            'RP': {'year1_base': 0.78, 'year1_age': -0.006, 'year2_base': 0.95, 'year3_base': 1.00},
            'INF': {'year1_base': 0.82, 'year1_age': -0.004, 'year2_base': 0.96, 'year3_base': 1.00},
            'C': {'year1_base': 0.75, 'year1_age': -0.010, 'year2_base': 0.90, 'year3_base': 0.98},
            'OF': {'year1_base': 0.85, 'year1_age': -0.003, 'year2_base': 0.98, 'year3_base': 1.00}
        }

        return self._apply_surgery_recovery_pattern(
            projections, injury_cases, ELBOW_COEFFICIENTS, 'Elbow Surgery'
        )

    def _apply_hip_surgery_recovery(self, projections: pd.DataFrame,
                                  injury_cases: pd.DataFrame) -> tuple[pd.DataFrame, int]:
        """
        Apply hip surgery recovery adjustments.

        Based on analysis: 24+ cases, 4-6 month recovery, affects mobility uniformly
        Recovery pattern: Year 1: 80-90%, Year 2: 95-100%, Year 3: 100%
        """
        HIP_COEFFICIENTS = {
            'SP': {'year1_base': 0.85, 'year1_age': -0.005, 'year2_base': 0.97, 'year3_base': 1.00},
            'RP': {'year1_base': 0.87, 'year1_age': -0.004, 'year2_base': 0.98, 'year3_base': 1.00},
            'INF': {'year1_base': 0.82, 'year1_age': -0.007, 'year2_base': 0.95, 'year3_base': 1.00},
            'C': {'year1_base': 0.80, 'year1_age': -0.008, 'year2_base': 0.93, 'year3_base': 0.99},
            'OF': {'year1_base': 0.83, 'year1_age': -0.006, 'year2_base': 0.96, 'year3_base': 1.00}
        }

        return self._apply_surgery_recovery_pattern(
            projections, injury_cases, HIP_COEFFICIENTS, 'Hip Surgery'
        )

    def _apply_back_surgery_recovery(self, projections: pd.DataFrame,
                                   injury_cases: pd.DataFrame) -> tuple[pd.DataFrame, int]:
        """
        Apply back surgery recovery adjustments.

        Based on analysis: 24+ cases, 3-6 month recovery, high recurrence risk (30-40%)
        Recovery pattern: Year 1: 75-85%, Year 2: 85-95%, Year 3: 90-98% (recurrence factor)
        """
        BACK_COEFFICIENTS = {
            'SP': {'year1_base': 0.78, 'year1_age': -0.008, 'year2_base': 0.88, 'year3_base': 0.94},
            'RP': {'year1_base': 0.82, 'year1_age': -0.006, 'year2_base': 0.92, 'year3_base': 0.96},
            'INF': {'year1_base': 0.83, 'year1_age': -0.005, 'year2_base': 0.93, 'year3_base': 0.97},
            'C': {'year1_base': 0.75, 'year1_age': -0.010, 'year2_base': 0.85, 'year3_base': 0.92},
            'OF': {'year1_base': 0.85, 'year1_age': -0.004, 'year2_base': 0.95, 'year3_base': 0.98}
        }

        return self._apply_surgery_recovery_pattern(
            projections, injury_cases, BACK_COEFFICIENTS, 'Back Surgery'
        )

    def _apply_oblique_strain_recovery(self, projections: pd.DataFrame,
                                     injury_cases: pd.DataFrame) -> tuple[pd.DataFrame, int]:
        """
        Apply oblique strain recovery adjustments.

        Based on analysis: 48+ cases, 2-6 week recovery, 20% recurrence rate
        Recovery pattern: Year 1: 88-95%, Year 2: 96-100%, Year 3: 100%
        """
        OBLIQUE_COEFFICIENTS = {
            'SP': {'year1_base': 0.90, 'year1_age': -0.003, 'year2_base': 0.98, 'year3_base': 1.00},
            'RP': {'year1_base': 0.92, 'year1_age': -0.002, 'year2_base': 0.99, 'year3_base': 1.00},
            'INF': {'year1_base': 0.88, 'year1_age': -0.004, 'year2_base': 0.96, 'year3_base': 1.00},
            'C': {'year1_base': 0.89, 'year1_age': -0.005, 'year2_base': 0.96, 'year3_base': 1.00},
            'OF': {'year1_base': 0.91, 'year1_age': -0.003, 'year2_base': 0.98, 'year3_base': 1.00}
        }

        return self._apply_surgery_recovery_pattern(
            projections, injury_cases, OBLIQUE_COEFFICIENTS, 'Oblique Strain'
        )

    def _apply_surgery_recovery_pattern(self, projections: pd.DataFrame, injury_cases: pd.DataFrame,
                                      coefficients: Dict, injury_name: str) -> tuple[pd.DataFrame, int]:
        """
        Generic method to apply surgery recovery patterns using position-specific coefficients.

        Args:
            projections: Player projections DataFrame
            injury_cases: Cases of specific injury type
            coefficients: Position-specific recovery coefficients
            injury_name: Name of injury for logging

        Returns:
            Tuple of (adjusted_projections, number_of_adjustments_made)
        """
        adjusted_projections = projections.copy()
        adjustments_made = 0

        for _, injury_case in injury_cases.iterrows():
            player_id = injury_case.get('mlbid')

            # Find corresponding player in projections
            player_proj = adjusted_projections[adjusted_projections['mlbid'] == player_id]
            if len(player_proj) == 0:
                continue

            player_idx = player_proj.index[0]
            player_age = player_proj.iloc[0].get('Age', 28)
            player_position = player_proj.iloc[0].get('Position', 'OF')

            # Map position to coefficient key
            coeff_key = self._map_position_to_coefficient_key(player_position)
            if coeff_key not in coefficients:
                coeff_key = 'OF'  # Default to outfielder coefficients

            coeffs = coefficients[coeff_key]

            # Calculate age-adjusted recovery factors
            age_adjustment = player_age - 28  # Centered at age 28

            year1_factor = coeffs['year1_base'] + (coeffs['year1_age'] * age_adjustment)
            year1_factor = max(0.4, min(1.0, year1_factor))  # Bound between 40-100%

            year2_factor = coeffs['year2_base']
            year3_factor = coeffs['year3_base']

            recovery_factors = {
                'year_1': year1_factor,
                'year_2': year2_factor,
                'year_3': year3_factor
            }

            # Apply recovery adjustments to projection years
            projection_cols = [col for col in adjusted_projections.columns
                             if col.startswith('projected_') and 'year_' in col]

            for col in projection_cols:
                if pd.isna(adjusted_projections.at[player_idx, col]):
                    continue

                year_num = int(col.split('_')[-1])
                recovery_factor = recovery_factors.get(f'year_{year_num}', 1.0)

                # Apply recovery adjustment
                original_proj = adjusted_projections.at[player_idx, col]
                adjusted_proj = original_proj * recovery_factor

                adjusted_projections.at[player_idx, col] = adjusted_proj

            adjustments_made += 1

        return adjusted_projections, adjustments_made

    def apply_general_injury_recovery(self, projections: pd.DataFrame,
                                    injury_data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply general injury category recovery adjustments for non-surgical injuries.

        Covers additional injury categories with meaningful recovery patterns:
        - Hamstring Strain, Wrist Injury, Groin Strain, Shoulder Strain
        - Knee Injury, Ankle Injury, Hand/Finger injuries

        Args:
            projections: DataFrame with player projections
            injury_data: Comprehensive injury database

        Returns:
            DataFrame with general injury recovery adjustments applied
        """
        if injury_data is None or len(injury_data) == 0:
            return projections

        adjusted_projections = projections.copy()

        # Process general injury categories
        general_injury_processors = [
            ('Hamstring Strain', self._apply_hamstring_strain_recovery),
            ('Wrist Injury', self._apply_wrist_injury_recovery),
            ('Groin Strain', self._apply_groin_strain_recovery),
            ('Shoulder Strain', self._apply_shoulder_strain_recovery),
            ('Knee Injury', self._apply_knee_injury_recovery),
            ('Ankle Injury', self._apply_ankle_injury_recovery),
            ('Hand/Finger Injury', self._apply_hand_finger_recovery)
        ]

        total_adjustments = 0
        for injury_type, processor_func in general_injury_processors:
            injury_cases = injury_data[injury_data['injury_type'] == injury_type].copy()
            if len(injury_cases) > 0:
                print(f"  Processing {len(injury_cases)} {injury_type} cases...")
                adjusted_projections, adjustments = processor_func(adjusted_projections, injury_cases)
                total_adjustments += adjustments

        if total_adjustments > 0:
            print(f"  Total general injury adjustments applied: {total_adjustments} players")

        return adjusted_projections

    def _apply_hamstring_strain_recovery(self, projections: pd.DataFrame,
                                       injury_cases: pd.DataFrame) -> tuple[pd.DataFrame, int]:
        """
        Apply hamstring strain recovery adjustments.

        Recovery pattern: Year 1: 90-95%, Year 2: 98-100%, Year 3: 100%
        Higher impact on speed-dependent positions (OF, INF base runners)
        """
        HAMSTRING_COEFFICIENTS = {
            'SP': {'year1_base': 0.96, 'year1_age': -0.002, 'year2_base': 1.00, 'year3_base': 1.00},
            'RP': {'year1_base': 0.95, 'year1_age': -0.003, 'year2_base': 1.00, 'year3_base': 1.00},
            'INF': {'year1_base': 0.91, 'year1_age': -0.004, 'year2_base': 0.98, 'year3_base': 1.00},
            'C': {'year1_base': 0.94, 'year1_age': -0.003, 'year2_base': 0.99, 'year3_base': 1.00},
            'OF': {'year1_base': 0.90, 'year1_age': -0.005, 'year2_base': 0.97, 'year3_base': 1.00}
        }

        return self._apply_surgery_recovery_pattern(
            projections, injury_cases, HAMSTRING_COEFFICIENTS, 'Hamstring Strain'
        )

    def _apply_wrist_injury_recovery(self, projections: pd.DataFrame,
                                   injury_cases: pd.DataFrame) -> tuple[pd.DataFrame, int]:
        """
        Apply wrist injury recovery adjustments.

        Recovery pattern: Year 1: 85-92%, Year 2: 95-98%, Year 3: 99-100%
        Higher impact on hitters than pitchers
        """
        WRIST_COEFFICIENTS = {
            'SP': {'year1_base': 0.92, 'year1_age': -0.003, 'year2_base': 0.98, 'year3_base': 1.00},
            'RP': {'year1_base': 0.90, 'year1_age': -0.004, 'year2_base': 0.97, 'year3_base': 1.00},
            'INF': {'year1_base': 0.87, 'year1_age': -0.005, 'year2_base': 0.95, 'year3_base': 0.99},
            'C': {'year1_base': 0.85, 'year1_age': -0.006, 'year2_base': 0.94, 'year3_base': 0.99},
            'OF': {'year1_base': 0.88, 'year1_age': -0.004, 'year2_base': 0.96, 'year3_base': 1.00}
        }

        return self._apply_surgery_recovery_pattern(
            projections, injury_cases, WRIST_COEFFICIENTS, 'Wrist Injury'
        )

    def _apply_groin_strain_recovery(self, projections: pd.DataFrame,
                                   injury_cases: pd.DataFrame) -> tuple[pd.DataFrame, int]:
        """
        Apply groin strain recovery adjustments.

        Recovery pattern: Year 1: 88-93%, Year 2: 96-99%, Year 3: 100%
        Affects mobility and rotational power
        """
        GROIN_COEFFICIENTS = {
            'SP': {'year1_base': 0.91, 'year1_age': -0.003, 'year2_base': 0.98, 'year3_base': 1.00},
            'RP': {'year1_base': 0.93, 'year1_age': -0.002, 'year2_base': 0.99, 'year3_base': 1.00},
            'INF': {'year1_base': 0.88, 'year1_age': -0.005, 'year2_base': 0.96, 'year3_base': 1.00},
            'C': {'year1_base': 0.90, 'year1_age': -0.004, 'year2_base': 0.97, 'year3_base': 1.00},
            'OF': {'year1_base': 0.89, 'year1_age': -0.004, 'year2_base': 0.96, 'year3_base': 1.00}
        }

        return self._apply_surgery_recovery_pattern(
            projections, injury_cases, GROIN_COEFFICIENTS, 'Groin Strain'
        )

    def _apply_shoulder_strain_recovery(self, projections: pd.DataFrame,
                                      injury_cases: pd.DataFrame) -> tuple[pd.DataFrame, int]:
        """
        Apply shoulder strain (non-surgical) recovery adjustments.

        Recovery pattern: Year 1: 85-95%, Year 2: 96-99%, Year 3: 100%
        Different from shoulder surgery - shorter but still meaningful impact
        """
        SHOULDER_STRAIN_COEFFICIENTS = {
            'SP': {'year1_base': 0.87, 'year1_age': -0.004, 'year2_base': 0.96, 'year3_base': 1.00},
            'RP': {'year1_base': 0.89, 'year1_age': -0.003, 'year2_base': 0.97, 'year3_base': 1.00},
            'INF': {'year1_base': 0.93, 'year1_age': -0.002, 'year2_base': 0.98, 'year3_base': 1.00},
            'C': {'year1_base': 0.91, 'year1_age': -0.003, 'year2_base': 0.97, 'year3_base': 1.00},
            'OF': {'year1_base': 0.95, 'year1_age': -0.002, 'year2_base': 0.99, 'year3_base': 1.00}
        }

        return self._apply_surgery_recovery_pattern(
            projections, injury_cases, SHOULDER_STRAIN_COEFFICIENTS, 'Shoulder Strain'
        )

    def _apply_knee_injury_recovery(self, projections: pd.DataFrame,
                                  injury_cases: pd.DataFrame) -> tuple[pd.DataFrame, int]:
        """
        Apply knee injury recovery adjustments.

        Recovery pattern: Year 1: 82-88%, Year 2: 92-96%, Year 3: 98-100%
        Affects mobility and defensive range
        """
        KNEE_COEFFICIENTS = {
            'SP': {'year1_base': 0.88, 'year1_age': -0.003, 'year2_base': 0.96, 'year3_base': 1.00},
            'RP': {'year1_base': 0.87, 'year1_age': -0.004, 'year2_base': 0.95, 'year3_base': 0.99},
            'INF': {'year1_base': 0.82, 'year1_age': -0.006, 'year2_base': 0.92, 'year3_base': 0.98},
            'C': {'year1_base': 0.80, 'year1_age': -0.008, 'year2_base': 0.90, 'year3_base': 0.97},
            'OF': {'year1_base': 0.84, 'year1_age': -0.005, 'year2_base': 0.94, 'year3_base': 0.99}
        }

        return self._apply_surgery_recovery_pattern(
            projections, injury_cases, KNEE_COEFFICIENTS, 'Knee Injury'
        )

    def _apply_ankle_injury_recovery(self, projections: pd.DataFrame,
                                   injury_cases: pd.DataFrame) -> tuple[pd.DataFrame, int]:
        """
        Apply ankle injury recovery adjustments.

        Recovery pattern: Year 1: 88-94%, Year 2: 96-99%, Year 3: 100%
        Affects base running and defensive mobility
        """
        ANKLE_COEFFICIENTS = {
            'SP': {'year1_base': 0.94, 'year1_age': -0.002, 'year2_base': 0.99, 'year3_base': 1.00},
            'RP': {'year1_base': 0.93, 'year1_age': -0.003, 'year2_base': 0.98, 'year3_base': 1.00},
            'INF': {'year1_base': 0.88, 'year1_age': -0.005, 'year2_base': 0.96, 'year3_base': 1.00},
            'C': {'year1_base': 0.90, 'year1_age': -0.004, 'year2_base': 0.97, 'year3_base': 1.00},
            'OF': {'year1_base': 0.89, 'year1_age': -0.004, 'year2_base': 0.96, 'year3_base': 1.00}
        }

        return self._apply_surgery_recovery_pattern(
            projections, injury_cases, ANKLE_COEFFICIENTS, 'Ankle Injury'
        )

    def _apply_hand_finger_recovery(self, projections: pd.DataFrame,
                                  injury_cases: pd.DataFrame) -> tuple[pd.DataFrame, int]:
        """
        Apply hand/finger injury recovery adjustments.

        Recovery pattern: Year 1: 90-96%, Year 2: 98-100%, Year 3: 100%
        Affects batting grip and throwing accuracy
        """
        HAND_FINGER_COEFFICIENTS = {
            'SP': {'year1_base': 0.94, 'year1_age': -0.002, 'year2_base': 0.99, 'year3_base': 1.00},
            'RP': {'year1_base': 0.95, 'year1_age': -0.002, 'year2_base': 0.99, 'year3_base': 1.00},
            'INF': {'year1_base': 0.91, 'year1_age': -0.003, 'year2_base': 0.98, 'year3_base': 1.00},
            'C': {'year1_base': 0.90, 'year1_age': -0.004, 'year2_base': 0.97, 'year3_base': 1.00},
            'OF': {'year1_base': 0.92, 'year1_age': -0.003, 'year2_base': 0.98, 'year3_base': 1.00}
        }

        return self._apply_surgery_recovery_pattern(
            projections, injury_cases, HAND_FINGER_COEFFICIENTS, 'Hand/Finger Injury'
        )

    def get_comprehensive_recovery_timeline(self, injury_type: str, age: float, position: str) -> Dict[str, Union[int, float, str]]:
        """
        Unified recovery timeline system for all injury types.

        Args:
            injury_type: Type of injury (e.g., 'Tommy John Surgery', 'Shoulder Surgery', etc.)
            age: Player age
            position: Player position

        Returns:
            Dictionary with comprehensive recovery timeline information
        """
        # Define recovery timeline parameters for all injury types
        INJURY_TIMELINE_PARAMETERS = {
            'Tommy John Surgery': {
                'base_days': {'SP': 399, 'RP': 383, 'INF': 349, 'C': 441, 'OF': 335},
                'age_effect': {'SP': 7.1, 'RP': 9.8, 'INF': 7.8, 'C': 12.1, 'OF': 6.0},
                'success_rate': {'SP': 0.896, 'RP': 0.912, 'INF': 0.963, 'C': 0.667, 'OF': 0.900},
                'severity': 'Critical',
                'category': 'Surgical'
            },
            'Shoulder Surgery': {
                'base_days': {'SP': 240, 'RP': 220, 'INF': 180, 'C': 200, 'OF': 160},
                'age_effect': {'SP': 8.0, 'RP': 7.5, 'INF': 5.0, 'C': 9.0, 'OF': 4.5},
                'success_rate': {'SP': 0.780, 'RP': 0.820, 'INF': 0.890, 'C': 0.750, 'OF': 0.910},
                'severity': 'Critical',
                'category': 'Surgical'
            },
            'Elbow Surgery': {
                'base_days': {'SP': 180, 'RP': 160, 'INF': 120, 'C': 140, 'OF': 100},
                'age_effect': {'SP': 6.0, 'RP': 5.5, 'INF': 4.0, 'C': 7.0, 'OF': 3.5},
                'success_rate': {'SP': 0.850, 'RP': 0.880, 'INF': 0.920, 'C': 0.800, 'OF': 0.940},
                'severity': 'High',
                'category': 'Surgical'
            },
            'Hip Surgery': {
                'base_days': {'SP': 150, 'RP': 140, 'INF': 130, 'C': 160, 'OF': 120},
                'age_effect': {'SP': 4.0, 'RP': 4.5, 'INF': 5.0, 'C': 6.0, 'OF': 4.0},
                'success_rate': {'SP': 0.950, 'RP': 0.950, 'INF': 0.940, 'C': 0.920, 'OF': 0.950},
                'severity': 'Medium-High',
                'category': 'Surgical'
            },
            'Back Surgery': {
                'base_days': {'SP': 120, 'RP': 110, 'INF': 100, 'C': 130, 'OF': 90},
                'age_effect': {'SP': 5.0, 'RP': 4.5, 'INF': 4.0, 'C': 6.5, 'OF': 3.5},
                'success_rate': {'SP': 0.720, 'RP': 0.750, 'INF': 0.800, 'C': 0.680, 'OF': 0.820},
                'severity': 'Medium-High',
                'category': 'Surgical'
            },
            'Oblique Strain': {
                'base_days': {'SP': 28, 'RP': 25, 'INF': 21, 'C': 30, 'OF': 19},
                'age_effect': {'SP': 1.5, 'RP': 1.2, 'INF': 1.8, 'C': 2.0, 'OF': 1.4},
                'success_rate': {'SP': 0.950, 'RP': 0.960, 'INF': 0.940, 'C': 0.930, 'OF': 0.950},
                'severity': 'Medium-High',
                'category': 'Surgical'
            },
            'Hamstring Strain': {
                'base_days': {'SP': 21, 'RP': 18, 'INF': 28, 'C': 25, 'OF': 32},
                'age_effect': {'SP': 1.0, 'RP': 1.2, 'INF': 2.0, 'C': 1.8, 'OF': 2.2},
                'success_rate': {'SP': 0.980, 'RP': 0.975, 'INF': 0.960, 'C': 0.970, 'OF': 0.955},
                'severity': 'Medium',
                'category': 'General'
            },
            'Wrist Injury': {
                'base_days': {'SP': 35, 'RP': 32, 'INF': 45, 'C': 50, 'OF': 40},
                'age_effect': {'SP': 1.8, 'RP': 2.0, 'INF': 2.5, 'C': 3.0, 'OF': 2.2},
                'success_rate': {'SP': 0.920, 'RP': 0.910, 'INF': 0.880, 'C': 0.860, 'OF': 0.900},
                'severity': 'Medium',
                'category': 'General'
            },
            'Groin Strain': {
                'base_days': {'SP': 24, 'RP': 21, 'INF': 30, 'C': 28, 'OF': 26},
                'age_effect': {'SP': 1.2, 'RP': 1.0, 'INF': 1.8, 'C': 1.6, 'OF': 1.5},
                'success_rate': {'SP': 0.960, 'RP': 0.970, 'INF': 0.940, 'C': 0.950, 'OF': 0.945},
                'severity': 'Medium',
                'category': 'General'
            },
            'Shoulder Strain': {
                'base_days': {'SP': 42, 'RP': 38, 'INF': 28, 'C': 35, 'OF': 25},
                'age_effect': {'SP': 2.5, 'RP': 2.2, 'INF': 1.5, 'C': 2.0, 'OF': 1.2},
                'success_rate': {'SP': 0.890, 'RP': 0.900, 'INF': 0.940, 'C': 0.920, 'OF': 0.950},
                'severity': 'Medium',
                'category': 'General'
            },
            'Knee Injury': {
                'base_days': {'SP': 38, 'RP': 35, 'INF': 55, 'C': 65, 'OF': 48},
                'age_effect': {'SP': 2.0, 'RP': 2.2, 'INF': 3.0, 'C': 4.0, 'OF': 2.8},
                'success_rate': {'SP': 0.900, 'RP': 0.890, 'INF': 0.840, 'C': 0.800, 'OF': 0.860},
                'severity': 'Medium-High',
                'category': 'General'
            },
            'Ankle Injury': {
                'base_days': {'SP': 28, 'RP': 25, 'INF': 35, 'C': 32, 'OF': 38},
                'age_effect': {'SP': 1.5, 'RP': 1.2, 'INF': 2.2, 'C': 2.0, 'OF': 2.5},
                'success_rate': {'SP': 0.940, 'RP': 0.950, 'INF': 0.910, 'C': 0.920, 'OF': 0.900},
                'severity': 'Medium',
                'category': 'General'
            },
            'Hand/Finger Injury': {
                'base_days': {'SP': 32, 'RP': 28, 'INF': 40, 'C': 45, 'OF': 35},
                'age_effect': {'SP': 1.8, 'RP': 1.5, 'INF': 2.0, 'C': 2.5, 'OF': 1.8},
                'success_rate': {'SP': 0.950, 'RP': 0.960, 'INF': 0.930, 'C': 0.920, 'OF': 0.940},
                'severity': 'Medium',
                'category': 'General'
            }
        }

        if injury_type not in INJURY_TIMELINE_PARAMETERS:
            return {
                'error': f'Unknown injury type: {injury_type}',
                'available_types': list(INJURY_TIMELINE_PARAMETERS.keys())
            }

        params = INJURY_TIMELINE_PARAMETERS[injury_type]
        coeff_key = self._map_position_to_coefficient_key(position)

        # Calculate recovery timeline
        base_days = params['base_days'].get(coeff_key, params['base_days']['OF'])
        age_effect = params['age_effect'].get(coeff_key, params['age_effect']['OF'])
        base_success = params['success_rate'].get(coeff_key, params['success_rate']['OF'])

        # Age adjustment
        age_adjustment = age - 28
        expected_days = base_days + (age_effect * age_adjustment)
        expected_months = expected_days / 30.44

        # Age penalty for success rate
        age_penalty = max(0, age_adjustment * 0.015)  # 1.5% penalty per year over 28
        success_probability = max(0.3, base_success - age_penalty)

        # Determine return timeline categories
        if expected_days <= 30:
            timeline_category = 'Short-term (< 1 month)'
        elif expected_days <= 90:
            timeline_category = 'Medium-term (1-3 months)'
        elif expected_days <= 180:
            timeline_category = 'Extended (3-6 months)'
        else:
            timeline_category = 'Long-term (> 6 months)'

        return {
            'injury_type': injury_type,
            'position_category': coeff_key,
            'severity': params['severity'],
            'injury_category': params['category'],
            'expected_recovery_days': int(expected_days),
            'expected_recovery_months': round(expected_months, 1),
            'timeline_category': timeline_category,
            'success_probability': round(success_probability, 3),
            'age_adjusted': age != 28,
            'baseline_days': base_days,
            'age_effect_days': round(age_effect * age_adjustment, 1)
        }

    def get_injury_risk_assessment(self, player_history: pd.DataFrame, injury_type: str = None) -> Dict[str, Union[float, List]]:
        """
        Assess injury risk based on player history and demographics.

        Args:
            player_history: Player's performance history
            injury_type: Specific injury type to assess (optional)

        Returns:
            Dictionary with risk assessment information
        """
        if len(player_history) == 0:
            return {'error': 'No player history provided'}

        # Get player demographics
        latest_record = player_history.sort_values('Season').iloc[-1]
        age = latest_record.get('Age', 28)
        position = latest_record.get('Position', 'OF')

        # Position-based injury risk factors
        POSITION_RISK_FACTORS = {
            'SP': {
                'high_risk': ['Tommy John Surgery', 'Shoulder Surgery', 'Back Surgery'],
                'medium_risk': ['Elbow Surgery', 'Oblique Strain', 'Shoulder Strain'],
                'multiplier': 1.2
            },
            'RP': {
                'high_risk': ['Tommy John Surgery', 'Shoulder Surgery', 'Elbow Surgery'],
                'medium_risk': ['Oblique Strain', 'Shoulder Strain', 'Back Surgery'],
                'multiplier': 1.1
            },
            'C': {
                'high_risk': ['Knee Injury', 'Back Surgery', 'Hand/Finger Injury'],
                'medium_risk': ['Hip Surgery', 'Shoulder Surgery', 'Wrist Injury'],
                'multiplier': 1.3
            },
            'INF': {
                'high_risk': ['Hamstring Strain', 'Groin Strain', 'Ankle Injury'],
                'medium_risk': ['Wrist Injury', 'Knee Injury', 'Hand/Finger Injury'],
                'multiplier': 1.0
            },
            'OF': {
                'high_risk': ['Hamstring Strain', 'Ankle Injury', 'Groin Strain'],
                'medium_risk': ['Wrist Injury', 'Knee Injury', 'Shoulder Strain'],
                'multiplier': 0.9
            }
        }

        coeff_key = self._map_position_to_coefficient_key(position)
        position_risks = POSITION_RISK_FACTORS.get(coeff_key, POSITION_RISK_FACTORS['OF'])

        # Age-based risk multiplier
        if age <= 25:
            age_multiplier = 0.8  # Lower risk when young
        elif age <= 30:
            age_multiplier = 1.0  # Baseline risk
        elif age <= 35:
            age_multiplier = 1.4  # Increased risk
        else:
            age_multiplier = 1.8  # High risk when older

        # Calculate overall risk score
        base_risk = 0.15  # 15% baseline annual injury risk
        position_multiplier = position_risks['multiplier']
        overall_risk = base_risk * position_multiplier * age_multiplier

        risk_assessment = {
            'overall_annual_risk': round(overall_risk, 3),
            'age': age,
            'position': position,
            'position_category': coeff_key,
            'high_risk_injuries': position_risks['high_risk'],
            'medium_risk_injuries': position_risks['medium_risk'],
            'age_multiplier': age_multiplier,
            'position_multiplier': position_multiplier,
            'risk_level': 'Low' if overall_risk < 0.12 else 'Medium' if overall_risk < 0.20 else 'High'
        }

        # Specific injury risk if requested
        if injury_type:
            specific_risk = base_risk * 0.1  # 10% of base risk for specific injury
            if injury_type in position_risks['high_risk']:
                specific_risk *= 3.0
            elif injury_type in position_risks['medium_risk']:
                specific_risk *= 1.5

            specific_risk *= age_multiplier * position_multiplier
            risk_assessment['specific_injury_risk'] = round(specific_risk, 4)
            risk_assessment['specific_injury_type'] = injury_type

        return risk_assessment