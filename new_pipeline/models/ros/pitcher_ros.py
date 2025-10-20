"""
Complete ROS Ensemble for Pitchers

Three-path ensemble:
- DirectROSForecaster (50% weight): MultiQuantileHistGB with automatic lags + all exogenous features
- Darts temporal models (40% weight): TCN + TSMixer + AutoARIMA on pure WAR time series
- Pure feature baseline (10% weight): MultiQuantileHistGB with all features (no time series conversion)

Data conversion strategy:
- DirectROSForecaster: ALL 51 ROS features as exogenous variables + automatic lags from WAR
- DartsTemporalEnsemble: WAR only - "learn clean career arcs without feature noise"
- MultiQuantileHistGB baseline: All features as numpy arrays
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from .base import BaseEnsemble
from .quantile_model import MultiQuantileHistGB
from .direct_forecaster import DirectROSForecaster
from .temporal_models import DartsTemporalEnsemble
from .data_utils import (
    convert_to_sktime_format,
    convert_to_darts_format,
    validate_time_series_data
)

# ROS Pitcher Features (51 total)
ROS_PITCHER_FEATURES = [
    # ===== Current Performance =====
    'BB%', 'K%', 'ERA', 'GB%', 'SwStr%', 'WPA/LI',
    'damage_control_ratio', 'Opportunity_Success',
    'strikeout_efficiency', 'contact_management',
    'Running_Control',

    # ===== Current Season Model Output =====
    'current_predicted_war_rate',  # Output from PitcherEnsemble

    # ===== Baseline Comparisons (ERA) =====
    'career_ERA', 'healthy_ERA', 'peak_ERA', 'recent_3yr_ERA',
    'ERA_vs_healthy', 'ERA_vs_recent', 'ERA_vs_peak',
    'maintenance_ratio_ERA',

    # ===== Elite Detection =====
    'elite_tier_level',  # 0-6 encoded (scrub to mvp_level)
    'is_injury_compromised_legend',  # 0/1
    'is_declining_veteran',  # 0/1
    'is_consistent_elite',  # 0/1
    'peak_WAR',  # Career peak
    'recent_2yr_avg',  # Last 2 years
    'deviation_from_peak',  # (current - peak) / peak
    'trajectory_slope',  # 5-year linear trend
    'trajectory_r_squared',  # Trend consistency

    # ===== Rookie Detection =====
    'is_qualifying_rookie',  # 0/1
    'rookie_tier_level',  # 0-3 encoded
    'years_experience',  # MLB seasons
    'debut_age',  # Age at debut
    'is_late_bloomer',  # 0/1 (debut >=25)

    # ===== Age Curves (Dynasty Guru methodology) =====
    'age',
    'age_phase',  # 0=ascending, 1=peak, 2=declining
    'position_peak_age',  # From POSITION_CURVES
    'years_from_peak',  # Can be negative
    'adjusted_peak_age',  # Shifted for late bloomers
    'age_performance_tier_encoded',  # young_elite, prime_average, etc.
    'base_age_factor',  # From _calculate_enhanced_age_factor
    'position_decline_rate',  # Annual % decline

    # ===== Injury Recovery =====
    'injury_flag',  # 0/1
    'injury_recovery_factor',  # 0.85-1.0
    'days_since_injury',
    'injury_severity_encoded',  # 0=none, 1=strain, 2=surgery
    'recurring_injury',  # 0/1 (same type in last 2 years)
    'season_ending_injury',  # 0/1 (injury prevents return this season)

    # ===== Usage Context =====
    'IP',  # Innings pitched to date
    'G',   # Games
    'GS',  # Games started
    'season_completion_pct'  # Percent of season completed (0.0-1.0)
]


class PitcherROSEnsemble(BaseEnsemble):
    """
    Complete ROS prediction ensemble for pitchers.

    Combines three approaches with validated weights:
    - DirectROSForecaster: 50% (time series + all features)
    - DartsTemporalEnsemble: 40% (pure WAR trajectory)
    - Baseline MultiQuantileHistGB: 10% (pure features)
    """

    def __init__(
        self,
        weights: Optional[List[float]] = None,
        feature_columns: Optional[List[str]] = None,
        target_column: str = 'WAR_per_162'
    ):
        """
        Initialize ROS ensemble.

        Args:
            weights: Weights for [Direct, Temporal, Baseline]
                Default: [0.5, 0.4, 0.1]
            feature_columns: List of feature columns (default: ROS_PITCHER_FEATURES)
            target_column: Target column name (default: 'WAR_per_162')
        """
        if weights is None:
            weights = [0.5, 0.4, 0.1]

        super().__init__(player_type='pitcher', weights=weights)

        self.feature_columns = feature_columns if feature_columns is not None else ROS_PITCHER_FEATURES
        self.target_column = target_column

        # Initialize component models
        self.direct_model = DirectROSForecaster(player_type='pitcher', lags=2)
        self.temporal_model = DartsTemporalEnsemble(player_type='pitcher')
        self.baseline_model = MultiQuantileHistGB(player_type='pitcher')

        self.component_models = [
            self.direct_model,
            self.temporal_model,
            self.baseline_model
        ]

        # Track whether temporal model was fitted (may be skipped if insufficient data)
        self.temporal_model_fitted = False

    def _segment_players_by_history(
        self,
        historical_df: pd.DataFrame,
        current_df: pd.DataFrame
    ) -> Dict[str, List[int]]:
        """
        Segment players into tiers based on historical data availability.

        Args:
            historical_df: Historical data with 'playerid' and 'Year' columns
            current_df: Current season data with 'playerid' column

        Returns:
            Dictionary with tier names as keys and lists of indices (into current_df) as values:
            - 'tier1': Indices of players with 4+ years (full ensemble)
            - 'tier2': Indices of players with exactly 3 years (temporal + baseline)
            - 'tier3': Indices of players with <3 years (baseline only)

        Example:
            >>> tiers = ensemble._segment_players_by_history(hist_df, current_df)
            >>> tiers
            {'tier1': [0, 1, 5, 8, ...], 'tier2': [2, 3, 9], 'tier3': [4, 6, 7]}
        """
        # Count years per player in historical data
        if 'playerid' not in historical_df.columns:
            raise ValueError("historical_df must have 'playerid' column")

        player_year_counts = historical_df.groupby('playerid')['Year'].nunique()

        tiers = {'tier1': [], 'tier2': [], 'tier3': []}

        for idx, row in current_df.iterrows():
            playerid = row.get('playerid', None)

            if playerid is None or playerid not in player_year_counts.index:
                # No historical data - baseline only
                tiers['tier3'].append(idx)
            else:
                n_years = player_year_counts[playerid]

                if n_years >= 4:
                    # Full ensemble (Direct + Temporal + Baseline)
                    tiers['tier1'].append(idx)
                elif n_years == 3:
                    # Temporal + Baseline only
                    tiers['tier2'].append(idx)
                else:
                    # <3 years - baseline only
                    tiers['tier3'].append(idx)

        return tiers

    def fit(
        self,
        historical_df: pd.DataFrame,
        feature_columns: Optional[List[str]] = None,
        target_column: Optional[str] = None
    ) -> 'PitcherROSEnsemble':
        """
        Fit complete ensemble to historical data.

        Args:
            historical_df: Historical data with columns:
                - playerid: Player identifier
                - Year: Year
                - All feature columns
                - Target column (WAR_per_162 or WAR_per_600)
            feature_columns: Optional override for feature columns
            target_column: Optional override for target column

        Returns:
            self (fitted ensemble)

        Example:
            >>> # Historical data: 2016-2024 seasons
            >>> ensemble = PitcherROSEnsemble()
            >>> ensemble.fit(historical_df)
            >>> # Ready to predict 2025 ROS
        """
        if feature_columns is None:
            feature_columns = self.feature_columns
        if target_column is None:
            target_column = self.target_column

        print(f"Fitting PitcherROSEnsemble on {len(historical_df)} samples...")

        # 1. Convert to sktime format for DirectROSForecaster
        print("  Converting to sktime format (DirectROSForecaster)...")
        y_sktime, X_sktime = convert_to_sktime_format(
            historical_df,
            feature_columns,
            target_column
        )

        # Validate and clean time series data
        # Require 4+ years for DirectROSForecaster (lags=2 requires n > window_length + fh = 2 + 1 = 3, so min 4)
        min_required = self.direct_model.lags + 2
        y_sktime, X_sktime = validate_time_series_data(y_sktime, X_sktime, min_length=min_required)

        # Fit DirectROSForecaster
        print(f"  Fitting DirectROSForecaster ({len(y_sktime)} samples after validation)...")
        self.direct_model.fit(y=y_sktime, X=X_sktime)

        # 2. Convert to Darts format for temporal ensemble
        print("  Converting to Darts format (Temporal ensemble)...")
        series_darts = convert_to_darts_format(
            historical_df,
            target_column,
            min_length=4
        )

        # Fit temporal ensemble (only if we have data)
        if not series_darts:
            print("  Warning: No players with sufficient temporal data (3+ years), skipping DartsTemporalEnsemble")
            print("  Ensemble will use DirectROSForecaster + Baseline only")
            self.temporal_model_fitted = False
        else:
            print(f"  Fitting DartsTemporalEnsemble ({len(series_darts)} player series)...")
            self.temporal_model.fit(series_list=series_darts)
            self.temporal_model_fitted = True

        # 3. Prepare baseline training data (numpy arrays)
        print("  Preparing baseline training data...")
        # Filter to available features
        available_features = [col for col in feature_columns if col in historical_df.columns]
        X_baseline = historical_df[available_features].values
        y_baseline = historical_df[target_column].values

        # Remove NaN rows
        valid_mask = ~np.isnan(y_baseline)
        X_baseline = X_baseline[valid_mask]
        y_baseline = y_baseline[valid_mask]

        # Fit baseline
        print(f"  Fitting baseline MultiQuantileHistGB ({len(y_baseline)} samples)...")
        self.baseline_model.fit(X_baseline, y_baseline)

        self.is_fitted = True
        print("Ensemble fitting complete.")
        return self

    def predict(
        self,
        current_df: pd.DataFrame,
        historical_df: Optional[pd.DataFrame] = None
    ) -> np.ndarray:
        """
        Generate ensemble ROS predictions with cascading fallback.

        Segments players by historical data availability:
        - Tier 1 (4+ years): 50% Direct + 40% Temporal + 10% Baseline
        - Tier 2 (3 years): 57% Temporal + 43% Baseline
        - Tier 3 (<3 years): 100% Baseline

        Args:
            current_df: Current season data (firsthalf stats)
            historical_df: Historical data for time series models (required for tiers 1-2)

        Returns:
            ROS WAR predictions (n_samples,)

        Example:
            >>> # Predict 2025 ROS for qualified pitchers (mixed history lengths)
            >>> X_2025 = feature_builder.build_features_batch(current_2025, historical_2016_2024)
            >>> ros_predictions = ensemble.predict(X_2025, historical_2016_2024)
            >>> ros_predictions
            array([3.8, 2.9, 4.5, ...])  # Secondhalf WAR predictions
        """
        if not self.is_fitted:
            raise ValueError("Ensemble not fitted. Call fit() first.")

        n_players = len(current_df)
        predictions = np.zeros(n_players)

        # Get baseline predictions for all players (used in all tiers)
        available_features = [col for col in self.feature_columns if col in current_df.columns]
        X_baseline = current_df[available_features].values
        baseline_pred = self.baseline_model.predict(X_baseline)

        # If no historical data, everyone gets baseline only
        if historical_df is None:
            print("  No historical data - using baseline predictions for all players")
            return baseline_pred

        # Segment players into tiers
        tiers = self._segment_players_by_history(historical_df, current_df)
        print(f"  Player tiers: Tier1={len(tiers['tier1'])}, Tier2={len(tiers['tier2'])}, Tier3={len(tiers['tier3'])}")

        # TIER 3: Baseline only (<3 years history)
        if tiers['tier3']:
            for idx in tiers['tier3']:
                predictions[idx] = baseline_pred[idx]

        # TIER 2: Temporal + Baseline (exactly 3 years)
        if tiers['tier2']:
            tier2_indices = tiers['tier2']

            # If temporal model wasn't fitted, fall back to baseline only
            if not self.temporal_model_fitted:
                for idx in tier2_indices:
                    predictions[idx] = baseline_pred[idx]
            else:
                tier2_playerids = current_df.iloc[tier2_indices]['playerid'].values

                # Filter historical data to tier2 players
                tier2_hist = historical_df[historical_df['playerid'].isin(tier2_playerids)]

                # Get temporal predictions
                series_darts = convert_to_darts_format(tier2_hist, self.target_column, min_length=4)

                # Create player mapping for correct alignment
                player_series_map = {series.static_covariates.iloc[0]['playerid'] if hasattr(series, 'static_covariates') else None: series
                                    for series in series_darts}

                # Renormalized weights: 40:10 -> 57:43 (0.57 temporal, 0.43 baseline)
                tier2_weights = [0.57, 0.43]

                for i, idx in enumerate(tier2_indices):
                    playerid = tier2_playerids[i]

                    # Try to get temporal prediction
                    if playerid in player_series_map:
                        try:
                            temporal_val = self.temporal_model.predict(n=1, series=player_series_map[playerid])[0]
                            # Check for NaN before combining
                            if not np.isnan(temporal_val) and not np.isnan(baseline_pred[idx]):
                                # Weighted combination
                                predictions[idx] = tier2_weights[0] * temporal_val + tier2_weights[1] * baseline_pred[idx]
                            elif not np.isnan(baseline_pred[idx]):
                                # Temporal is NaN, use baseline only
                                predictions[idx] = baseline_pred[idx]
                            else:
                                # Both NaN - shouldn't happen, but set to 0
                                predictions[idx] = 0.0
                        except:
                            # Fallback to baseline if temporal fails
                            predictions[idx] = baseline_pred[idx]
                    else:
                        # No temporal series available - use baseline
                        predictions[idx] = baseline_pred[idx]

        # TIER 1: Full ensemble (4+ years)
        if tiers['tier1']:
            tier1_indices = tiers['tier1']
            tier1_playerids = current_df.iloc[tier1_indices]['playerid'].values

            # Filter historical data to tier1 players
            tier1_hist = historical_df[historical_df['playerid'].isin(tier1_playerids)]

            # Combine with current data for sktime format
            tier1_current = current_df.iloc[tier1_indices]
            combined_df = pd.concat([tier1_hist, tier1_current], ignore_index=True)

            # Convert to sktime format for DirectROSForecaster
            y_sktime, X_sktime = convert_to_sktime_format(
                combined_df,
                self.feature_columns,
                self.target_column
            )

            # Get direct predictions
            try:
                direct_pred_tier1 = self.direct_model.predict(X=X_sktime)
                # Log NaN predictions from DirectROSForecaster
                if direct_pred_tier1 is not None:
                    nan_count = np.isnan(direct_pred_tier1).sum()
                    if nan_count > 0:
                        print(f"  Warning: DirectROSForecaster returned {nan_count}/{len(direct_pred_tier1)} NaN predictions for Tier1 players")
            except:
                # Fallback if direct forecaster fails
                direct_pred_tier1 = None

            # Get temporal predictions (if temporal model was fitted)
            if self.temporal_model_fitted:
                series_darts = convert_to_darts_format(tier1_hist, self.target_column, min_length=4)
                player_series_map = {series.static_covariates.iloc[0]['playerid'] if hasattr(series, 'static_covariates') else None: series
                                    for series in series_darts}
            else:
                player_series_map = {}

            # Original weights: 50% Direct, 40% Temporal, 10% Baseline
            tier1_weights = self.weights  # [0.5, 0.4, 0.1]

            for i, idx in enumerate(tier1_indices):
                playerid = tier1_playerids[i]

                # Collect available predictions
                preds_available = []
                weights_available = []

                # Direct prediction (filter NaN)
                if direct_pred_tier1 is not None and i < len(direct_pred_tier1):
                    if not np.isnan(direct_pred_tier1[i]):
                        preds_available.append(direct_pred_tier1[i])
                        weights_available.append(tier1_weights[0])

                # Temporal prediction (only if model was fitted, already has try-except for NaN)
                if self.temporal_model_fitted and playerid in player_series_map:
                    try:
                        temporal_val = self.temporal_model.predict(n=1, series=player_series_map[playerid])[0]
                        if not np.isnan(temporal_val):
                            preds_available.append(temporal_val)
                            weights_available.append(tier1_weights[1])
                    except:
                        pass

                # Baseline (filter NaN as safety)
                if not np.isnan(baseline_pred[idx]):
                    preds_available.append(baseline_pred[idx])
                    weights_available.append(tier1_weights[2])

                # Renormalize weights and combine
                if preds_available:
                    weights_array = np.array(weights_available)
                    weights_array = weights_array / weights_array.sum()  # Renormalize
                    predictions[idx] = np.dot(preds_available, weights_array)
                else:
                    # Shouldn't happen (baseline always available), but fallback
                    predictions[idx] = baseline_pred[idx]

        return predictions

    def predict_with_elite_adjustments(
        self,
        current_df: pd.DataFrame,
        historical_df: Optional[pd.DataFrame] = None,
        season_pct: float = 0.59,
        blend_ratio: float = 0.70,
        team_games_dict: Optional[Dict[str, int]] = None,
        league_median_games: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate ROS predictions with elite player blend ratio adjustments.

        Applies tier-based post-processing to base ensemble predictions:
        1. Get base predictions from ensemble
        2. For each player, check if they qualify as elite candidate
        3. If elite: blend current WAR rate with baseline using blend_ratio
        4. If not elite: use baseline prediction as-is

        Args:
            current_df: Current season data (must include 'WAR', 'IP', 'G', 'GS', 'playerid')
            historical_df: Historical data for elite candidate detection
            season_pct: Season completion percentage (0.0-1.0, default 0.59 = ~95 games)
            blend_ratio: Weight for current rate in blending (default 0.70)
                        Final prediction = blend_ratio * current_rate_proj + (1-blend_ratio) * baseline

        Returns:
            Adjusted ROS WAR predictions (n_samples,)

        Example:
            >>> # Predict 2025 ROS with elite adjustments at All-Star break
            >>> ros_predictions = ensemble.predict_with_elite_adjustments(
            ...     current_2025, historical_2016_2024, season_pct=0.59, blend_ratio=0.70
            ... )
        """
        from .tier_thresholds import (
            get_thresholds,
            calculate_war_rate,
            is_elite_candidate,
            calculate_blended_prediction,
            subclassify_swing_pitcher
        )
        from new_pipeline.common.projections.usage_projections import calculate_remaining_usage, get_team_games_from_data

        # Get team games data if not provided
        if team_games_dict is None or league_median_games is None:
            team_games_dict, league_median_games = get_team_games_from_data(current_df)

        # Get base ensemble predictions
        base_predictions = self.predict(current_df, historical_df)

        # If no historical data, can't do elite detection - return base predictions
        if historical_df is None:
            print("  No historical data - returning base predictions without elite adjustments")
            return base_predictions

        # Get tier thresholds for current season progress
        adjusted_predictions = base_predictions.copy()

        # Build player history lookup (one row per year, using 0.75 split as proxy for full season)
        player_history = {}
        for playerid in historical_df['playerid'].unique():
            player_hist = historical_df[historical_df['playerid'] == playerid]
            # For multi-split data, keep only 0.75 split (closest to full season) per year
            if 'split_point' in player_hist.columns:
                player_hist = player_hist[player_hist['split_point'] == 0.75]
            player_history[playerid] = player_hist

        # Process each player
        for idx, row in current_df.iterrows():
            playerid = row['playerid']
            base_pred = base_predictions[idx]

            # Determine role (with swing pitcher subclassification)
            from new_pipeline.common.projections.usage_projections import classify_pitcher_role
            base_role = classify_pitcher_role(row['GS'], row['G'], row['IP'])

            if base_role == 'swing':
                role = subclassify_swing_pitcher(row['GS'], row['G'], row['IP'])
            else:
                role = base_role

            # Get thresholds for this role
            good_threshold, elite_threshold = get_thresholds(role, season_pct)

            # Calculate current WAR rate
            current_usage = row['IP']
            current_war_rate = calculate_war_rate(row['WAR'], current_usage, role)

            # Check if elite candidate
            history = player_history.get(playerid, pd.DataFrame())
            is_candidate, reason = is_elite_candidate(
                history, current_war_rate, role, elite_threshold, good_threshold
            )

            if is_candidate:
                # Calculate remaining usage projection
                remaining_usage = calculate_remaining_usage(
                    row, 'pitcher', team_games_dict, league_median_games
                )

                # Apply blend ratio adjustment
                adjusted_pred = calculate_blended_prediction(
                    current_war_rate, remaining_usage, base_pred, blend_ratio, role
                )

                adjusted_predictions[idx] = adjusted_pred

        return adjusted_predictions

    def predict_with_uncertainty(
        self,
        current_df: pd.DataFrame,
        historical_df: Optional[pd.DataFrame] = None,
        apply_elite_adjustments: bool = False,
        season_pct: float = 0.59,
        blend_ratio: float = 0.70,
        team_games_dict: Optional[Dict[str, int]] = None,
        league_median_games: Optional[int] = None
    ) -> Dict[str, np.ndarray]:
        """
        Generate predictions with uncertainty bands.

        Uses tier-based approach like predict() to handle players with varying history lengths.

        Args:
            current_df: Current season data
            historical_df: Historical data (optional)
            apply_elite_adjustments: If True, apply blend ratio adjustments to elite players (default: False)
            season_pct: Season completion percentage for elite adjustments (default: 0.59)
            blend_ratio: Blend ratio for elite adjustments (default: 0.70)

        Returns:
            Dictionary with:
            - 'mean': Ensemble mean predictions (with elite adjustments if enabled)
            - 'q10', 'q25', 'q50', 'q75', 'q90': Quantile predictions
            - 'std': Standard deviation across components
            - 'uncertainty_band': q90 - q10

        Example:
            >>> # Without elite adjustments
            >>> preds = ensemble.predict_with_uncertainty(X_2025, historical_df)
            >>> preds['mean']
            array([3.8, 2.9, ...])

            >>> # With elite adjustments
            >>> preds = ensemble.predict_with_uncertainty(
            ...     X_2025, historical_df,
            ...     apply_elite_adjustments=True, season_pct=0.59, blend_ratio=0.70
            ... )
            >>> preds['mean']  # Elite players adjusted upward
            array([4.2, 2.9, ...])
        """
        if not self.is_fitted:
            raise ValueError("Ensemble not fitted. Call fit() first.")

        n_players = len(current_df)

        # Get baseline predictions and quantiles for ALL players (always available)
        available_features = [col for col in self.feature_columns if col in current_df.columns]
        X_baseline = current_df[available_features].values
        baseline_pred = self.baseline_model.predict(X_baseline)
        quantile_preds = self.baseline_model.predict_quantiles(X_baseline)

        # Use predict() or predict_with_elite_adjustments() for ensemble mean
        if apply_elite_adjustments:
            mean_pred = self.predict_with_elite_adjustments(
                current_df, historical_df, season_pct, blend_ratio,
                team_games_dict, league_median_games
            )
        else:
            mean_pred = self.predict(current_df, historical_df)

        # Calculate component std by tracking available predictions per player
        # Initialize with baseline std (will be overwritten for players with more components)
        component_std = np.zeros(n_players)

        if historical_df is not None:
            # Segment players into tiers
            tiers = self._segment_players_by_history(historical_df, current_df)

            # TIER 3: Baseline only - std is 0 (single component)
            for idx in tiers['tier3']:
                component_std[idx] = 0.0

            # TIER 2: Temporal + Baseline
            if tiers['tier2'] and self.temporal_model_fitted:
                tier2_indices = tiers['tier2']
                tier2_playerids = current_df.iloc[tier2_indices]['playerid'].values
                tier2_hist = historical_df[historical_df['playerid'].isin(tier2_playerids)]
                series_darts = convert_to_darts_format(tier2_hist, self.target_column, min_length=4)
                player_series_map = {
                    series.static_covariates.iloc[0]['playerid'] if hasattr(series, 'static_covariates') else None: series
                    for series in series_darts
                }

                for i, idx in enumerate(tier2_indices):
                    playerid = tier2_playerids[i]
                    if playerid in player_series_map:
                        try:
                            temporal_val = self.temporal_model.predict(n=1, series=player_series_map[playerid])[0]
                            # Std of [temporal, baseline]
                            component_std[idx] = np.std([temporal_val, baseline_pred[idx]])
                        except:
                            component_std[idx] = 0.0
                    else:
                        component_std[idx] = 0.0
            elif tiers['tier2']:
                # Temporal not fitted - baseline only
                for idx in tiers['tier2']:
                    component_std[idx] = 0.0

            # TIER 1: Full ensemble (Direct + Temporal + Baseline)
            if tiers['tier1']:
                tier1_indices = tiers['tier1']
                tier1_playerids = current_df.iloc[tier1_indices]['playerid'].values
                tier1_hist = historical_df[historical_df['playerid'].isin(tier1_playerids)]

                # Get direct predictions
                tier1_current = current_df.iloc[tier1_indices]
                combined_df = pd.concat([tier1_hist, tier1_current], ignore_index=True)
                y_sktime, X_sktime = convert_to_sktime_format(
                    combined_df,
                    self.feature_columns,
                    self.target_column
                )
                try:
                    direct_pred_tier1 = self.direct_model.predict(X=X_sktime)
                except:
                    direct_pred_tier1 = None

                # Get temporal predictions
                if self.temporal_model_fitted:
                    series_darts = convert_to_darts_format(tier1_hist, self.target_column, min_length=4)
                    player_series_map = {
                        series.static_covariates.iloc[0]['playerid'] if hasattr(series, 'static_covariates') else None: series
                        for series in series_darts
                    }
                else:
                    player_series_map = {}

                for i, idx in enumerate(tier1_indices):
                    playerid = tier1_playerids[i]
                    preds_for_std = [baseline_pred[idx]]  # Always include baseline

                    # Add direct if available
                    if direct_pred_tier1 is not None and i < len(direct_pred_tier1):
                        preds_for_std.append(direct_pred_tier1[i])

                    # Add temporal if available
                    if self.temporal_model_fitted and playerid in player_series_map:
                        try:
                            temporal_val = self.temporal_model.predict(n=1, series=player_series_map[playerid])[0]
                            preds_for_std.append(temporal_val)
                        except:
                            pass

                    # Calculate std from available components
                    if len(preds_for_std) > 1:
                        component_std[idx] = np.std(preds_for_std)
                    else:
                        component_std[idx] = 0.0
        else:
            # No historical data - baseline only for all players
            component_std = np.zeros(n_players)

        # Scale quantiles to match ensemble mean
        # baseline quantiles reflect uncertainty structure but may be centered differently
        # scale_factor shifts quantiles to align with ensemble prediction
        scale_factor = np.where(
            np.abs(baseline_pred) > 1e-6,  # Avoid division by zero
            mean_pred / baseline_pred,
            1.0  # If baseline is ~0, don't scale
        )

        # Apply scaling to all quantiles
        scaled_quantiles = {
            q: quantile_preds[q] * scale_factor
            for q in [0.1, 0.25, 0.5, 0.75, 0.9]
        }

        # Enforce monotonicity: q10 <= q25 <= q50 <= q75 <= q90
        # Independent quantile models can produce inconsistent orderings
        # Sort quantiles per player to fix inversions (e.g., q50 > q90)
        quantile_array = np.column_stack([
            scaled_quantiles[0.1],
            scaled_quantiles[0.25],
            scaled_quantiles[0.5],
            scaled_quantiles[0.75],
            scaled_quantiles[0.9]
        ])
        quantile_array_sorted = np.sort(quantile_array, axis=1)

        return {
            'mean': mean_pred,
            'q10': quantile_array_sorted[:, 0],
            'q25': quantile_array_sorted[:, 1],
            'q50': quantile_array_sorted[:, 2],
            'q75': quantile_array_sorted[:, 3],
            'q90': quantile_array_sorted[:, 4],
            'std': component_std,
            'uncertainty_band': quantile_array_sorted[:, 4] - quantile_array_sorted[:, 0]
        }

    def get_component_predictions(
        self,
        current_df: pd.DataFrame,
        historical_df: Optional[pd.DataFrame] = None
    ) -> Dict[str, np.ndarray]:
        """
        Get individual predictions from each component.

        Useful for debugging and understanding ensemble behavior.

        Args:
            current_df: Current season data
            historical_df: Historical data (optional)

        Returns:
            Dictionary with 'direct', 'temporal', 'baseline' predictions

        Example:
            >>> comp_preds = ensemble.get_component_predictions(X_2025, historical_df)
            >>> comp_preds['direct']
            array([3.9, 2.8, ...])
            >>> comp_preds['temporal']
            array([3.7, 3.0, ...])
            >>> comp_preds['baseline']
            array([3.8, 2.9, ...])
        """
        if not self.is_fitted:
            raise ValueError("Ensemble not fitted. Call fit() first.")

        # Baseline
        available_features = [col for col in self.feature_columns if col in current_df.columns]
        X_baseline = current_df[available_features].values
        baseline_pred = self.baseline_model.predict(X_baseline)

        # Direct
        if historical_df is not None:
            try:
                combined_df = pd.concat([historical_df, current_df], ignore_index=True)
                y_sktime, X_sktime = convert_to_sktime_format(
                    combined_df,
                    self.feature_columns,
                    self.target_column
                )
                direct_pred = self.direct_model.predict(X=X_sktime)
            except:
                # Return NaN array if DirectROSForecaster fails (irregular time series)
                direct_pred = np.full(len(current_df), np.nan)
        else:
            direct_pred = baseline_pred.copy()

        # Temporal
        if historical_df is not None:
            series_darts = convert_to_darts_format(
                historical_df,
                self.target_column,
                min_length=4
            )
            temporal_pred = np.array([
                self.temporal_model.predict(n=1, series=series)
                for series in series_darts[:len(current_df)]
            ])
        else:
            temporal_pred = baseline_pred.copy()

        return {
            'direct': direct_pred,
            'temporal': temporal_pred,
            'baseline': baseline_pred
        }
