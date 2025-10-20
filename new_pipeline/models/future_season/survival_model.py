"""
Survival Model for Retirement Probability Prediction

Uses Cox Proportional Hazards model to predict retirement probability.
Copied from future_season_modules/future_projections.py lines 750-840.

See FUTURE_PROJECTIONS_MODEL_ARCHITECTURE.md Module 2 for design specs.
See FUTURE_PROJECTIONS_MIGRATION_GUIDE.md Section 3C for migration notes.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

try:
    from lifelines import CoxPHFitter
    from lifelines.utils import concordance_index
    LIFELINES_AVAILABLE = True
except ImportError:
    print("Warning: lifelines not installed. Survival model will not be available.")
    print("Install with: pip install lifelines")
    LIFELINES_AVAILABLE = False


class SurvivalModel:
    """
    Retirement probability model using Cox Proportional Hazards.

    Predicts P(retire | age, performance) for future projections.

    Features:
    - age_at_end: Player age at end of season
    - final_war: WAR in most recent season
    - career_war: Cumulative career WAR
    - peak_war: Best single-season WAR
    - war_decline: Difference between peak and recent WAR
    """

    def __init__(self):
        """Initialize survival model."""
        if not LIFELINES_AVAILABLE:
            raise ImportError("lifelines library required for survival model. Install with: pip install lifelines")

        self.model = None
        self.is_fitted = False
        self.concordance_index_func = concordance_index

    def prepare_survival_data(
        self,
        historical_df: pd.DataFrame,
        current_year: int = 2024
    ) -> pd.DataFrame:
        """
        Prepare survival data from historical player data.

        Creates career histories with retirement events.

        Args:
            historical_df: Historical player data from load_historical_player_data()
            current_year: Current year for censoring (default: 2024)

        Returns:
            DataFrame with columns:
                - playerid
                - duration: Career length (years)
                - event: 1 if retired, 0 if still active (censored)
                - age_at_end: Age at end of career
                - final_war: WAR in final season
                - career_war: Cumulative career WAR
                - peak_war: Best single-season WAR
                - war_decline: Peak - recent WAR
        """
        survival_data = []

        for playerid, player_df in historical_df.groupby('playerid'):
            player_df = player_df.sort_values('Year')

            # Career duration
            career_start = player_df['Year'].min()
            career_end = player_df['Year'].max()
            duration = career_end - career_start + 1

            # Retirement status (1 = retired before current year, 0 = still active/censored)
            event = 1 if career_end < current_year else 0

            # Final season stats
            final_season = player_df.iloc[-1]
            age_at_end = final_season['Age']
            final_war = final_season['WAR']

            # Career stats
            career_war = player_df['WAR'].sum()
            peak_war = player_df['WAR'].max()

            # War decline (compare recent to peak)
            if len(player_df) >= 3:
                recent_war = player_df['WAR'].tail(2).mean()
                war_decline = peak_war - recent_war
            else:
                war_decline = 0.0

            survival_data.append({
                'playerid': playerid,
                'duration': duration,
                'event': event,
                'age_at_end': age_at_end,
                'final_war': final_war,
                'career_war': career_war,
                'peak_war': peak_war,
                'war_decline': war_decline
            })

        survival_df = pd.DataFrame(survival_data)

        # Ensure minimum duration > 0 (Cox PH requirement)
        survival_df['duration'] = survival_df['duration'].clip(lower=0.1)

        return survival_df

    def train(self, survival_df: pd.DataFrame) -> Dict[str, float]:
        """
        Train Cox PH model on survival data.

        Args:
            survival_df: Output from prepare_survival_data()

        Returns:
            Training metrics: {'concordance_index': float, 'n_observations': int, 'n_events': int}
        """
        print("Training survival model (Cox PH)...")
        print(f"  Observations: {len(survival_df)}")
        print(f"  Retirement events: {survival_df['event'].sum()}")
        print(f"  Censored (still active): {(~survival_df['event'].astype(bool)).sum()}")
        print(f"  Event rate: {survival_df['event'].mean():.3f}")

        # Survival features
        survival_features = ['age_at_end', 'final_war', 'career_war', 'peak_war', 'war_decline']

        # Prepare model data
        model_data = survival_df[['duration', 'event'] + survival_features].copy()

        # Remove any features with zero variance
        for feature in survival_features:
            if model_data[feature].std() == 0:
                print(f"   Warning: Removing zero-variance feature: {feature}")
                model_data = model_data.drop(columns=[feature])
                survival_features.remove(feature)

        # Fit Cox Proportional Hazards model
        # Copy from future_projections.py lines 788-794
        self.model = CoxPHFitter(penalizer=0.1)  # Regularization for stability
        try:
            self.model.fit(
                model_data,
                duration_col='duration',
                event_col='event',
                fit_options={'step_size': 0.1, 'max_steps': 100}
            )
            self.is_fitted = True
        except Exception as e:
            print(f"   Warning: Cox model fitting failed ({str(e)}). Using simplified approach.")
            # Fallback to simpler features
            simple_features = ['age_at_end', 'final_war', 'career_war']
            available_features = [f for f in simple_features if f in model_data.columns]

            if available_features:
                simple_data = model_data[['duration', 'event'] + available_features].copy()
                self.model = CoxPHFitter(penalizer=0.1)
                self.model.fit(simple_data, duration_col='duration', event_col='event')
                self.is_fitted = True
            else:
                raise RuntimeError("Unable to fit survival model with available features")

        # Calculate concordance index
        try:
            final_features = [col for col in self.model.params_.index if col != 'intercept']
            if final_features:
                test_data = model_data[final_features]
                c_index = concordance_index(
                    model_data['duration'],
                    -self.model.predict_partial_hazard(test_data),
                    model_data['event']
                )
            else:
                c_index = 0.5
        except Exception as e:
            print(f"   Warning: Could not calculate concordance index: {str(e)}")
            c_index = 0.5

        print(f"   Cox PH model fitted successfully")
        print(f"   Concordance index: {c_index:.3f}")

        metrics = {
            'concordance_index': c_index,
            'n_observations': len(model_data),
            'n_events': int(model_data['event'].sum())
        }

        return metrics

    def predict_survival_probability(
        self,
        player_features: pd.DataFrame,
        years_ahead: int = 1
    ) -> np.ndarray:
        """
        Predict survival probability (probability of still playing) for given years ahead.

        Args:
            player_features: DataFrame with survival features
            years_ahead: Number of years to predict

        Returns:
            Survival probabilities (n_players,)
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call train() first.")

        # Get survival function predictions
        try:
            survival_probs = self.model.predict_survival_function(
                player_features,
                times=[years_ahead]
            ).values.flatten()
        except Exception as e:
            print(f"Warning: Survival prediction failed ({str(e)}). Using default probability.")
            # Default to 0.9 survival probability (90% chance of continuing)
            survival_probs = np.full(len(player_features), 0.9)

        return survival_probs

    def get_summary(self) -> Optional[str]:
        """
        Get model summary.

        Returns:
            Model summary string or None if not fitted
        """
        if not self.is_fitted:
            return None

        return str(self.model.summary)
