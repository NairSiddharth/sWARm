"""
Three-Path Pitcher Ensemble - Phase 1 Standalone Implementation.

This module implements rate-based WAR prediction for pitchers using separate
RF + Keras ensembles for starters and relievers, with blending for mixed roles.

IMPORTANT: This is a STANDALONE TESTING FILE. Do not integrate into production
until validation confirms >30% improvement over current method.

Model Choice: RF + Keras (Phase 1)
Training: Rate-based (WAR per 162 IP) to avoid train/test mismatch
Routing: Three-path based on GS/G ratio

Testing notebook: sWARm_CS_pitching.ipynb
Validation script: current_season_modules/modeling/validate_phase1.py
Documentation: docs/pitcher_war_prediction_fix.md
"""

# Standard library imports
import warnings
from typing import Dict, List, Optional

# Third-party imports
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# TensorFlow/Keras imports
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
except ImportError:
    warnings.warn(
        "TensorFlow not available. Install with: pip install tensorflow",
        ImportWarning,
        stacklevel=2
    )

# Constants
QUALIFIED_IP = 162.0  # MLB standard: 1 IP per team game
RANDOM_STATE = 42  # For reproducibility

# Pitcher features (rate-based - NO IP)
# Matches actual feature set: 9 features after removing IP
PITCHER_RATE_FEATURES = [
    'BB%', 'K%', 'K-BB%', 'ERA',
    'damage_control_ratio',
    'Opportunity_Success',
    'Contact_Quality_Index',
    'HBP%',
    'Statcast_Launch_Quality_Index'
]

# Role routing thresholds
PURE_RELIEVER_THRESHOLD = 0.1  # GS/G < 0.1
PURE_STARTER_THRESHOLD = 0.7   # GS/G > 0.7


# =============================================================================
# Data Preparation Functions
# =============================================================================

def normalize_pitcher_targets(
    war_values: np.ndarray,
    ip_values: np.ndarray,
    baseline_ip: float = QUALIFIED_IP
) -> np.ndarray:
    """
    Normalize WAR to rate per baseline IP.

    Args:
        war_values: Array of WAR values
        ip_values: Array of innings pitched
        baseline_ip: IP baseline to normalize to (default: 162)

    Returns:
        Array of WAR per baseline IP

    Example:
        >>> war = np.array([6.0, 2.5])
        >>> ip = np.array([200, 60])
        >>> normalize_pitcher_targets(war, ip, 162)
        array([4.86, 6.75])
    """
    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        war_per_162 = np.where(
            ip_values == 0,
            0,
            (war_values / ip_values) * baseline_ip
        )

    return war_per_162


def denormalize_pitcher_war(
    war_per_162: float,
    actual_ip: float,
    baseline_ip: float = QUALIFIED_IP
) -> float:
    """
    Convert rate-based WAR back to absolute WAR for given IP.

    Args:
        war_per_162: WAR rate per baseline IP
        actual_ip: Actual innings pitched
        baseline_ip: IP baseline used for normalization (default: 162)

    Returns:
        Absolute WAR for the actual IP

    Example:
        >>> denormalize_pitcher_war(4.86, 120, 162)
        3.6
    """
    return war_per_162 * (actual_ip / baseline_ip)


def calculate_role_ratio(gs: np.ndarray, g: np.ndarray) -> np.ndarray:
    """
    Calculate GS/G ratio for role classification.

    Args:
        gs: Games started
        g: Games pitched

    Returns:
        Array of GS/G ratios (0 to 1)

    Raises:
        ValueError: If G contains zero values
    """
    if np.any(g == 0):
        raise ValueError("Games pitched (G) cannot be zero")

    return gs / g


def classify_role_by_ratio(role_ratio: float) -> str:
    """
    Classify pitcher role based on GS/G ratio.

    Args:
        role_ratio: GS/G ratio

    Returns:
        Role classification: 'reliever', 'starter', or 'mixed'

    Example:
        >>> classify_role_by_ratio(0.05)
        'reliever'
        >>> classify_role_by_ratio(0.9)
        'starter'
        >>> classify_role_by_ratio(0.4)
        'mixed'
    """
    if role_ratio < PURE_RELIEVER_THRESHOLD:
        return 'reliever'
    elif role_ratio > PURE_STARTER_THRESHOLD:
        return 'starter'
    else:
        return 'mixed'


# =============================================================================
# Three-Path Pitcher Ensemble Class
# =============================================================================

class ThreePathPitcherEnsemble:
    """
    Three-path ensemble predictor for pitcher WAR/WARP.

    Uses separate RF + Keras ensembles for starters and relievers, with
    blending for mixed-role pitchers. All models trained on WAR per 162 IP
    to avoid train/test distribution mismatch.

    Architecture:
        - Pure Reliever (GS/G < 0.1): Reliever-specific RF + Keras
        - Pure Starter (GS/G > 0.7): Starter-specific RF + Keras
        - Mixed (0.1 <= GS/G <= 0.7): Linearly blended predictions

    Attributes:
        random_state: Random seed for reproducibility
        starter_models: Dict of starter models by metric type
        reliever_models: Dict of reliever models by metric type
        scalers: Dict of feature scalers by role and metric
        ensemble_weights: Weights for RF and Keras models

    Example:
        >>> ensemble = ThreePathPitcherEnsemble()
        >>> ensemble.train_starter_ensemble(X_train, y_train, 'war')
        >>> ensemble.train_reliever_ensemble(X_train, y_train, 'war')
        >>> prediction = ensemble.predict(features, GS=19, G=19, IP=120)
    """

    def __init__(self, random_state: int = RANDOM_STATE):
        """
        Initialize three-path pitcher ensemble.

        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state

        # Model storage by role and metric
        self.starter_models: Dict[str, Dict[str, any]] = {}
        self.reliever_models: Dict[str, Dict[str, any]] = {}
        self.scalers: Dict[str, StandardScaler] = {}

        # Ensemble weights (same as hitter ensemble)
        self.ensemble_weights = {
            'war': {'randomforest': 0.3, 'keras': 0.7},
            'warp': {'randomforest': 0.7, 'keras': 0.3}
        }

    def train_starter_ensemble(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        metric_type: str,
        validation_split: float = 0.2,
        epochs: int = 150,
        batch_size: int = 64
    ) -> None:
        """
        Train RF + Keras ensemble on starter data.

        Args:
            X_train: Feature matrix (rate stats, no IP)
            y_train: Target (WAR per 162 IP)
            metric_type: 'war' or 'warp'
            validation_split: Fraction for validation
            epochs: Keras training epochs
            batch_size: Keras batch size

        Raises:
            ValueError: If metric_type invalid or insufficient data
        """
        if metric_type not in ['war', 'warp']:
            raise ValueError(f"Invalid metric_type: {metric_type}")

        if len(X_train) < 50:
            raise ValueError(
                f"Insufficient starter data: {len(X_train)} samples (need >= 50)"
            )

        # Scale features
        scaler_key = f"starter_{metric_type}"
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)
        self.scalers[scaler_key] = scaler

        # Train RandomForest
        rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=self.random_state,
            n_jobs=-1
        )
        rf_model.fit(X_scaled, y_train)

        # Train Keras with early stopping
        keras_model = self._build_keras_model(X_scaled.shape[1])
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True
        )

        keras_model.fit(
            X_scaled, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stop],
            verbose=0
        )

        # Store models
        self.starter_models[metric_type] = {
            'randomforest': rf_model,
            'keras': keras_model
        }

    def train_reliever_ensemble(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        metric_type: str,
        validation_split: float = 0.2,
        epochs: int = 150,
        batch_size: int = 64
    ) -> None:
        """
        Train RF + Keras ensemble on reliever data.

        Args:
            X_train: Feature matrix (rate stats, no IP)
            y_train: Target (WAR per 162 IP)
            metric_type: 'war' or 'warp'
            validation_split: Fraction for validation
            epochs: Keras training epochs
            batch_size: Keras batch size

        Raises:
            ValueError: If metric_type invalid or insufficient data
        """
        if metric_type not in ['war', 'warp']:
            raise ValueError(f"Invalid metric_type: {metric_type}")

        if len(X_train) < 50:
            raise ValueError(
                f"Insufficient reliever data: {len(X_train)} samples (need >= 50)"
            )

        # Scale features
        scaler_key = f"reliever_{metric_type}"
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)
        self.scalers[scaler_key] = scaler

        # Train RandomForest
        rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=self.random_state,
            n_jobs=-1
        )
        rf_model.fit(X_scaled, y_train)

        # Train Keras with early stopping
        keras_model = self._build_keras_model(X_scaled.shape[1])
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True
        )

        keras_model.fit(
            X_scaled, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stop],
            verbose=0
        )

        # Store models
        self.reliever_models[metric_type] = {
            'randomforest': rf_model,
            'keras': keras_model
        }

    def _build_keras_model(self, input_dim: int) -> Sequential:
        """
        Build Keras neural network (same architecture as hitter ensemble).

        Args:
            input_dim: Number of input features

        Returns:
            Compiled Keras Sequential model

        Architecture:
            - Input layer: input_dim neurons
            - Hidden: 128 -> 64 -> 32 -> 16 neurons with dropout
            - Output: 1 neuron (regression)
            - Activation: ReLU for hidden layers
            - Optimizer: Adam
            - Loss: MSE
        """
        model = Sequential([
            Dense(128, activation='relu', input_dim=input_dim),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1)
        ])

        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model

    def predict(
        self,
        features: np.ndarray,
        GS: int,
        G: int,
        IP: float,
        metric_type: str = 'war',
        projected_ip: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Predict current and projected WAR for a pitcher.

        Args:
            features: Feature vector (rate stats, no IP)
            GS: Games started
            G: Games pitched
            IP: Current innings pitched
            metric_type: 'war' or 'warp'
            projected_ip: Projected full season IP (optional)

        Returns:
            Dictionary containing:
                - role: Pitcher role classification
                - role_ratio: GS/G ratio
                - war_per_162: Skill rate prediction
                - current_war: Current WAR based on actual IP
                - projected_war: Full season projection (if projected_ip given)
                - current_ip: Actual IP
                - projected_ip: Projected IP (if given)

        Raises:
            ValueError: If models not trained for metric_type
            ValueError: If G is zero

        Example:
            >>> result = ensemble.predict(
            ...     features=np.array([0.28, 0.015, 0.265, 2.2, ...]),
            ...     GS=19, G=19, IP=120, metric_type='war',
            ...     projected_ip=195
            ... )
            >>> print(f"Current: {result['current_war']:.2f} WAR")
            >>> print(f"Projected: {result['projected_war']:.2f} WAR")
        """
        if metric_type not in self.starter_models:
            raise ValueError(
                f"Models not trained for {metric_type}. "
                f"Call train_starter_ensemble() and train_reliever_ensemble() first."
            )

        if G == 0:
            raise ValueError("Games pitched (G) cannot be zero")

        # Calculate role
        role_ratio = GS / G
        role = classify_role_by_ratio(role_ratio)

        # Predict skill rate (WAR per 162 IP)
        war_per_162 = self._predict_war_rate(features, role_ratio, metric_type)

        # Calculate current WAR
        current_war = denormalize_pitcher_war(war_per_162, IP)

        # Build result dictionary
        result = {
            'role': role,
            'role_ratio': role_ratio,
            'war_per_162': war_per_162,
            'current_war': current_war,
            'current_ip': IP
        }

        # Add projection if requested
        if projected_ip is not None:
            result['projected_war'] = denormalize_pitcher_war(
                war_per_162, projected_ip
            )
            result['projected_ip'] = projected_ip

        return result

    def _predict_war_rate(
        self,
        features: np.ndarray,
        role_ratio: float,
        metric_type: str
    ) -> float:
        """
        Predict WAR per 162 IP using role-based routing.

        Args:
            features: Feature vector
            role_ratio: GS/G ratio
            metric_type: 'war' or 'warp'

        Returns:
            WAR per 162 IP prediction

        Routing Logic:
            - GS/G < 0.1: Pure reliever model
            - GS/G > 0.7: Pure starter model
            - 0.1 <= GS/G <= 0.7: Linear blend between models
        """
        # Reshape features if needed
        if features.ndim == 1:
            features = features.reshape(1, -1)

        if role_ratio < PURE_RELIEVER_THRESHOLD:
            # Pure reliever path
            return self._get_ensemble_prediction(
                features, 'reliever', metric_type
            )

        elif role_ratio > PURE_STARTER_THRESHOLD:
            # Pure starter path
            return self._get_ensemble_prediction(
                features, 'starter', metric_type
            )

        else:
            # Mixed role - blend predictions
            blend_weight = (role_ratio - PURE_RELIEVER_THRESHOLD) / (
                PURE_STARTER_THRESHOLD - PURE_RELIEVER_THRESHOLD
            )

            reliever_pred = self._get_ensemble_prediction(
                features, 'reliever', metric_type
            )
            starter_pred = self._get_ensemble_prediction(
                features, 'starter', metric_type
            )

            return (1 - blend_weight) * reliever_pred + blend_weight * starter_pred

    def _get_ensemble_prediction(
        self,
        features: np.ndarray,
        role: str,
        metric_type: str
    ) -> float:
        """
        Get weighted ensemble prediction from RF and Keras models.

        Args:
            features: Feature vector
            role: 'starter' or 'reliever'
            metric_type: 'war' or 'warp'

        Returns:
            Weighted ensemble prediction
        """
        # Select appropriate models
        models = self.starter_models if role == 'starter' else self.reliever_models
        scaler_key = f"{role}_{metric_type}"

        # Scale features
        X_scaled = self.scalers[scaler_key].transform(features)

        # Get predictions from both models
        rf_pred = models[metric_type]['randomforest'].predict(X_scaled)[0]
        keras_pred = models[metric_type]['keras'].predict(X_scaled, verbose=0)[0][0]

        # Weighted ensemble
        weights = self.ensemble_weights[metric_type]
        ensemble_pred = (
            weights['randomforest'] * rf_pred +
            weights['keras'] * keras_pred
        )

        return float(ensemble_pred)


# =============================================================================
# Training and Prediction Helper Functions
# =============================================================================

def prepare_pitcher_data_for_training(
    pitcher_df: pd.DataFrame,
    features: List[str],
    target_col: str = 'WAR',
    holdout_year: Optional[int] = None
) -> Dict[str, np.ndarray]:
    """
    Prepare pitcher data for rate-based ensemble training.

    Args:
        pitcher_df: DataFrame with pitcher statistics
        features: List of feature column names (should not include IP)
        target_col: Target column name ('WAR' or 'WARP')
        holdout_year: Optional year to hold out for validation

    Returns:
        Dictionary containing:
            - X: Feature matrix
            - y: Normalized target (WAR per 162 IP)
            - GS: Games started
            - G: Games pitched
            - IP: Innings pitched
            - years: Season years
            - role_ratio: GS/G ratios
            - train_mask: Boolean mask for training data (if holdout_year given)

    Raises:
        ValueError: If required columns missing
        ValueError: If IP contains zeros or negative values
    """
    # Validate required columns
    required_cols = features + [target_col, 'GS', 'G', 'IP', 'Season']
    missing_cols = set(required_cols) - set(pitcher_df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Filter out invalid IP values
    valid_ip_mask = pitcher_df['IP'] > 0
    if not valid_ip_mask.all():
        n_invalid = (~valid_ip_mask).sum()
        warnings.warn(
            f"Filtering {n_invalid} rows with IP <= 0",
            UserWarning,
            stacklevel=2
        )
        pitcher_df = pitcher_df[valid_ip_mask].copy()

    # Extract arrays
    X = pitcher_df[features].values
    y = pitcher_df[target_col].values
    GS = pitcher_df['GS'].values
    G = pitcher_df['G'].values
    IP = pitcher_df['IP'].values
    years = pitcher_df['Season'].values

    # Normalize targets
    y_normalized = normalize_pitcher_targets(y, IP)

    # Calculate role ratios
    role_ratios = calculate_role_ratio(GS, G)

    # Create result dictionary
    result = {
        'X': X,
        'y': y_normalized,
        'GS': GS,
        'G': G,
        'IP': IP,
        'years': years,
        'role_ratio': role_ratios
    }

    # Add train mask if holdout specified
    if holdout_year is not None:
        result['train_mask'] = years != holdout_year

    return result


def train_three_path_ensemble(
    pitcher_data_dict: Dict[str, pd.DataFrame],
    holdout_year: Optional[int] = None
) -> ThreePathPitcherEnsemble:
    """
    Train three-path pitcher ensemble on historical data.

    Args:
        pitcher_data_dict: Dictionary with 'war' and 'warp' DataFrames
        holdout_year: Optional year to hold out for validation

    Returns:
        Trained ThreePathPitcherEnsemble instance

    Raises:
        ValueError: If required data missing or insufficient samples

    Example:
        >>> pitcher_data = {'war': war_df, 'warp': warp_df}
        >>> ensemble = train_three_path_ensemble(pitcher_data, holdout_year=2024)
    """
    ensemble = ThreePathPitcherEnsemble()

    for metric_type in ['war', 'warp']:
        if metric_type not in pitcher_data_dict:
            warnings.warn(
                f"Skipping {metric_type} - not in pitcher_data_dict",
                UserWarning,
                stacklevel=2
            )
            continue

        # Prepare data
        data = prepare_pitcher_data_for_training(
            pitcher_data_dict[metric_type],
            PITCHER_RATE_FEATURES,
            target_col=metric_type.upper(),
            holdout_year=holdout_year
        )

        # Split by role
        role_ratios = data['role_ratio']
        starter_mask = role_ratios > PURE_STARTER_THRESHOLD
        reliever_mask = role_ratios < PURE_RELIEVER_THRESHOLD

        # Apply holdout if specified
        if holdout_year is not None:
            train_mask = data['train_mask']
            starter_train_mask = starter_mask & train_mask
            reliever_train_mask = reliever_mask & train_mask
        else:
            starter_train_mask = starter_mask
            reliever_train_mask = reliever_mask

        # Extract starter training data
        X_starters = data['X'][starter_train_mask]
        y_starters = data['y'][starter_train_mask]

        if len(X_starters) < 50:
            raise ValueError(
                f"Insufficient starter data for {metric_type}: "
                f"{len(X_starters)} samples (need >= 50)"
            )

        # Extract reliever training data
        X_relievers = data['X'][reliever_train_mask]
        y_relievers = data['y'][reliever_train_mask]

        if len(X_relievers) < 50:
            raise ValueError(
                f"Insufficient reliever data for {metric_type}: "
                f"{len(X_relievers)} samples (need >= 50)"
            )

        # Train ensembles
        print(f"Training {metric_type.upper()} starter ensemble "
              f"({len(X_starters)} samples)...")
        ensemble.train_starter_ensemble(X_starters, y_starters, metric_type)

        print(f"Training {metric_type.upper()} reliever ensemble "
              f"({len(X_relievers)} samples)...")
        ensemble.train_reliever_ensemble(X_relievers, y_relievers, metric_type)

    return ensemble


if __name__ == "__main__":
    print("Three-Path Pitcher Ensemble - Phase 1")
    print("=" * 60)
    print("This is a standalone testing file.")
    print("Run validation via: validate_phase1.py")
    print("Testing notebook: sWARm_CS_pitching.ipynb")
    print("\nModel: RF + Keras (rate-based)")
    print(f"Features: {len(PITCHER_RATE_FEATURES)} rate stats (no IP)")
    print(f"Routing: GS/G ratio ({PURE_RELIEVER_THRESHOLD}/{PURE_STARTER_THRESHOLD})")
    print("=" * 60)
