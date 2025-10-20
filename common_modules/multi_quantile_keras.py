"""
Multi-quantile Keras models for regression with sklearn compatibility.

This module provides:
- Custom TensorFlow quantile loss for simultaneous multi-quantile prediction
- Keras model builder with configurable quantiles and architecture
- Wrapper class to extract individual quantiles for sklearn StackingRegressor

Quantile regression addresses systematic underprediction bias by penalizing
underpredictions more heavily than overpredictions for quantiles > 0.5.

Architecture follows CODING_PRINCIPLES.md:
- Google-style docstrings for all functions
- Type hints for function signatures
- Error handling with try/except blocks
- No magic numbers (all as named constants)
"""

# Standard library imports
from typing import List, Tuple, Optional, Callable

# Third-party imports
import numpy as np
import tensorflow as tf
import keras  # Keras 3 standalone
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.base import BaseEstimator, RegressorMixin

# Constants (per CODING_PRINCIPLES.md)
DEFAULT_QUANTILES = [0.5, 0.75, 0.9]
DEFAULT_QUANTILE_WEIGHTS = [0.2, 0.3, 0.5]  # Emphasize upper quantiles
DEFAULT_HIDDEN_LAYERS = (128, 64, 32, 16)
DEFAULT_DROPOUT_RATES = (0.3, 0.3, 0.2, 0.0)
DEFAULT_LEARNING_RATE = 0.001


@keras.saving.register_keras_serializable()
def multi_quantile_loss(
    quantiles: List[float],
    weights: Optional[List[float]] = None
) -> Callable:
    """
    Create multi-quantile loss function for Keras.

    Quantile loss penalizes underprediction more heavily than overprediction
    for quantiles > 0.5, making it suitable for addressing elite pitcher
    underprediction bias.

    The loss for quantile q is:
        L(y, ŷ) = q * (y - ŷ)     if y > ŷ  (underprediction)
                  (q-1) * (y - ŷ) if y ≤ ŷ  (overprediction)

    For q=0.9: underprediction penalty is 9x overprediction penalty.

    Args:
        quantiles: List of target quantiles (e.g., [0.5, 0.75, 0.9])
            Each value must be in (0, 1)
        weights: Optional loss weights per quantile (default: equal weighting)
            Must sum to 1.0 if provided

    Returns:
        TensorFlow loss function compatible with model.compile()

    Raises:
        ValueError: If quantiles not in (0, 1) or weights don't sum to 1

    Example:
        >>> loss_fn = multi_quantile_loss([0.5, 0.9], weights=[0.3, 0.7])
        >>> model.compile(optimizer='adam', loss=loss_fn)
    """
    # Validate quantiles
    if any(q <= 0 or q >= 1 for q in quantiles):
        raise ValueError(
            f"All quantiles must be in (0, 1), got: {quantiles}"
        )

    # Set default weights if not provided
    if weights is None:
        weights = [1.0 / len(quantiles)] * len(quantiles)
    else:
        # Validate weights sum to 1
        if not np.isclose(sum(weights), 1.0):
            raise ValueError(
                f"Weights must sum to 1.0, got sum={sum(weights)}"
            )

    def loss(y_true, y_pred):
        """
        Compute weighted multi-quantile loss.

        Args:
            y_true: True target values (batch_size,)
            y_pred: Predicted quantiles (batch_size, num_quantiles)

        Returns:
            Scalar loss value
        """
        # Ensure y_true has correct shape for broadcasting
        # y_true: (batch_size,) -> (batch_size, 1)
        y_true_expanded = tf.expand_dims(y_true, axis=-1)

        losses = []
        for i, q in enumerate(quantiles):
            # Extract predictions for this quantile
            y_pred_q = y_pred[:, i:i+1]

            # Compute error
            error = y_true_expanded - y_pred_q

            # Asymmetric quantile loss
            q_loss = tf.reduce_mean(
                tf.maximum(q * error, (q - 1) * error)
            )

            # Apply weight
            losses.append(weights[i] * q_loss)

        # Sum weighted losses
        return tf.add_n(losses)

    # Set function name for debugging
    loss.__name__ = f'multi_quantile_loss_{quantiles}'

    return loss


def build_multi_quantile_keras(
    input_dim: int,
    quantiles: Optional[List[float]] = None,
    quantile_weights: Optional[List[float]] = None,
    hidden_layers: Optional[Tuple[int, ...]] = None,
    dropout_rates: Optional[Tuple[float, ...]] = None,
    learning_rate: float = DEFAULT_LEARNING_RATE
) -> Sequential:
    """
    Build Keras neural network for multi-quantile regression.

    Architecture:
    - Input layer (input_dim neurons)
    - Hidden layers with ReLU activation and dropout
    - Output layer (len(quantiles) neurons, no activation)

    Args:
        input_dim: Number of input features
        quantiles: Target quantiles (default: [0.5, 0.75, 0.9])
        quantile_weights: Loss weights (default: [0.2, 0.3, 0.5])
        hidden_layers: Tuple of layer sizes (default: (128, 64, 32, 16))
        dropout_rates: Dropout per layer (default: (0.3, 0.3, 0.2, 0))
        learning_rate: Adam optimizer learning rate

    Returns:
        Compiled Keras Sequential model

    Raises:
        ValueError: If hidden_layers and dropout_rates have different lengths

    Example:
        >>> model = build_multi_quantile_keras(
        ...     input_dim=10,
        ...     quantiles=[0.5, 0.9],
        ...     quantile_weights=[0.4, 0.6]
        ... )
        >>> model.fit(X_train, y_train, epochs=100)
    """
    # Set defaults
    if quantiles is None:
        quantiles = DEFAULT_QUANTILES
    if quantile_weights is None:
        quantile_weights = DEFAULT_QUANTILE_WEIGHTS
    if hidden_layers is None:
        hidden_layers = DEFAULT_HIDDEN_LAYERS
    if dropout_rates is None:
        dropout_rates = DEFAULT_DROPOUT_RATES

    # Validate hidden_layers and dropout_rates match
    if len(hidden_layers) != len(dropout_rates):
        raise ValueError(
            f"hidden_layers ({len(hidden_layers)}) and "
            f"dropout_rates ({len(dropout_rates)}) must have same length"
        )

    # Build model
    model = Sequential()

    # Input + first hidden layer
    model.add(Dense(
        hidden_layers[0],
        activation='relu',
        input_dim=input_dim
    ))
    if dropout_rates[0] > 0:
        model.add(Dropout(dropout_rates[0]))

    # Remaining hidden layers
    for units, dropout in zip(hidden_layers[1:], dropout_rates[1:]):
        model.add(Dense(units, activation='relu'))
        if dropout > 0:
            model.add(Dropout(dropout))

    # Output layer: one neuron per quantile (no activation)
    model.add(Dense(len(quantiles)))

    # Compile with multi-quantile loss
    loss_fn = multi_quantile_loss(quantiles, quantile_weights)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss_fn
        # Note: No metrics - multi-output loss doesn't work with standard metrics
    )

    return model


class MultiQuantileExtractor(BaseEstimator, RegressorMixin):
    """
    Extract single quantile from multi-quantile Keras for sklearn StackingRegressor.

    StackingRegressor expects single-output estimators. This wrapper allows
    a multi-quantile Keras model to provide separate outputs for each quantile.

    The wrapper ensures the underlying Keras model is only trained once, then
    multiple extractors can reference the same trained model.

    Attributes:
        keras_model: Underlying KerasRegressor with multi-quantile output
        quantile_idx: Index of quantile to extract (0, 1, 2, ...)
        _is_fitted: Internal flag to prevent re-training

    Example:
        >>> from scikeras.wrappers import KerasRegressor
        >>> keras_full = KerasRegressor(
        ...     model=build_multi_quantile_keras,
        ...     model__input_dim=10,
        ...     epochs=100
        ... )
        >>>
        >>> # Create 3 extractors for use in StackingRegressor
        >>> keras_q50 = MultiQuantileExtractor(keras_full, quantile_idx=0)
        >>> keras_q75 = MultiQuantileExtractor(keras_full, quantile_idx=1)
        >>> keras_q90 = MultiQuantileExtractor(keras_full, quantile_idx=2)
        >>>
        >>> # Use in stacking
        >>> from sklearn.ensemble import StackingRegressor
        >>> from xgboost import XGBRegressor
        >>> stacking = StackingRegressor(
        ...     estimators=[
        ...         ('keras_q50', keras_q50),
        ...         ('keras_q75', keras_q75),
        ...         ('keras_q90', keras_q90)
        ...     ],
        ...     final_estimator=XGBRegressor(...)
        ... )
    """

    # sklearn requirement: declare estimator type as class attribute
    _estimator_type = "regressor"

    def __init__(self, keras_model, quantile_idx: int):
        """
        Initialize quantile extractor.

        Args:
            keras_model: KerasRegressor with multi-quantile output
            quantile_idx: Index of quantile to extract (0, 1, 2, ...)

        Raises:
            ValueError: If quantile_idx is negative
        """
        if quantile_idx < 0:
            raise ValueError(
                f"quantile_idx must be non-negative, got {quantile_idx}"
            )

        self.keras_model = keras_model
        self.quantile_idx = quantile_idx
        self._is_fitted = False

    def fit(self, X, y, **kwargs):
        """
        Train underlying Keras model (only once).

        Args:
            X: Training features
            y: Training targets
            **kwargs: Additional arguments passed to keras_model.fit()

        Returns:
            self
        """
        # Check if the underlying Keras model is already fitted
        # We store a flag on the keras_model itself to share across extractors
        if not hasattr(self.keras_model, '_multi_quantile_fitted'):
            try:
                self.keras_model.fit(X, y, **kwargs)
                # Mark the shared model as fitted
                self.keras_model._multi_quantile_fitted = True
                self._is_fitted = True
            except Exception as e:
                raise RuntimeError(
                    f"Failed to train Keras model: {e}"
                ) from e
        else:
            # Model already trained by another extractor
            self._is_fitted = True

        return self

    def predict(self, X):
        """
        Extract single quantile prediction.

        Args:
            X: Features to predict on

        Returns:
            1D array of predictions for specified quantile

        Raises:
            RuntimeError: If model not fitted yet
        """
        if not self._is_fitted:
            raise RuntimeError(
                "Model not fitted yet. Call fit() before predict()."
            )

        try:
            # Get all quantile predictions
            all_quantiles = self.keras_model.predict(X)

            # [DIAGNOSTIC] Log raw Keras output
            print(f"[MultiQuantileExtractor q{self.quantile_idx}] Raw Keras output shape: {all_quantiles.shape}")
            if X.shape[0] <= 3:  # Only print values for small batches
                print(f"[MultiQuantileExtractor q{self.quantile_idx}] Raw values: {all_quantiles}")

            # Keras/SciKeras may return flattened output (n_samples * n_quantiles,)
            # We need to reshape to (n_samples, n_quantiles)
            n_samples = X.shape[0]

            if len(all_quantiles.shape) == 1:
                # Flattened output - reshape to (n_samples, n_quantiles)
                # The model outputs 3 quantiles, so total elements = n_samples * 3
                n_quantiles = all_quantiles.shape[0] // n_samples
                all_quantiles = all_quantiles.reshape(n_samples, n_quantiles)
                print(f"[MultiQuantileExtractor q{self.quantile_idx}] Reshaped to: {all_quantiles.shape}")

            # Extract column for this quantile
            # Return 1D array with shape (n_samples,)
            predictions = all_quantiles[:, self.quantile_idx]

            # [DIAGNOSTIC] Log extracted quantile
            print(f"[MultiQuantileExtractor q{self.quantile_idx}] Extracted predictions shape: {predictions.shape}")
            if X.shape[0] <= 3:
                print(f"[MultiQuantileExtractor q{self.quantile_idx}] Extracted values: {predictions}")

            return predictions

        except Exception as e:
            raise RuntimeError(
                f"Prediction failed for quantile_idx={self.quantile_idx}: {e}"
            ) from e

    def __sklearn_tags__(self):
        """
        Provide sklearn tags for version 1.7+ compatibility.

        sklearn 1.7+ uses get_tags() which calls this method to determine
        estimator type instead of checking _estimator_type attribute.
        """
        from sklearn.utils._tags import Tags, RegressorTags, InputTags, TargetTags

        # Get parent tags if available
        if hasattr(super(), '__sklearn_tags__'):
            parent_tags = super().__sklearn_tags__()
        else:
            # Create default tags for a regressor
            parent_tags = Tags(
                estimator_type='regressor',
                target_tags=TargetTags(required=True),
                regressor_tags=RegressorTags(),
                input_tags=InputTags(two_d_array=True)
            )

        # Ensure estimator_type is set to 'regressor'
        parent_tags.estimator_type = 'regressor'
        return parent_tags

    def get_params(self, deep=True):
        """
        Get parameters for this estimator (required by sklearn).

        Args:
            deep: If True, return parameters of sub-objects

        Returns:
            Dictionary of parameter names to values
        """
        return {
            'keras_model': self.keras_model,
            'quantile_idx': self.quantile_idx
        }

    def set_params(self, **params):
        """
        Set parameters for this estimator (required by sklearn).

        Args:
            **params: Estimator parameters

        Returns:
            self
        """
        for key, value in params.items():
            setattr(self, key, value)
        return self
