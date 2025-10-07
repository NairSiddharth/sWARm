"""
Keras utilities for multi-quantile regression with AdamW optimizer.

Provides:
- Multi-quantile loss function (copied from common_modules)
- AdamW-based model builder with Swish activation and BatchNorm
- Training callbacks for early stopping and learning rate scheduling
"""

from typing import List, Optional, Callable
import numpy as np
import tensorflow as tf
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


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


def build_multi_quantile_keras_adamw(input_dim: int) -> Sequential:
    """
    Build Keras neural network for multi-quantile regression with AdamW.

    UPDATES from basic implementation:
    - Architecture: 256→128→64→32→3 (wider for more capacity)
    - Optimizer: AdamW instead of Adam (proper weight decay)
    - Activation: Swish instead of ReLU (smooth gradients, better for outliers)
    - Added BatchNormalization layers (faster convergence, more stable)

    Architecture:
    - Input layer (input_dim neurons)
    - Hidden layers: 256, 128, 64, 32 with Swish + BatchNorm + Dropout
    - Output layer: 3 neurons for [q50, q75, q90] quantiles

    Args:
        input_dim: Number of input features

    Returns:
        Compiled Keras Sequential model

    Example:
        >>> model = build_multi_quantile_keras_adamw(input_dim=12)
        >>> model.fit(X_train, y_train, epochs=200, callbacks=get_keras_callbacks())
    """
    model = Sequential([
        # Layer 1: Wider for more capacity
        Dense(256, input_dim=input_dim),  # NO kernel_regularizer (AdamW handles it)
        BatchNormalization(),  # Faster convergence
        Activation('swish'),  # Smooth gradients, better for outliers
        Dropout(0.3),

        # Layer 2
        Dense(128),
        BatchNormalization(),
        Activation('swish'),
        Dropout(0.3),

        # Layer 3
        Dense(64),
        BatchNormalization(),
        Activation('swish'),
        Dropout(0.2),

        # Layer 4
        Dense(32),
        Activation('swish'),
        # No dropout on last hidden layer

        # Output: 3 quantiles [q50, q75, q90]
        Dense(3)  # No activation for regression
    ])

    # AdamW optimizer with weight decay (replaces L2 regularization)
    optimizer = AdamW(
        learning_rate=0.0005,  # Lower LR for fine-tuning
        weight_decay=0.01,  # Global regularization (replaces L2)
        beta_1=0.9,  # Default (momentum)
        beta_2=0.999,  # Default (RMSprop component)
        epsilon=1e-7,  # Default (numerical stability)
        clipnorm=1.0  # Gradient clipping for elite outliers
    )

    # Multi-quantile loss with emphasis on upper quantiles
    loss_fn = multi_quantile_loss(
        quantiles=[0.5, 0.75, 0.9],
        weights=[0.2, 0.3, 0.5]  # Emphasize upper quantiles for elites
    )

    model.compile(
        optimizer=optimizer,
        loss=loss_fn
    )

    return model


def get_keras_callbacks():
    """
    Get training callbacks for Keras models.

    Returns:
        list: [EarlyStopping, ReduceLROnPlateau] callbacks

    Example:
        >>> callbacks = get_keras_callbacks()
        >>> model.fit(X_train, y_train, epochs=200, callbacks=callbacks)
    """
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=25,  # More patience for LR schedule
        min_delta=1e-4,  # Minimum improvement threshold
        restore_best_weights=True,
        verbose=1
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,  # Halve LR when plateauing
        patience=10,  # Wait 10 epochs before reducing
        min_lr=1e-6,  # Don't go below this
        verbose=1
    )

    return [early_stop, reduce_lr]
