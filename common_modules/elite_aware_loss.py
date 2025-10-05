"""
Custom Keras loss function for Phase 3: Elite-aware loss.

Penalizes elite pitcher underprediction more heavily than average errors,
forcing the neural network to prioritize accuracy for high-WAR pitchers.
"""

import tensorflow as tf


def create_elite_aware_loss(elite_threshold=3.5, elite_penalty=2.5):
    """
    Factory function to create elite-aware loss with custom parameters.

    Args:
        elite_threshold: WAR threshold for elite classification (default: 3.5)
        elite_penalty: Multiplier for elite errors (default: 2.5x)

    Returns:
        Loss function compatible with Keras model.compile()

    Example:
        >>> loss_fn = create_elite_aware_loss(elite_threshold=3.5, elite_penalty=2.5)
        >>> model.compile(optimizer='adam', loss=loss_fn, metrics=['mae'])
    """
    def elite_aware_loss(y_true, y_pred):
        """
        Elite-aware MSE loss function.

        Standard MSE for all samples, with extra penalty for elite underprediction.

        For y_true < elite_threshold: loss = (y_true - y_pred)²
        For y_true ≥ elite_threshold: loss = (y_true - y_pred)² * elite_penalty

        Args:
            y_true: Actual WAR values (tensor)
            y_pred: Predicted WAR values (tensor)

        Returns:
            Weighted MSE loss (scalar tensor)
        """
        error = y_true - y_pred

        # Standard MSE for all samples
        base_loss = tf.square(error)

        # Extra penalty for underpredicting elites
        elite_mask = tf.cast(y_true >= elite_threshold, tf.float32)
        elite_extra_penalty = elite_mask * tf.square(error) * (elite_penalty - 1.0)

        # Combined loss
        total_loss = base_loss + elite_extra_penalty

        return tf.reduce_mean(total_loss)

    # Set function name for better error messages
    elite_aware_loss.__name__ = f'elite_aware_loss_t{elite_threshold}_p{elite_penalty}'

    return elite_aware_loss


# Default loss function (can be imported directly)
elite_aware_loss = create_elite_aware_loss(elite_threshold=3.5, elite_penalty=2.5)
