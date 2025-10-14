"""
Custom metrics for ROS model training and evaluation.

Provides torchmetrics-compatible metrics for tracking bias during training.
"""

import torch
from torchmetrics import Metric, MetricCollection, MeanAbsoluteError, R2Score


class MeanError(Metric):
    """
    Computes mean error (bias).

    Positive value = systematic overprediction
    Negative value = systematic underprediction
    Zero = unbiased (ideal)

    Formula: mean(y_pred - y_true)
    """

    def __init__(self):
        super().__init__()
        self.add_state("sum_errors", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """Update metric state with batch predictions."""
        errors = preds - target
        self.sum_errors += errors.sum()
        self.total += target.numel()

    def compute(self):
        """Compute final metric value."""
        return self.sum_errors / self.total


class EliteBias(Metric):
    """
    Computes bias specifically for elite players (WAR > threshold).

    Tracks whether model underpredicts elites.
    Negative value indicates elite underprediction (the problem we're trying to solve).
    """

    def __init__(self, elite_threshold: float = 4.0):
        super().__init__()
        self.elite_threshold = elite_threshold
        self.add_state("sum_elite_errors", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("elite_count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """Update metric for elite players only."""
        # Identify elite players in this batch
        elite_mask = target > self.elite_threshold

        if elite_mask.sum() > 0:
            elite_errors = preds[elite_mask] - target[elite_mask]
            self.sum_elite_errors += elite_errors.sum()
            self.elite_count += elite_mask.sum()

    def compute(self):
        """Compute elite bias (returns 0 if no elite players seen)."""
        if self.elite_count == 0:
            return torch.tensor(0.0)
        return self.sum_elite_errors / self.elite_count


class EliteMAE(Metric):
    """
    Mean Absolute Error specifically for elite players.

    Useful to track alongside elite bias to understand
    both accuracy and direction of errors for elite players.
    """

    def __init__(self, elite_threshold: float = 4.0):
        super().__init__()
        self.elite_threshold = elite_threshold
        self.add_state("sum_elite_abs_errors", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("elite_count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """Update metric for elite players only."""
        elite_mask = target > self.elite_threshold

        if elite_mask.sum() > 0:
            elite_abs_errors = torch.abs(preds[elite_mask] - target[elite_mask])
            self.sum_elite_abs_errors += elite_abs_errors.sum()
            self.elite_count += elite_mask.sum()

    def compute(self):
        """Compute elite MAE (returns 0 if no elite players seen)."""
        if self.elite_count == 0:
            return torch.tensor(0.0)
        return self.sum_elite_abs_errors / self.elite_count


def get_ros_metrics(elite_threshold: float = 4.0) -> MetricCollection:
    """
    Get standard ROS metric collection.

    Includes:
    - MAE: Overall mean absolute error
    - R2: Coefficient of determination
    - Bias: Overall mean error (overprediction/underprediction)
    - Elite_Bias: Bias specifically for elite players (WAR > threshold)
    - Elite_MAE: MAE specifically for elite players

    Args:
        elite_threshold: WAR threshold for elite players (default: 4.0)

    Returns:
        MetricCollection with all metrics

    Example:
        >>> metrics = get_ros_metrics(elite_threshold=4.0)
        >>> tcn = TCNModel(..., torch_metrics=metrics)
        >>> # Metrics tracked during training and validation
    """
    return MetricCollection({
        'MAE': MeanAbsoluteError(),
        'R2': R2Score(),
        'Bias': MeanError(),
        'Elite_Bias': EliteBias(elite_threshold),
        'Elite_MAE': EliteMAE(elite_threshold)
    })
