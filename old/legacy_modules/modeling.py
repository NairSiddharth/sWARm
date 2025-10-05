"""
Legacy Modeling Module
=====================

Provides modeling utilities for old/legacy analysis scripts.
Contains functions and classes referenced by files in the old/ directory.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional


class ModelResults:
    """
    Container class for storing model results and metadata.
    Used by legacy analysis scripts in the old/ directory.
    """

    def __init__(self, model_type: str = "unknown"):
        self.model_type = model_type
        self.results = {}
        self.metrics = {}
        self.metadata = {}

    def add_result(self, model_name: str, predictions: List[float],
                   actuals: List[float], features: List[str] = None):
        """Add model results."""
        self.results[model_name] = {
            'predictions': predictions,
            'actuals': actuals,
            'features': features or []
        }

        # Calculate basic metrics
        if len(predictions) == len(actuals) and len(predictions) > 0:
            predictions = np.array(predictions)
            actuals = np.array(actuals)

            mse = np.mean((predictions - actuals) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(predictions - actuals))

            # R-squared
            ss_res = np.sum((actuals - predictions) ** 2)
            ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

            self.metrics[model_name] = {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'n_samples': len(predictions)
            }

    def get_best_model(self, metric: str = 'r2') -> Optional[str]:
        """Get the best model based on specified metric."""
        if not self.metrics:
            return None

        if metric == 'r2':
            return max(self.metrics.keys(), key=lambda k: self.metrics[k]['r2'])
        elif metric in ['mse', 'rmse', 'mae']:
            return min(self.metrics.keys(), key=lambda k: self.metrics[k][metric])
        else:
            return list(self.metrics.keys())[0]  # Default to first model

    def get_model_summary(self, model_name: str) -> Dict[str, Any]:
        """Get summary information for a specific model."""
        if model_name not in self.results:
            return {}

        summary = {
            'model_name': model_name,
            'model_type': self.model_type,
            'metrics': self.metrics.get(model_name, {}),
            'n_features': len(self.results[model_name].get('features', [])),
            'features': self.results[model_name].get('features', [])
        }

        return summary


def select_best_models_by_category(results: Dict[str, Any],
                                 categories: List[str] = None,
                                 metric: str = 'r2') -> Dict[str, str]:
    """
    Select the best models by category based on performance metric.

    Args:
        results: Dictionary of model results or ModelResults objects
        categories: List of categories to consider (e.g., ['hitter', 'pitcher'])
        metric: Metric to use for selection ('r2', 'mse', 'rmse', 'mae')

    Returns:
        Dictionary mapping category to best model name
    """
    best_models = {}

    # Default categories if none provided
    if categories is None:
        categories = ['hitter', 'pitcher', 'overall']

    for category in categories:
        category_models = {}

        # Extract models for this category
        for model_name, result in results.items():
            if category.lower() in model_name.lower() or category == 'overall':
                if isinstance(result, ModelResults):
                    best_model = result.get_best_model(metric)
                    if best_model and result.metrics.get(best_model):
                        category_models[model_name] = result.metrics[best_model][metric]
                elif isinstance(result, dict) and 'metrics' in result:
                    if metric in result['metrics']:
                        category_models[model_name] = result['metrics'][metric]
                elif isinstance(result, dict) and metric in result:
                    category_models[model_name] = result[metric]

        # Select best model for this category
        if category_models:
            if metric == 'r2':  # Higher is better
                best_models[category] = max(category_models.keys(),
                                          key=lambda k: category_models[k])
            else:  # Lower is better (mse, rmse, mae)
                best_models[category] = min(category_models.keys(),
                                          key=lambda k: category_models[k])
        else:
            # Fallback: use first available model
            available_models = list(results.keys())
            if available_models:
                best_models[category] = available_models[0]

    return best_models


def create_model_comparison_summary(results: Dict[str, Any]) -> pd.DataFrame:
    """
    Create a DataFrame summarizing model performance for comparison.

    Args:
        results: Dictionary of model results

    Returns:
        DataFrame with model comparison metrics
    """
    summary_data = []

    for model_name, result in results.items():
        if isinstance(result, ModelResults):
            best_model = result.get_best_model('r2')
            if best_model and result.metrics.get(best_model):
                metrics = result.metrics[best_model]
                summary_data.append({
                    'Model': model_name,
                    'Best_Submodel': best_model,
                    'R2': metrics.get('r2', 0),
                    'RMSE': metrics.get('rmse', float('inf')),
                    'MAE': metrics.get('mae', float('inf')),
                    'N_Samples': metrics.get('n_samples', 0),
                    'N_Features': len(result.results[best_model].get('features', []))
                })
        elif isinstance(result, dict) and 'metrics' in result:
            metrics = result['metrics']
            summary_data.append({
                'Model': model_name,
                'Best_Submodel': model_name,
                'R2': metrics.get('r2', 0),
                'RMSE': metrics.get('rmse', float('inf')),
                'MAE': metrics.get('mae', float('inf')),
                'N_Samples': metrics.get('n_samples', 0),
                'N_Features': metrics.get('n_features', 0)
            })

    if summary_data:
        df = pd.DataFrame(summary_data)
        df = df.sort_values('R2', ascending=False)
        return df
    else:
        return pd.DataFrame(columns=['Model', 'Best_Submodel', 'R2', 'RMSE', 'MAE', 'N_Samples', 'N_Features'])


def evaluate_model_stability(results: Dict[str, Any], threshold: float = 0.1) -> Dict[str, bool]:
    """
    Evaluate model stability based on performance variance.

    Args:
        results: Dictionary of model results
        threshold: Maximum acceptable coefficient of variation

    Returns:
        Dictionary mapping model names to stability status (True = stable)
    """
    stability = {}

    for model_name, result in results.items():
        if isinstance(result, ModelResults) and len(result.metrics) > 1:
            # Calculate coefficient of variation for R2 across submodels
            r2_values = [metrics.get('r2', 0) for metrics in result.metrics.values()]
            if len(r2_values) > 1:
                mean_r2 = np.mean(r2_values)
                std_r2 = np.std(r2_values)
                cv = std_r2 / mean_r2 if mean_r2 != 0 else float('inf')
                stability[model_name] = cv <= threshold
            else:
                stability[model_name] = True  # Single model is considered stable
        else:
            stability[model_name] = True  # Default to stable for simple results

    return stability


# Legacy compatibility functions
def get_model_performance_summary(results, metric='r2'):
    """Legacy function for backward compatibility."""
    return create_model_comparison_summary(results)

def select_top_models(results, n=5, metric='r2'):
    """Legacy function to select top N models."""
    summary_df = create_model_comparison_summary(results)
    if len(summary_df) == 0:
        return []

    if metric == 'r2':
        top_models = summary_df.nlargest(n, 'R2')
    else:
        # For error metrics, smaller is better
        metric_col = metric.upper()
        if metric_col in summary_df.columns:
            top_models = summary_df.nsmallest(n, metric_col)
        else:
            top_models = summary_df.head(n)  # Fallback

    return top_models['Model'].tolist()