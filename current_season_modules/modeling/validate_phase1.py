"""
Phase 1 Validation Script - Three-Way Comparison.

Compares:
1. Original Method (RF+Keras, volume-based)
2. Phase 1 Method (RF+Keras, rate-based, three-path)
3. FanGraphs Actuals (ground truth)

Decision Criteria:
- Elite starters (WAR > 4.0): >30% error reduction required
- Relievers: Maintain accuracy within 10%
- Overall: >15% RMSE improvement required
- Full season: No >10% regression

Usage:
    python -m current_season_modules.modeling.validate_phase1 \\
        --test-year 2024 \\
        --output validation_results.txt

Author: oWAR Development Team
Date: October 2025
"""

# Standard library imports
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Third-party imports
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Local imports
from current_season_modules.modeling.pitcher_roles_ensemble_standalone import (
    ThreePathPitcherEnsemble,
    PITCHER_RATE_FEATURES,
    train_three_path_ensemble
)
from current_season_modules.modeling.ensemble_modeling import (
    EnsembleWARPredictor,
    create_ensemble_for_data
)
from current_season_modules.modeling import prepare_data_for_kfold
from common_modules.logging import get_logger

logger = get_logger(__name__)


# =============================================================================
# Validation Data Structures
# =============================================================================

class PitcherTestCase:
    """Container for pitcher test case data."""

    def __init__(
        self,
        name: str,
        features: np.ndarray,
        GS: int,
        G: int,
        IP: float,
        actual_war: float,
        season: int,
        role: str
    ):
        """
        Initialize pitcher test case.

        Args:
            name: Player name
            features: Feature vector
            GS: Games started
            G: Games pitched
            IP: Innings pitched
            actual_war: Actual WAR (ground truth)
            season: Season year
            role: Role classification
        """
        self.name = name
        self.features = features
        self.GS = GS
        self.G = G
        self.IP = IP
        self.actual_war = actual_war
        self.season = season
        self.role = role


class ValidationResults:
    """Container for validation results."""

    def __init__(self):
        """Initialize validation results storage."""
        self.detailed_results: List[Dict] = []
        self.segment_metrics: Dict = {}
        self.decision: str = ""
        self.summary: str = ""


# =============================================================================
# Metrics Calculation Functions
# =============================================================================

def calculate_metrics(
    predictions: np.ndarray,
    actuals: np.ndarray
) -> Dict[str, float]:
    """
    Calculate regression metrics.

    Args:
        predictions: Predicted values
        actuals: Actual values

    Returns:
        Dictionary with RMSE, MAE, R², and count
    """
    # Filter out NaN values
    valid_mask = ~(np.isnan(predictions) | np.isnan(actuals))
    preds_clean = predictions[valid_mask]
    actuals_clean = actuals[valid_mask]

    if len(actuals_clean) == 0:
        return {
            'rmse': np.nan,
            'mae': np.nan,
            'r2': np.nan,
            'count': 0
        }

    rmse = np.sqrt(mean_squared_error(actuals_clean, preds_clean))
    mae = mean_absolute_error(actuals_clean, preds_clean)

    # R² can fail if variance is zero
    try:
        r2 = r2_score(actuals_clean, preds_clean)
    except (ValueError, ZeroDivisionError):
        r2 = np.nan

    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'count': len(actuals_clean)
    }


def calculate_improvement(
    original_metric: float,
    phase1_metric: float
) -> float:
    """
    Calculate percentage improvement (negative means worse).

    Args:
        original_metric: Baseline metric (higher is worse)
        phase1_metric: New metric (higher is worse)

    Returns:
        Percentage improvement (positive = better)

    Example:
        >>> calculate_improvement(2.5, 1.5)  # 40% improvement
        40.0
    """
    if original_metric == 0:
        return 0.0

    improvement = ((original_metric - phase1_metric) / original_metric) * 100
    return improvement


# =============================================================================
# Data Loading Functions
# =============================================================================

def load_pitcher_data(
    data_dir: Path,
    seasons: List[int],
    metric: str = 'WAR'
) -> pd.DataFrame:
    """
    Load pitcher data for specified seasons.

    Args:
        data_dir: Path to MLB Player Data directory
        seasons: List of season years to load
        metric: 'WAR' or 'WARP'

    Returns:
        DataFrame with pitcher statistics

    Raises:
        FileNotFoundError: If data files not found
    """
    dfs = []

    for season in seasons:
        # Try multiple possible file paths
        possible_paths = [
            data_dir / f"FanGraphs_Data/pitching/fangraphs_pitching_{metric.lower()}_{season}.csv",
            data_dir / f"pitching_{metric.lower()}_{season}.csv",
            data_dir / f"pitcher_{metric.lower()}_{season}.csv"
        ]

        data_path = None
        for path in possible_paths:
            if path.exists():
                data_path = path
                break

        if data_path is None:
            logger.warning(f"Could not find data for season {season}")
            continue

        try:
            df = pd.read_csv(data_path)
            df['Season'] = season
            dfs.append(df)
            logger.info(f"Loaded {len(df)} pitchers from {season}")
        except Exception as e:
            logger.error(f"Error loading {data_path}: {e}")
            continue

    if not dfs:
        raise FileNotFoundError(
            f"No pitcher data found in {data_dir} for seasons {seasons}"
        )

    combined_df = pd.concat(dfs, ignore_index=True)
    logger.info(f"Total pitchers loaded: {len(combined_df)}")

    return combined_df


def load_fangraphs_actuals(
    data_dir: Path,
    season: int,
    metric: str = 'WAR'
) -> Dict[str, float]:
    """
    Load FanGraphs actual WAR/WARP values for validation.

    Args:
        data_dir: Path to MLB Player Data directory
        season: Season year
        metric: 'WAR' or 'WARP'

    Returns:
        Dictionary mapping player names to actual values
    """
    df = load_pitcher_data(data_dir, [season], metric)

    # Create name -> WAR mapping
    actuals = {}
    for _, row in df.iterrows():
        name = row.get('Name', row.get('PlayerName', ''))
        war_value = row.get(metric, np.nan)

        if name and not pd.isna(war_value):
            actuals[name] = war_value

    logger.info(f"Loaded {len(actuals)} FanGraphs {metric} values for {season}")
    return actuals


# =============================================================================
# Model Training and Prediction Functions
# =============================================================================

def train_original_ensemble(
    pitcher_df: pd.DataFrame,
    holdout_year: int
) -> EnsembleWARPredictor:
    """
    Train original ensemble (volume-based).

    Args:
        pitcher_df: Pitcher data DataFrame
        holdout_year: Year to hold out for testing

    Returns:
        Trained EnsembleWARPredictor
    """
    logger.info("Training ORIGINAL ensemble (volume-based)...")

    ensemble = EnsembleWARPredictor()

    for metric_type in ['war', 'warp']:
        # Prepare data (volume-based - includes IP in features)
        target_col = metric_type.upper()

        # Original features include IP
        original_features = ['IP'] + PITCHER_RATE_FEATURES

        # Filter valid data
        valid_mask = (
            (pitcher_df['IP'] > 0) &
            (~pitcher_df[target_col].isna()) &
            (pitcher_df['Season'] != holdout_year)
        )

        train_df = pitcher_df[valid_mask].copy()

        X_train = train_df[original_features].values
        y_train = train_df[target_col].values  # Volume-based target
        groups_train = train_df['Season'].values

        # Train ensemble
        ensemble.train_ensemble(
            X_train, y_train, groups_train,
            metric_type, 'pitcher',
            holdout_validation=False
        )

    ensemble.is_trained = True
    logger.info("Original ensemble training complete")

    return ensemble


def train_phase1_ensemble(
    pitcher_df: pd.DataFrame,
    holdout_year: int
) -> ThreePathPitcherEnsemble:
    """
    Train Phase 1 ensemble (rate-based, three-path).

    Args:
        pitcher_df: Pitcher data DataFrame
        holdout_year: Year to hold out for testing

    Returns:
        Trained ThreePathPitcherEnsemble
    """
    logger.info("Training PHASE 1 ensemble (rate-based, three-path)...")

    # Prepare data for both metrics
    pitcher_data_dict = {}

    for metric_type in ['war', 'warp']:
        metric_df = pitcher_df[pitcher_df[metric_type.upper()].notna()].copy()
        pitcher_data_dict[metric_type] = metric_df

    # Train using standalone function
    from current_season_modules.modeling.pitcher_roles_ensemble_standalone import (
        train_three_path_ensemble
    )

    ensemble = train_three_path_ensemble(pitcher_data_dict, holdout_year)

    logger.info("Phase 1 ensemble training complete")
    return ensemble


# =============================================================================
# Comparison and Validation Functions
# =============================================================================

def compare_predictions(
    test_cases: List[PitcherTestCase],
    original_ensemble: EnsembleWARPredictor,
    phase1_ensemble: ThreePathPitcherEnsemble,
    metric_type: str = 'war'
) -> List[Dict]:
    """
    Generate predictions from both models and compare.

    Args:
        test_cases: List of pitcher test cases
        original_ensemble: Original volume-based ensemble
        phase1_ensemble: Phase 1 rate-based ensemble
        metric_type: 'war' or 'warp'

    Returns:
        List of result dictionaries for each test case
    """
    results = []

    for case in test_cases:
        try:
            # Original prediction (volume-based)
            orig_result = original_ensemble.predict_ensemble(
                case.features,
                metric_type,
                'pitcher'
            )
            orig_pred = orig_result['ensemble']

            # Phase 1 prediction (rate-based)
            phase1_result = phase1_ensemble.predict(
                features=case.features[1:],  # Remove IP feature
                GS=case.GS,
                G=case.G,
                IP=case.IP,
                metric_type=metric_type
            )
            phase1_pred = phase1_result['current_war']

            # Calculate errors
            orig_error = abs(orig_pred - case.actual_war)
            phase1_error = abs(phase1_pred - case.actual_war)
            improvement = orig_error - phase1_error
            pct_improvement = (improvement / orig_error * 100) if orig_error > 0 else 0

            results.append({
                'name': case.name,
                'role': case.role,
                'season': case.season,
                'actual': case.actual_war,
                'original_pred': orig_pred,
                'phase1_pred': phase1_pred,
                'original_error': orig_error,
                'phase1_error': phase1_error,
                'improvement': improvement,
                'pct_improvement': pct_improvement,
                'GS': case.GS,
                'G': case.G,
                'IP': case.IP,
                'role_ratio': case.GS / case.G if case.G > 0 else 0
            })

        except Exception as e:
            logger.error(f"Error predicting for {case.name}: {e}")
            continue

    return results


def analyze_segment(
    results: List[Dict],
    segment_name: str,
    filter_func: callable
) -> Dict:
    """
    Analyze results for a specific segment.

    Args:
        results: List of result dictionaries
        segment_name: Name of segment
        filter_func: Function to filter results for this segment

    Returns:
        Dictionary with segment analysis
    """
    segment_results = [r for r in results if filter_func(r)]

    if not segment_results:
        return {
            'name': segment_name,
            'count': 0,
            'metrics': {},
            'status': 'NO_DATA'
        }

    # Extract arrays
    actuals = np.array([r['actual'] for r in segment_results])
    orig_preds = np.array([r['original_pred'] for r in segment_results])
    phase1_preds = np.array([r['phase1_pred'] for r in segment_results])

    # Calculate metrics
    orig_metrics = calculate_metrics(orig_preds, actuals)
    phase1_metrics = calculate_metrics(phase1_preds, actuals)

    # Calculate improvements
    rmse_improvement = calculate_improvement(
        orig_metrics['rmse'],
        phase1_metrics['rmse']
    )

    mae_improvement = calculate_improvement(
        orig_metrics['mae'],
        phase1_metrics['mae']
    )

    return {
        'name': segment_name,
        'count': len(segment_results),
        'original_metrics': orig_metrics,
        'phase1_metrics': phase1_metrics,
        'rmse_improvement_pct': rmse_improvement,
        'mae_improvement_pct': mae_improvement,
        'status': 'ANALYZED'
    }


def apply_decision_matrix(segment_analyses: Dict) -> str:
    """
    Apply decision matrix based on threshold criteria.

    Args:
        segment_analyses: Dictionary with segment analysis results

    Returns:
        Decision recommendation string
    """
    # Extract improvement percentages
    elite_improvement = segment_analyses['elite_starters']['rmse_improvement_pct']
    reliever_change = segment_analyses['relievers']['rmse_improvement_pct']
    overall_improvement = segment_analyses['all_pitchers']['rmse_improvement_pct']

    # Apply thresholds
    elite_pass = elite_improvement > 30
    elite_partial = 10 <= elite_improvement <= 30
    reliever_maintained = abs(reliever_change) >= -10  # Allow up to 10% worse
    overall_pass = overall_improvement > 15

    # Decision logic
    if elite_pass and reliever_maintained and overall_pass:
        return "✅ INTEGRATE Phase 1 - All thresholds met"
    elif elite_partial and reliever_maintained and overall_pass:
        return "⚠️  Try Phase 2 (tree ensemble) - Partial improvement"
    elif not reliever_maintained:
        return "❌ DO NOT INTEGRATE - Reliever accuracy regressed"
    elif not overall_pass:
        return "❌ DO NOT INTEGRATE - Overall performance insufficient"
    elif not elite_pass and not elite_partial:
        return "❌ ABANDON rate-based approach - No meaningful improvement"
    else:
        return "⚠️  MANUAL REVIEW REQUIRED - Mixed results"


# =============================================================================
# Output Formatting Functions
# =============================================================================

def format_detailed_results(results: List[Dict], top_n: int = 20) -> str:
    """
    Format detailed results for individual pitchers.

    Args:
        results: List of result dictionaries
        top_n: Number of top cases to show

    Returns:
        Formatted string
    """
    # Sort by improvement (best first)
    sorted_results = sorted(
        results,
        key=lambda x: x['improvement'],
        reverse=True
    )

    output = ["=" * 80]
    output.append("DETAILED RESULTS - Top Improvements")
    output.append("=" * 80)
    output.append("")

    for i, result in enumerate(sorted_results[:top_n], 1):
        output.append(f"{i}. {result['name']} ({result['season']})")
        output.append(f"   Role: {result['role']} (GS/G={result['role_ratio']:.2f})")
        output.append(f"   Actual WAR: {result['actual']:.2f}")
        output.append(f"   Original:   {result['original_pred']:.2f} "
                      f"(error: {result['original_error']:.2f})")
        output.append(f"   Phase 1:    {result['phase1_pred']:.2f} "
                      f"(error: {result['phase1_error']:.2f})")
        output.append(f"   Improvement: {result['improvement']:+.2f} WAR "
                      f"({result['pct_improvement']:+.1f}%)")
        output.append("")

    return "\n".join(output)


def format_segment_analysis(segment_analyses: Dict) -> str:
    """
    Format segment analysis results.

    Args:
        segment_analyses: Dictionary with segment analyses

    Returns:
        Formatted string
    """
    output = ["=" * 80]
    output.append("SEGMENT ANALYSIS")
    output.append("=" * 80)
    output.append("")

    for segment_name, analysis in segment_analyses.items():
        if analysis['count'] == 0:
            continue

        output.append(f"{segment_name.upper().replace('_', ' ')} (N={analysis['count']})")
        output.append("-" * 40)

        orig = analysis['original_metrics']
        phase1 = analysis['phase1_metrics']

        output.append(f"  Original RMSE:  {orig['rmse']:.3f}")
        output.append(f"  Phase 1 RMSE:   {phase1['rmse']:.3f}")
        output.append(f"  Improvement:    {analysis['rmse_improvement_pct']:+.1f}%")
        output.append("")

        output.append(f"  Original MAE:   {orig['mae']:.3f}")
        output.append(f"  Phase 1 MAE:    {phase1['mae']:.3f}")
        output.append(f"  Improvement:    {analysis['mae_improvement_pct']:+.1f}%")
        output.append("")

        output.append(f"  Original R²:    {orig['r2']:.3f}")
        output.append(f"  Phase 1 R²:     {phase1['r2']:.3f}")
        output.append("")

    return "\n".join(output)


# =============================================================================
# Main Validation Function
# =============================================================================

def validate_phase1(
    data_dir: Path,
    test_year: int,
    training_years: List[int],
    output_path: Optional[Path] = None
) -> ValidationResults:
    """
    Run full Phase 1 validation with three-way comparison.

    Args:
        data_dir: Path to MLB Player Data directory
        test_year: Year to test on (held out from training)
        training_years: Years to train on
        output_path: Optional path to save results

    Returns:
        ValidationResults object
    """
    logger.info("=" * 80)
    logger.info("PHASE 1 VALIDATION - Three-Way Comparison")
    logger.info("=" * 80)
    logger.info(f"Test year: {test_year}")
    logger.info(f"Training years: {min(training_years)}-{max(training_years)}")
    logger.info("")

    # Load data
    all_years = training_years + [test_year]
    pitcher_df = load_pitcher_data(data_dir, all_years, 'WAR')

    # Train both ensembles
    original_ensemble = train_original_ensemble(pitcher_df, test_year)
    phase1_ensemble = train_phase1_ensemble(pitcher_df, test_year)

    # Create test cases from test year
    test_df = pitcher_df[pitcher_df['Season'] == test_year].copy()

    test_cases = []
    for _, row in test_df.iterrows():
        try:
            # Original features (with IP)
            orig_features = np.array([
                row['IP'],
                row['BB%'],
                row['K%'],
                row.get('K-BB%', row['K%'] - row['BB%']),
                row['ERA'],
                row['damage_control_ratio'],
                row['Opportunity_Success'],
                row['Contact_Quality_Index'],
                row['HBP%'],
                row.get('WP', 5.0),
                row['Statcast_Launch_Quality_Index']
            ])

            # Classify role
            role_ratio = row['GS'] / row['G'] if row['G'] > 0 else 0
            if role_ratio < 0.1:
                role = 'reliever'
            elif role_ratio > 0.7:
                role = 'starter'
            else:
                role = 'mixed'

            test_case = PitcherTestCase(
                name=row.get('Name', 'Unknown'),
                features=orig_features,
                GS=row['GS'],
                G=row['G'],
                IP=row['IP'],
                actual_war=row['WAR'],
                season=test_year,
                role=role
            )
            test_cases.append(test_case)

        except Exception as e:
            logger.error(f"Error creating test case: {e}")
            continue

    logger.info(f"Created {len(test_cases)} test cases")

    # Run comparisons
    results = compare_predictions(
        test_cases,
        original_ensemble,
        phase1_ensemble,
        'war'
    )

    # Analyze segments
    segment_analyses = {
        'elite_starters': analyze_segment(
            results,
            'Elite Starters (WAR > 4.0)',
            lambda r: r['actual'] > 4.0 and r['role'] == 'starter'
        ),
        'relievers': analyze_segment(
            results,
            'Pure Relievers (GS/G < 0.1)',
            lambda r: r['role'] == 'reliever'
        ),
        'all_pitchers': analyze_segment(
            results,
            'All Pitchers',
            lambda r: True
        )
    }

    # Apply decision matrix
    decision = apply_decision_matrix(segment_analyses)

    # Format output
    output_lines = []
    output_lines.append(format_segment_analysis(segment_analyses))
    output_lines.append("")
    output_lines.append("=" * 80)
    output_lines.append("DECISION")
    output_lines.append("=" * 80)
    output_lines.append(decision)
    output_lines.append("=" * 80)
    output_lines.append("")
    output_lines.append(format_detailed_results(results))

    output_text = "\n".join(output_lines)

    # Print to console
    print(output_text)

    # Save to file if requested
    if output_path:
        with open(output_path, 'w') as f:
            f.write(output_text)
        logger.info(f"Results saved to {output_path}")

    # Create results object
    validation_results = ValidationResults()
    validation_results.detailed_results = results
    validation_results.segment_metrics = segment_analyses
    validation_results.decision = decision
    validation_results.summary = output_text

    return validation_results


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    """CLI entry point for validation script."""
    parser = argparse.ArgumentParser(
        description="Validate Phase 1 pitcher ensemble against original method"
    )

    parser.add_argument(
        '--data-dir',
        type=Path,
        default=Path('MLB Player Data'),
        help='Path to MLB Player Data directory'
    )

    parser.add_argument(
        '--test-year',
        type=int,
        default=2024,
        help='Year to hold out for testing'
    )

    parser.add_argument(
        '--train-start',
        type=int,
        default=2016,
        help='First year for training data'
    )

    parser.add_argument(
        '--train-end',
        type=int,
        default=2023,
        help='Last year for training data'
    )

    parser.add_argument(
        '--output',
        type=Path,
        default=Path('phase1_validation_results.txt'),
        help='Output file path'
    )

    args = parser.parse_args()

    # Create training years list
    training_years = list(range(args.train_start, args.train_end + 1))

    # Run validation
    try:
        results = validate_phase1(
            data_dir=args.data_dir,
            test_year=args.test_year,
            training_years=training_years,
            output_path=args.output
        )

        # Return exit code based on decision
        if "INTEGRATE" in results.decision:
            sys.exit(0)  # Success
        else:
            sys.exit(1)  # Failure

    except Exception as e:
        logger.error(f"Validation failed: {e}", exc_info=True)
        sys.exit(2)


if __name__ == "__main__":
    main()
