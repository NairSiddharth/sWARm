"""
Animated Analysis Module for oWAR (Windows Compatible)

This module contains functions for creating sophisticated animated visualizations
showing model prediction evolution over time.
"""

def create_animated_model_comparison(model_results=None):
    """
    Create sophisticated animated visualizations showing model prediction evolution over time.
    This provides temporal analysis of how different models perform across the data period,
    with animations progressing chronologically from data start to end.

    Args:
        model_results: ModelResults object containing prediction data. If None, will try to get from globals.

    Returns:
        animation_results: Dictionary containing animation metadata and results
    """
    from modules.modeling import select_best_models_by_category
    from modules.data_visualization import plot_war_warp_animated
    import inspect

    print("CREATING ANIMATED TEMPORAL MODEL COMPARISON")
    print("="*60)

    # Initialize animation_results to avoid UnboundLocalError
    animation_results = None

    # Get model_results from parameter or globals
    if model_results is None:
        # Get the calling frame's globals to access notebook variables
        frame = inspect.currentframe()
        try:
            # Go up the call stack to find model_results
            caller_globals = frame.f_back.f_globals
            if 'model_results' in caller_globals:
                model_results = caller_globals['model_results']
            else:
                print("WARNING: No model results available. Please run model training first.")
                return animation_results
        finally:
            del frame

    # Verify model results are available and valid
    if not hasattr(model_results, 'results') or len(model_results.results) == 0:
        print("WARNING: No model results available. Please run model training first.")
        return animation_results

    print("SUCCESS: Model results found - proceeding with animated analysis...")

    # Get the best performing models for comparison
    try:
        best_models = select_best_models_by_category(model_results)
        print(f"SELECTED MODELS: {[m.upper() for m in best_models]}")
    except Exception as e:
        print(f"WARNING: Error selecting best models: {e}")
        # Fallback to available models if auto-selection fails
        available_models = list(set([key.split('_')[0] for key in model_results.results.keys()]))
        best_models = available_models[:4]  # Limit to prevent overcrowding
        print(f"USING AVAILABLE MODELS: {[m.upper() for m in best_models]}")

    # Create animated temporal analysis with enhanced aesthetics
    print("\nGenerating animated visualizations with chronological progression...")
    print("- Cinematic bubble animation showing prediction accuracy evolution")
    print("- Performance heatmap tracking model improvement over time")
    print("- 3D temporal surface revealing prediction patterns")
    print("- All animations progress chronologically from data start to end")

    # Execute the animated visualization function with error handling
    try:
        animation_results = plot_war_warp_animated(
            model_results=model_results,
            season_col="Season",  # Ensure chronological ordering by season
            model_names=best_models,
            show_hitters=True,   # Include hitter predictions
            show_pitchers=True   # Include pitcher predictions
        )
    except Exception as e:
        print(f"ERROR: Animation creation failed: {e}")
        animation_results = None
        return animation_results

    # Display additional temporal insights
    if animation_results:
        print(f"\nTEMPORAL ANALYSIS INSIGHTS:")
        print(f"- Total observations: {animation_results.get('total_observations', 'N/A')}")
        if animation_results.get('temporal_range'):
            print(f"- Time period: {animation_results['temporal_range'][0]} to {animation_results['temporal_range'][-1]}")
        print(f"- Visual features: {', '.join(animation_results.get('aesthetic_features', ['N/A']))}")

        print(f"\nCHRONOLOGICAL PROGRESSION:")
        print(f"- Animation frames advance in temporal order")
        print(f"- Each frame represents a season/year in your dataset")
        print(f"- Smooth transitions show prediction evolution over time")
        print(f"- Interactive controls allow manual navigation")

        print(f"\nCOMPARATIVE ANALYSIS FEATURES:")
        print(f"- Side-by-side model performance visualization")
        print(f"- Dynamic accuracy zones showing prediction quality")
        print(f"- Color-coded error gradients for immediate insight")
        print(f"- Player-level hover details for granular analysis")
    else:
        print("\nWARNING: Animation function returned None - check for data issues")

    print(f"\nANIMATED TEMPORAL COMPARISON COMPLETE!")
    print(f"- Animation created successfully" if animation_results else "- Animation creation failed")
    if animation_results and animation_results.get('temporal_range'):
        print(f"- Chronological progression from {animation_results['temporal_range'][0]} to {animation_results['temporal_range'][-1]}")
    print(f"- Interactive features enabled for detailed exploration" if animation_results else "- Please check data and try again")

    return animation_results


def test_module_functionality():
    """
    Test the animated analysis module with minimal data.
    """
    from modules.modeling import ModelResults

    print("TESTING ANIMATED ANALYSIS MODULE")
    print("="*50)

    # Create minimal test data
    model_results = ModelResults()

    # Add complete test data for one model
    model_results.store_results(
        'test_model', 'hitter', 'war',
        y_true=[2.5, 3.1, 1.8],
        y_pred=[2.3, 2.9, 1.9],
        player_names=['Player A', 'Player B', 'Player C'],
        seasons=['2021', '2021', '2022']
    )
    model_results.store_results(
        'test_model', 'hitter', 'warp',
        y_true=[2.4, 3.0, 1.7],
        y_pred=[2.2, 2.8, 1.8],
        player_names=['Player A', 'Player B', 'Player C'],
        seasons=['2021', '2021', '2022']
    )
    model_results.store_results(
        'test_model', 'pitcher', 'war',
        y_true=[1.5, 2.1, 0.8],
        y_pred=[1.3, 1.9, 0.9],
        player_names=['Pitcher A', 'Pitcher B', 'Pitcher C'],
        seasons=['2021', '2021', '2022']
    )
    model_results.store_results(
        'test_model', 'pitcher', 'warp',
        y_true=[1.4, 2.0, 0.7],
        y_pred=[1.2, 1.8, 0.8],
        player_names=['Pitcher A', 'Pitcher B', 'Pitcher C'],
        seasons=['2021', '2021', '2022']
    )

    print("SUCCESS: Created test model results")
    print(f"Available keys: {list(model_results.results.keys())}")

    # Test the function with explicit model_results parameter
    try:
        result = create_animated_model_comparison(model_results=model_results)
        print("SUCCESS: Function executed without scope errors")
        print(f"Result type: {type(result)}")
        return True
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Run test when module is executed directly
    test_module_functionality()