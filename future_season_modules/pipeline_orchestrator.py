"""
Pipeline Orchestrator Module for Future Season Projections

This module handles high-level workflow orchestration for SYSTEM 2: Future Performance Projections,
extracted from integration.py for better modularity and maintainability.

Original functionality preserved with no modifications.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
import warnings
import json
import pickle
import hashlib
from datetime import datetime
from pathlib import Path


class ModelHistoryManager:
    """
    Manages model versioning, history tracking, and metadata for project archaeology.
    """

    def __init__(self, base_dir: str = "models"):
        """
        Initialize model history manager.

        Args:
            base_dir: Base directory for model storage
        """
        self.base_dir = Path(base_dir)
        self.history_dir = self.base_dir / "history"
        self.war_dir = self.history_dir / "war"
        self.warp_dir = self.history_dir / "warp"
        self.versions_file = self.history_dir / "versions.json"

        # Create directory structure
        self.war_dir.mkdir(parents=True, exist_ok=True)
        self.warp_dir.mkdir(parents=True, exist_ok=True)

    def _calculate_model_hash(self, model_obj) -> str:
        """
        Calculate hash of model object for change detection.

        Args:
            model_obj: Model object to hash

        Returns:
            SHA256 hash of model
        """
        try:
            model_bytes = pickle.dumps(model_obj)
            return hashlib.sha256(model_bytes).hexdigest()[:16]
        except Exception:
            # Fallback to timestamp if serialization fails
            return datetime.now().strftime("%Y%m%d_%H%M%S")

    def _get_version_registry(self) -> Dict:
        """
        Load or create version registry.

        Returns:
            Version registry dictionary
        """
        if self.versions_file.exists():
            with open(self.versions_file, 'r') as f:
                return json.load(f)
        else:
            return {
                "war_models": {},
                "warp_models": {},
                "created": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat()
            }

    def _update_version_registry(self, registry: Dict):
        """
        Update version registry file.

        Args:
            registry: Updated registry dictionary
        """
        registry["last_updated"] = datetime.now().isoformat()
        with open(self.versions_file, 'w') as f:
            json.dump(registry, f, indent=2)

    def _determine_version(self, model_type: str, model_hash: str, year_range: str) -> int:
        """
        Determine next version number for a model.

        Args:
            model_type: 'war' or 'warp'
            model_hash: Hash of the model
            year_range: Year range string like '2016-2024'

        Returns:
            Version number (1 if new, existing if hash matches)
        """
        registry = self._get_version_registry()
        model_key = f"{model_type}_models"

        # Check if this exact model already exists
        for version_info in registry[model_key].values():
            if (version_info.get("model_hash") == model_hash and
                version_info.get("year_range") == year_range):
                return int(version_info["version"])

        # Find next version number
        if registry[model_key]:
            return max(int(v["version"]) for v in registry[model_key].values()) + 1
        else:
            return 1

    def save_model_with_history(self,
                               model_obj,
                               model_type: str,
                               training_metrics: Dict,
                               year_range: str,
                               metadata: Dict = None) -> str:
        """
        Save model with complete history tracking.

        Args:
            model_obj: Model object to save
            model_type: 'war' or 'warp'
            training_metrics: Performance metrics from training
            year_range: Year range string like '2016-2024'
            metadata: Additional metadata dictionary

        Returns:
            Path to saved model file
        """
        # Calculate model hash for change detection
        model_hash = self._calculate_model_hash(model_obj)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Determine version
        version = self._determine_version(model_type, model_hash, year_range)

        # Check if this exact model version already exists
        registry = self._get_version_registry()
        model_key = f"{model_type}_models"

        for existing_path, version_info in registry[model_key].items():
            if (version_info.get("model_hash") == model_hash and
                version_info.get("year_range") == year_range and
                version_info.get("version") == version):
                print(f"Model {model_type} v{version} already exists at {existing_path}")
                return existing_path

        # Create filename
        filename = f"{model_type}_model_v{version}_{year_range}_{timestamp}.pkl"
        metadata_filename = f"{model_type}_model_v{version}_{year_range}_{timestamp}_metadata.json"

        # Determine save directory
        save_dir = self.war_dir if model_type == "war" else self.warp_dir
        model_path = save_dir / filename
        metadata_path = save_dir / metadata_filename

        # Save model
        with open(model_path, 'wb') as f:
            pickle.dump(model_obj, f)

        # Prepare metadata
        model_metadata = {
            "model_type": model_type,
            "version": version,
            "year_range": year_range,
            "timestamp": timestamp,
            "created_at": datetime.now().isoformat(),
            "model_hash": model_hash,
            "training_metrics": training_metrics,
            "file_size_bytes": model_path.stat().st_size,
            "model_class": model_obj.__class__.__name__,
            "model_attributes": {
                "max_projection_years": getattr(model_obj, 'max_projection_years', None),
                "use_dynasty_guru": getattr(model_obj, 'use_dynasty_guru', None)
            }
        }

        # Add custom metadata if provided
        if metadata:
            model_metadata["custom_metadata"] = metadata

        # Save metadata
        with open(metadata_path, 'w') as f:
            json.dump(model_metadata, f, indent=2)

        # Update version registry
        registry[model_key][str(model_path)] = {
            "version": version,
            "year_range": year_range,
            "timestamp": timestamp,
            "model_hash": model_hash,
            "created_at": datetime.now().isoformat(),
            "metadata_file": str(metadata_path)
        }
        self._update_version_registry(registry)

        print(f"Model saved to history: {model_path}")
        print(f"Metadata saved to: {metadata_path}")

        return str(model_path)

    def list_model_versions(self, model_type: str = None) -> Dict:
        """
        List all model versions with metadata.

        Args:
            model_type: Filter by 'war' or 'warp', or None for all

        Returns:
            Dictionary of model versions
        """
        registry = self._get_version_registry()

        if model_type:
            return registry.get(f"{model_type}_models", {})
        else:
            return {
                "war_models": registry.get("war_models", {}),
                "warp_models": registry.get("warp_models", {})
            }

    def get_latest_model(self, model_type: str, year_range: str = None) -> Optional[str]:
        """
        Get path to latest model of specified type.

        Args:
            model_type: 'war' or 'warp'
            year_range: Optional year range filter

        Returns:
            Path to latest model file or None
        """
        models = self.list_model_versions(model_type)

        if not models:
            return None

        # Filter by year range if specified
        if year_range:
            filtered_models = {
                path: info for path, info in models.items()
                if info.get("year_range") == year_range
            }
        else:
            filtered_models = models

        if not filtered_models:
            return None

        # Return path with latest timestamp
        latest_entry = max(filtered_models.items(),
                          key=lambda x: x[1]["created_at"])
        return latest_entry[0]

    def compare_models(self, path1: str, path2: str) -> Dict:
        """
        Compare two model versions.

        Args:
            path1: Path to first model
            path2: Path to second model

        Returns:
            Comparison results dictionary
        """
        # Load metadata for both models
        metadata1_path = path1.replace('.pkl', '_metadata.json')
        metadata2_path = path2.replace('.pkl', '_metadata.json')

        comparison = {"error": "Could not load metadata for comparison"}

        try:
            with open(metadata1_path, 'r') as f:
                meta1 = json.load(f)
            with open(metadata2_path, 'r') as f:
                meta2 = json.load(f)

            comparison = {
                "model1": {
                    "path": path1,
                    "version": meta1.get("version"),
                    "timestamp": meta1.get("timestamp"),
                    "year_range": meta1.get("year_range"),
                    "model_hash": meta1.get("model_hash")
                },
                "model2": {
                    "path": path2,
                    "version": meta2.get("version"),
                    "timestamp": meta2.get("timestamp"),
                    "year_range": meta2.get("year_range"),
                    "model_hash": meta2.get("model_hash")
                },
                "same_hash": meta1.get("model_hash") == meta2.get("model_hash"),
                "metrics_comparison": {
                    "model1_metrics": meta1.get("training_metrics"),
                    "model2_metrics": meta2.get("training_metrics")
                }
            }
        except Exception as e:
            comparison["error"] = str(e)

        return comparison


class PipelineOrchestrator:
    """
    Handles high-level workflow orchestration for projection pipeline including:
    - Model training coordination
    - Model validation workflows
    - Batch projection generation
    - Complete pipeline execution
    """

    def __init__(self, system_pipeline=None):
        """
        Initialize pipeline orchestrator.

        Args:
            system_pipeline: Main System2Pipeline instance to coordinate
        """
        self.system_pipeline = system_pipeline
        self.model_history = ModelHistoryManager()

    def train_projection_model(self, data: pd.DataFrame) -> Dict[str, Union[float, Dict]]:
        """
        Train the complete confidence-aware joint longitudinal-survival model.

        Args:
            data: Training dataset (used for both training and confidence calculation)

        Returns:
            Dictionary containing training metrics for both models
        """
        print("Training separate confidence-aware WAR and WARP projection models...")

        # Store training data for confidence scoring and constraints
        self.system_pipeline.training_data = data

        # Separate WAR and WARP data for independent model training
        war_data = data[data['DataSource'] == 'WAR'].copy()
        warp_data = data[data['DataSource'] == 'WARP'].copy()

        print(f"  WAR training data: {len(war_data)} records")
        print(f"  WARP training data: {len(warp_data)} records")

        # Train WAR model with confidence features
        if len(war_data) > 100:
            war_data['TARGET_METRIC'] = war_data['WAR']

            # Clean data before training - remove records with missing critical fields
            pre_clean_count = len(war_data)
            war_data_clean = war_data[
                war_data['Age'].notna() &
                war_data['Position'].notna() &
                (war_data['Position'] != '') &
                war_data['TARGET_METRIC'].notna()
            ].copy()
            dropped_count = pre_clean_count - len(war_data_clean)
            if dropped_count > 0:
                print(f"  Dropped {dropped_count} WAR records with missing Age/Position/TARGET_METRIC")

            from .future_projections import FutureProjectionAgeCurve
            self.system_pipeline.war_model = FutureProjectionAgeCurve(
                max_projection_years=self.system_pipeline.max_projection_years,
                use_dynasty_guru=self.system_pipeline.use_dynasty_guru
            )
            war_metrics = self.system_pipeline.war_model.fit_joint_model(war_data_clean, training_data=data)
        else:
            self.system_pipeline.war_model = None
            war_metrics = {'error': 'insufficient_data'}

        # Train WARP model with confidence features
        if len(warp_data) > 100:
            warp_data['TARGET_METRIC'] = warp_data['WARP']

            # Clean data before training - remove records with missing critical fields
            pre_clean_count = len(warp_data)
            warp_data_clean = warp_data[
                warp_data['Age'].notna() &
                warp_data['Position'].notna() &
                (warp_data['Position'] != '') &
                warp_data['TARGET_METRIC'].notna()
            ].copy()
            dropped_count = pre_clean_count - len(warp_data_clean)
            if dropped_count > 0:
                print(f"  Dropped {dropped_count} WARP records with missing Age/Position/TARGET_METRIC")

            from .future_projections import FutureProjectionAgeCurve
            self.system_pipeline.warp_model = FutureProjectionAgeCurve(
                max_projection_years=self.system_pipeline.max_projection_years,
                use_dynasty_guru=self.system_pipeline.use_dynasty_guru
            )
            warp_metrics = self.system_pipeline.warp_model.fit_joint_model(warp_data_clean, training_data=data)
        else:
            self.system_pipeline.warp_model = None
            warp_metrics = {'error': 'insufficient_data'}

        # Set primary model for backward compatibility
        if len(war_data) >= len(warp_data) and self.system_pipeline.war_model:
            self.system_pipeline.projection_model = self.system_pipeline.war_model
        elif self.system_pipeline.warp_model:
            self.system_pipeline.projection_model = self.system_pipeline.warp_model
        else:
            raise ValueError("Unable to train either model")

        combined_metrics = {
            'war_model': war_metrics,
            'warp_model': warp_metrics
        }

        self.system_pipeline.model_performance = combined_metrics

        # Initialize constraint optimizer after models are trained
        if hasattr(self.system_pipeline, 'constraint_optimizer'):
            try:
                from .constraint_optimizer import ConstraintOptimizer
            except ImportError:
                from constraint_optimizer import ConstraintOptimizer
            self.system_pipeline.constraint_optimizer = ConstraintOptimizer(
                war_model=self.system_pipeline.war_model,
                warp_model=self.system_pipeline.warp_model,
                elite_adjuster=self.system_pipeline.elite_adjuster
            )

        print("Confidence-aware separate model training complete!")
        return combined_metrics

    def validate_model(self, data: pd.DataFrame, n_splits: int = 5) -> Dict[str, Union[float, Dict]]:
        """
        Perform temporal cross-validation of the joint model.

        Args:
            data: Complete dataset for validation
            n_splits: Number of validation folds

        Returns:
            Dictionary containing validation metrics
        """
        print(f"Validating both models with {n_splits}-fold temporal cross-validation...")

        validation_results = {}

        # Prepare separate datasets for validation
        war_data = data[data['DataSource'] == 'WAR'].copy()
        warp_data = data[data['DataSource'] == 'WARP'].copy()

        # Validate WAR model if it exists
        if hasattr(self.system_pipeline, 'war_model') and self.system_pipeline.war_model is not None and len(war_data) > 100:
            print("Validating WAR model...")
            war_data['TARGET_METRIC'] = war_data['WAR']
            war_validation = self.system_pipeline.validator.validate_joint_model(
                self.system_pipeline.war_model, war_data, n_splits, data  # Pass full dataset for confidence
            )
            validation_results['war_model_validation'] = war_validation

        # Validate WARP model if it exists
        if hasattr(self.system_pipeline, 'warp_model') and self.system_pipeline.warp_model is not None and len(warp_data) > 100:
            print("Validating WARP model...")
            warp_data['TARGET_METRIC'] = warp_data['WARP']

            # Clean WARP data for validation - filter out NaN values
            pre_clean_warp = len(warp_data)
            warp_data_clean = warp_data[
                warp_data['Age'].notna() &
                warp_data['Position'].notna() &
                (warp_data['Position'] != '') &
                warp_data['TARGET_METRIC'].notna()
            ].copy()
            dropped_warp = pre_clean_warp - len(warp_data_clean)
            if dropped_warp > 0:
                print(f"  Dropped {dropped_warp} WARP records with missing Age/Position/TARGET_METRIC for validation")

            if len(warp_data_clean) > 50:  # Ensure sufficient data for validation
                warp_validation = self.system_pipeline.validator.validate_joint_model(
                    self.system_pipeline.warp_model, warp_data_clean, n_splits, data  # Pass full dataset for confidence
                )
                validation_results['warp_model_validation'] = warp_validation
            else:
                print(f"  Insufficient clean WARP data for validation ({len(warp_data_clean)} records)")
                validation_results['warp_model_validation'] = {'error': 'insufficient_clean_data'}

        # For backward compatibility, also include primary model validation
        if hasattr(self.system_pipeline, 'projection_model') and self.system_pipeline.projection_model is not None:
            # Copy validation results to top level for backward compatibility
            if self.system_pipeline.projection_model == self.system_pipeline.war_model:
                if 'war_model_validation' in validation_results:
                    validation_results.update(validation_results['war_model_validation'])
            elif 'warp_model_validation' in validation_results:
                validation_results.update(validation_results['warp_model_validation'])

        print("Model validation complete!")
        return validation_results

    def batch_generate_projections(self,
                                 target_season: int,
                                 years_ahead: int = 3,
                                 min_career_length: int = 2) -> pd.DataFrame:
        """
        Generate projections for all eligible players.

        Args:
            target_season: Base season to project from
            years_ahead: Number of years to project
            min_career_length: Minimum career length for inclusion

        Returns:
            DataFrame with projections for all eligible players
        """
        if not (self.system_pipeline.war_model or self.system_pipeline.warp_model):
            raise ValueError("Models must be trained before generating projections")

        print(f"\\nGenerating {years_ahead}-year projections from {target_season}...")

        # Get player data from training_data to ensure data quality
        player_data = self.system_pipeline.training_data.copy()

        # Filter to target season and active players
        current_season_data = player_data[player_data['Season'] == target_season].copy()

        # Apply minimum career length filter
        career_lengths = player_data.groupby('mlbid')['Season'].nunique()
        eligible_players = career_lengths[career_lengths >= min_career_length].index
        current_season_data = current_season_data[current_season_data['mlbid'].isin(eligible_players)]

        print(f"Eligible players for projection: {len(current_season_data)}")

        all_projections = []

        for _, player_row in current_season_data.iterrows():
            player_id = player_row['mlbid']
            age = player_row['Age']

            # Generate projections using both models
            player_projections = {
                'mlbid': player_id,
                'Name': player_row['Name'],
                'Age': age,
                'Position': player_row['Position'],
                'Current_WAR': player_row.get('WAR', np.nan),
                'Current_WARP': player_row.get('WARP', np.nan)
            }

            # WAR projections
            if self.system_pipeline.war_model:
                # Get player's current data from training data
                player_data = self.system_pipeline.training_data[
                    (self.system_pipeline.training_data['mlbid'] == player_id) &
                    (self.system_pipeline.training_data['Season'] == target_season)
                ]
                if len(player_data) > 0:
                    current_row = player_data.iloc[0]
                    player_history = self.system_pipeline.training_data[
                        self.system_pipeline.training_data['mlbid'] == player_id
                    ].sort_values('Season').copy()
                    # Ensure TARGET_METRIC is set for WAR projections
                    player_history['TARGET_METRIC'] = player_history['WAR']

                    war_projections = self.system_pipeline.war_model.predict_performance_path(
                        current_row.get('Age', 27),
                        current_row.get('Position', 'OF'),
                        current_row.get('WAR', 0),
                        player_history,
                        years_ahead,
                        self.system_pipeline.training_data
                    )
                    for year_idx, war_value in enumerate(war_projections):
                        player_projections[f'projected_WAR_year_{year_idx + 1}'] = war_value

            # WARP projections
            if self.system_pipeline.warp_model:
                # Get player's current data from training data
                player_data = self.system_pipeline.training_data[
                    (self.system_pipeline.training_data['mlbid'] == player_id) &
                    (self.system_pipeline.training_data['Season'] == target_season)
                ]
                if len(player_data) > 0:
                    current_row = player_data.iloc[0]
                    player_history = self.system_pipeline.training_data[
                        self.system_pipeline.training_data['mlbid'] == player_id
                    ].sort_values('Season').copy()
                    # Ensure TARGET_METRIC is set for WARP projections
                    player_history['TARGET_METRIC'] = player_history['WARP']

                    warp_projections = self.system_pipeline.warp_model.predict_performance_path(
                        current_row.get('Age', 27),
                        current_row.get('Position', 'OF'),
                        current_row.get('WARP', 0),
                        player_history,
                        years_ahead,
                        self.system_pipeline.training_data
                    )
                    for year_idx, warp_value in enumerate(warp_projections):
                        player_projections[f'projected_WARP_year_{year_idx + 1}'] = warp_value

            all_projections.append(player_projections)

        projections_df = pd.DataFrame(all_projections)
        print(f"Projections generated for {len(projections_df)} players")

        # Apply elite player adjustments if enabled
        if hasattr(self.system_pipeline, 'elite_adjuster') and self.system_pipeline.elite_adjuster is not None:
            print("\\nApplying elite player adjustments...")
            projections_df = self.system_pipeline.elite_adjuster.adjust_elite_projections(
                projections_df,
                confidence_scores=None,  # Will be calculated inside the method
                training_data=self.system_pipeline.training_data
            )

        # Apply injury recovery adjustments if enabled and data available
        if self.system_pipeline.use_injury_modeling:
            # Load injury data if not already loaded
            if not hasattr(self.system_pipeline, 'injury_data') or self.system_pipeline.injury_data is None:
                print("\\nLoading injury data...")
                try:
                    self.system_pipeline.injury_data = self.system_pipeline.load_injury_data([target_season-2, target_season-1, target_season])
                except Exception as e:
                    print(f"Failed to load injury data: {e}")
                    self.system_pipeline.injury_data = None

            if self.system_pipeline.injury_data is not None and not self.system_pipeline.injury_data.empty:
                print(f"\\nApplying injury recovery adjustments... ({len(self.system_pipeline.injury_data)} injury records)")
                projections_df = self.system_pipeline._apply_injury_recovery_adjustments(projections_df, target_season)
            else:
                print("\\nInjury modeling enabled but no injury data available - skipping injury adjustments")

        # Apply zero-sum constraint if enabled (AFTER elite adjustments and injury recovery)
        if self.system_pipeline.use_zero_sum_constraint:
            print("\\nApplying zero-sum WAR constraint optimization...")

            # Calculate original totals for comparison
            original_war_total = projections_df['projected_WAR_year_1'].sum()
            original_warp_total = projections_df['projected_WARP_year_1'].sum()

            # Prepare training data with TARGET_METRIC for constraint optimizer
            constraint_training_data = self.system_pipeline.training_data.copy()
            # Set TARGET_METRIC based on data source for confidence calculation
            war_mask = constraint_training_data['DataSource'] == 'WAR'
            warp_mask = constraint_training_data['DataSource'] == 'WARP'
            constraint_training_data.loc[war_mask, 'TARGET_METRIC'] = constraint_training_data.loc[war_mask, 'WAR']
            constraint_training_data.loc[warp_mask, 'TARGET_METRIC'] = constraint_training_data.loc[warp_mask, 'WARP']

            # Apply constraint to both WAR and WARP projections using constraint optimizer
            projections_df = self.system_pipeline.constraint_optimizer.apply_zero_sum_war_constraint(
                projections_df, training_data=constraint_training_data
            )

            # Recalculate totals
            adjusted_war_total = projections_df['projected_WAR_year_1'].sum()
            adjusted_warp_total = projections_df['projected_WARP_year_1'].sum()

            print(f"WAR total adjustment: {original_war_total:.1f} -> {adjusted_war_total:.1f}")
            print(f"WARP total adjustment: {original_warp_total:.1f} -> {adjusted_warp_total:.1f}")

        return projections_df

    def run_complete_pipeline(self,
                            years: Optional[List[int]] = None,
                            validation_splits: int = 5,
                            save_model: bool = True,
                            model_path: str = "system2_projection_model.pkl") -> Dict[str, Union[float, Dict]]:
        """
        Execute the complete SYSTEM 2 projection pipeline.

        Args:
            years: Years to include in analysis (default 2016-2024)
            validation_splits: Number of cross-validation folds
            save_model: Whether to save the trained model
            model_path: Path to save the model

        Returns:
            Dictionary containing pipeline results
        """
        if years is None:
            years = list(range(2016, 2025))

        print("SYSTEM 2: COMPLETE PIPELINE EXECUTION")
        print("=" * 50)

        results = {}

        # Step 1: Load data
        print("\\n1. Loading complete dataset...")
        raw_data = self.system_pipeline.load_complete_dataset(
            years=years,
            player_types=['hitters', 'pitchers']
        )
        results['data_loaded'] = len(raw_data)

        # Step 2: Prepare features
        print("\\n2. Preparing projection features...")
        processed_data = self.system_pipeline.prepare_projection_features(raw_data)
        results['features_prepared'] = len(processed_data)

        # Step 3: Prepare training data
        print("\\n3. Preparing training data...")
        training_data = self.system_pipeline.prepare_training_data(processed_data)
        results['training_data_prepared'] = len(training_data)

        # Step 4: Train models
        print("\\n4. Training projection models...")
        training_metrics = self.train_projection_model(training_data)
        results['training_metrics'] = training_metrics

        # Step 5: Validate models
        print("\\n5. Validating models...")
        validation_results = self.validate_model(training_data, validation_splits)
        results['validation_results'] = validation_results

        # Step 6: Load injury data if enabled
        if self.system_pipeline.use_injury_modeling:
            print("\\n6. Loading injury data...")
            try:
                injury_data = self.system_pipeline.load_injury_data(years)
                if injury_data is not None and not injury_data.empty:
                    results['injury_data_loaded'] = len(injury_data)
                    print(f"Injury data loaded: {len(injury_data)} records")
                else:
                    results['injury_data_loaded'] = 0
                    print("No injury data available")
            except Exception as e:
                print(f"Injury data loading failed: {e}")
                results['injury_data_error'] = str(e)

        # Step 7: Generate projections for most recent year
        print("\\n7. Generating future projections...")
        target_year = max(years)
        projections = self.batch_generate_projections(
            target_season=target_year,
            years_ahead=self.system_pipeline.max_projection_years,
            min_career_length=2
        )
        results['projections_generated'] = len(projections)

        # Step 8: Save models with history tracking if requested
        if save_model:
            year_range = f"{min(years)}-{max(years)}"
            saved_paths = []

            # Save WAR model if it exists
            if hasattr(self.system_pipeline, 'war_model') and self.system_pipeline.war_model is not None:
                war_path = self.model_history.save_model_with_history(
                    model_obj=self.system_pipeline.war_model,
                    model_type="war",
                    training_metrics=training_metrics.get('war_model', {}),
                    year_range=year_range,
                    metadata={
                        "pipeline_config": {
                            "use_zero_sum_constraint": getattr(self.system_pipeline, 'use_zero_sum_constraint', False),
                            "use_injury_modeling": getattr(self.system_pipeline, 'use_injury_modeling', False),
                            "use_dynasty_guru": getattr(self.system_pipeline, 'use_dynasty_guru', False)
                        },
                        "validation_results": validation_results.get('war_model_validation', {}),
                        "data_summary": {
                            "total_records": results.get('data_loaded', 0),
                            "training_records": results.get('training_data_prepared', 0),
                            "projection_records": results.get('projections_generated', 0)
                        }
                    }
                )
                saved_paths.append(war_path)

            # Save WARP model if it exists
            if hasattr(self.system_pipeline, 'warp_model') and self.system_pipeline.warp_model is not None:
                warp_path = self.model_history.save_model_with_history(
                    model_obj=self.system_pipeline.warp_model,
                    model_type="warp",
                    training_metrics=training_metrics.get('warp_model', {}),
                    year_range=year_range,
                    metadata={
                        "pipeline_config": {
                            "use_zero_sum_constraint": getattr(self.system_pipeline, 'use_zero_sum_constraint', False),
                            "use_injury_modeling": getattr(self.system_pipeline, 'use_injury_modeling', False),
                            "use_dynasty_guru": getattr(self.system_pipeline, 'use_dynasty_guru', False)
                        },
                        "validation_results": validation_results.get('warp_model_validation', {}),
                        "data_summary": {
                            "total_records": results.get('data_loaded', 0),
                            "training_records": results.get('training_data_prepared', 0),
                            "projection_records": results.get('projections_generated', 0)
                        }
                    }
                )
                saved_paths.append(warp_path)

            # Maintain backward compatibility with old model_path
            if saved_paths:
                # Also save primary model using old method for compatibility
                try:
                    self.system_pipeline.projection_model.save_model(model_path)
                    print(f"Legacy model also saved to: {model_path}")
                except Exception as e:
                    print(f"Legacy model save failed (non-critical): {e}")

                results['models_saved'] = saved_paths
                results['model_saved'] = model_path  # Backward compatibility
            else:
                print("No models available to save")
                results['model_saved'] = None

        print("\\n" + "=" * 50)
        print("SYSTEM 2 PIPELINE COMPLETE")
        print("=" * 50)

        return results


# Required imports for the orchestrator to work properly
# These will be imported by the main integration.py file
def _import_dependencies():
    """Import required dependencies for the orchestrator."""
    try:
        from .future_projections import FutureProjectionAgeCurve
        return True
    except ImportError:
        return False