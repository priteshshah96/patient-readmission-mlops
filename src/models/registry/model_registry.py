import os
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
import logging
from datetime import datetime
import time
import json

# MLflow for model registry
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from mlflow.entities.model_registry import ModelVersion
from mlflow.exceptions import MlflowException

# Model loading utilities
import joblib
from sklearn.base import BaseEstimator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    MLflow Model Registry for Patient Readmission Prediction Models
    Handles model versioning, lifecycle management, and deployment staging.
    """
    
    def __init__(self, tracking_uri: str = "file:./mlruns"):
        """Initialize model registry with MLflow tracking URI."""
        self.tracking_uri = tracking_uri
        mlflow.set_tracking_uri(tracking_uri)
        self.client = MlflowClient()
        
        # Model registry configuration
        self.registered_model_name = "patient_readmission_predictor"
        self.experiment_name = "patient_readmission_prediction"
        
        logger.info(f"✅ Model Registry initialized with tracking URI: {tracking_uri}")
        
    def create_or_get_registered_model(self) -> str:
        """Create registered model if it doesn't exist."""
        try:
            # Try to get existing registered model
            registered_model = self.client.get_registered_model(self.registered_model_name)
            logger.info(f"📋 Using existing registered model: {self.registered_model_name}")
            return registered_model.name
            
        except MlflowException:
            # Create new registered model
            description = """
            Patient 30-Day Readmission Prediction Model
            
            This model predicts the likelihood of a patient being readmitted to the hospital 
            within 30 days of discharge. Built using XGBoost with SMOTE for class balancing.
            
            Key Performance Metrics:
            - Recall: 67.2% (primary metric for healthcare)
            - Precision: 16.8%
            - ROC-AUC: 67.4%
            - F1-Score: 26.8%
            
            Features Used: 15 most predictive features including:
            - number_inpatient, number_emergency
            - discharge_disposition_id, number_diagnoses
            - time_in_hospital, num_medications
            - diabetesMed, metformin, num_lab_procedures
            - change, number_outpatient, age
            - num_procedures, admission_type_id, repaglinide
            """
            
            registered_model = self.client.create_registered_model(
                name=self.registered_model_name,
                description=description.strip()
            )
            
            logger.info(f"🎉 Created new registered model: {self.registered_model_name}")
            return registered_model.name
    
    def find_best_model_run(self, min_recall: float = 0.6) -> Optional[str]:
        """Find the best model run based on recall performance."""
        try:
            # Get experiment
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if not experiment:
                logger.error(f"Experiment {self.experiment_name} not found!")
                return None
            
            # Search for runs with good recall - returns DataFrame
            runs_df = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string=f"metrics.recall >= {min_recall}",
                order_by=["metrics.recall DESC"],
                max_results=10
            )
            
            # Check if DataFrame is empty
            if runs_df.empty:
                logger.warning(f"No runs found with recall >= {min_recall}")
                # Try with lower threshold
                runs_df = mlflow.search_runs(
                    experiment_ids=[experiment.experiment_id],
                    order_by=["metrics.recall DESC"],
                    max_results=5
                )
            
            # Check again if DataFrame is empty
            if runs_df.empty:
                logger.error("No model runs found!")
                return None
            
            # Get the best run (first row)
            best_run = runs_df.iloc[0]
            logger.info(f"🏆 Best model found:")
            logger.info(f"   Run ID: {best_run['run_id']}")
            logger.info(f"   Recall: {best_run['metrics.recall']:.4f}")
            logger.info(f"   Precision: {best_run['metrics.precision']:.4f}")
            logger.info(f"   F1-Score: {best_run['metrics.f1_score']:.4f}")
            logger.info(f"   ROC-AUC: {best_run['metrics.roc_auc']:.4f}")
            
            return str(best_run['run_id'])
            
        except Exception as e:
            logger.error(f"Error finding best model: {e}")
            return None
    
    def register_best_model(self, run_id: Optional[str] = None, stage: str = "Staging") -> Optional[ModelVersion]:
        """Register the best model in MLflow Model Registry."""
        
        # Find best run if not provided
        if run_id is None:
            run_id = self.find_best_model_run()
            if run_id is None:
                logger.error("No suitable model run found for registration")
                return None
        
        try:
            # Ensure registered model exists
            model_name = self.create_or_get_registered_model()
            
            # Register model version
            model_uri = f"runs:/{run_id}/model"
            
            # Get run details for version description
            run = mlflow.get_run(run_id)
            metrics = run.data.metrics
            params = run.data.params
            
            description = f"""
            XGBoost Patient Readmission Predictor
            
            Performance Metrics:
            - Recall: {metrics.get('recall', 'N/A'):.4f} (Primary Healthcare Metric)
            - Precision: {metrics.get('precision', 'N/A'):.4f}
            - F1-Score: {metrics.get('f1_score', 'N/A'):.4f}
            - ROC-AUC: {metrics.get('roc_auc', 'N/A'):.4f}
            - Accuracy: {metrics.get('accuracy', 'N/A'):.4f}
            
            Model Configuration:
            - Algorithm: XGBoost with SMOTE
            - Features: 15 selected features
            - Training Samples: {params.get('training_samples', 'N/A')}
            - Class Imbalance Ratio: {params.get('class_imbalance_ratio', 'N/A')}
            
            Training Details:
            - Training Time: {metrics.get('training_time_seconds', 'N/A'):.2f}s
            - Prediction Time: {metrics.get('prediction_time', 'N/A'):.4f}s
            - Run ID: {run_id}
            - Registered: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            """
            
            # Register model without description first
            model_version = mlflow.register_model(
                model_uri=model_uri,
                name=model_name
            )
            
            # Update the model version description separately
            self.client.update_model_version(
                name=model_name,
                version=model_version.version,
                description=description.strip()
            )
            
            logger.info(f"🎉 Model registered successfully!")
            logger.info(f"   Model Name: {model_name}")
            logger.info(f"   Version: {model_version.version}")
            logger.info(f"   Run ID: {run_id}")
            
            # Add tags to the model version
            self.client.set_model_version_tag(
                name=model_name,
                version=model_version.version,
                key="model_type",
                value="xgboost_with_smote"
            )
            
            self.client.set_model_version_tag(
                name=model_name,
                version=model_version.version,
                key="use_case",
                value="healthcare_readmission_prediction"
            )
            
            self.client.set_model_version_tag(
                name=model_name,
                version=model_version.version,
                key="primary_metric",
                value="recall"
            )
            
            # Transition to staging if requested
            if stage and stage != "None":
                self.transition_model_stage(model_name, model_version.version, stage)
            
            return model_version
            
        except Exception as e:
            logger.error(f"Error registering model: {e}")
            return None
    
    def transition_model_stage(self, model_name: str, version: str, stage: str, 
                              archive_existing: bool = True) -> bool:
        """Transition model to a specific stage (Staging, Production, Archived)."""
        
        valid_stages = ["Staging", "Production", "Archived"]
        if stage not in valid_stages:
            logger.error(f"Invalid stage: {stage}. Must be one of {valid_stages}")
            return False
        
        try:
            # Archive existing models in the target stage if requested
            if archive_existing and stage in ["Staging", "Production"]:
                existing_models = self.client.get_latest_versions(
                    name=model_name,
                    stages=[stage]
                )
                
                for existing_model in existing_models:
                    logger.info(f"📦 Archiving existing {stage} model version {existing_model.version}")
                    self.client.transition_model_version_stage(
                        name=model_name,
                        version=existing_model.version,
                        stage="Archived",
                        archive_existing_versions=False
                    )
            
            # Transition new model to target stage
            self.client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage=stage,
                archive_existing_versions=False
            )
            
            logger.info(f"✅ Model {model_name} version {version} transitioned to {stage}")
            return True
            
        except Exception as e:
            logger.error(f"Error transitioning model stage: {e}")
            return False
    
    def get_production_model(self) -> Optional[Dict[str, Any]]:
        """Get the current production model."""
        try:
            production_models = self.client.get_latest_versions(
                name=self.registered_model_name,
                stages=["Production"]
            )
            
            if not production_models:
                logger.warning("No production model found")
                return None
            
            prod_model = production_models[0]
            
            # Get model details
            model_details = {
                'name': prod_model.name,
                'version': prod_model.version,
                'stage': prod_model.current_stage,
                'run_id': prod_model.run_id,
                'model_uri': f"models:/{prod_model.name}/{prod_model.version}",
                'creation_timestamp': prod_model.creation_timestamp,
                'last_updated_timestamp': prod_model.last_updated_timestamp,
                'description': prod_model.description
            }
            
            logger.info(f"🏭 Production model: {prod_model.name} v{prod_model.version}")
            return model_details
            
        except Exception as e:
            logger.error(f"Error getting production model: {e}")
            return None
    
    def load_model_from_registry(self, stage: str = "Production") -> Optional[BaseEstimator]:
        """Load model from registry by stage."""
        try:
            model_uri = f"models:/{self.registered_model_name}/{stage}"
            # Use explicit import to fix type checking warning
            import mlflow.sklearn as mlflow_sklearn
            model = mlflow_sklearn.load_model(model_uri)
            
            logger.info(f"📥 Loaded {stage} model from registry")
            return model
            
        except Exception as e:
            logger.error(f"Error loading model from registry: {e}")
            return None
    
    def list_all_model_versions(self) -> List[Dict[str, Any]]:
        """List all versions of the registered model."""
        try:
            model_versions = self.client.search_model_versions(
                filter_string=f"name='{self.registered_model_name}'"
            )
            
            versions_info = []
            for version in model_versions:
                # Get run metrics with proper error handling
                try:
                    if version.run_id:  # Check if run_id exists
                        run = mlflow.get_run(version.run_id)
                        metrics = run.data.metrics
                    else:
                        metrics = {}
                except Exception:
                    metrics = {}
                
                version_info = {
                    'version': version.version,
                    'stage': version.current_stage,
                    'run_id': version.run_id,
                    'creation_time': datetime.fromtimestamp(version.creation_timestamp / 1000),
                    'recall': metrics.get('recall', 'N/A'),
                    'precision': metrics.get('precision', 'N/A'),
                    'f1_score': metrics.get('f1_score', 'N/A'),
                    'roc_auc': metrics.get('roc_auc', 'N/A')
                }
                versions_info.append(version_info)
            
            # Sort by version number (descending)
            versions_info.sort(key=lambda x: int(x['version']), reverse=True)
            
            return versions_info
            
        except Exception as e:
            logger.error(f"Error listing model versions: {e}")
            return []
    
    def print_model_registry_status(self):
        """Print a comprehensive status report of the model registry."""
        
        logger.info("="*70)
        logger.info("🏛️  MODEL REGISTRY STATUS REPORT")
        logger.info("="*70)
        
        try:
            # Get registered model info
            registered_model = self.client.get_registered_model(self.registered_model_name)
            logger.info(f"📋 Registered Model: {registered_model.name}")
            
            # Handle potential None description
            description = registered_model.description or "No description"
            logger.info(f"📝 Description: {description[:100]}...")
            
            # Get all versions
            versions = self.list_all_model_versions()
            logger.info(f"📊 Total Versions: {len(versions)}")
            
            if versions:
                logger.info("\n📈 MODEL VERSIONS:")
                logger.info("-" * 70)
                logger.info(f"{'Ver':<4} {'Stage':<12} {'Recall':<8} {'Precision':<10} {'F1':<8} {'Created':<12}")
                logger.info("-" * 70)
                
                for version in versions:
                    created = version['creation_time'].strftime('%m/%d/%Y') if isinstance(version['creation_time'], datetime) else 'N/A'
                    recall = f"{version['recall']:.3f}" if isinstance(version['recall'], (int, float)) else 'N/A'
                    precision = f"{version['precision']:.3f}" if isinstance(version['precision'], (int, float)) else 'N/A'
                    f1 = f"{version['f1_score']:.3f}" if isinstance(version['f1_score'], (int, float)) else 'N/A'
                    
                    logger.info(f"{version['version']:<4} {version['stage']:<12} {recall:<8} {precision:<10} {f1:<8} {created:<12}")
            
            # Production model info
            prod_model = self.get_production_model()
            if prod_model:
                logger.info(f"\n🏭 PRODUCTION MODEL:")
                logger.info(f"   Version: {prod_model['version']}")
                logger.info(f"   Run ID: {prod_model['run_id']}")
                logger.info(f"   Model URI: {prod_model['model_uri']}")
            else:
                logger.info(f"\n⚠️  NO PRODUCTION MODEL DEPLOYED")
            
        except MlflowException:
            logger.warning(f"No registered model found: {self.registered_model_name}")
        
        logger.info("="*70)
    
    def promote_to_production(self, version: Optional[str] = None) -> bool:
        """Promote a model version to production."""
        
        if version is None:
            # Get latest staging model
            staging_models = self.client.get_latest_versions(
                name=self.registered_model_name,
                stages=["Staging"]
            )
            
            if not staging_models:
                logger.error("No staging model found to promote")
                return False
            
            version = staging_models[0].version
            logger.info(f"🚀 Promoting staging model version {version} to production")
        
        # Transition to production
        success = self.transition_model_stage(
            model_name=self.registered_model_name,
            version=version,
            stage="Production",
            archive_existing=True
        )
        
        if success:
            logger.info(f"🎉 Model version {version} is now in PRODUCTION!")
        
        return success


def main():
    """Main function to demonstrate model registry functionality."""
    
    print("🏛️  PATIENT READMISSION MODEL REGISTRY")
    print("="*60)
    print(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    # Initialize model registry
    registry = ModelRegistry()
    
    # Register the best model
    logger.info("🔍 Finding and registering best model...")
    model_version = registry.register_best_model(stage="Staging")
    
    if model_version:
        logger.info("✅ Model registration completed successfully!")
        
        # Show registry status
        registry.print_model_registry_status()
        
        # Optional: Promote to production
        try:
            print(f"\n🤔 Would you like to promote version {model_version.version} to Production? (y/n): ", end="")
            response = input().lower().strip()
            
            if response in ['y', 'yes']:
                registry.promote_to_production(model_version.version)
                print("\n📊 Updated registry status:")
                registry.print_model_registry_status()
        except KeyboardInterrupt:
            print("\nSkipping production promotion.")
        
        print(f"\n🎉 Model Registry Setup Complete!")
        print(f"📋 Model Name: {registry.registered_model_name}")
        print(f"🏷️  Latest Version: {model_version.version}")
        print(f"📊 View in MLflow UI: http://localhost:5000")
        print("="*60)
        
    else:
        logger.error("❌ Model registration failed!")


if __name__ == "__main__":
    main()