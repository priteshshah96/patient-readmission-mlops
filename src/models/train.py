import os
import pandas as pd
# import tqdm
import numpy as np
from typing import Dict, Any, Tuple, Optional
import logging
from datetime import datetime
import time

# ML libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    roc_auc_score, precision_recall_curve, auc,
    accuracy_score, precision_score, recall_score, f1_score
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# MLflow for experiment tracking
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

# Try to import XGBoost, skip if not available
try:
    import xgboost as xgb
    import mlflow.xgboost
    XGBOOST_AVAILABLE = True
    print("✅ XGBoost loaded successfully")
except ImportError as e:
    XGBOOST_AVAILABLE = False
    print(f"⚠️ XGBoost not available: {e}")
    print("   Continuing with Logistic Regression and Random Forest only")

# Try to import tqdm for progress bars
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("⚠️ tqdm not available, progress bars disabled")

# Import our preprocessing module
from .preprocessing import DataPreprocessor

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ReadmissionModelTrainer:
    """Handles model training with MLflow experiment tracking and progress monitoring."""
    
    def __init__(self, experiment_name: str = "patient_readmission_prediction"):
        self.experiment_name = experiment_name
        self.preprocessor = DataPreprocessor()
        self.progress_bar = None
        
        # Set up MLflow - FIXED: Respect Docker MLflow server
        try:
            # Use environment variable or default to Docker MLflow server
            tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(experiment_name)
            logger.info(f"✅ MLflow experiment '{experiment_name}' ready")
            logger.info(f"🔗 Tracking URI: {tracking_uri}")
        except Exception as e:
            logger.warning(f"⚠️ MLflow setup issue: {e}")
            # Fallback to local only if Docker MLflow fails
            logger.info("🔄 Falling back to local file tracking")
            mlflow.set_tracking_uri("file:./mlruns")
            mlflow.set_experiment(experiment_name)
        
        # Progress tracking
        self.total_steps = 0
        self.current_step = 0
        
    def update_progress(self, description: str):
        """Update progress bar with current step."""
        self.current_step += 1
        if self.progress_bar and TQDM_AVAILABLE:
            self.progress_bar.set_description(f"Step {self.current_step}/{self.total_steps}: {description}")
            self.progress_bar.update(1)
        logger.info(f"[{self.current_step}/{self.total_steps}] {description}")
    
    def setup_class_imbalance_handling(self, strategy: str = "smote") -> SMOTE:
        """Setup class imbalance handling strategy."""
        
        if strategy == "smote":
            # SMOTE for oversampling minority class
            return SMOTE(random_state=42, sampling_strategy=0.3)  # Target 30% positive class
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def train_xgboost_only(self, X_train: pd.DataFrame, y_train: pd.Series) -> Optional[Dict[str, Any]]:
        """Train XGBoost model only (research shows best performance for this dataset)."""
        
        if not XGBOOST_AVAILABLE:
            logger.error("XGBoost not available!")
            return None
            
        self.update_progress("Training XGBoost")
        
        try:
            # Create pipeline with SMOTE
            smote = self.setup_class_imbalance_handling("smote")
            
            # Calculate scale_pos_weight for class imbalance
            scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
            
            model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=scale_pos_weight,
                random_state=42,
                eval_metric='logloss',
                verbosity=0  # Reduce output
            )
            
            # Create pipeline
            pipeline = ImbPipeline([
                ('smote', smote),
                ('classifier', model)
            ])
            
            # Train model with timing
            start_time = time.time()
            pipeline.fit(X_train, y_train)
            training_time = time.time() - start_time
            
            logger.info(f"XGBoost training completed in {training_time:.2f} seconds")
            
            return {
                'model': pipeline,
                'model_name': 'xgboost',
                'training_time': training_time,
                'params': {
                    'n_estimators': 100,
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'scale_pos_weight': float(scale_pos_weight),
                    'smote_sampling_strategy': 0.3
                }
            }
        except Exception as e:
            logger.error(f"XGBoost training failed: {e}")
            return None
    
    def evaluate_model(self, model: Any, X_test: pd.DataFrame, y_test: pd.Series, 
                      model_name: str) -> Dict[str, float]:
        """Comprehensive model evaluation with healthcare-focused metrics."""
        
        self.update_progress(f"Evaluating {model_name}")
        
        try:
            # Get predictions with timing
            start_time = time.time()
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            prediction_time = time.time() - start_time
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)  # Critical for healthcare
            f1 = f1_score(y_test, y_pred, zero_division=0)
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            
            # Precision-Recall AUC (better for imbalanced datasets)
            precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_pred_proba)
            pr_auc = auc(recall_vals, precision_vals)
            
            # Confusion matrix details
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            metrics = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,  # Most important for readmission prediction
                'f1_score': f1,
                'roc_auc': roc_auc,
                'pr_auc': pr_auc,
                'specificity': specificity,
                'prediction_time': prediction_time,
                'true_positives': int(tp),
                'false_positives': int(fp),
                'true_negatives': int(tn),
                'false_negatives': int(fn)
            }
            
            # Log detailed results
            logger.info(f"{model_name} Results:")
            logger.info(f"  Accuracy: {accuracy:.4f}")
            logger.info(f"  Precision: {precision:.4f}")
            logger.info(f"  Recall: {recall:.4f} ⭐ (Most Important)")
            logger.info(f"  F1-Score: {f1:.4f}")
            logger.info(f"  ROC-AUC: {roc_auc:.4f}")
            logger.info(f"  PR-AUC: {pr_auc:.4f}")
            logger.info(f"  Prediction time: {prediction_time:.4f}s")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating {model_name}: {e}")
            return {}
    
    def log_to_mlflow(self, model_info: Dict[str, Any], metrics: Dict[str, float], 
                     X_train: pd.DataFrame, y_train: pd.Series) -> str:
        """Log model and metrics to MLflow."""
        
        self.update_progress(f"Logging {model_info['model_name']} to MLflow")
        
        try:
            with mlflow.start_run(run_name=f"{model_info['model_name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
                
                # Log parameters
                mlflow.log_params(model_info['params'])
                
                # Log metrics
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        mlflow.log_metric(key, value)
                
                # Log training time
                if 'training_time' in model_info:
                    mlflow.log_metric('training_time_seconds', model_info['training_time'])
                
                # Log model - FIXED: Always use sklearn logging for pipelines
                model = model_info['model']
                
                try:
                    signature = infer_signature(X_train, y_train)
                    # Since all our models are wrapped in ImbPipeline, always use sklearn logging
                    mlflow.sklearn.log_model(
                        model, 
                        "model", 
                        signature=signature,
                        input_example=X_train.head(5)
                    )
                except Exception as model_log_error:
                    logger.warning(f"Could not log model artifacts: {model_log_error}")
                    # Just log the model without signature/example if there are issues
                    mlflow.sklearn.log_model(model, "model")
                
                # Log additional info
                mlflow.log_param("training_samples", len(X_train))
                mlflow.log_param("n_features", X_train.shape[1])
                mlflow.log_param("class_imbalance_ratio", f"{(y_train==0).sum()}:{(y_train==1).sum()}")
                
                run_id = mlflow.active_run().info.run_id
                logger.info(f"Model logged to MLflow with run_id: {run_id}")
                
                return run_id
                
        except Exception as e:
            logger.error(f"Error logging to MLflow: {e}")
            return "failed"
    
    def train_xgboost_pipeline(self) -> Dict[str, Any]:
        """Train only XGBoost model and return results."""
        
        # Calculate total steps for progress tracking (just XGBoost)
        self.total_steps = 4  # Data + train + eval + log
        self.current_step = 0
        
        # Initialize progress bar
        if TQDM_AVAILABLE:
            self.progress_bar = tqdm(total=self.total_steps, desc="XGBoost Training Pipeline", 
                                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
        
        logger.info("🚀 Starting XGBoost model training...")
        logger.info("="*50)
        
        # 1. Preprocess data
        self.update_progress("Loading and preprocessing data")
        start_time = time.time()
        try:
            X_train, X_test, y_train, y_test = self.preprocessor.run_full_pipeline()
            preprocessing_time = time.time() - start_time
            logger.info(f"Data preprocessing completed in {preprocessing_time:.2f} seconds")
        except Exception as e:
            logger.error(f"Data preprocessing failed: {e}")
            if self.progress_bar:
                self.progress_bar.close()
            raise
        
        # 2. Train XGBoost
        logger.info(f"\n{'='*50}")
        logger.info(f"🎯 TRAINING XGBOOST")
        logger.info(f"{'='*50}")
        
        try:
            # Train model
            model_info = self.train_xgboost_only(X_train, y_train)
            
            if model_info is None:
                logger.error("XGBoost training failed!")
                if self.progress_bar:
                    self.progress_bar.close()
                raise RuntimeError("XGBoost training failed")
            
            # Evaluate model
            metrics = self.evaluate_model(model_info['model'], X_test, y_test, 'xgboost')
            
            if not metrics:
                logger.error("XGBoost evaluation failed!")
                if self.progress_bar:
                    self.progress_bar.close()
                raise RuntimeError("XGBoost evaluation failed")
            
            # Log to MLflow
            run_id = self.log_to_mlflow(model_info, metrics, X_train, y_train)
            
            # Store results
            result = {
                'model_info': model_info,
                'metrics': metrics,
                'run_id': run_id
            }
            
        except Exception as e:
            logger.error(f"Failed to train XGBoost: {e}")
            if self.progress_bar:
                self.progress_bar.close()
            raise
        
        # Close progress bar
        if self.progress_bar:
            self.progress_bar.close()
        
        # Final summary
        total_time = time.time() - start_time
        training_time = model_info.get('training_time', 0)
        
        logger.info(f"\n{'='*50}")
        logger.info(f"🎉 XGBOOST TRAINING COMPLETED!")
        logger.info(f"{'='*50}")
        logger.info(f"⏱️  Total execution time: {total_time:.2f} seconds")
        logger.info(f"🏋️  Training time: {training_time:.2f} seconds")
        logger.info(f"📊 Data preprocessing time: {preprocessing_time:.2f} seconds")
        logger.info(f"🎯 Recall: {metrics['recall']:.4f}")
        logger.info(f"📈 F1-Score: {metrics['f1_score']:.4f}")
        logger.info(f"🔗 MLflow Run ID: {run_id}")
        logger.info(f"{'='*50}")
        
        return {
            'model_name': 'xgboost',
            'result': result,
            'timing': {
                'total_time': total_time,
                'preprocessing_time': preprocessing_time,
                'training_time': training_time
            },
            'data_shapes': {
                'X_train': X_train.shape,
                'X_test': X_test.shape,
                'y_train_distribution': y_train.value_counts().to_dict(),
                'y_test_distribution': y_test.value_counts().to_dict()
            }
        }


# Main execution for XGBoost only
if __name__ == "__main__":
    
    # Create logs directory
    os.makedirs('logs', exist_ok=True)
    
    print("🚀 PATIENT READMISSION PREDICTION - XGBOOST TRAINING")
    print("="*60)
    print(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"💾 Dataset: 101,766 patient records")
    print(f"🎯 Target: 30-day readmission prediction")
    print(f"🏥 Class imbalance: 8:1 (88.8% no readmission)")
    print(f"🔧 XGBoost Available: {XGBOOST_AVAILABLE}")
    print(f"📊 Progress Bars: {TQDM_AVAILABLE}")
    print("="*60)
    
    if not XGBOOST_AVAILABLE:
        print("❌ XGBoost not available! Cannot proceed.")
        exit(1)
    
    # Initialize trainer
    trainer = ReadmissionModelTrainer()
    
    # Train XGBoost only
    try:
        results = trainer.train_xgboost_pipeline()
        
        # Print final summary
        print(f"\n🎉 XGBOOST TRAINING COMPLETE!")
        print(f"🎯 Recall: {results['result']['metrics']['recall']:.4f}")
        print(f"📈 F1-Score: {results['result']['metrics']['f1_score']:.4f}")
        print(f"⏱️ Total Time: {results['timing']['total_time']:.2f} seconds")
        print(f"📊 Check MLflow UI for detailed results and model artifacts!")
        print(f"🔗 Run: mlflow ui --port 5000")
        print("="*60)
        
    except Exception as e:
        print(f"❌ XGBoost training failed: {e}")
        logger.error(f"XGBoost training pipeline failed: {e}")
        exit(1)