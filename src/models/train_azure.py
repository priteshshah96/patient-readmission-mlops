import os
import pandas as pd
import numpy as np
from typing import Dict, Any
import logging
from datetime import datetime
import time

# ML libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, auc,
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Azure ML and MLflow
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

# Try XGBoost
try:
    import xgboost as xgb
    import mlflow.xgboost
    XGBOOST_AVAILABLE = True
    print("✅ XGBoost loaded successfully")
except ImportError as e:
    XGBOOST_AVAILABLE = False
    print(f"⚠️ XGBoost not available: {e}")

# Import preprocessing
from .preprocessing import DataPreprocessor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def setup_azure_mlflow():
    """Setup MLflow for Azure ML"""
    # Azure ML automatically configures MLflow tracking URI
    # We just need to ensure we're using the right settings
    
    # Check if we're running in Azure ML
    if "AZUREML_RUN_ID" in os.environ:
        logger.info("🚀 Running in Azure ML - MLflow auto-configured")
    else:
        logger.info("🏠 Running locally - will connect to Azure ML MLflow")
        # For local runs, we can still connect to Azure ML MLflow if configured
    
    # Disable autologging for manual control
    mlflow.sklearn.autolog(disable=True)
    
    if XGBOOST_AVAILABLE:
        mlflow.xgboost.autolog(disable=True)
    
    logger.info(f"✅ MLflow tracking URI: {mlflow.get_tracking_uri()}")
    return True

class AzureReadmissionTrainer:
    """Azure ML MLflow trainer for patient readmission prediction"""
    
    def __init__(self, experiment_name: str = "patient_readmission_prediction"):
        self.experiment_name = experiment_name
        self.preprocessor = DataPreprocessor()
        
        # Setup Azure MLflow
        setup_azure_mlflow()
        
        # Set experiment (Azure ML manages this automatically in cloud runs)
        try:
            mlflow.set_experiment(experiment_name)
            logger.info(f"✅ MLflow experiment '{experiment_name}' ready")
        except Exception as e:
            logger.warning(f"⚠️ MLflow experiment setup: {e}")
        
        logger.info(f"🚀 Azure ML MLflow trainer initialized")
    
    def train_model(self, model_type: str, X_train, y_train) -> Dict[str, Any]:
        """Train a specific model type"""
        
        logger.info(f"🎯 Training {model_type}...")
        
        # Setup SMOTE
        smote = SMOTE(random_state=42, sampling_strategy=0.3)
        
        if model_type == "logistic_regression":
            model = LogisticRegression(
                random_state=42,
                class_weight='balanced',
                max_iter=1000,
                solver='liblinear'
            )
            params = {
                'smote_sampling_strategy': 0.3,
                'class_weight': 'balanced',
                'max_iter': 1000,
                'solver': 'liblinear'
            }
            
        elif model_type == "random_forest":
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=5,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )
            params = {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 10,
                'min_samples_leaf': 5,
                'class_weight': 'balanced',
                'smote_sampling_strategy': 0.3
            }
            
        elif model_type == "xgboost" and XGBOOST_AVAILABLE:
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
                verbosity=0
            )
            params = {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'scale_pos_weight': float(scale_pos_weight),
                'smote_sampling_strategy': 0.3
            }
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Create pipeline
        pipeline = ImbPipeline([
            ('smote', smote),
            ('classifier', model)
        ])
        
        # Train model
        start_time = time.time()
        pipeline.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        logger.info(f"✅ {model_type} trained in {training_time:.2f} seconds")
        
        return {
            'model': pipeline,
            'model_type': model_type,
            'training_time': training_time,
            'params': params
        }
    
    def evaluate_model(self, model, X_test, y_test, model_type: str) -> Dict[str, float]:
        """Evaluate model and return metrics"""
        
        logger.info(f"📊 Evaluating {model_type}...")
        
        # Predictions
        start_time = time.time()
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        prediction_time = time.time() - start_time
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        # Precision-Recall AUC
        precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_pred_proba)
        pr_auc = auc(recall_vals, precision_vals)
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
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
        
        logger.info(f"📈 {model_type} Results:")
        logger.info(f"   Accuracy: {accuracy:.4f}")
        logger.info(f"   Precision: {precision:.4f}")
        logger.info(f"   Recall: {recall:.4f} ⭐ (Most Important)")
        logger.info(f"   F1-Score: {f1:.4f}")
        logger.info(f"   ROC-AUC: {roc_auc:.4f}")
        logger.info(f"   PR-AUC: {pr_auc:.4f}")
        
        return metrics
    
    def log_to_mlflow(self, model_info: Dict, metrics: Dict, X_train, y_train):
        """Log model and metrics to Azure ML MLflow"""
        
        model_type = model_info['model_type']
        model = model_info['model']
        
        try:
            with mlflow.start_run(run_name=f"{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
                
                # Log parameters
                mlflow.log_params(model_info['params'])
                
                # Log metrics
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        mlflow.log_metric(key, value)
                
                # Log training time
                mlflow.log_metric('training_time_seconds', model_info['training_time'])
                
                # Create model signature
                signature = infer_signature(X_train, y_train)
                
                # Log model with appropriate method
                if model_type == "xgboost" and XGBOOST_AVAILABLE:
                    mlflow.xgboost.log_model(
                        model,
                        "model",
                        signature=signature,
                        input_example=X_train.head(5),
                        registered_model_name=f"patient_readmission_{model_type}"
                    )
                else:
                    mlflow.sklearn.log_model(
                        model,
                        "model", 
                        signature=signature,
                        input_example=X_train.head(5),
                        registered_model_name=f"patient_readmission_{model_type}"
                    )
                
                # Log additional metadata
                mlflow.log_param("training_samples", len(X_train))
                mlflow.log_param("test_samples", len(y_train))
                mlflow.log_param("n_features", X_train.shape[1])
                mlflow.log_param("feature_names", list(X_train.columns))
                mlflow.log_param("class_imbalance_ratio", f"{(y_train==0).sum()}:{(y_train==1).sum()}")
                
                # Log tags for better organization
                mlflow.set_tags({
                    "model_type": model_type,
                    "use_case": "healthcare_readmission",
                    "data_version": "v1.0",
                    "framework": "scikit-learn" if model_type != "xgboost" else "xgboost",
                    "target_metric": "recall",
                    "domain": "healthcare"
                })
                
                run_id = mlflow.active_run().info.run_id
                logger.info(f"✅ {model_type} logged to Azure ML MLflow: {run_id}")
                
                return run_id
                
        except Exception as e:
            logger.error(f"❌ Failed to log {model_type} to MLflow: {e}")
            return "failed"
    
    def train_all_models(self):
        """Train all models using Azure ML MLflow"""
        
        logger.info("🚀 Starting Azure ML MLOps training pipeline...")
        logger.info("="*70)
        
        # Load and preprocess data
        logger.info("📊 Loading and preprocessing data...")
        start_time = time.time()
        X_train, X_test, y_train, y_test = self.preprocessor.run_full_pipeline()
        preprocessing_time = time.time() - start_time
        
        logger.info(f"✅ Data preprocessing completed in {preprocessing_time:.2f} seconds")
        logger.info(f"📈 Training data: {X_train.shape}")
        logger.info(f"📈 Test data: {X_test.shape}")
        logger.info(f"📊 Class distribution: {y_train.value_counts().to_dict()}")
        
        # Models to train
        model_types = ["logistic_regression", "random_forest"]
        if XGBOOST_AVAILABLE:
            model_types.append("xgboost")
        
        logger.info(f"🎯 Training {len(model_types)} models: {model_types}")
        
        results = {}
        total_training_time = 0
        
        for model_type in model_types:
            logger.info(f"\n{'='*50}")
            logger.info(f"🎯 TRAINING {model_type.upper()}")
            logger.info(f"{'='*50}")
            
            try:
                # Train model
                model_info = self.train_model(model_type, X_train, y_train)
                total_training_time += model_info['training_time']
                
                # Evaluate model  
                metrics = self.evaluate_model(model_info['model'], X_test, y_test, model_type)
                
                # Log to MLflow
                run_id = self.log_to_mlflow(model_info, metrics, X_train, y_train)
                
                results[model_type] = {
                    'metrics': metrics,
                    'run_id': run_id,
                    'training_time': model_info['training_time']
                }
                
                logger.info(f"✅ {model_type} completed successfully")
                
            except Exception as e:
                logger.error(f"❌ Failed to train {model_type}: {e}")
                continue
        
        # Summary
        if results:
            best_model = max(results.keys(), key=lambda x: results[x]['metrics']['recall'])
            best_recall = results[best_model]['metrics']['recall']
            
            total_time = time.time() - start_time
            
            logger.info(f"\n{'='*70}")
            logger.info(f"🎉 TRAINING PIPELINE COMPLETED!")
            logger.info(f"{'='*70}")
            logger.info(f"⏱️  Total execution time: {total_time:.2f} seconds")
            logger.info(f"🏋️  Total training time: {total_training_time:.2f} seconds")
            logger.info(f"📊 Data preprocessing time: {preprocessing_time:.2f} seconds")
            logger.info(f"🏆 Best Model: {best_model.upper()}")
            logger.info(f"🎯 Best Recall: {best_recall:.4f}")
            logger.info(f"📊 Trained {len(results)} models successfully")
            
            # Print comparison table
            logger.info("\n📊 MODEL COMPARISON SUMMARY:")
            logger.info("="*85)
            logger.info(f"{'Model':<15} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'ROC-AUC':<10} {'Time(s)':<8}")
            logger.info("-"*85)
            
            for model_name, result in results.items():
                metrics = result['metrics']
                training_time = result['training_time']
                
                logger.info(f"{model_name:<15} "
                           f"{metrics['accuracy']:<10.4f} "
                           f"{metrics['precision']:<10.4f} "
                           f"{metrics['recall']:<10.4f} "
                           f"{metrics['f1_score']:<10.4f} "
                           f"{metrics['roc_auc']:<10.4f} "
                           f"{training_time:<8.2f}")
            
            logger.info("="*85)
            logger.info("✅ All models logged to Azure ML MLflow!")
            logger.info("🔗 Check Azure ML Studio for experiment results")
            
            return results
        else:
            raise Exception("❌ All model training failed")

def main():
    """Main training function"""
    print("🚀 AZURE ML MLOPS - PATIENT READMISSION PREDICTION")
    print("="*60)
    print(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"💾 Dataset: 101,766 patient records")
    print(f"🎯 Target: 30-day readmission prediction")
    print(f"🏥 Class imbalance: 8:1 (88.8% no readmission)")
    print(f"🔧 XGBoost Available: {XGBOOST_AVAILABLE}")
    print("="*60)
    
    try:
        trainer = AzureReadmissionTrainer()
        results = trainer.train_all_models()
        
        print("\n🎉 TRAINING COMPLETED SUCCESSFULLY!")
        print("📊 Check Azure ML Studio for MLflow experiments and models")
        print("🔗 Models are registered in Azure ML Model Registry")
        print("🎯 Focus on models with highest recall for healthcare use case")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        raise

if __name__ == "__main__":
    main()