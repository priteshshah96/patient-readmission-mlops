import logging
import time
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import mlflow
import mlflow.sklearn
from sklearn.base import BaseEstimator
import os 

from .models import PatientData, PredictionResponse, ReadmissionRisk
from .config import get_settings

logger = logging.getLogger(__name__)


class PredictionService:
    """Service for loading ML model and making predictions."""
    
    def __init__(self):
        self.settings = get_settings()
        self.model: Optional[BaseEstimator] = None
        self.model_version: Optional[str] = None
        self.model_info: Dict[str, Any] = {}
        self.feature_names: List[str] = [
            'number_inpatient', 'number_emergency', 'discharge_disposition_id',
            'number_diagnoses', 'time_in_hospital', 'num_medications',
            'diabetesMed', 'metformin', 'num_lab_procedures', 'change',
            'number_outpatient', 'age', 'num_procedures', 'admission_type_id', 'repaglinide'
        ]
        self.load_time: Optional[datetime] = None
        
    def load_model(self) -> bool:
        """Load model from MLflow registry."""
        try:
            logger.info(f"Loading model from MLflow registry...")
            logger.info(f"Model: {self.settings.model_name}, Stage: {self.settings.model_stage}")
            
            # Set MLflow tracking URI
            mlflow.set_tracking_uri(self.settings.mlflow_tracking_uri)
            
            # Load model from registry
            model_uri = f"models:/{self.settings.model_name}/{self.settings.model_stage}"
            
            start_time = time.time()
            self.model = mlflow.sklearn.load_model(model_uri)
            load_time = time.time() - start_time
            
            # Get model version info
            client = mlflow.tracking.MlflowClient()
            try:
                latest_versions = client.get_latest_versions(
                    name=self.settings.model_name,
                    stages=[self.settings.model_stage]
                )
                if latest_versions:
                    model_version_info = latest_versions[0]
                    self.model_version = model_version_info.version
                    
                    # Get run info for additional metadata
                    run = mlflow.get_run(model_version_info.run_id)
                    self.model_info = {
                        'version': self.model_version,
                        'stage': self.settings.model_stage,
                        'run_id': model_version_info.run_id,
                        'metrics': run.data.metrics,
                        'params': run.data.params,
                        'creation_time': model_version_info.creation_timestamp
                    }
                else:
                    self.model_version = "unknown"
                    self.model_info = {'version': 'unknown', 'stage': self.settings.model_stage}
                    
            except Exception as e:
                logger.warning(f"Could not get model version info: {e}")
                self.model_version = "unknown"
                self.model_info = {'version': 'unknown', 'stage': self.settings.model_stage}
            
            self.load_time = datetime.now()
            
            logger.info(f"✅ Model loaded successfully in {load_time:.2f}s")
            logger.info(f"   Model version: {self.model_version}")
            logger.info(f"   Model stage: {self.settings.model_stage}")
            
            # Test prediction to ensure model works
            self._test_model()
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            self.model = None
            self.model_version = None
            return False
    
    def _test_model(self) -> None:
        """Test model with dummy data to ensure it works."""
        try:
            # Create dummy patient data
            dummy_data = pd.DataFrame({
                'number_inpatient': [1],
                'number_emergency': [0],
                'discharge_disposition_id': [1],
                'number_diagnoses': [5],
                'time_in_hospital': [3],
                'num_medications': [15],
                'diabetesMed': [1],
                'metformin': [1],
                'num_lab_procedures': [20],
                'change': [0],
                'number_outpatient': [2],
                'age': [65],
                'num_procedures': [1],
                'admission_type_id': [1],
                'repaglinide': [0]
            })
            
            # Test prediction
            prediction = self.model.predict(dummy_data)
            probabilities = self.model.predict_proba(dummy_data)
            
            logger.info(f"✅ Model test successful - Prediction: {prediction[0]}, Probability: {probabilities[0][1]:.3f}")
            
        except Exception as e:
            logger.error(f"❌ Model test failed: {e}")
            raise
    
    def is_model_loaded(self) -> bool:
        """Check if model is loaded and ready."""
        return self.model is not None
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        if not self.is_model_loaded():
            return {"status": "Model not loaded"}
        
        return {
            "model_name": self.settings.model_name,
            "model_version": self.model_version,
            "model_stage": self.settings.model_stage,
            "features_count": len(self.feature_names),
            "features_used": self.feature_names,
            "load_time": self.load_time.isoformat() if self.load_time else None,
            "model_info": self.model_info
        }
    
    def _prepare_features(self, patient_data: PatientData) -> pd.DataFrame:
        """Convert patient data to model input format."""
        # Create DataFrame with features in correct order
        features_dict = {
            'number_inpatient': patient_data.number_inpatient,
            'number_emergency': patient_data.number_emergency,
            'discharge_disposition_id': patient_data.discharge_disposition_id,
            'number_diagnoses': patient_data.number_diagnoses,
            'time_in_hospital': patient_data.time_in_hospital,
            'num_medications': patient_data.num_medications,
            'diabetesMed': patient_data.diabetesMed,
            'metformin': patient_data.metformin,
            'num_lab_procedures': patient_data.num_lab_procedures,
            'change': patient_data.change,
            'number_outpatient': patient_data.number_outpatient,
            'age': patient_data.age,
            'num_procedures': patient_data.num_procedures,
            'admission_type_id': patient_data.admission_type_id,
            'repaglinide': patient_data.repaglinide
        }
        
        return pd.DataFrame([features_dict])
    
    def _get_risk_level(self, probability: float) -> ReadmissionRisk:
        """Determine risk level based on probability."""
        if probability >= 0.7:
            return ReadmissionRisk.HIGH
        elif probability >= 0.4:
            return ReadmissionRisk.MEDIUM
        else:
            return ReadmissionRisk.LOW
    
    def _get_confidence(self, probability: float) -> str:
        """Get confidence level description."""
        if probability >= 0.8 or probability <= 0.2:
            return "High"
        elif probability >= 0.6 or probability <= 0.4:
            return "Medium"
        else:
            return "Low"
    
    def predict_single(self, patient_data: PatientData, patient_id: Optional[str] = None) -> PredictionResponse:
        """Make prediction for a single patient."""
        if not self.is_model_loaded():
            raise RuntimeError("Model not loaded. Please load model first.")
        
        try:
            # Prepare features
            features_df = self._prepare_features(patient_data)
            
            # Make prediction
            start_time = time.time()
            prediction = self.model.predict(features_df)[0]
            probabilities = self.model.predict_proba(features_df)[0]
            prediction_time = time.time() - start_time
            
            # Get probability of positive class (readmission)
            probability = float(probabilities[1])
            
            # Determine risk level and confidence
            risk_level = self._get_risk_level(probability)
            confidence = self._get_confidence(probability)
            
            logger.debug(f"Prediction completed in {prediction_time:.3f}s - "
                        f"Result: {prediction}, Probability: {probability:.3f}, Risk: {risk_level}")
            
            return PredictionResponse(
                patient_id=patient_id,
                prediction=int(prediction),
                probability=probability,
                risk_level=risk_level,
                confidence=confidence,
                model_version=str(self.model_version) if self.model_version else "unknown"
            )
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise RuntimeError(f"Prediction failed: {str(e)}")
    
    def predict_batch(self, patients_data: List[PatientData]) -> Tuple[List[PredictionResponse], Dict[str, Any]]:
        """Make predictions for multiple patients."""
        if not self.is_model_loaded():
            raise RuntimeError("Model not loaded. Please load model first.")
        
        if len(patients_data) > self.settings.batch_size_limit:
            raise ValueError(f"Batch size {len(patients_data)} exceeds limit {self.settings.batch_size_limit}")
        
        try:
            start_time = time.time()
            
            # Prepare all features at once
            features_list = []
            for patient_data in patients_data:
                features_dict = {
                    'number_inpatient': patient_data.number_inpatient,
                    'number_emergency': patient_data.number_emergency,
                    'discharge_disposition_id': patient_data.discharge_disposition_id,
                    'number_diagnoses': patient_data.number_diagnoses,
                    'time_in_hospital': patient_data.time_in_hospital,
                    'num_medications': patient_data.num_medications,
                    'diabetesMed': patient_data.diabetesMed,
                    'metformin': patient_data.metformin,
                    'num_lab_procedures': patient_data.num_lab_procedures,
                    'change': patient_data.change,
                    'number_outpatient': patient_data.number_outpatient,
                    'age': patient_data.age,
                    'num_procedures': patient_data.num_procedures,
                    'admission_type_id': patient_data.admission_type_id,
                    'repaglinide': patient_data.repaglinide
                }
                features_list.append(features_dict)
            
            features_df = pd.DataFrame(features_list)
            
            # Make batch predictions
            predictions = self.model.predict(features_df)
            probabilities = self.model.predict_proba(features_df)
            
            processing_time = time.time() - start_time
            
            # Create response objects
            responses = []
            risk_counts = {"high": 0, "medium": 0, "low": 0}
            
            for i, (prediction, prob_array) in enumerate(zip(predictions, probabilities)):
                probability = float(prob_array[1])
                risk_level = self._get_risk_level(probability)
                confidence = self._get_confidence(probability)
                
                risk_counts[risk_level.value] += 1
                
                response = PredictionResponse(
                    patient_id=f"patient_{i}",
                    prediction=int(prediction),
                    probability=probability,
                    risk_level=risk_level,
                    confidence=confidence,
                    model_version=self.model_version or "unknown"
                )
                responses.append(response)
            
            batch_info = {
                "total_patients": len(patients_data),
                "high_risk_count": risk_counts["high"],
                "medium_risk_count": risk_counts["medium"],
                "low_risk_count": risk_counts["low"],
                "processing_time_ms": processing_time * 1000
            }
            
            logger.info(f"Batch prediction completed for {len(patients_data)} patients in {processing_time:.3f}s")
            
            return responses, batch_info
            
        except Exception as e:
            logger.error(f"Batch prediction failed: {e}")
            raise RuntimeError(f"Batch prediction failed: {str(e)}")


# Global prediction service instance
prediction_service = PredictionService()


def get_prediction_service() -> PredictionService:
    """Get the global prediction service instance."""
    return prediction_service