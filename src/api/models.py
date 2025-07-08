from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


class ReadmissionRisk(str, Enum):
    """Risk levels for readmission prediction."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class PatientData(BaseModel):
    """Input schema for patient data used in readmission prediction.
    
    Based on the top 15 features selected by the ML pipeline.
    """
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "number_inpatient": 1,
                "number_emergency": 0,
                "discharge_disposition_id": 1,
                "number_diagnoses": 5,
                "time_in_hospital": 3,
                "num_medications": 15,
                "diabetesMed": 1,
                "metformin": 1,
                "num_lab_procedures": 20,
                "change": 0,
                "number_outpatient": 2,
                "age": 65,
                "num_procedures": 1,
                "admission_type_id": 1,
                "repaglinide": 0
            }
        }
    )
    
    # Patient history features
    number_inpatient: int = Field(
        ..., 
        ge=0, 
        le=10,
        description="Number of inpatient visits in the year prior"
    )
    
    number_emergency: int = Field(
        ..., 
        ge=0, 
        le=10,
        description="Number of emergency visits in the year prior"
    )
    
    number_outpatient: int = Field(
        ..., 
        ge=0, 
        le=50,
        description="Number of outpatient visits in the year prior"
    )
    
    # Current encounter features
    discharge_disposition_id: int = Field(
        ..., 
        ge=1, 
        le=30,
        description="Discharge disposition code"
    )
    
    admission_type_id: int = Field(
        ..., 
        ge=1, 
        le=8,
        description="Type of admission"
    )
    
    number_diagnoses: int = Field(
        ..., 
        ge=1, 
        le=20,
        description="Number of diagnoses"
    )
    
    time_in_hospital: int = Field(
        ..., 
        ge=1, 
        le=14,
        description="Length of stay in days"
    )
    
    num_medications: int = Field(
        ..., 
        ge=0, 
        le=50,
        description="Number of medications"
    )
    
    num_lab_procedures: int = Field(
        ..., 
        ge=0, 
        le=100,
        description="Number of lab procedures"
    )
    
    num_procedures: int = Field(
        ..., 
        ge=0, 
        le=10,
        description="Number of procedures"
    )
    
    age: int = Field(
        ..., 
        ge=0, 
        le=100,
        description="Patient age"
    )
    
    # Medication features (binary: 0 or 1)
    diabetesMed: int = Field(
        ..., 
        ge=0, 
        le=1,
        description="Diabetes medication prescribed (0=No, 1=Yes)"
    )
    
    metformin: int = Field(
        ..., 
        ge=0, 
        le=1,
        description="Metformin prescribed (0=No, 1=Yes)"
    )
    
    repaglinide: int = Field(
        ..., 
        ge=0, 
        le=1,
        description="Repaglinide prescribed (0=No, 1=Yes)"
    )
    
    change: int = Field(
        ..., 
        ge=0, 
        le=1,
        description="Change in diabetic medications (0=No, 1=Yes)"
    )


class PredictionRequest(BaseModel):
    """Request schema for single patient prediction."""
    patient_data: PatientData
    return_probabilities: bool = Field(
        default=True,
        description="Whether to return prediction probabilities"
    )
    patient_id: Optional[str] = Field(
        default=None,
        description="Optional patient identifier for tracking"
    )


class PredictionResponse(BaseModel):
    """Response schema for single patient prediction."""
    patient_id: Optional[str]
    prediction: int = Field(description="Predicted readmission (0=No, 1=Yes)")
    probability: float = Field(description="Probability of readmission")
    risk_level: ReadmissionRisk = Field(description="Risk level category")
    confidence: str = Field(description="Confidence level")
    timestamp: datetime = Field(default_factory=datetime.now)
    model_version: str = Field(description="Model version used")
    
    model_config = ConfigDict(
    json_encoders={
        datetime: lambda v: v.isoformat()
    }
)


class BatchPredictionRequest(BaseModel):
    """Request schema for batch predictions."""
    patients: List[PatientData] = Field(
        ...,
        max_items=100,
        description="List of patient data (max 100 patients)"
    )
    return_probabilities: bool = Field(
        default=True,
        description="Whether to return prediction probabilities"
    )


class BatchPredictionResponse(BaseModel):
    """Response schema for batch predictions."""
    predictions: List[PredictionResponse]
    total_patients: int
    high_risk_count: int
    medium_risk_count: int
    low_risk_count: int
    processing_time_ms: float
    timestamp: datetime = Field(default_factory=datetime.now)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ModelInfoResponse(BaseModel):
    """Response schema for model information."""
    model_name: str
    model_version: str
    model_stage: str
    features_count: int
    features_used: List[str]
    performance_metrics: Dict[str, float]
    last_updated: datetime
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class HealthResponse(BaseModel):
    """Response schema for health check."""
    status: str = Field(description="Service status")
    timestamp: datetime = Field(default_factory=datetime.now)
    model_loaded: bool = Field(description="Whether ML model is loaded")
    model_version: Optional[str] = Field(description="Loaded model version")
    uptime_seconds: float = Field(description="Service uptime in seconds")
    memory_usage_mb: float = Field(description="Memory usage in MB")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ErrorResponse(BaseModel):
    """Error response schema."""
    error: str = Field(description="Error message")
    error_code: str = Field(description="Error code")
    timestamp: datetime = Field(default_factory=datetime.now)
    request_id: Optional[str] = Field(description="Request ID for tracking")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }