# FIXED: src/api/config.py
from pydantic_settings import BaseSettings
from typing import Optional
import os


class Settings(BaseSettings):
    """Application configuration settings."""
    
    # API Configuration
    app_name: str = "Patient Readmission Prediction API"
    app_version: str = "1.0.0"
    debug: bool = False
    
    # MLflow Configuration - FIXED: Default to Docker MLflow
    mlflow_tracking_uri: str = "http://mlflow:5000"  # Changed from "file:./mlruns"
    model_name: str = "patient_readmission_predictor"
    model_stage: str = "Production"  # or "Staging"
    
    # Model Configuration
    prediction_threshold: float = 0.5
    batch_size_limit: int = 100
    
    # API Security
    api_key: Optional[str] = None
    cors_origins: list = ["*"]
    
    # Logging
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Health Check
    health_check_interval: int = 60  # seconds
    
    # Azure Storage (from your .env file)
    azure_storage_connection_string: Optional[str] = None
    azure_storage_account_name: Optional[str] = None
    azure_container_name: Optional[str] = None
    project_name: Optional[str] = None
    data_source_url: Optional[str] = None
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "allow"  # Allow extra fields from .env


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get application settings."""
    return settings