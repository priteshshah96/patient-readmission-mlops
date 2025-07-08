
"""
Patient Readmission Prediction API

A production-ready FastAPI service for serving XGBoost readmission predictions.
"""

__version__ = "1.0.0"
__author__ = "Patient Readmission MLOps Team"

from .main import app
from .config import get_settings
from .prediction import get_prediction_service

__all__ = [
    "app",
    "get_settings", 
    "get_prediction_service"
]