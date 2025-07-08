import logging
import time
import traceback
from datetime import datetime
from typing import List
import uvicorn

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exception_handlers import http_exception_handler

from src.api.models import (
    PredictionRequest, PredictionResponse, BatchPredictionRequest, 
    BatchPredictionResponse, ModelInfoResponse, ErrorResponse
)
from src.api.prediction import get_prediction_service, PredictionService
from src.api.health import health_router
from src.api.config import get_settings, Settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize settings
settings = get_settings()

# Create FastAPI app
app = FastAPI(
    title="Patient Readmission Prediction API",
    description="""
    🏥 **Healthcare ML API for 30-Day Readmission Prediction**
    
    This API serves a production-ready XGBoost model that predicts the likelihood 
    of patient readmission within 30 days of discharge.
    
    ## Key Features
    - **High Recall Model**: Optimized to catch 67%+ of readmissions
    - **Real-time Predictions**: Single patient and batch prediction endpoints
    - **Production Ready**: Model versioning with MLflow registry
    - **Healthcare Focused**: Risk levels and confidence scores
    - **Comprehensive Monitoring**: Health checks and detailed metrics
    
    ## Model Information
    - **Algorithm**: XGBoost with SMOTE for class balancing
    - **Features**: 15 most predictive patient and encounter features
    - **Target**: 30-day readmission prediction (binary classification)
    - **Performance**: Prioritizes recall for patient safety
    
    ## Usage
    1. Use `/predict` for single patient predictions
    2. Use `/predict/batch` for multiple patients (up to 100)
    3. Monitor service health with `/health` endpoints
    4. Get model information with `/model/info`
    """,
    version=settings.app_version,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Include health check router
app.include_router(health_router)

# Request tracking middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests for monitoring."""
    start_time = time.time()
    
    # Process request
    response = await call_next(request)
    
    # Log request details
    process_time = time.time() - start_time
    logger.info(
        f"{request.method} {request.url.path} - "
        f"Status: {response.status_code} - "
        f"Time: {process_time:.3f}s"
    )
    
    # Add timing header
    response.headers["X-Process-Time"] = str(process_time)
    
    return response


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle all unhandled exceptions."""
    logger.error(f"Unhandled exception: {exc}")
    logger.error(f"Traceback: {traceback.format_exc()}")
    
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            error_code="INTERNAL_ERROR",
            request_id=str(time.time())
        ).dict()
    )


# Custom HTTP exception handler
@app.exception_handler(HTTPException)
async def custom_http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with custom error format."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            error_code=f"HTTP_{exc.status_code}",
            request_id=str(time.time())
        ).dict()
    )


@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup."""
    logger.info("🚀 Starting Patient Readmission Prediction API")
    logger.info(f"Version: {settings.app_version}")
    logger.info(f"Model: {settings.model_name}")
    logger.info(f"Stage: {settings.model_stage}")
    
    # Load the ML model
    prediction_service = get_prediction_service()
    success = prediction_service.load_model()
    
    if success:
        logger.info("✅ Model loaded successfully - API ready!")
    else:
        logger.error("❌ Failed to load model - API starting in degraded mode")


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up on shutdown."""
    logger.info("🛑 Shutting down Patient Readmission Prediction API")


@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint redirect to docs."""
    return {
        "message": "Patient Readmission Prediction API",
        "version": settings.app_version,
        "docs": "/docs",
        "health": "/health",
        "model_info": "/model/info"
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_readmission(
    request: PredictionRequest,
    prediction_service: PredictionService = Depends(get_prediction_service)
) -> PredictionResponse:
    """
    Predict readmission risk for a single patient.
    
    Returns prediction with probability, risk level, and confidence score.
    Optimized for high recall to identify at-risk patients.
    """
    try:
        if not prediction_service.is_model_loaded():
            raise HTTPException(
                status_code=503,
                detail="Model not loaded. Service unavailable."
            )
        
        # Make prediction
        result = prediction_service.predict_single(
            request.patient_data, 
            request.patient_id
        )
        
        logger.info(f"Prediction completed - Patient: {request.patient_id}, "
                   f"Risk: {result.risk_level}, Probability: {result.probability:.3f}")
        
        return result
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch_readmission(
    request: BatchPredictionRequest,
    background_tasks: BackgroundTasks,
    prediction_service: PredictionService = Depends(get_prediction_service)
) -> BatchPredictionResponse:
    """
    Predict readmission risk for multiple patients (up to 100).
    
    Returns predictions with summary statistics and risk distribution.
    Useful for batch processing and population health analysis.
    """
    try:
        if not prediction_service.is_model_loaded():
            raise HTTPException(
                status_code=503,
                detail="Model not loaded. Service unavailable."
            )
        
        if len(request.patients) > settings.batch_size_limit:
            raise HTTPException(
                status_code=400,
                detail=f"Batch size {len(request.patients)} exceeds limit {settings.batch_size_limit}"
            )
        
        # Make batch predictions
        predictions, batch_info = prediction_service.predict_batch(request.patients)
        
        # Log batch processing
        def log_batch_result():
            logger.info(f"Batch prediction completed - "
                       f"Patients: {batch_info['total_patients']}, "
                       f"High risk: {batch_info['high_risk_count']}, "
                       f"Time: {batch_info['processing_time_ms']:.0f}ms")
        
        background_tasks.add_task(log_batch_result)
        
        return BatchPredictionResponse(
            predictions=predictions,
            **batch_info
        )
        
    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Batch prediction failed: {str(e)}"
        )


@app.get("/model/info", response_model=ModelInfoResponse)
async def get_model_info(
    prediction_service: PredictionService = Depends(get_prediction_service)
) -> ModelInfoResponse:
    """
    Get information about the loaded ML model.
    
    Returns model version, performance metrics, and feature information.
    """
    try:
        if not prediction_service.is_model_loaded():
            raise HTTPException(
                status_code=503,
                detail="Model not loaded. Service unavailable."
            )
        
        model_info = prediction_service.get_model_info()
        metrics = model_info.get('model_info', {}).get('metrics', {})
        
        return ModelInfoResponse(
            model_name=model_info['model_name'],
            model_version=model_info['model_version'],
            model_stage=model_info['model_stage'],
            features_count=model_info['features_count'],
            features_used=model_info['features_used'],
            performance_metrics=metrics,
            last_updated=datetime.fromisoformat(model_info['load_time']) if model_info.get('load_time') else datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Failed to get model info: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get model info: {str(e)}"
        )


@app.post("/model/reload")
async def reload_model(
    prediction_service: PredictionService = Depends(get_prediction_service)
):
    """
    Reload the ML model from MLflow registry.
    
    Useful for deploying new model versions without restarting the service.
    """
    try:
        logger.info("🔄 Reloading model from MLflow registry...")
        success = prediction_service.load_model()
        
        if success:
            return {
                "status": "success",
                "message": "Model reloaded successfully",
                "model_version": prediction_service.model_version,
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise HTTPException(
                status_code=500,
                detail="Failed to reload model"
            )
            
    except Exception as e:
        logger.error(f"Model reload failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Model reload failed: {str(e)}"
        )


# Development server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",  # Fixed: use full module path
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )