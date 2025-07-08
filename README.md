# Patient Readmission Prediction - MLOps Pipeline

A production-ready machine learning system for predicting 30-day hospital readmissions using XGBoost and MLflow model registry.

## Overview

This project implements an end-to-end MLOps pipeline for healthcare readmission prediction, focusing on high recall to identify at-risk patients. The system uses advanced techniques like SMOTE for class imbalance and MLflow for model lifecycle management.

## Features

- **XGBoost Classification** with SMOTE for handling imbalanced healthcare data
- **MLflow Model Registry** for version control and deployment management
- **FastAPI Service** with comprehensive prediction endpoints
- **Production-Ready Model Management** with CLI tools
- **Healthcare-Focused Metrics** prioritizing recall for patient safety
- **Comprehensive Logging** and experiment tracking
- **Type-Safe Code** with MyPy and modern Python practices
- **Azure Blob Storage Integration** for data pipeline

## Technology Stack

- **ML Framework**: XGBoost, scikit-learn, imbalanced-learn
- **Experiment Tracking**: MLflow
- **API Framework**: FastAPI
- **Data Storage**: Azure Blob Storage
- **Development**: Python 3.11+, UV package manager
- **Code Quality**: Ruff, MyPy, pre-commit hooks
- **Data Processing**: Pandas, Polars, NumPy

## Project Structure

```
├── src/
│   ├── data/
│   │   └── preprocessing.py          # Data cleaning and feature engineering
│   ├── models/
│   │   ├── train.py                 # Main training pipeline
│   │   ├── train_azure.py           # Azure ML specific training
│   │   └── registry/                # Model registry management
│   │       ├── model_registry.py        # MLflow model versioning
│   │       └── model_management.py      # CLI for model deployment
│   └── api/                         # FastAPI service endpoints
│       ├── main.py                  # FastAPI application
│       ├── models.py                # Pydantic models
│       ├── config.py                # Configuration settings
│       ├── prediction.py            # Prediction service
│       └── health.py                # Health check endpoints
├── tests/                           # Unit and integration tests
├── notebooks/                       # Data exploration and analysis
├── configs/                         # Configuration files
├── infrastructure/                  # Docker and deployment configs
├── logs/                           # Application and training logs
├── reports/                        # Generated model reports
├── download_data.py                # Data pipeline script
├── pyproject.toml                  # Project dependencies
├── .env.example                    # Environment variables template
├── .gitignore                      # Git ignore rules
├── .pre-commit-config.yaml         # Code quality hooks
└── README.md                       # Project documentation
```

## Quick Start

### Prerequisites

- Python 3.11+
- UV package manager (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
- Azure Storage Account (for data storage)
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd patient-readmission-mlops
   ```

2. **Set up environment**
   ```bash
   # Install UV package manager
   curl -LsSf https://astral.sh/uv/install.sh | sh
   
   # Create virtual environment and install dependencies
   uv sync
   
   # Install pre-commit hooks
   uv run pre-commit install
   ```

3. **Configure environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your Azure credentials and MLflow settings
   ```

4. **Download and prepare data**
   ```bash
   uv run python download_data.py
   ```

## API Service Usage

### Start the API Server

```bash
# Development mode with auto-reload
uv run uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

# Production mode
uv run python -m src.api.main
```

### API Endpoints

**Base URL**: `http://localhost:8000`

#### Health Checks
- `GET /health` - Basic health status
- `GET /health/detailed` - Comprehensive system information
- `GET /health/ready` - Kubernetes-style readiness probe
- `GET /health/live` - Kubernetes-style liveness probe

#### Predictions
- `POST /predict` - Single patient prediction
- `POST /predict/batch` - Batch predictions (up to 100 patients)

#### Model Management
- `GET /model/info` - Model version and performance metrics
- `POST /model/reload` - Reload model from MLflow registry

#### Documentation
- `GET /docs` - Interactive API documentation (Swagger UI)
- `GET /redoc` - Alternative API documentation

### Example API Usage

**Single Prediction:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "patient_id": "12345",
    "patient_data": {
      "race": "Caucasian",
      "gender": "Female",
      "age": "[70-80)",
      "admission_type_id": 1,
      "discharge_disposition_id": 1,
      "admission_source_id": 7,
      "time_in_hospital": 1,
      "num_lab_procedures": 41,
      "num_procedures": 0,
      "num_medications": 1,
      "number_outpatient": 0,
      "number_emergency": 0,
      "number_inpatient": 0,
      "diag_1": "250.83",
      "diag_2": "250.01",
      "diag_3": "255",
      "number_diagnoses": 1,
      "max_glu_serum": "None",
      "A1Cresult": "None",
      "metformin": "No",
      "repaglinide": "No",
      "nateglinide": "No",
      "chlorpropamide": "No",
      "glimepiride": "No",
      "acetohexamide": "No",
      "glipizide": "No",
      "glyburide": "No",
      "tolbutamide": "No",
      "pioglitazone": "No",
      "rosiglitazone": "No",
      "acarbose": "No",
      "miglitol": "No",
      "troglitazone": "No",
      "tolazamide": "No",
      "examide": "No",
      "citoglipton": "No",
      "insulin": "Down",
      "glyburide_metformin": "No",
      "glipizide_metformin": "No",
      "glimepiride_pioglitazone": "No",
      "metformin_rosiglitazone": "No",
      "metformin_pioglitazone": "No",
      "change": "Ch",
      "diabetesMed": "Yes"
    }
  }'
```

**Response:**
```json
{
  "patient_id": "12345",
  "prediction": 0,
  "probability": 0.23,
  "risk_level": "low",
  "confidence": 0.85,
  "model_version": "v1.2.0"
}
```

## Training Pipeline

### Local Training

```bash
# Start MLflow server
mlflow ui --host 0.0.0.0 --port 5000

# Run training pipeline
uv run python -m src.models.train

# Check model registry
uv run python src/models/registry/model_management.py status
```

### Model Management CLI

```bash
# List all model versions
uv run python src/models/registry/model_management.py list

# Deploy to production
uv run python src/models/registry/model_management.py deploy --version 3

# Validate model for production
uv run python src/models/registry/model_management.py validate --version 3

# Rollback production
uv run python src/models/registry/model_management.py rollback

# Compare model versions
uv run python src/models/registry/model_management.py compare --version1 2 --version2 3
```

## Deployment Options

### Local Development
```bash
# Start MLflow server
mlflow ui --host 0.0.0.0 --port 5000

# Start API server
uv run python -m src.api.main

# Access endpoints
curl http://localhost:8000/health
curl http://localhost:8000/docs
```

### Production Deployment (Next Phase)

1. **Container Deployment**
   - Docker containerization
   - Multi-stage builds for optimization
   - Health checks and monitoring

2. **Azure Container Apps**
   - Serverless container deployment
   - Auto-scaling based on demand
   - Integrated monitoring

3. **CI/CD Pipeline**
   - GitHub Actions workflow
   - Automated testing and deployment
   - Model validation gates

## Monitoring and Observability

### Logging
- **Structured Logging**: JSON format for easy parsing
- **Multiple Log Levels**: DEBUG, INFO, WARNING, ERROR
- **Request Tracking**: All API requests logged with timing
- **Error Handling**: Comprehensive exception handling

### Health Monitoring
- **Kubernetes-style probes**: `/health/ready` and `/health/live`
- **Model Status**: Monitor model loading and performance
- **System Metrics**: CPU, memory, and disk usage
- **Processing Time**: Request timing headers

### API Metrics
- **Response Times**: X-Process-Time header on all responses
- **Error Rates**: Automatic error tracking and logging
- **Model Performance**: Track prediction accuracy over time
- **Batch Processing**: Monitor batch prediction efficiency

## Security and Compliance

### Data Protection
- **Environment Variables**: Sensitive data in .env files
- **Input Validation**: Pydantic models for request validation
- **Error Sanitization**: Safe error messages without data leakage
- **CORS Configuration**: Controlled cross-origin access

### Healthcare Compliance
- **HIPAA Considerations**: Framework for healthcare data handling
- **Model Interpretability**: Feature importance and decision explanations
- **Audit Logging**: Track all predictions and model changes
- **Data Minimization**: Only required patient features collected

## Performance Characteristics

### Model Performance
- **Recall**: Optimized for high recall (0.85+) to identify at-risk patients
- **Precision**: Balanced precision to minimize false positives
- **ROC-AUC**: Consistent performance across risk thresholds
- **Class Imbalance**: SMOTE technique handles imbalanced datasets

### API Performance
- **Single Predictions**: < 100ms response time
- **Batch Processing**: Up to 100 patients per request
- **Memory Efficiency**: Model pre-loading for faster predictions
- **Scalability**: Stateless design for horizontal scaling

## Troubleshooting

### Common Issues

**API won't start**
```bash
# Check if model is available
uv run python -c "from src.api.prediction import get_prediction_service; print(get_prediction_service().is_model_loaded())"

# Check MLflow connection
uv run python -c "import mlflow; print(mlflow.get_tracking_uri())"
```

**Model not loading**
```bash
# Check MLflow server
curl http://localhost:5000

# Verify model in registry
uv run python src/models/registry/model_management.py status
```

**Prediction errors**
```bash
# Check API logs
tail -f logs/api.log

# Test health endpoint
curl http://localhost:8000/health/detailed
```

### Performance Optimization

- **Memory Usage**: Model pre-loading reduces prediction latency
- **Batch Processing**: Use batch endpoint for multiple predictions
- **Async Processing**: FastAPI async endpoints for better concurrency
- **Caching**: Pre-computed features reduce processing time

## Contributing

We welcome contributions! Please follow these guidelines:

### Getting Started
1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Set up development environment: `uv sync`
4. Install pre-commit hooks: `uv run pre-commit install`

### Development Standards
- Follow PEP 8 style guidelines (enforced by Ruff)
- Add type hints for all functions
- Write comprehensive tests for new features
- Update documentation for API changes
- Ensure all pre-commit hooks pass

### Pull Request Process
1. Update README.md with details of changes
2. Ensure all tests pass: `uv run pytest`
3. Run code quality checks: `uv run pre-commit run --all-files`
4. Update version numbers following semantic versioning
5. Create pull request with clear description

### Code Review Guidelines
- Focus on code clarity and maintainability
- Verify healthcare domain expertise in medical features
- Ensure proper error handling and logging
- Check for potential security vulnerabilities

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this project in academic research, please cite:

```bibtex
@software{patient_readmission_mlops,
  title={Patient Readmission Prediction MLOps Pipeline},
  author={[Pritesh Shah]},
  year={2025},
  url={https://github.com/[priteshshah96]/patient-readmission-mlops}
}
```

## Acknowledgments

- **UCI Machine Learning Repository** for providing the diabetes dataset
- **MLflow Community** for excellent experiment tracking tools
- **XGBoost Team** for the high-performance gradient boosting framework
- **FastAPI Team** for the excellent API framework
- **Healthcare ML Community** for research on readmission prediction
- **SMOTE Authors** for class imbalance handling techniques

## Support and Contact

- **Issues**: Report bugs and request features via [GitHub Issues](../../issues)
- **Discussions**: Ask questions in [GitHub Discussions](../../discussions)
- **Documentation**: Full API documentation available at `/docs`
- **Email**: [priteshshahwork@gmail.com] for critical issues

## Roadmap

### Phase 1: Core ML Pipeline ✅
- [x] Data preprocessing and feature engineering
- [x] XGBoost model training with SMOTE
- [x] MLflow experiment tracking
- [x] Model registry and lifecycle management

### Phase 2: API Development ✅
- [x] FastAPI service for model serving
- [x] Input validation and error handling
- [x] Health check endpoints
- [x] API documentation with OpenAPI
- [x] Prediction endpoints (single and batch)
- [x] Model management endpoints

### Phase 3: Deployment 🚧
- [ ] Docker containerization
- [ ] Azure Container Apps deployment
- [ ] CI/CD pipeline with GitHub Actions
- [ ] Production monitoring and alerting

### Phase 4: Advanced Features 📋
- [ ] Model retraining pipeline
- [ ] A/B testing framework
- [ ] Real-time prediction streaming
- [ ] Advanced model interpretability

---

**Built with ❤️ for improving healthcare outcomes through machine learning**