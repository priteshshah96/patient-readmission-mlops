# 🏥 Patient Readmission Prediction - Production MLOps Pipeline

[![Docker](https://img.shields.io/badge/Docker-Ready-blue)](https://www.docker.com/)
[![MLflow](https://img.shields.io/badge/MLflow-Model%20Registry-orange)](https://mlflow.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Production%20API-green)](https://fastapi.tiangolo.com/)
[![XGBoost](https://img.shields.io/badge/XGBoost-67.2%25%20Recall-red)](https://xgboost.readthedocs.io/)

> **Complete end-to-end MLOps pipeline for predicting 30-day hospital readmissions with production-ready Docker deployment, MLflow model registry, and FastAPI serving.**

## 🚀 **Docker Quick Start**

### **One-Command Deployment**
```bash
# Clone repository
git clone https://github.com/priteshshah96/patient-readmission-mlops
cd patient-readmission-mlops

# Setup environment
cp .env.example .env
# Edit .env with your Azure storage credentials

# Deploy everything with Docker
docker-compose up -d

# Train model
docker-compose exec api python -m src.models.train

# 🎉 Ready! Access services:
# - 🌐 API: http://localhost:8000
# - 📊 MLflow: http://localhost:5000  
# - 📖 Docs: http://localhost:8000/docs
# - 💚 Health: http://localhost:8000/health
```

### **What's Deployed**
- 🐳 **MLflow Server** - SQLite backend with Model Registry
- ⚡ **FastAPI API** - Production-ready with health monitoring  
- 🤖 **Complete Pipeline** - Training → Registry → Serving
- 📊 **Model Management** - Version control and staging

### **Test the System**
```bash
# Health check
curl http://localhost:8000/health/ready
# Response: {"status":"ready","model_loaded":true}

# Make prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "number_inpatient": 2,
    "number_emergency": 1,
    "time_in_hospital": 5,
    "number_diagnoses": 8,
    "num_medications": 15
  }'
# Response: {"prediction": 1, "probability": 0.742, "risk_level": "high"}
```

## 📊 **Model Performance**
- **🎯 Recall: 67.2%** - Identifies 2/3 of at-risk patients  
- **📈 ROC-AUC: 67.4%** - Strong discrimination ability
- **⚖️ Precision: 16.8%** - Balanced false positives
- **🏥 Healthcare-Optimized** - SMOTE handles class imbalance

## 🏗️ **Architecture**

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Pipeline │    │  MLflow Server  │    │                 │
│                 │    │                 │    │   FastAPI App   │
│ • Azure Blob    │───▶│ • Model Registry│───▶│                 │
│ • Data Cleaning │    │ • Experiment    │    │ • Health Checks │
│ • Feature Eng.  │    │   Tracking      │    │ • Predictions   │
│ • SMOTE         │    │ • Versioning    │    │ • Monitoring    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🛠️ **Technology Stack**

| Component | Technology | Purpose |
|-----------|------------|---------|
| **ML Framework** | XGBoost + scikit-learn | High-performance gradient boosting |
| **Class Balancing** | SMOTE | Handle imbalanced healthcare data |
| **Experiment Tracking** | MLflow | Model versioning & registry |
| **API Framework** | FastAPI | Production-ready API serving |
| **Database** | SQLite | MLflow backend storage |
| **Containerization** | Docker + Compose | Reproducible deployment |
| **Data Storage** | Azure Blob Storage | Scalable data pipeline |

## 📈 **Model Features**

The model uses **15 carefully selected features** for optimal performance:

1. `number_inpatient` - Previous inpatient visits
2. `number_emergency` - Emergency department visits  
3. `discharge_disposition_id` - How patient was discharged
4. `number_diagnoses` - Total number of diagnoses
5. `time_in_hospital` - Length of current stay
6. `num_medications` - Number of medications
7. `diabetesMed` - Diabetes medication prescribed
8. `metformin` - Specific diabetes medication
9. `num_lab_procedures` - Laboratory tests performed
10. `change` - Medication change during stay
11. `number_outpatient` - Outpatient visits
12. `age` - Patient age group
13. `num_procedures` - Medical procedures performed
14. `admission_type_id` - Type of admission
15. `repaglinide` - Another diabetes medication

## 🏭 **Production Features**

### Model Management
- **Automated Registration** - Models automatically registered in MLflow
- **Version Control** - Track model performance across versions
- **Stage Management** - Development → Staging → Production workflow
- **Rollback Capability** - Easy rollback to previous versions

### API Features
- **Input Validation** - Pydantic models ensure data quality
- **Error Handling** - Comprehensive error responses
- **Health Monitoring** - Kubernetes-style health checks
- **Performance Metrics** - Request timing and success rates
- **Batch Processing** - Handle multiple predictions efficiently

### Monitoring & Observability
- **Structured Logging** - JSON logs for easy parsing
- **Health Endpoints** - `/health/live`, `/health/ready`
- **Model Status** - Track model loading and performance
- **Request Tracking** - All API requests logged with timing

## 🔧 **Local Development**

### Prerequisites
- Python 3.11+
- UV package manager
- Docker & Docker Compose
- Azure Storage Account

### Setup
```bash
# Install dependencies
uv sync

# Start MLflow locally
mlflow ui --host 0.0.0.0 --port 5000

# Run API in development mode
uv run python -m src.api.main
```

### Model Training
```bash
# Train new model version
uv run python -m src.models.train

# Manage model registry
uv run python src/models/registry/model_management.py status
```

## 📊 **Project Structure**

```
├── src/
│   ├── api/                     # FastAPI application
│   │   ├── main.py             # FastAPI app & routing
│   │   ├── models.py           # Pydantic models
│   │   ├── prediction.py       # Prediction service
│   │   ├── health.py           # Health check endpoints
│   │   └── config.py           # Configuration
│   ├── data/
│   │   └── preprocessing.py    # Data pipeline
│   └── models/
│       ├── train.py            # Model training
│       └── registry/           # Model management
├── docker-compose.yml          # Service orchestration
├── Dockerfile                  # API container
├── .env.example               # Environment template
├── pyproject.toml             # Dependencies
└── README.md                  # This file
```

## 🔗 **API Endpoints**

### Health Checks
- `GET /health` - Basic health status
- `GET /health/ready` - Readiness probe
- `GET /health/live` - Liveness probe
- `GET /health/detailed` - Comprehensive system info

### Predictions
- `POST /predict` - Single patient prediction
- `POST /predict/batch` - Batch predictions

### Model Management
- `GET /model/info` - Model version and metrics
- `POST /model/reload` - Reload from registry

### Documentation
- `GET /docs` - Interactive API docs (Swagger)
- `GET /redoc` - Alternative documentation

## 🧪 **Testing**

### Health Check
```bash
curl http://localhost:8000/health/ready
```

### Single Prediction
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "patient_id": "12345",
    "patient_data": {
      "number_inpatient": 2,
      "number_emergency": 1,
      "discharge_disposition_id": 1,
      "number_diagnoses": 8,
      "time_in_hospital": 5,
      "num_medications": 15,
      "diabetesMed": 1,
      "metformin": 0,
      "num_lab_procedures": 20,
      "change": 1,
      "number_outpatient": 2,
      "age": 65,
      "num_procedures": 1,
      "admission_type_id": 1,
      "repaglinide": 0
    }
  }'
```

**Response:**
```json
{
  "patient_id": "12345",
  "prediction": 1,
  "probability": 0.742,
  "risk_level": "high",
  "confidence": 0.85,
  "model_version": "2"
}
```

## 🔒 **Security & Compliance**

### Data Protection
- Environment variables for sensitive data
- Input validation with Pydantic models
- Safe error messages without data leakage
- CORS configuration for controlled access

### Healthcare Compliance
- HIPAA considerations framework
- Model interpretability features
- Audit logging for all predictions
- Data minimization principles

## 🤝 **Contributing**

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Install dependencies: `uv sync`
4. Install pre-commit hooks: `uv run pre-commit install`
5. Make your changes
6. Run tests: `uv run pytest`
7. Submit pull request

## 📄 **License**

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 **Acknowledgments**

- **UCI Machine Learning Repository** for the diabetes dataset
- **MLflow Community** for experiment tracking tools
- **XGBoost Team** for the ML framework
- **FastAPI Team** for the API framework
- **Healthcare ML Community** for research insights

---

**Built with ❤️ for improving healthcare outcomes through machine learning**