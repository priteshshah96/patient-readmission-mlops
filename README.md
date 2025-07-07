# Patient Readmission Prediction - MLOps Pipeline

A production-ready machine learning system for predicting 30-day hospital readmissions using XGBoost and MLflow model registry.

## Overview

This project implements an end-to-end MLOps pipeline for healthcare readmission prediction, focusing on high recall to identify at-risk patients. The system uses advanced techniques like SMOTE for class imbalance and MLflow for model lifecycle management.

## Features

- **XGBoost Classification** with SMOTE for handling imbalanced healthcare data
- **MLflow Model Registry** for version control and deployment management
- **Automated Data Pipeline** with Azure Blob Storage integration
- **Production-Ready Model Management** with CLI tools
- **Healthcare-Focused Metrics** prioritizing recall for patient safety
- **Comprehensive Logging** and experiment tracking
- **Type-Safe Code** with MyPy and modern Python practices

## Technology Stack

- **ML Framework**: XGBoost, scikit-learn, imbalanced-learn
- **Experiment Tracking**: MLflow
- **Data Storage**: Azure Blob Storage
- **API Framework**: FastAPI (planned)
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
│   └── api/                         # FastAPI service endpoints (planned)
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
   # Install dependencies
   uv sync
   
   # Activate virtual environment
   source .venv/bin/activate  # Linux/Mac
   # or .venv\Scripts\activate  # Windows
   ```

3. **Configure environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your Azure storage credentials
   ```

4. **Install development tools**
   ```bash
   uv run pre-commit install
   ```

### Data Setup

1. **Download and prepare data**
   ```bash
   python download_data.py
   ```
   This will download the UCI Diabetes dataset and upload it to Azure Blob Storage.

2. **Verify data pipeline**
   ```bash
   python -c "from src.data.preprocessing import DataPreprocessor; dp = DataPreprocessor(); print('Data pipeline ready')"
   ```

### Model Training

1. **Train the XGBoost model**
   ```bash
   python -m src.models.train
   ```
   This will:
   - Load and preprocess the data
   - Train XGBoost with SMOTE for class balancing
   - Log experiments to MLflow
   - Evaluate model performance with healthcare metrics

2. **Register model in MLflow**
   ```bash
   python src/models/registry/model_registry.py
   ```

3. **Start MLflow UI**
   ```bash
   mlflow ui --host 0.0.0.0 --port 5000
   ```
   Access at `http://localhost:5000` to view experiments and models.

### Model Management

The project includes a comprehensive CLI for model lifecycle management:

```bash
# Check model registry status
python src/models/registry/model_management.py status

# Register best model from experiments
python src/models/registry/model_management.py register

# Validate model for production
python src/models/registry/model_management.py validate --version 1

# Deploy model to production
python src/models/registry/model_management.py deploy --version 1

# Compare model versions
python src/models/registry/model_management.py compare --version 1 --version2 2

# Generate comprehensive model report
python src/models/registry/model_management.py report

# Archive old model versions
python src/models/registry/model_management.py archive --keep 3

# Rollback production model
python src/models/registry/model_management.py rollback --version 1
```

## Configuration

### Environment Variables

Create a `.env` file based on `.env.example`:

```bash
# Azure Storage Configuration
AZURE_STORAGE_CONNECTION_STRING=DefaultEndpointsProtocol=https;AccountName=...
AZURE_STORAGE_ACCOUNT_NAME=your_account_name
AZURE_CONTAINER_NAME=patient-data

# Data Source
DATA_SOURCE_URL=https://archive.ics.uci.edu/static/public/296/diabetes+130-us+hospitals+for+years+1999-2008.zip

# Project Settings
PROJECT_NAME=patient-readmission-mlops
```

### Model Hyperparameters

Key parameters in the training pipeline (adjustable in `src/models/train.py`):

```python
# XGBoost Parameters
n_estimators = 100
max_depth = 6
learning_rate = 0.1
subsample = 0.8
colsample_bytree = 0.8

# SMOTE Configuration
sampling_strategy = 0.3  # Target 30% positive class

# Feature Selection
n_features = 15  # Top N most predictive features

# Data Split
test_size = 0.2  # 80/20 train/test split
```

## Dataset Information

**Source**: UCI Machine Learning Repository  
**Name**: Diabetes 130-US hospitals for years 1999-2008  
**Records**: ~101,766 patient encounters  
**Features**: 50+ including demographics, diagnoses, medications, procedures  
**Target**: 30-day readmission (binary classification)  
**Class Distribution**: ~11% readmission rate (imbalanced)

### Key Features Used

The model uses 15 most predictive features:
1. `number_inpatient` - Prior inpatient visits
2. `number_emergency` - Emergency visits
3. `discharge_disposition_id` - Discharge destination
4. `number_diagnoses` - Number of diagnoses
5. `time_in_hospital` - Length of stay
6. `num_medications` - Number of medications
7. `diabetesMed` - Diabetes medication prescribed
8. `metformin` - Metformin prescribed
9. `num_lab_procedures` - Laboratory procedures
10. `change` - Change in diabetic medications
11. `number_outpatient` - Outpatient visits
12. `age` - Patient age group
13. `num_procedures` - Number of procedures
14. `admission_type_id` - Type of admission
15. `repaglinide` - Repaglinide prescribed

## ML Pipeline Details

### 1. Data Preprocessing
- **Missing Value Imputation**: KNN imputation for numerical features
- **Feature Selection**: Mutual information-based selection of top 15 features
- **Outlier Handling**: Capping at 95th percentile
- **Feature Scaling**: StandardScaler for numerical features
- **Data Validation**: Type checking and range validation

### 2. Class Imbalance Handling
- **SMOTE Oversampling**: Synthetic minority oversampling technique
- **Balanced Weights**: Class-balanced weights in XGBoost
- **Stratified Splitting**: Maintains class distribution in train/test splits

### 3. Model Training
- **Algorithm**: XGBoost Classifier
- **Cross-Validation**: 5-fold stratified cross-validation
- **Early Stopping**: Prevents overfitting
- **Hyperparameter Tuning**: Grid search for optimal parameters

### 4. Evaluation Metrics
Healthcare-focused evaluation prioritizing recall:
- **Recall (Primary)**: Sensitivity for catching readmissions
- **Precision**: Positive predictive value
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under ROC curve
- **PR-AUC**: Area under precision-recall curve (better for imbalanced data)

### 5. Model Registry
- **Automated Versioning**: MLflow model registry
- **Stage Management**: Staging → Production → Archived
- **Performance Tracking**: Compare metrics across versions
- **Deployment Safety**: Validation checks before production
- **Rollback Capability**: Quick reversion to previous versions

## Development Workflow

### Code Quality

The project enforces high code quality standards:

```bash
# Run linting
uv run ruff check src/

# Fix linting issues
uv run ruff check src/ --fix

# Run type checking
uv run mypy src/

# Format code
uv run ruff format src/

# Run all pre-commit hooks
uv run pre-commit run --all-files
```

### Testing

```bash
# Run unit tests
uv run pytest tests/

# Run with coverage
uv run pytest --cov=src tests/

# Run specific test file
uv run pytest tests/test_models.py -v
```

### Adding New Features

1. Create feature branch: `git checkout -b feature/new-feature`
2. Make changes with proper type hints
3. Add comprehensive tests
4. Update documentation
5. Run pre-commit hooks: `uv run pre-commit run --all-files`
6. Create pull request

## Deployment Options

### Local Development
```bash
# Start MLflow server
mlflow ui --host 0.0.0.0 --port 5000

# Run training pipeline
python -m src.models.train

# Access model registry
python src/models/registry/model_management.py status
```

### Production Deployment (Planned)

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
- **Rotation**: Automatic log rotation to manage disk space
- **Centralized**: All components log to consistent format

### Metrics Tracking
- **Model Performance**: Ongoing accuracy monitoring
- **Data Drift**: Detection of input data changes
- **Prediction Distribution**: Monitor prediction patterns
- **System Health**: API response times and error rates

### Alerts
- **Performance Degradation**: Alert when metrics drop
- **System Errors**: Immediate notification of failures
- **Data Quality Issues**: Alert on data anomalies

## Security and Compliance

### Data Protection
- **Environment Variables**: Sensitive data in .env files
- **Access Controls**: Azure RBAC for data access
- **Encryption**: Data encrypted in transit and at rest
- **Audit Logging**: Track all data access and model changes

### Healthcare Compliance
- **HIPAA Considerations**: Framework for healthcare data handling
- **Model Interpretability**: Feature importance and decision explanations
- **Bias Detection**: Regular audits for algorithmic bias
- **Documentation**: Comprehensive model documentation for regulatory review

## Troubleshooting

### Common Issues

**MLflow UI not accessible**
```bash
# Check if MLflow server is running
ps aux | grep mlflow

# Restart MLflow server
mlflow ui --host 0.0.0.0 --port 5000
```

**Model training fails**
```bash
# Check data availability
python -c "from src.data.preprocessing import DataPreprocessor; dp = DataPreprocessor(); dp.load_data()"

# Verify environment variables
python -c "import os; print(os.getenv('AZURE_STORAGE_CONNECTION_STRING'))"
```

**Import errors**
```bash
# Ensure virtual environment is activated
source .venv/bin/activate

# Reinstall dependencies
uv sync
```

### Performance Optimization

- **Memory Usage**: Use Polars for large dataset processing
- **Training Speed**: Utilize XGBoost's parallel processing
- **Model Size**: Feature selection reduces model complexity
- **Inference Speed**: Pre-load models for faster predictions

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
  author={[Your Name]},
  year={2025},
  url={https://github.com/[username]/patient-readmission-mlops}
}
```

## Acknowledgments

- **UCI Machine Learning Repository** for providing the diabetes dataset
- **MLflow Community** for excellent experiment tracking tools
- **XGBoost Team** for the high-performance gradient boosting framework
- **Healthcare ML Community** for research on readmission prediction
- **SMOTE Authors** for class imbalance handling techniques

## Support and Contact

- **Issues**: Report bugs and request features via [GitHub Issues](../../issues)
- **Discussions**: Ask questions in [GitHub Discussions](../../discussions)
- **Documentation**: Full API documentation available at [docs/](docs/)
- **Email**: [your-email@domain.com] for critical issues

## Roadmap

### Phase 1: Core ML Pipeline ✅
- [x] Data preprocessing and feature engineering
- [x] XGBoost model training with SMOTE
- [x] MLflow experiment tracking
- [x] Model registry and lifecycle management

### Phase 2: API Development 🚧
- [ ] FastAPI service for model serving
- [ ] Input validation and error handling
- [ ] Authentication and rate limiting
- [ ] API documentation with OpenAPI

### Phase 3: Deployment 📋
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