# Patient Readmission Prediction - MLOps Project

## What We're Building

A machine learning system that predicts if hospital patients will be readmitted within 30 days. This helps hospitals prepare better care and reduce costs.

## Technology Stack

- **Python 3.11+** - Programming language
- **Azure Blob Storage** - Cloud data storage
- **Databricks** - Data processing and ML platform
- **MLflow** - Track ML experiments
- **FastAPI** - Create APIs
- **Docker** - Package our application

## Project Structure

```
patient-readmission-mlops/
├── src/                    # Main Python code
│   ├── data/              # Data processing scripts
│   ├── models/            # ML model code
│   └── api/               # API code
├── notebooks/             # Jupyter notebooks for experiments
├── configs/              # Configuration files
├── tests/                # Test files
├── infrastructure/       # Docker and deployment files
├── docs/                 # Documentation
├── download_data.py      # Azure data pipeline script
├── .env.example          # Environment variables template
└── pyproject.toml        # Project dependencies
```

## What We've Done So Far

### ✅ Project Setup
- Created GitHub repository
- Set up Python virtual environment with `uv`
- Installed modern development tools (Ruff, MyPy, Pre-commit)

### ✅ Development Environment
- Pre-commit hooks working (auto-formats code, checks for errors)
- Type checking enabled
- Fast linting with Ruff

### ✅ Dependencies Installed
- Core ML libraries: scikit-learn, XGBoost, MLflow
- Data processing: Polars, pandas
- API framework: FastAPI
- Azure integration: azure-storage-blob, python-dotenv

### ✅ Cloud Data Pipeline
- **Azure Storage Account**: `patientdata06082025`
- **Container**: `patient-data` 
- **Dataset**: UCI Diabetes 130-US hospitals (18.3 MB)
- **Files**: `diabetic_data.csv`, `IDS_mapping.csv`
- **Location**: `raw-data/` folder in Azure Blob Storage

## Next Steps

1. **Connect Databricks** - Set up ML platform and connect to Azure storage
2. **Data Exploration** - Analyze the diabetes dataset 
3. **Data Processing** - Clean and prepare the data
4. **Feature Engineering** - Create ML features
5. **Model Training** - Build ML models
6. **API Development** - Create prediction service
7. **Deployment** - Deploy to Azure

## Quick Start

### Environment Setup
```bash
# Clone the project
git clone <repository-url>
cd patient-readmission-mlops

# Set up environment
uv sync
source .venv/bin/activate  # Linux/Mac
# or .venv\Scripts\activate  # Windows

# Install pre-commit hooks
uv run pre-commit install

# Test everything works
uv run pre-commit run --all-files
```

### Data Pipeline Setup

1. **Create Azure Storage Account** (or use existing)
2. **Copy environment template**:
   ```bash
   cp .env.example .env
   ```
3. **Add your Azure connection string** to `.env`:
   ```
   AZURE_STORAGE_CONNECTION_STRING=your_connection_string_here
   AZURE_STORAGE_ACCOUNT_NAME=your_storage_account_name
   AZURE_CONTAINER_NAME=patient-data
   ```
4. **Run the data pipeline**:
   ```bash
   python download_data.py
   ```

## Current Status

**Phase**: Data Pipeline Complete
**Progress**: 30% complete
**Working**: 
- Development environment
- Code quality tools
- Azure Blob Storage integration
- Dataset uploaded to cloud

**Next**: Set up Databricks and begin data analysis

## Dataset Information

**Source**: UCI Machine Learning Repository
**Name**: Diabetes 130-US hospitals for years 1999-2008
**Size**: 18.3 MB
**Records**: ~100,000 patient encounters
**Features**: Patient demographics, diagnoses, medications, readmission status

The dataset is now stored in Azure Blob Storage and ready for analysis and ML model development.