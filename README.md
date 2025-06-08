# Patient Readmission Prediction - MLOps Project

## What We're Building

A machine learning system that predicts if hospital patients will be readmitted within 30 days. This helps hospitals prepare better care and reduce costs.

## Technology Stack

- **Python 3.11+** - Programming language
- **Azure** - Cloud platform (free tier)
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
├── data/                  # Dataset storage
│   ├── raw/              # Original data files
│   └── processed/        # Cleaned data
├── configs/              # Configuration files
├── tests/                # Test files
├── infrastructure/       # Docker and deployment files
└── docs/                 # Documentation
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
- Azure integration: azure-storage-blob

## Next Steps

1. **Download Dataset** - Get the hospital readmission data
2. **Set up Azure Storage** - Store data in the cloud
3. **Connect Databricks** - Set up ML platform
4. **Data Processing** - Clean and prepare the data
5. **Model Training** - Build ML models
6. **API Development** - Create prediction service
7. **Deployment** - Deploy to Azure

## Quick Start

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

## What We're Doing Next

### Immediate Goal
Set up the data pipeline - download the dataset and store it in Azure Blob Storage.

### Step-by-Step Plan

**Step 1: Complete the Download Script**
- Finish the `download_data.py` file we started
- Add code to download the UCI diabetes dataset
- Upload it to Azure Blob Storage (not just local files)
- Add error handling

**Step 2: Set up Azure Storage**
- Log into Azure portal
- Create a Storage Account
- Create a container called "raw-data"
- Get connection string

**Step 3: Test Data Pipeline**
- Run our script to download and store data
- Verify data appears in Azure Blob Storage
- Check file size and format

**Step 4: Set up Databricks**
- Sign up for free Databricks account
- Create a cluster
- Connect to our Azure storage
- Test we can read the data

## Current Status

**Phase**: Initial Setup Complete
**Progress**: 15% complete
**Working**: Development environment, code quality tools
**Next**: Complete download script and Azure Storage setup