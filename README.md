cat > README.md << 'EOF'
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
EOF

