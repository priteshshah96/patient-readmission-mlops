"""
Data preprocessing pipeline for patient readmission prediction.
Based on EDA insights from 101,766 patient records.

Key preprocessing steps:
1. Load cleaned data from Azure
2. Remove target leakage columns
3. Handle outliers in emergency visits
4. Feature selection based on correlation analysis
5. Split data with stratification for class imbalance
"""

import os
import polars as pl
import pandas as pd
import numpy as np
from io import StringIO
from typing import Tuple, List
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/preprocessing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Handles all data preprocessing for the readmission model."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.selected_features = []
        
    def load_data_from_azure(self) -> pl.DataFrame:
        """Load cleaned dataset from Azure Blob Storage."""
        load_dotenv()
        
        connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        container_name = "patient-data"
        blob_path = "cleaned-data/diabetic_data_cleaned_knn_imputed.csv"
        
        if not connection_string:
            raise ValueError("AZURE_STORAGE_CONNECTION_STRING not found in environment")
        
        logger.info("Loading data from Azure Blob Storage...")
        
        # Get data from Azure
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        blob_client = blob_service_client.get_blob_client(
            container=container_name, 
            blob=blob_path
        )
        
        # Download and read with Polars
        blob_data = blob_client.download_blob().readall()
        csv_string = blob_data.decode('utf-8')
        df = pl.read_csv(StringIO(csv_string))
        
        logger.info(f"Data loaded successfully: {df.shape[0]:,} rows, {df.shape[1]} columns")
        return df
    
    def remove_leakage_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        """Remove columns that cause target leakage."""
        leakage_columns = [
            'readmitted',  # Original target variable (high correlation -0.74)
            'encounter_id',  # Not predictive
            'patient_nbr'    # Not predictive
        ]
        
        columns_to_drop = [col for col in leakage_columns if col in df.columns]
        
        if columns_to_drop:
            df = df.drop(columns_to_drop)
            logger.info(f"Removed leakage columns: {columns_to_drop}")
        
        return df
    
    def handle_outliers(self, df: pl.DataFrame) -> pl.DataFrame:
        """Handle extreme outliers based on EDA insights."""
        
        # Cap emergency visits (EDA showed 30+ visits = 100% readmission)
        # This creates unrealistic perfect prediction
        emergency_95th = df['number_emergency'].quantile(0.95)
        
        original_max = df['number_emergency'].max()
        df = df.with_columns([
            pl.when(pl.col('number_emergency') > emergency_95th)
            .then(emergency_95th)
            .otherwise(pl.col('number_emergency'))
            .alias('number_emergency')
        ])
        
        logger.info(f"Capped emergency visits at 95th percentile: {emergency_95th:.0f} (was {original_max})")
        
        # Similar capping for inpatient visits if needed
        inpatient_95th = df['number_inpatient'].quantile(0.95)
        original_inpatient_max = df['number_inpatient'].max()
        
        df = df.with_columns([
            pl.when(pl.col('number_inpatient') > inpatient_95th)
            .then(inpatient_95th)
            .otherwise(pl.col('number_inpatient'))
            .alias('number_inpatient')
        ])
        
        logger.info(f"Capped inpatient visits at 95th percentile: {inpatient_95th:.0f} (was {original_inpatient_max})")
        
        return df
    
    def select_top_features(self, df: pl.DataFrame, top_n: int = 15) -> List[str]:
        """Select top N features based on correlation with target."""
        
        # Calculate correlations with target
        correlations = []
        target_col = 'readmitted_30_days'
        
        for col in df.columns:
            if col != target_col:
                try:
                    corr = df.select([
                        pl.corr(col, target_col).alias('correlation')
                    ])[0, 0]
                    
                    if not np.isnan(corr):  # Skip NaN correlations
                        correlations.append((col, abs(corr)))
                except:
                    continue
        
        # Sort by absolute correlation and select top N
        correlations.sort(key=lambda x: x[1], reverse=True)
        top_features = [feature for feature, _ in correlations[:top_n]]
        
        logger.info(f"Selected top {len(top_features)} features:")
        for i, (feature, corr) in enumerate(correlations[:top_n]):
            logger.info(f"  {i+1:2d}. {feature:<25} | {corr:.4f}")
        
        self.selected_features = top_features
        return top_features
    
    def prepare_for_modeling(self, df: pl.DataFrame, feature_list: List[str] = None) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare final dataset for modeling."""
        
        if feature_list is None:
            feature_list = self.selected_features
        
        # Convert to pandas for sklearn compatibility
        df_pandas = df.to_pandas()
        
        # Check which features actually exist in the dataframe
        available_features = [col for col in feature_list if col in df_pandas.columns]
        missing_features = [col for col in feature_list if col not in df_pandas.columns]
        
        if missing_features:
            logger.warning(f"Missing features (will be skipped): {missing_features}")
        
        if not available_features:
            raise ValueError("No valid features found in the dataset!")
        
        # Separate features and target
        X = df_pandas[available_features]
        y = df_pandas['readmitted_30_days']
        
        logger.info(f"Final dataset prepared:")
        logger.info(f"  Features: {X.shape[1]} (requested: {len(feature_list)})")
        logger.info(f"  Samples: {X.shape[0]:,}")
        logger.info(f"  Target distribution: {y.value_counts().to_dict()}")
        logger.info(f"  Feature names: {list(X.columns)}")
        
        return X, y
    
    def split_data(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, 
                   random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Split data with stratification to preserve class balance."""
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=random_state,
            stratify=y  # Important for imbalanced dataset
        )
        
        logger.info(f"Data split completed:")
        logger.info(f"  Training set: {X_train.shape[0]:,} samples")
        logger.info(f"  Test set: {X_test.shape[0]:,} samples")
        logger.info(f"  Train readmission rate: {y_train.mean():.3f}")
        logger.info(f"  Test readmission rate: {y_test.mean():.3f}")
        
        return X_train, X_test, y_train, y_test
    
    def scale_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Scale features using StandardScaler fitted on training data."""
        
        # Fit scaler on training data only
        X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        
        # Transform test data using fitted scaler
        X_test_scaled = pd.DataFrame(
            self.scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )
        
        logger.info("Feature scaling completed")
        
        return X_train_scaled, X_test_scaled
    
    def run_full_pipeline(self, top_n_features: int = 15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Run the complete preprocessing pipeline."""
        
        logger.info("Starting full preprocessing pipeline...")
        
        # 1. Load data
        df = self.load_data_from_azure()
        
        # 2. Remove leakage
        df = self.remove_leakage_columns(df)
        
        # 3. Handle outliers
        df = self.handle_outliers(df)
        
        # 4. Feature selection
        top_features = self.select_top_features(df, top_n_features)
        
        # 5. Prepare for modeling
        X, y = self.prepare_for_modeling(df, top_features)
        
        # 6. Split data
        X_train, X_test, y_train, y_test = self.split_data(X, y)
        
        # 7. Scale features
        X_train_scaled, X_test_scaled = self.scale_features(X_train, X_test)
        
        logger.info("Preprocessing pipeline completed successfully!")
        
        return X_train_scaled, X_test_scaled, y_train, y_test


# Example usage
if __name__ == "__main__":
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Run preprocessing
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.run_full_pipeline()
    
    print(f"\nPreprocessing complete!")
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    print(f"Ready for model training!")