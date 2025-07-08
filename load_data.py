import pandas as pd
import os
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv
from io import StringIO

def load_diabetes_data():
    load_dotenv()
    
    connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    container_name = "patient-data"
    
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    blob_client = blob_service_client.get_blob_client(
        container=container_name, 
        blob="raw-data/diabetic_data.csv"
    )
    
    blob_data = blob_client.download_blob().readall()
    df = pd.read_csv(StringIO(blob_data.decode('utf-8')))
    
    return df

if __name__ == "__main__":
    df = load_diabetes_data()
    print(f"Dataset loaded: {df.shape}")
    print("\nFirst 5 rows:")
    print(df.head())