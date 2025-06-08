import os
import requests
import zipfile
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def download_and_upload_data():
    """
    Downloads the diabetes dataset and uploads it straight to Azure.
    Uses environment variables for configuration.
    """

    # Get configuration from environment variables
    dataset_url = os.getenv("DATA_SOURCE_URL")
    connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    container_name = os.getenv("AZURE_CONTAINER_NAME", "patient-data")

    if not dataset_url:
        print("Error: DATA_SOURCE_URL not found in environment variables")
        return False

    if not connection_string:
        print(
            "Error: AZURE_STORAGE_CONNECTION_STRING not found in environment variables"
        )
        print("Make sure you have a .env file with your Azure connection string")
        return False

    print(f"Downloading dataset from {dataset_url}")

    # Download the zip file
    response = requests.get(dataset_url)
    if response.status_code != 200:
        print(f"Failed to download: {response.status_code}")
        return False

    print("Downloaded successfully. Now uploading to Azure...")

    # Connect to Azure
    blob_client = BlobServiceClient.from_connection_string(connection_string)

    # Make sure container exists
    try:
        blob_client.create_container(container_name)
        print(f"Created container: {container_name}")
    except Exception as e:
        if "ContainerAlreadyExists" in str(e):
            print(f"Using existing container: {container_name}")
        else:
            print(f"Error creating container: {e}")
            return False

    # Save zip file temporarily so we can extract it
    zip_filename = "diabetes_data.zip"
    with open(zip_filename, "wb") as f:
        f.write(response.content)

    # Extract and upload each file
    try:
        with zipfile.ZipFile(zip_filename, "r") as zip_file:
            file_list = zip_file.namelist()
            print(f"Found {len(file_list)} files in the dataset")

            for filename in file_list:
                if filename.endswith("/"):  # skip folders
                    continue

                print(f"Uploading {filename}...")

                # Read the file from the zip
                file_data = zip_file.read(filename)

                # Upload to Azure with a clean path
                blob_path = f"raw-data/{filename}"
                blob = blob_client.get_blob_client(
                    container=container_name, blob=blob_path
                )

                blob.upload_blob(file_data, overwrite=True)

                # Show file size so we know it worked
                size_mb = len(file_data) / (1024 * 1024)
                print(f"Uploaded {filename} ({size_mb:.1f} MB)")

        print("\nAll done! Dataset is now in Azure Blob Storage")
        print(f"Container: {container_name}")
        print("Path: raw-data/")

        return True

    except Exception as e:
        print(f"Error processing files: {e}")
        return False
    finally:
        # Clean up the temp zip file
        if os.path.exists(zip_filename):
            os.remove(zip_filename)


def list_files_in_azure():
    """Check what files we have in Azure storage"""

    connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    container_name = os.getenv("AZURE_CONTAINER_NAME", "patient-data")

    if not connection_string:
        print(
            "Error: AZURE_STORAGE_CONNECTION_STRING not found in environment variables"
        )
        return

    blob_client = BlobServiceClient.from_connection_string(connection_string)

    print(f"\nFiles in {container_name}:")
    try:
        container = blob_client.get_container_client(container_name)
        blobs = container.list_blobs()

        total_size = 0
        file_count = 0

        for blob in blobs:
            size_mb = blob.size / (1024 * 1024)
            print(f"  {blob.name} ({size_mb:.1f} MB)")
            total_size += blob.size
            file_count += 1

        total_mb = total_size / (1024 * 1024)
        print(f"\nTotal: {file_count} files, {total_mb:.1f} MB")

    except Exception as e:
        print(f"Couldn't list files: {e}")


if __name__ == "__main__":
    # First download and upload the data
    success = download_and_upload_data()

    if success:
        # Then show what we uploaded
        list_files_in_azure()
    else:
        print("Something went wrong with the upload")
