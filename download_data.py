import requests
import zipfile
import os


def download_dataset():
    os.makedirs("data/raw", exist_ok=True)

    url = "https://archive.ics.uci.edu/static/public/296/diabetes+130-us+hospitals+for+years+1999-2008.zip"
    print("Downloading dataset...")
    response = requests.get(url)

    with open("data/raw/diabetes_datset.zip", "wb") as f:
        f.write(response.content)

    with zipfile.ZipFile("data/raw/diabetes_datset.zip", "r") as zip_ref:
        zip_ref.extractall("data/raw")

    print("✅ Dataset downloaded and extracted to data/raw/")


if __name__ == "__main__":
    download_dataset()
