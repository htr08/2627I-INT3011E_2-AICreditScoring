from pathlib import Path
import hashlib
import zipfile
import requests


URL = "https://www.kaggle.com/api/v1/datasets/download/uciml/default-of-credit-card-clients-dataset"

ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT_DIR / "data" / "raw"

ZIP_PATH = RAW_DIR / "default-of-credit-card-clients.zip"
CHECKSUM_PATH = RAW_DIR / "default-of-credit-card-clients.sha256"


def calculate_sha256(file_path):
    sha256 = hashlib.sha256()

    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


def download_data():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print("Downloading dataset...")

    response = requests.get(URL, stream=True)
    response.raise_for_status()

    with open(ZIP_PATH, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

    print("Download completed.")


def extract_data():
    with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall(RAW_DIR)

    print("Dataset extracted.")


def check_checksum():
    checksum = calculate_sha256(ZIP_PATH)

    if CHECKSUM_PATH.exists():
        expected = CHECKSUM_PATH.read_text().strip()

        if checksum != expected:
            raise ValueError("Checksum verification failed!")

        print("Checksum OK.")
    else:
        CHECKSUM_PATH.write_text(checksum)
        print("Checksum saved.")


if __name__ == "__main__":
    download_data()
    check_checksum()
    extract_data()
    print("Done!")