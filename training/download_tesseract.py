import urllib.request
import os

def download_tesseract():
    url = "https://github.com/UB-Mannheim/tesseract/releases/download/v5.4.0.20240606/tesseract-ocr-w64-setup-5.4.0.20240606.exe"
    dest = os.path.join(os.getcwd(), "tesseract_setup.exe")
    print(f"Downloading Tesseract installer from: {url}...")
    urllib.request.urlretrieve(url, dest)
    print(f"Successfully downloaded to: {dest}")

if __name__ == "__main__":
    download_tesseract()
