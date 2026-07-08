import os
import requests
import zipfile

def download_emnist_zip():
    cache_dir = "C:\\Users\\rreji\\.cache\\emnist"
    os.makedirs(cache_dir, exist_ok=True)
    
    zip_path = os.path.join(cache_dir, "emnist.zip")
    
    # Check if a valid emnist.zip already exists
    if os.path.exists(zip_path):
        size_mb = os.path.getsize(zip_path) / (1024 * 1024)
        print(f"Checking existing cached file: {zip_path} ({size_mb:.2f} MB)")
        
        if size_mb > 500:
            try:
                # Quickly test if it is a valid zip file
                print("Testing if cached ZIP is valid...")
                with zipfile.ZipFile(zip_path) as zf:
                    # Test zip integrity
                    first_bad_file = zf.testzip()
                    if first_bad_file is None:
                        print("Cached EMNIST ZIP is valid. Skipping download.")
                        return
                    else:
                        print(f"Warning: Cached ZIP has a bad file: {first_bad_file}. Re-downloading...")
            except Exception as e:
                print(f"Warning: Failed to verify ZIP file: {e}. Re-downloading...")
    
    # URL for EMNIST gzip.zip
    url = "https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip"
    
    print(f"Downloading EMNIST dataset from {url}...")
    print(f"Saving to {zip_path}")
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):  # 1MB chunks
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"Downloaded: {downloaded / (1024 * 1024):.2f}MB / {total_size / (1024 * 1024):.2f}MB ({percent:.1f}%)", end='\r')
                    else:
                        print(f"Downloaded: {downloaded / (1024 * 1024):.2f}MB", end='\r')
        print("\nDownload complete! EMNIST dataset successfully cached.")
    except Exception as e:
        print(f"\nError downloading EMNIST dataset: {e}")
        # Clean up partial download
        if os.path.exists(zip_path):
            os.remove(zip_path)
        raise e

if __name__ == '__main__':
    download_emnist_zip()
