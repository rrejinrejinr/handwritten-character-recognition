import os
import shutil
import subprocess

def setup_and_download():
    # 1. Setup kaggle.json in the user's home directory
    home = os.path.expanduser("~")
    kaggle_dir = os.path.join(home, ".kaggle")
    os.makedirs(kaggle_dir, exist_ok=True)
    
    src_json = "C:\\Users\\rreji\\Downloads\\kaggle.json"
    dst_json = os.path.join(kaggle_dir, "kaggle.json")
    
    if os.path.exists(src_json):
        shutil.copy(src_json, dst_json)
        print(f"Copied kaggle.json to {dst_json}")
        # On Windows, we don't strictly need chmod, but let's make sure it's writeable/readable
        os.chmod(dst_json, 0o600)
    else:
        print("Warning: kaggle.json not found in C:\\Users\\rreji\\Downloads.")
        
    # 2. Ensure kaggle is installed in Python 3.11
    try:
        import kaggle
    except ImportError:
        print("Installing kaggle CLI...")
        subprocess.check_call([
            "C:\\Users\\rreji\\AppData\\Local\\Programs\\Python\\Python311\\python.exe", 
            "-m", "pip", "install", "kaggle"
        ])
        import kaggle
        
    # 3. Download the dataset
    target_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "mnist_data"))
    os.makedirs(target_path, exist_ok=True)
    
    print("Downloading hojjatk/mnist-dataset from Kaggle...")
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files('hojjatk/mnist-dataset', path=target_path, unzip=True)
    print(f"MNIST Dataset successfully downloaded and extracted to {target_path}")

if __name__ == '__main__':
    setup_and_download()
