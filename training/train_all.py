import os
import subprocess
import sys

def main():
    python_exe = "C:\\Users\\rreji\\AppData\\Local\\Programs\\Python\\Python311\\python.exe"
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 1. Download Kaggle MNIST first
    download_script = os.path.join(base_dir, "download_mnist.py")
    print(f"--- Running MNIST Download from Kaggle ({download_script}) ---")
    subprocess.check_call([python_exe, download_script])
    
    # 2. Download EMNIST from the working NIST mirror
    download_emnist_script = os.path.join(base_dir, "download_emnist.py")
    print(f"\n--- Running EMNIST Download from Working Mirror ({download_emnist_script}) ---")
    subprocess.check_call([python_exe, download_emnist_script])
    
    # 3. Train Digits model
    digits_script = os.path.join(base_dir, "train_digits.py")
    print(f"\n--- Training Digits Model ({digits_script}) ---")
    subprocess.check_call([python_exe, digits_script])
    
    # 4. Train Characters model
    char_script = os.path.join(base_dir, "train_characters.py")
    print(f"\n--- Training Characters Model ({char_script}) ---")
    subprocess.check_call([python_exe, char_script])
    
    print("\n=============================================")
    print("All training tasks completed successfully!")
    print("=============================================")

if __name__ == '__main__':
    main()
