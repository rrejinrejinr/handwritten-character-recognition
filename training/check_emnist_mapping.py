import numpy as np
from emnist import extract_training_samples

def check_mapping():
    print("Loading EMNIST balanced split...")
    x_train, y_train = extract_training_samples('balanced')
    
    # We want to print a sample image for class index 10 (A), 20 (K), 23 (N), 35 (Z)
    target_classes = [10, 20, 23, 35]
    
    for cls in target_classes:
        idx = np.where(y_train == cls)[0][0]
        img = x_train[idx]
        
        print(f"\n=====================================")
        print(f"Sample for Class Index: {cls}")
        print(f"=====================================")
        downsampled = img[::2, ::2]
        for row in downsampled:
            line = "".join(["# block" if val > 100 else "      " for val in row])
            # Keep it compact
            line = "".join(["#" if val > 100 else " " for val in row])
            print(line)

if __name__ == '__main__':
    check_mapping()
