import numpy as np
from emnist import extract_training_samples

# EMNIST Balanced mapping (47 classes)
EMNIST_CLASSES = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    'a', 'b', 'd', 'e', 'f', 'g', 'h', 'n', 'q', 'r', 't'
]

def test_letters():
    print("Loading EMNIST balanced...")
    x_train, y_train = extract_training_samples('balanced')
    
    # Let's find index of 'Z' (class index 35) and 'A' (class index 10)
    z_idx = np.where(y_train == 35)[0][0]
    a_idx = np.where(y_train == 10)[0][0]
    
    z_img = x_train[z_idx]
    a_img = x_train[a_idx]
    
    def print_digit(img, title):
        print(f"\n=== {title} ===")
        # Threshold and downsample to 14x14 for console print
        downsampled = img[::2, ::2]
        for row in downsampled:
            line = "".join(["#" if val > 100 else " " for val in row])
            print(line)
            
    print("--- Testing Letter Z ---")
    print_digit(z_img, "Raw Z (Unprocessed)")
    print_digit(np.transpose(z_img), "Z Transposed Only (np.transpose)")
    print_digit(np.fliplr(np.transpose(z_img)), "Z Transposed + Horizontal Flip")
    print_digit(np.flipud(np.transpose(z_img)), "Z Transposed + Vertical Flip")
    
    print("\n--- Testing Letter A ---")
    print_digit(a_img, "Raw A (Unprocessed)")
    print_digit(np.transpose(a_img), "A Transposed Only (np.transpose)")
    print_digit(np.fliplr(np.transpose(a_img)), "A Transposed + Horizontal Flip")
    print_digit(np.flipud(np.transpose(a_img)), "A Transposed + Vertical Flip")

if __name__ == '__main__':
    test_letters()
