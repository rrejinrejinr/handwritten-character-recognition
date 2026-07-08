import numpy as np
import tensorflow as tf
from emnist import extract_training_samples

def test_orientation():
    # Load EMNIST digits split (which corresponds to MNIST digits 0-9)
    print("Loading EMNIST digits...")
    emnist_images, emnist_labels = extract_training_samples('digits')
    
    # Load Keras MNIST dataset (ground truth orientation)
    print("Loading standard Keras MNIST...")
    (mnist_images, mnist_labels), _ = tf.keras.datasets.mnist.load_data()
    
    # Let's take the first digit '5' or another digit from both datasets and compare
    # We find the first occurrence of digit 5 in both
    emnist_5_idx = np.where(emnist_labels == 5)[0][0]
    mnist_5_idx = np.where(mnist_labels == 5)[0][0]
    
    emnist_5 = emnist_images[emnist_5_idx]
    mnist_5 = mnist_images[mnist_5_idx]
    
    print("\n--- Testing EMNIST Orientations ---")
    
    # Transformation A: np.transpose only (our previous version)
    transposed_only = np.transpose(emnist_5)
    
    # Transformation B: np.transpose + horizontal flip (np.fliplr)
    transposed_and_flipped = np.fliplr(np.transpose(emnist_5))
    
    # Let's count matching pixels with MNIST (roughly, or print a textual representation of the digit)
    # We can print a low-res text representation of mnist_5, transposed_only, and transposed_and_flipped
    # using characters for non-zero pixels!
    
    def print_digit(img, title):
        print(f"\n=== {title} ===")
        # Threshold and downsample to 14x14 for console print
        downsampled = img[::2, ::2]
        for row in downsampled:
            line = "".join(["#" if val > 100 else " " for val in row])
            print(line)
            
    print_digit(mnist_5, "Standard MNIST (Ground Truth)")
    print_digit(emnist_5, "Raw EMNIST (Unprocessed)")
    print_digit(transposed_only, "EMNIST with Transpose Only")
    print_digit(transposed_and_flipped, "EMNIST with Transpose + Horizontal Flip")

if __name__ == '__main__':
    test_orientation()
