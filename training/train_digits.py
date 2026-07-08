import os
import gzip
import gc
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from tensorflow.keras import layers, models

def load_idx_file(filepath):
    """Parses idx files (either gzip-compressed or raw binary)."""
    is_gz = filepath.endswith('.gz')
    open_func = gzip.open if is_gz else open
    
    with open_func(filepath, 'rb') as f:
        magic_number = int.from_bytes(f.read(4), 'big')
        
        if magic_number == 2051:  # Image IDX file
            num_items = int.from_bytes(f.read(4), 'big')
            rows = int.from_bytes(f.read(4), 'big')
            cols = int.from_bytes(f.read(4), 'big')
            data = np.frombuffer(f.read(), dtype=np.uint8)
            return data.reshape(num_items, rows, cols)
        elif magic_number == 2049:  # Label IDX file
            num_items = int.from_bytes(f.read(4), 'big')
            data = np.frombuffer(f.read(), dtype=np.uint8)
            return data
        else:
            raise ValueError(f"Unknown magic number {magic_number} in file {filepath}")

def load_kaggle_mnist(data_dir):
    """Finds and loads the MNIST dataset files from the local directory."""
    files = os.listdir(data_dir)
    
    train_img_path = None
    train_lbl_path = None
    test_img_path = None
    test_lbl_path = None
    
    for f in files:
        f_lower = f.lower()
        full_path = os.path.join(data_dir, f)
        if 'train-images' in f_lower or 'train_images' in f_lower:
            train_img_path = full_path
        elif 'train-labels' in f_lower or 'train_labels' in f_lower:
            train_lbl_path = full_path
        elif 't10k-images' in f_lower or 't10k_images' in f_lower or 'test-images' in f_lower:
            test_img_path = full_path
        elif 't10k-labels' in f_lower or 't10k_labels' in f_lower or 'test-labels' in f_lower:
            test_lbl_path = full_path
            
    if not all([train_img_path, train_lbl_path, test_img_path, test_lbl_path]):
        raise FileNotFoundError(f"Could not locate all 4 MNIST files in {data_dir}. Found files: {files}")
        
    print("Loading local Kaggle MNIST files...")
    x_train = load_idx_file(train_img_path)
    y_train = load_idx_file(train_lbl_path)
    x_test = load_idx_file(test_img_path)
    y_test = load_idx_file(test_lbl_path)
    
    return (x_train, y_train), (x_test, y_test)

def train_digit_model():
    print("TensorFlow version:", tf.__version__)
    
    # Define directories
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "mnist_data")
    
    # 1. Acquire dataset (ensure downloaded)
    if not os.path.exists(data_dir) or len(os.listdir(data_dir)) < 4:
        print("Kaggle MNIST data not found locally. Running downloader...")
        try:
            from download_mnist import setup_and_download
            setup_and_download()
        except Exception as e:
            print(f"Failed to auto-download Kaggle dataset: {e}")
            
    # Load dataset
    try:
        (x_train, y_train), (x_test, y_test) = load_kaggle_mnist(data_dir)
    except Exception as e:
        print(f"Error loading local Kaggle MNIST dataset: {e}")
        print("Falling back to standard tf.keras.datasets.mnist.load_data()...")
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    # 2. Preprocess data (Optimized for ultra-low RAM)
    # Subset to 10,000 training samples and 2,000 test samples to fit in limited RAM
    print("Limiting dataset sizes to conserve memory...")
    x_train = x_train[:10000]
    y_train = y_train[:10000]
    x_test = x_test[:2000]
    y_test = y_test[:2000]
    
    # Keep as uint8 in memory (4x memory reduction) and reshape for Conv2D
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    
    print(f"Train data shape: {x_train.shape} (uint8)")
    print(f"Test data shape: {x_test.shape} (uint8)")
    
    # 3. Build the CNN model with a lightweight LeNet-like architecture
    model = models.Sequential([
        # Batch-by-batch normalization inside model to avoid full-array float32 casting
        layers.Input(shape=(28, 28, 1)),
        layers.Rescaling(1.0 / 255.0),
        
        # Block 1
        layers.Conv2D(16, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),
        
        # Block 2
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),
        
        # Classifier
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(10, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary()
    
    # Force Garbage Collection
    gc.collect()
    
    # 4. Train model (4 epochs, batch size 64)
    print("Training model...")
    history = model.fit(
        x_train, y_train,
        epochs=4,
        batch_size=64,
        validation_split=0.1
    )
    
    # 5. Evaluate model
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    print(f"\nTest accuracy: {test_acc:.4f}, Test loss: {test_loss:.4f}")
    
    # Create directories for outputs if they don't exist
    plots_dir = os.path.join(base_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    models_dir = os.path.abspath(os.path.join(base_dir, '..', 'ml-service', 'models'))
    os.makedirs(models_dir, exist_ok=True)
    
    # 6. Save Plots
    print("Generating training plots...")
    plt.figure(figsize=(12, 4))
    
    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Digit Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Digit Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'digits_history.png'))
    plt.close()
    
    # 7. Confusion Matrix
    print("Generating confusion matrix...")
    y_pred = model.predict(x_test, batch_size=64)
    y_pred_classes = np.argmax(y_pred, axis=1)
    cm = confusion_matrix(y_test, y_pred_classes)
    
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Digits Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(10)
    plt.xticks(tick_marks, range(10))
    plt.yticks(tick_marks, range(10))
    
    thresh = cm.max() / 2.0
    for i in range(10):
        for j in range(10):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
            
    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(plots_dir, 'digits_confusion_matrix.png'))
    plt.close()
    
    # 8. Save Model
    model_path = os.path.join(models_dir, 'digits_model.h5')
    print(f"Saving model to {model_path}...")
    model.save(model_path)
    print("Digit model training complete!")

if __name__ == '__main__':
    train_digit_model()
