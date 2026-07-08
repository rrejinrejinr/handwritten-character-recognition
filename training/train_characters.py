import os
import gc
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from tensorflow.keras import layers, models

# EMNIST Balanced mapping (47 classes)
EMNIST_CLASSES = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    'a', 'b', 'd', 'e', 'f', 'g', 'h', 'n', 'q', 'r', 't'
]

def train_character_model():
    print("TensorFlow version:", tf.__version__)
    
    # 1. Load EMNIST Balanced dataset
    print("Loading EMNIST Balanced dataset...")
    try:
        from emnist import extract_training_samples, extract_test_samples
        x_train, y_train = extract_training_samples('balanced')
        x_test, y_test = extract_test_samples('balanced')
    except ImportError:
        print("emnist package not found. Attempting to install...")
        import subprocess
        subprocess.check_call(["pip", "install", "emnist"])
        from emnist import extract_training_samples, extract_test_samples
        x_train, y_train = extract_training_samples('balanced')
        x_test, y_test = extract_test_samples('balanced')
        
    print(f"Original EMNIST shapes - train: {x_train.shape}, test: {x_test.shape}")
    
    # We train on the full dataset now to achieve >85% accuracy.

    # Keep as uint8 in memory (4x memory reduction) and reshape for Conv2D
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    
    print(f"Preprocessed train shape: {x_train.shape} (uint8)")
    print(f"Preprocessed test shape: {x_test.shape} (uint8)")
    
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
        layers.Dense(47, activation='softmax') # 47 classes
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
        epochs=8,
        batch_size=128,
        validation_split=0.1
    )
    
    # 5. Evaluate model
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    print(f"\nTest accuracy: {test_acc:.4f}, Test loss: {test_loss:.4f}")
    
    # Create directories for outputs
    plots_dir = os.path.join(os.path.dirname(__file__), 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'ml-service', 'models'))
    os.makedirs(models_dir, exist_ok=True)
    
    # 6. Save Plots
    print("Generating training plots...")
    plt.figure(figsize=(12, 4))
    
    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Character Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Character Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'char_history.png'))
    plt.close()
    
    # 7. Confusion Matrix
    print("Generating confusion matrix...")
    y_pred = model.predict(x_test, batch_size=64)
    y_pred_classes = np.argmax(y_pred, axis=1)
    cm = confusion_matrix(y_test, y_pred_classes)
    
    plt.figure(figsize=(16, 14))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Characters Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(47)
    plt.xticks(tick_marks, EMNIST_CLASSES, rotation=90)
    plt.yticks(tick_marks, EMNIST_CLASSES)
    
    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(plots_dir, 'char_confusion_matrix.png'))
    plt.close()
    
    # 8. Save Model
    model_path = os.path.join(models_dir, 'char_model.h5')
    print(f"Saving model to {model_path}...")
    model.save(model_path)
    print("Character model training complete!")

if __name__ == '__main__':
    train_character_model()
