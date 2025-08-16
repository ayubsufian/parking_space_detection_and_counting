import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    roc_curve,
    auc,
    classification_report
)
from tensorflow.keras.preprocessing import image_dataset_from_directory

IMG_HEIGHT = 48
IMG_WIDTH = 48
BATCH_SIZE = 32
TEST_DATA_DIR = 'data/test' # Path to your test dataset root (contains 'empty' and 'occupied' subfolders)

# --- Load the trained model ---
model_path = 'model/model_final.h5'
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}. Make sure it's in the 'model/' directory.")
model = tf.keras.models.load_model(model_path)

# --- Load Test Data from Directory ---
if not os.path.exists(TEST_DATA_DIR):
    raise FileNotFoundError(f"Test data directory not found: {TEST_DATA_DIR}. "
                            "Please ensure 'data/test' exists and contains 'empty' and 'occupied' subdirectories.")

print(f"Loading test images from: {TEST_DATA_DIR} (labels inferred from subdirectories)")
test_ds = image_dataset_from_directory(
    TEST_DATA_DIR,
    labels='inferred',       # Labels are inferred from subdirectory names
    label_mode='int',        # Labels will be integer encoded (e.g., 0 for 'empty', 1 for 'occupied')
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    shuffle=False,           # CRITICAL: Maintain order for consistent label and prediction alignment
    interpolation='bilinear' # Use the same interpolation as you did during training
)

class_labels = test_ds.class_names
print(f"Inferred class names (and their integer mapping): {class_labels}")

# Preprocess images (normalize pixels to 0-1)
# The dataset yields (image_batch, label_batch) tuples
def preprocess_image_with_label(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

test_ds = test_ds.map(preprocess_image_with_label)

# Convert the TensorFlow dataset to NumPy arrays for model prediction and evaluation
# This collects all images and their labels into NumPy arrays in memory.
print("Converting image dataset to NumPy arrays for evaluation...")
test_images_list = []
test_labels_list = []
for images_batch, labels_batch in test_ds:
    test_images_list.append(images_batch.numpy())
    test_labels_list.append(labels_batch.numpy())

test_images = np.concatenate(test_images_list, axis=0)
test_labels = np.concatenate(test_labels_list, axis=0)
print(f"Successfully loaded {len(test_images)} test images and {len(test_labels)} labels.")

predictions = model.predict(test_images)

occupied_class_index = class_labels.index('occupied')

predicted_probabilities = predictions[:, occupied_class_index] 
predicted_classes = np.argmax(predictions, axis=1)             

test_labels_int = test_labels.astype(int)
predicted_classes_int = predicted_classes.astype(int)

accuracy = accuracy_score(test_labels_int, predicted_classes_int)
print(f'\nAccuracy: {accuracy:.4f}')

fpr, tpr, thresholds = roc_curve(test_labels_int, predicted_probabilities)
roc_auc = auc(fpr, tpr)
print(f'AUC: {roc_auc:.4f}')

print("\nClassification Report:\n")
print(classification_report(test_labels_int, predicted_classes_int, target_names=class_labels))