import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model_final = load_model('model/model_final.h5')

# Define the class dictionary
class_dictionary = {0: 'Empty', 1: 'Full'}

# Predictive system
def make_prediction(image_path):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    # Resize the image to the model's expected input size
    image = cv2.resize(image, (48, 48))
    
    # Normalize the image
    img = image / 255.0

    # Expand dimensions to create a batch of size 1
    img = np.expand_dims(img, axis=0)  # Shape: (1, 48, 48, 3)

    # Make prediction
    class_predicted = model_final.predict(img)
    intId = np.argmax(class_predicted[0])  # Get the class with the highest probability
    label = class_dictionary[intId]
    return label

# Test the function
print(make_prediction("assets/spot26.jpg"))