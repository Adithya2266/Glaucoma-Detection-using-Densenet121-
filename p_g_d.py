import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from keras.models import load_model

# Load the saved model
model_path = 'glaucoma_detection_densenet1.h5'
model = load_model(model_path)
print("Model loaded successfully!")

# Function to preprocess the input image
def preprocess_image(image_path, target_size=(224, 224)):
    """
    Preprocess the image for model prediction.
    
    Parameters:
        image_path (str): Path to the image file.
        target_size (tuple): Target size for resizing the image.
        
    Returns:
        np.array: Preprocessed image.
    """
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Function to make predictions
def predict_glaucoma(image_path):
    """
    Predict if the input eye image has glaucoma or not.
    
    Parameters:
        image_path (str): Path to the eye image.
        
    Returns:
        str: Prediction result ("Glaucoma Detected" or "No Glaucoma").
    """
    preprocessed_img = preprocess_image(image_path)
    prediction = model.predict(preprocessed_img)[0][0]
    return "Glaucoma Detected" if prediction >= 0.5 else "No Glaucoma"

# Predict for all images in the training dataset
def predict_all_images_in_directory(directory_path):
    """
    Predict glaucoma status for all images in a directory.
    
    Parameters:
        directory_path (str): Path to the dataset directory.
    """
    for subdir in os.listdir(directory_path):
        subdir_path = os.path.join(directory_path, subdir)
        if os.path.isdir(subdir_path):
            print(f"\nProcessing folder: {subdir}")
            for img_file in os.listdir(subdir_path):
                img_path = os.path.join(subdir_path, img_file)
                if img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                    try:
                        result = predict_glaucoma(img_path)
                        print(f"Image: {img_file} - Prediction: {result}")
                    except Exception as e:
                        print(f"Error processing {img_file}: {e}")

# Set the path to the training dataset directory
train_dir = 'dataset (divided)/Train'  # Replace with your training dataset path

print("Predicting for the entire training dataset...")
predict_all_images_in_directory(train_dir)
