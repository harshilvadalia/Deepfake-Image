# It accepts an image path as input and returns the prediction label and confidence score. The model is loaded from the file resnet_model_new.h5, which should be present in the same directory as the script. The model is used to make a prediction on the input image, and the prediction label and confidence score are returned. The confidence score is the probability of the prediction being correct. The function handles exceptions and prints an error message if an exception occurs during prediction.
# This is being used within the app.py file to make predictions on uploaded images. The app.py file is a Flask web application that allows users to upload images and get predictions from the model. The predict_image function is imported into the app.py file to make predictions on the uploaded images. The function is called with the path to the uploaded image, and the prediction label and confidence score are returned to the user.

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow logging

# Import tensorflow after setting environment variables
import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.models import load_model

# Configure metal plugin
try:
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    pass

def predict_image(img_path):
    try:
        # Load and preprocess image
        img = load_img(img_path, target_size=(224, 224))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Load model
        model = load_model('efficient_resnet_model_new.h5', compile=False)
        
        # Make prediction
        prediction = model.predict(img_array, verbose=0)
        pred_value = prediction[0][0]
        
        if pred_value > 0.5:
            label = 'REAL'
            confidence = pred_value
        else:
            label = 'FAKE'
            confidence = 1 - pred_value
            
        confidence_percent = round(float(confidence) * 100, 2)
        return label, confidence_percent
        
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        raise e