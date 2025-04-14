#This is just to check the performance of the model on the validation data. The model is loaded from the file best_model.h5, which should be present in the same directory as the script. The validation data is loaded from the file newval_data.csv, which contains the image names and labels for the validation data. The validation data is preprocessed using the ImageDataGenerator class from Keras, and the model is used to make predictions on the validation data. The predictions are evaluated using the F1 score and confusion matrix. The F1 score is a measure of the model's accuracy, and the confusion matrix shows the number of true positives, true negatives, false positives, and false negatives. The F1 score and confusion matrix are printed to the console, and the confusion matrix is plotted using matplotlib.
#We get F1 score and confusion matrix to evaluate the model performance on the validation data. The F1 score is a measure of the model's accuracy, and the confusion matrix shows the number of true positives, true negatives, false positives, and false negatives. The F1 score and confusion matrix are printed to the console, and the confusion matrix is plotted using matplotlib.

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow logging

import tensorflow as tf
import numpy as np
import pandas as pd  # Import pandas
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt  # Import matplotlib

# Load the trained model without compiling
model = load_model('resnet_model_final.h5', compile=False)

# Compile the model manually
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Prepare the validation data generator
val_datagen = ImageDataGenerator(rescale=1./255)
val_data = pd.read_csv('NewData/newval_data.csv')
val_generator = val_datagen.flow_from_dataframe(
    val_data,
    directory='NewData/newval',
    x_col='image_name',
    y_col='label',
    target_size=(224, 224),
    class_mode='binary',
    batch_size=16,
    shuffle=False  # Important to keep the order for evaluation
)

# Make predictions
val_preds = model.predict(val_generator)
val_preds = (val_preds > 0.5).astype(int)

# Calculate F1 score
f1 = f1_score(val_generator.classes, val_preds)
print(f"Validation F1 Score: {f1:.4f}")

# Calculate confusion matrix
conf_matrix = confusion_matrix(val_generator.classes, val_preds)
print("Confusion Matrix:")
print(conf_matrix)

# Plot confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=val_generator.class_indices.keys())
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()