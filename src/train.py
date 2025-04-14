import os
os.environ["DEVICE"] = "GPU"
os.environ["METAL_DEVICE_WRAPPER_TYPE"] = "1"
os.environ["METAL_DEVICE_ID"] = "0"

import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from keras.optimizers import Adam

# Load data
train_data = pd.read_csv('NewData/newtrain_data.csv')
val_data = pd.read_csv('NewData/newval_data.csv')

# Image data generators
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_dataframe(
    train_data,
    directory='NewData/newtrain',
    x_col='image_name',
    y_col='label',
    target_size=(224, 224),
    class_mode='binary',
    batch_size=16  # Adjust as needed
)

val_generator = val_datagen.flow_from_dataframe(
    val_data,
    directory='NewData/newval',
    x_col='image_name',
    y_col='label',
    target_size=(224, 224),
    class_mode='binary',
    batch_size=16  # Adjust as needed
)

# Load ResNet50 model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = Flatten()(base_model.output)
x = Dense(1, activation='sigmoid')(x)
model = Model(inputs=base_model.input, outputs=x)

# Compile model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# Train model
model.fit(train_generator, validation_data=val_generator, epochs=100)

# Save model
model.save('resnet_model_new.h5')