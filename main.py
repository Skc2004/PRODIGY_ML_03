import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

train_dir = 'train'
test_dir = 'test1'

img_size = (150, 150)
batch_size = 32

def create_dataset(directory):
    images = []
    labels = []
    for filename in os.listdir(directory):
        if filename.startswith('cat'):
            labels.append(0)
        elif filename.startswith('dog'):
            labels.append(1)
        else:
            continue

        img_path = os.path.join(directory, filename)
        img = load_img(img_path, target_size=img_size)
        img = img_to_array(img)
        img = img / 255.0
        images.append(img)

    return np.array(images), np.array(labels)

images, labels = create_dataset(train_dir)

X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

train_datagen = ImageDataGenerator()
val_datagen = ImageDataGenerator()

train_generator = train_datagen.flow(X_train, y_train, batch_size=batch_size)
validation_generator = val_datagen.flow(X_val, y_val, batch_size=batch_size)

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid'),
])

model.compile(loss='binary_crossentropy',
              optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=1e-4),
              metrics=['accuracy'])

history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    validation_data=validation_generator,
    validation_steps=len(validation_generator),
    epochs=20,
)

model.save('cats_and_dogs_classifier.h5')
