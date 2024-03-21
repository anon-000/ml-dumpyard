

# Dataset
# https://www.kaggle.com/datasets/karakaggle/kaggle-cat-vs-dog-dataset

import os

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator


model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(150, 150, 3)),
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])


model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(learning_rate=0.001),
              metrics=['accuracy'])

TRAINING_DIR = 'dataset/training'
# All images will be rescaled by 1./255
train_datagen = ImageDataGenerator(rescale=1./255)
# Flow training images in batches of 128 using train_datagen generator
train_generator = train_datagen.flow_from_directory(
    TRAINING_DIR,  # This is the source directory for training images
    target_size=(150, 150),  # All images will be resized to 150x150
    batch_size=100,
    class_mode='binary')

TESTING_DIR = 'dataset/testing'
# All images will be rescaled by 1./255
test_datagen = ImageDataGenerator(rescale=1./255)
# Flow training images in batches of 128 using train_datagen generator
test_generator = test_datagen.flow_from_directory(
    TESTING_DIR,  # This is the source directory for training images
    target_size=(150, 150),  # All images will be resized to 150x150
    batch_size=100,
    class_mode='binary')

history = model.fit(train_generator,
                              epochs=15,
                              verbose=1,
                              validation_data=test_generator)

model.save('cats_dogs_model.h5')


# -----------------------------------------------------------
# Retrieve a list of list results on training and test data
# sets for each training epoch
# -----------------------------------------------------------
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))  # Get number of epochs

# ------------------------------------------------
# Plot training and validation accuracy per epoch
# ------------------------------------------------
plt.plot(epochs, acc, 'r', "Training Accuracy")
plt.plot(epochs, val_acc, 'b', "Validation Accuracy")
plt.title('Training and validation accuracy')
plt.figure()

# ------------------------------------------------
# Plot training and validation loss per epoch
# ------------------------------------------------
plt.plot(epochs, loss, 'r', "Training Loss")
plt.plot(epochs, val_loss, 'b', "Validation Loss")
plt.figure()



