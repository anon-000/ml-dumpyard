import tensorflow as tf
from keras.preprocessing import image as image_processor
import os
import numpy as np


# Path to the saved model file
model_path = 'cats_dogs_model.h5'

# Load the saved model
model = tf.keras.models.load_model(model_path)


# classes = model.predict(images, batch_size=10)
# print(classes[0])
# if classes[0] > 0.5:
#     print(fn + " is a human")
# else:
#     print(fn + " is a horse")

directory = './'
# Get a list of all files in the directory
files = os.listdir(directory)

# Filter out only image files
image_files = [f for f in files if f.endswith(
    '.jpeg') or f.endswith('.png') or f.endswith('.jpg')]

print(image_files)


# Iterate over each image file
for image_file in image_files:

    # predicting images
    image_path = os.path.join(directory, image_file)
    img = image_processor.load_img(image_path, target_size=(150, 150))
    x = image_processor.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    images = np.vstack([x])
    classes = model.predict(images, batch_size=10)
    print(classes[0])
    if classes[0] > 0.5:
        print(image_file + " is a Dog")
    else:
        print(image_file + " is a Cat")
