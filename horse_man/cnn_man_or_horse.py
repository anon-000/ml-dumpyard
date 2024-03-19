import os

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_horse_dir = os.path.join('/tmp/horse-or-human/horses')

train_human_dir = os.path.join('/tmp/horse-or-human/humans')


train_horse_names = os.listdir(train_horse_dir)
# print(train_horse_names[:10])
train_human_names = os.listdir(train_human_dir)
# print(train_human_names[:10])


print('total training horse images:', len(train_horse_names))
print('total training human images:', len(train_human_names))


# TRAINING_DIR = "/tmp/horse-or-human/"
# training_datagen = ImageDataGenerator(
#     rescale=1./255,
#     rotation_range=40,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True,
#     fill_mode='nearest')

# ----------------------------------- START OF IMAGE PLOT --------------->>>>
# Parameters for our graph; we'll output images in a 4x4 configuration
# nrows = 4
# ncols = 4

# # Index for iterating over images
# pic_index = 0

# # Set up matplotlib fig, and size it to fit 4x4 pics
# fig = plt.gcf()
# fig.set_size_inches(ncols * 4, nrows * 4)

# pic_index += 8


# next_horse_pix = [os.path.join(train_horse_dir, fname)
#                   for fname in train_horse_names[pic_index-8:pic_index]]


# # print(next_horse_pix)
# next_human_pix = [os.path.join(train_human_dir, fname)
#                   for fname in train_human_names[pic_index-8:pic_index]]


# # print(list(enumerate(next_horse_pix+next_human_pix)))
# for i, img_path in enumerate(next_horse_pix+next_human_pix):
#     # Set up subplot; subplot indices start at 1
#     sp = plt.subplot(nrows, ncols, i + 1)
#     sp.axis('Off')  # Don't show axes (or gridlines)

#     img = mpimg.imread(img_path)
#     # print(type(img))
#     plt.imshow(img)

# plt.show()
# ----------------------------------- END OF IMAGE PLOT --------------->>>>


model = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 300x300 with 3 bytes color
    # This is the first convolution
    tf.keras.layers.Input(shape=(300, 300, 3)),
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The second convolution
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The third convolution
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The fourth convolution
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The fifth convolution
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='relu'),
    # Only 1 output neuron. It will contain a value from 0-1 where 0 for 1 class ('horses') and 1 for the other ('humans')
    tf.keras.layers.Dense(1, activation='sigmoid')
])


model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(learning_rate=0.001),
              metrics=['acc'])


# All images will be rescaled by 1./255
train_datagen = ImageDataGenerator(rescale=1./255)

# Flow training images in batches of 128 using train_datagen generator
train_generator = train_datagen.flow_from_directory(
    '/tmp/horse-or-human/',  # This is the source directory for training images
    target_size=(300, 300),  # All images will be resized to 150x150
    batch_size=128,
    # Since we use binary_crossentropy loss, we need binary labels
    class_mode='binary')


history = model.fit(
    train_generator,
    steps_per_epoch=8,
    epochs=15,
    verbose=1)


model.save('horse_man.h5')
