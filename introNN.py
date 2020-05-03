# TensorFlow and tf.keras
# https://www.tensorflow.org/tutorials/keras/classification
import tensorflow as tf
from tensorflow import keras

# to save and load model
import os

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print('Tensorflow version: ' + tf.__version__)

# import dataset
fashion_mnist = keras.datasets.fashion_mnist

# split dataset
(train_images, train_labels), (test_images,
                               test_labels) = fashion_mnist.load_data()

# print(train_images)

# store label classes
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress',
               'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print('Train labels shape: ', train_labels.shape)
print('Train images shaps: ', train_images.shape)

# get length
# print(len(train_labels))

# display image data
# plt.figure()
# plt.imshow(train_images[0])
# plt.colorbar()
# plt.grid(False)
# plt.show()

# image values are 0-255
# scale values to 0-1

train_images = train_images / 255.0
test_images = test_images / 255.0

# display the first 25 images from the training set
# plt.figure(figsize=(10, 10))
# for i in range(25):
#     plt.subplot(5, 5, i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(train_images[i], cmap=plt.cm.binary)
#     plt.xlabel(class_names[train_labels[i]])
# plt.show()

# Build the model

# Building the neural network requires configuring the layers of the model, then compiling the model.

# Set up the layers

# The basic building block of a neural network is the layer. Layers EXTRACT REPRESENTATIONS from the data fed into them.

# Most of deep learning consists of chaining together simple layers. Most layers, such as tf.keras.layers.Dense, have parameters that are learned during training.

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10)
])

# first layer is Flatten. It converts 2D data (28 x 28) to 1 D data of 784 pixels.

# the other two layers are two neural Keras Dense layers.
# the first neural layer has 128 neurons.
# the second (and last) returns a logit array of length 10. EACH NODE CONTAINS A SCORE that indicates the current image belongs to one of the 10 classes.

# Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(
                  from_logits=True),
              metrics=['accuracy'])

# 3 components: optimiser, loss function and metric.
# Loss function measures how accurate the model is during training. You want to minimize this function to "steer" the model in the right direction.
# Optimizer —This is how the model is updated based on the data it sees and its loss function.
# Metrics —Used to monitor the training and testing steps. The following example uses accuracy, the fraction of the images that are correctly classified.

# Train the model
model.fit(train_images, train_labels, epochs=20)


# evaluate model accuracy
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print('\nTest accuracy:', test_acc)

# Savinf model
# print(model.summary())
model.save('models/intro_model')
print('Model saved successfully.')
