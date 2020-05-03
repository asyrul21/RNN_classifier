# TensorFlow and tf.keras
# https://www.tensorflow.org/tutorials/keras/save_and_load
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


# loading a trained model
model = tf.keras.models.load_model('models/intro_model')

# Check its architecture
# print(model.summary())

# evaluate model accuracy
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print('\nTest accuracy:', test_acc)
