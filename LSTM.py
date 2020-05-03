import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

# Import LSTM model
# https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM

# https://www.tensorflow.org/tutorials/text/text_classification_rnn
# pip install -q tf-nightly


# get dataset
# dataset, info = tfds.load('imdb_reviews/subwords8k', with_info=True,
#                           as_supervised=True)
# train_examples, test_examples = dataset['train'], dataset['test']


# import dataset
fashion_mnist = keras.datasets.fashion_mnist

# split dataset
(train_images, train_labels), (test_images,
                               test_labels) = fashion_mnist.load_data()

# encoder to convert text into numeric
# encoder = info.features['text'].encoder

# scale
train_images = train_images / 255.0
test_images = test_images / 255.0

print('Train images shaps: ', train_images.shape)
print('Train labels shape: ', train_labels.shape)

# # prepare training data
# BUFFER_SIZE = 10000
# BATCH_SIZE = 64

# # Use the padded_batch method to zero-pad the sequences to the length of the longest string in the batch:

# train_dataset = (train_examples
#                  .shuffle(BUFFER_SIZE)
#                  .padded_batch(BATCH_SIZE, padded_shapes=([None], [])))

# test_dataset = (test_examples
#                 .padded_batch(BATCH_SIZE,  padded_shapes=([None], [])))


# # Create the model
# # Build a tf.keras.Sequential model and start with an embedding layer.

# # An embedding layer stores one vector per word. When called, it converts the sequences of word indices to sequences of vectors. These vectors are trainable. After training (on enough data), words with similar meanings often have similar vectors.

# # A recurrent neural network (RNN) processes sequence input by iterating through the elements. RNNs pass the outputs from one timestep to their input—and then to the next.

# # The tf.keras.layers.Bidirectional wrapper can also be used with an RNN layer. This propagates the input forward and backwards through the RNN layer and then concatenates the output. This helps the RNN to learn long range dependencies.

model = tf.keras.Sequential([
    # tf.keras.layers.Embedding(encoder.vocab_size, 64),
    tf.keras.layers.LSTM(128, return_sequences=True,
                         dropout=0.2),
    # tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(128, dropout=0.2),
    # tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

# compile model
model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              optimizer=tf.keras.optimizers.Adam(lr=1e-3),
              metrics=['accuracy'])

#   decay=1e-5

# train model
# model.fit(train_dataset, epochs=10,
#           validation_data=test_dataset,
#           validation_steps=30)

# LST expects not 1D input data, but 2D.
model.fit(train_images, train_labels, epochs=10,
          validation_data=(test_images, test_labels))


# # evaluate
# test_loss, test_acc = model.evaluate(test_dataset)

# print('Test Loss: {}'.format(test_loss))
# print('Test Accuracy: {}'.format(test_acc))


# evaluate model accuracy
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

print('\nTest accuracy:', test_acc)
