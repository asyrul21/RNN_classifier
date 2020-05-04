import tensorflow as tf
from tensorflow import keras
import numpy as np

from PreProcessor import PreProcessor

print("Question Classifier")

pp = PreProcessor('train_5500.txt')
trainData, trainLabels, testData, testLabels = pp.preProcess()

# print(len(trainData))
# print(len(trainLabels))
# print(len(testData))
# print(len(testLabels))

print(trainData.shape)
print(testData.shape)

# model
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(128, return_sequences=True,
                         dropout=0.4),
    # tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(128, dropout=0.4),
    # tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(50, activation='softmax')
])

# compile model
model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              optimizer=tf.keras.optimizers.Adam(lr=1e-3),
              metrics=['accuracy'])

#   decay=1e-5

# LST expects not 1D input data, but 2D.
model.fit(trainData, trainLabels, epochs=100,
          validation_data=(testData, testLabels))

# evaluate model accuracy
test_loss, test_acc = model.evaluate(testData, testLabels, verbose=2)
print('\nTest accuracy:', test_acc)
