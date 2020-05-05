import tensorflow as tf
from tensorflow import keras
import numpy as np
from SupportClasses.DatasetTxt import DatasetTxt
from SupportClasses.DatasetCsv import DatasetCsv

print("Question Classifier")

# 2 args = file and label position - first or last
# data = DatasetTxt('train_5500.txt', 'first')
# data.filterByParentClass('HUM', save=True)

data = DatasetCsv('HUM.csv', 'first')
# print(data.formattedData[:10])

# if you want to filter data by classes
# data.filterByParentClass('HUM', save=True)

########################
# Data Loading
########################
# 2 args = embeddingMode(none, parent, or child) and traindataSlplit,
trainData, trainLabels, testData, testLabels = data.load('child')
classSize = len(data.labelDictionary)

print('Label dictionary size:', classSize)
print(data.labelDictionary)
# print(trainData.shape)
# print(testData.shape)

########################
# NN Model Configurations
########################
model = tf.keras.Sequential([
    # tf.keras.layers.LSTM(128, return_sequences=True,
    #                      dropout=0.4),
    # tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(128, dropout=0.2),
    # tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(classSize, activation='softmax')
])

# compile model
model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              optimizer=tf.keras.optimizers.Adam(lr=1e-3),
              metrics=['accuracy'])

#   decay=1e-5

# LST expects not 1D input data, but 2D.
model.fit(trainData, trainLabels, epochs=30,
          validation_data=(testData, testLabels))

# evaluate model accuracy
test_loss, test_acc = model.evaluate(testData, testLabels, verbose=2)
print('\nTest accuracy:', test_acc)
