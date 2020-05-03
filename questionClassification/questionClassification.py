from csv import reader

import tensorflow as tf
from tensorflow import keras
import numpy as np

# pip install tensorflow_datasets
import tensorflow_datasets as tfds

print("Question Classifier")
allQuestions = []

# open file
with open('train_5500.csv', 'r') as read_obj:
    csv_reader = reader(read_obj)
    header = next(csv_reader)
    # Check file as empty
    if header != None:
        # Iterate over each row after the header in the csv
        for row in csv_reader:
            allQuestions.append(row)


train_data = allQuestions[:int(len(allQuestions) * 0.8)]
test_data = allQuestions[int(-len(allQuestions) * 0.2):]

print(len(train_data))
print(len(test_data))
