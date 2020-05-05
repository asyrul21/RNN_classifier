# RNN_classifier

This classifier was built to support Chatbot's intent classificiation. We use tensorflow with Python3. The steps will be outlined below:

## Tensorflow Installation

1. Update PIP
```bash
pip install --upgrade pip
```

2. Install Tensorflow. This will take some time.
```bash
pip install tensorflow
```

3. Print Tensorflow version
```python
print(tf.__version__);
```

## First Neural Network
1. First we need to load the data and do some preprocessing. The data needs to be scaled to only have values from 0-1.

```python
# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

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
print(len(train_labels))

# display image data
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

# image values are 0-255
# scale values to 0-1

train_images = train_images / 255.0
test_images = test_images / 255.0

# display the first 25 images from the training set
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()
```

2. Now we need to build the model. Building the neural network requires configuring the layers of the model, then compiling the model.
  * Set up layers
```python
model = keras.Sequential([
keras.layers.Flatten(input_shape=(28, 28)),
keras.layers.Dense(128, activation='relu'),
keras.layers.Dense(10)
])

# first layer is Flatten. It converts 2D data (28 x 28) to 1 D data of 784 pixels.

# the other two layers are two neural Keras Dense layers.
# the first neural layer has 128 neurons.
# the second (and last) returns a logit array of length 10. EACH NODE CONTAINS A SCORE that indicates the current image belongs to one of the 10 classes
```
  * Compile the model
```python
model.compile(optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=True),
            metrics=['accuracy'])

# 3 components: optimiser, loss function and metric.
# Loss function measures how accurate the model is during training. You want to minimize this function to "steer" the model in the right direction.
# Optimizer —This is how the model is updated based on the data it sees and its loss function.
# Metrics —Used to monitor the training and testing steps. The following example uses accuracy, the fraction of the images that are correctly classified.
```

3. Train the model
```python
model.fit(train_images, train_labels, epochs=10)
```

4. Evaluate model accuracy with test data
```python
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print('\nTest accuracy:', test_acc)
```

5. Make predictions with the model
We need to first attach a Softmax layer to convert model's linear output (Logits) to probabilities.
```python
probability_model = tf.keras.Sequential([model, 
                                         tf.keras.layers.Softmax()])

                                         
# predict
predictions = probability_model.predict(test_images)                                         
```

# Saving and Loading Model

https://www.tensorflow.org/tutorials/keras/save_and_load

1. first we need to install pyyaml and h5py
```bash
pip install -q pyyaml h5py  # Required to save models in HDF5 format
```

2. Save model after training it
```python
# This may generate warnings related to saving the state of the optimizer.
# These warnings (and similar warnings throughout this notebook)
# are in place to discourage outdated usage, and can be ignored.

model.save('models/intro_model')
print('Model saved successfully.')
```

3. Load the model in a new file
```python
# loading a trained model
model = tf.keras.models.load_model('models/intro_model')

# Check its architecture
# print(model.summary())

# evaluate model accuracy
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print('\nTest accuracy:', test_acc)
```

# Tensorflow LSTM

RNN model can be built such as below:
```python
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(encoder.vocab_size, 64),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])
```
We build a tf.keras.Sequential model and start with the embedding layer. An embedding layer stores ONE VECTOR PER WORD. When called, it **CONVERT THE SEQUENCE OF WORD INDICES TO SEQUENCES OF VECTORSc**.

```python
sample_string = 'Hello TensorFlow.'

encoded_string = encoder.encode(sample_string)
print('Encoded string is {}'.format(encoded_string))

original_string = encoder.decode(encoded_string)
print('The original string: "{}"'.format(original_string))
```
```bash
Encoded string is [4025, 222, 6307, 2327, 4043, 2120, 7975]  <= **sequence of word indices**
The original string: "Hello TensorFlow."
```
# Question Classifier Example
1. To run this example you must first ensure that the glove file *glove.6B.100d.txt* is present in the QuestionClassification folder. Due to the huge size I cant push this file to git repo.

2. In QuestionClassifier.py, create dataset instance.
```python
# 2 args = file and label position - first or last
data = DatasetCsv('HUM.csv', 'first')

#if your data file is txt use
data = DatasetTxt('train_5500.txt', 'first')
```

3. You can also filter the data by parent label. In first argument specify the name of parent class. In the second argument you can set if you want to save the filtered dataset to csv. Otherwise, the function just return a list of data.
```python
data.filterByParentClass('HUM', save=True)
```

4. You can now get train and test data by loading the data. You can specify two arguments: first the embedding mode - either none, parent or child. If you want to load the labels based on the child label, specify 'child'. The secodn argument is the training data split. By default it is 0.8, meaning 80% training and 20% test. You can change this by giving a value between 0-1.
```python
# 2 args = embeddingMode(none, parent, or child) and traindataSlplit,
trainData, trainLabels, testData, testLabels = data.load('child')
```
  - You can also view the label dictionary
  ```python
  classSize = len(data.labelDictionary)

  print('Label dictionary size:', classSize)
  print(data.labelDictionary)
  ```

5. You can now build, compile and train an LSTM NN model
```python
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(128, dropout=0.2),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(classSize, activation='softmax')
])

# compile model
model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              optimizer=tf.keras.optimizers.Adam(lr=1e-3),
              metrics=['accuracy'])

# LST expects not 1D input data, but 2D.
model.fit(trainData, trainLabels, epochs=30,
          validation_data=(testData, testLabels))

# evaluate model accuracy
test_loss, test_acc = model.evaluate(testData, testLabels, verbose=2)
print('\nTest accuracy:', test_acc)
```
6. You can save the model if you want by referring 'Save and Load Model' section above.


# Refences
1. Understanding Logits

"If the model is solving a multi-class classification problem, logits typically become an input to the softmax function"

https://developers.google.com/machine-learning/glossary#logits


2. Romove NON AsCII from string

https://stackoverflow.com/questions/20078816/replace-non-ascii-characters-with-a-single-space

```python
def remove_non_ascii_2(text):
    return re.sub(r'[^\x00-\x7F]',' ', text)
```

3. Git Large File Problem Solution
https://medium.com/@marcosantonocito/fixing-the-gh001-large-files-detected-you-may-want-to-try-git-large-file-storage-43336b983272

```bash
git filter-branch -f --index-filter 'git rm --cached --ignore-unmatch questionClassification/glove.6B.100d.txt'
```

4. Importing classes from different folder

https://stackoverflow.com/questions/456481/cant-get-python-to-import-from-a-different-folder

Create __init__.py at the root module

