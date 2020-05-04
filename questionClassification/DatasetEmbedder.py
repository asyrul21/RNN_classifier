from SentenceEmbedder import SentenceEmbedder
from LabelEmbedder import LabelEmbedder
from VectorPadder import VectorPadder
import numpy as np


class DatasetEmbedder:
    def __init__(self, dataset, labelPosition='last'):
        self.dataset = dataset
        self.labelPosition = labelPosition
        self.sentenceEmbedder = self.__createSentenceEmbedder()
        self.labelEmbedder = self.__createLabelEmbedder()
        self.vectorPadder = self.__createVectorPadder()

    def __createSentenceEmbedder(self):
        return SentenceEmbedder()

    def __createLabelEmbedder(self):
        return LabelEmbedder(self.dataset, self.labelPosition)

    def __createVectorPadder(self):
        return VectorPadder(self.getEmbeddedData())

    # labels
    def getAllLabels(self):
        return self.labelEmbedder.getAllLabels()

    def getEmbeddedLabels(self):
        return self.labelEmbedder.embedLabelToAllLabels()

    def getParentLabels(self):
        return self.labelEmbedder.getParentLabels()

    # data
    def getEmbeddedData(self):
        embedding = []

        for data in self.dataset:
            sentence = data[1]
            sentenceVector = self.sentenceEmbedder.getVector(sentence)
            # print('Length of sentence vector:', len(sentenceVector))
            embedding.append(sentenceVector)

        return np.asarray(embedding)

    def getPaddedData(self):
        return self.vectorPadder.padVectors()


# tests
# dataset = np.array([['ST:food', 'I love to eat pasta'],
#                     ['ST:clothes', 'My shirt is blue and my jacket is green'], ['TH:device', 'iPhones are the best smarphones'],  ['TH:device', 'I use a macbook'], ['TH:device', 'I dont like beats headphones']])


# de = DatasetEmbedder(dataset, labelPosition='first')

# # resData = de.getEmbeddedData()
# resLabel = de.getEmbeddedLabels()

# print(resData)

# print(de.getAllLabels())
# print(resLabel)

# print(de.getParentLabels())
# print(resLabel)

# paddedData = de.getPaddedData()

# for item in paddedData:
#     print(len(item))

# print(paddedData)
