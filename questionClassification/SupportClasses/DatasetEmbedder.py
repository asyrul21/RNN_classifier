from SupportClasses.SentenceEmbedder import SentenceEmbedder
from SupportClasses.LabelEmbedder import LabelEmbedder
from SupportClasses.ParentLabelEmbedder import ParentLabelEmbedder
from SupportClasses.ChildLabelEmbedder import ChildLabelEmbedder
from SupportClasses.VectorPadder import VectorPadder
import numpy as np


class DatasetEmbedder:
    def __init__(self, dataset, labelPosition='last', embeddingMode='none'):
        self.dataset = dataset
        self.labelPosition = labelPosition
        self.embeddingModes = ['none', 'parent', 'child']
        self.embeddingMode = embeddingMode
        self.sentenceEmbedder = self.__createSentenceEmbedder()
        self.labelEmbedder = self.__createLabelEmbedder()
        self.vectorPadder = self.__createVectorPadder()

        if(self.embeddingMode not in self.embeddingModes):
            raise ValueError(
                'Embedding mode must only either be none, parent or child.')

    def __createSentenceEmbedder(self):
        return SentenceEmbedder()

    def __createLabelEmbedder(self):
        if(self.embeddingMode == 'none'):
            return LabelEmbedder(self.dataset, self.labelPosition)
        elif(self.embeddingMode == 'parent'):
            return ParentLabelEmbedder(self.dataset, self.labelPosition)
        elif(self.embeddingMode == 'child'):
            return ChildLabelEmbedder(self.dataset, self.labelPosition)

    def __createVectorPadder(self):
        return VectorPadder(self.getEmbeddedData())

    # labels
    def getLabels(self):
        return self.labelEmbedder.getLabels()

    def getLabelDictionary(self):
        return self.labelEmbedder.getLabelDictionay()

    def getEmbeddedLabels(self):
        return self.labelEmbedder.embedLabels()

    # data
    def getEmbeddedData(self):
        embedding = []

        for data in self.dataset:
            sentence = data[1]
            sentenceVector = self.sentenceEmbedder.getVector(sentence)
            embedding.append(sentenceVector)

        return np.asarray(embedding)

    def getPaddedData(self):
        return self.vectorPadder.padVectors()


# tests
# dataset = np.array([['ST:food', 'I love to eat pasta'],
#                     ['ST:clothes', 'My shirt is blue and my jacket is green'], ['TH:device', 'iPhones are the best smarphones'],  ['TH:device', 'I use a macbook'], ['TH:device', 'I dont like beats headphones']])


# print(dataset)

# de = DatasetEmbedder(dataset, labelPosition='first', embeddingMode='none')
# # resData = de.getEmbeddedData()
# resLabel = de.getEmbeddedLabels()

# # print(resData)
# print(de.getLabelDictionary())
# print(resLabel)
