from abc import abstractmethod
from SupportClasses.DatasetLabels import DatasetLabels
from SupportClasses.ILabel import ILabel
import numpy as np


class LabelEmbedder(ILabel):
    def __init__(self, dataset, labelPosition='last'):
        self.dataset = dataset
        self.labelPosition = labelPosition
        self.datasetLabels = self._createDatasetLabels()
        self.labelDictionary = self.datasetLabels.labelDictionary

    def _createDatasetLabels(self):
        return DatasetLabels(self.dataset, self.labelPosition)

    def getLabels(self):
        return self.datasetLabels.allLabels

    def getLabelDictionay(self):
        return self.labelDictionary

    # embedding
    def embedLabels(self):
        labelIndices = []
        for label in self.datasetLabels.allLabels:
            labelIndices.append(self.labelDictionary[label])

        return np.asarray(labelIndices)


# dataset = np.array([['food', 'I love to eat pasta'],
#                     ['clothes', 'My shirt is blue and my jacket is green'], ['device', 'iPhones are the best smarphones'],  ['device', 'I use a macbook'], ['device', 'I dont like beats headphones']])

# le = LabelEmbedder(dataset, labelPosition='first')
# print(dataset)
# print(le.getAllLabels())
# print(le.getLabelDictionay())
# print(le.embedLabelToParentLabels())
