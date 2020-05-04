# from DatasetLabels import DatasetLabels
from HierarchicalDatasetLabels import HierarchicalDatasetLabels
import numpy as np


class LabelEmbedder:
    def __init__(self, dataset, labelPosition='last'):
        self.dataset = dataset
        self.labelPosition = labelPosition
        self.datasetLabels = self.__createDatasetLabels()
        self.labelDictionary = self.datasetLabels.labelDictionary

    def __createDatasetLabels(self):
        # return DatasetLabels(self.dataset, self.labelPosition)
        return HierarchicalDatasetLabels(self.dataset, self.labelPosition)

    def getAllLabels(self):
        return self.datasetLabels.allLabels

    def getLabelDictionay(self):
        return self.labelDictionary

    def getParentLabels(self):
        return self.datasetLabels.parentLabels

    def getParentLabelDictionary(self):
        return self.datasetLabels.parentLabelDictionary

    def getChildLabels(self):
        return self.datasetLabels.childLabels

    def getChildLabelDictionary(self):
        return self.datasetLabels.childLabelDictionary

    # embedding
    def embedLabelToAllLabels(self):
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
