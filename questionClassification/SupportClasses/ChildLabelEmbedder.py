from SupportClasses.LabelEmbedder import LabelEmbedder
from SupportClasses.DatasetChildLabels import DatasetChildLabels
from SupportClasses.ILabel import ILabel
import numpy as np


class ChildLabelEmbedder(ILabel):
    def __init__(self, dataset, labelPosition='last'):
        # # super
        # LabelEmbedder.__init__(self, dataset, labelPosition)

        self.dataset = dataset
        self.labelPosition = labelPosition
        self.datasetLabels = self._createDatasetLabels()
        self.labelDictionary = self.datasetLabels.labelDictionary

    # override method
    def _createDatasetLabels(self):
        return DatasetChildLabels(self.dataset, self.labelPosition)

    def getLabels(self):
        return self.datasetLabels.allLabels

    def getLabelDictionay(self):
        return self.labelDictionary

    # embedding
    def embedLabels(self):
        labelIndices = []
        for label in self.datasetLabels.allLabels:
            # childLabel = label.split(':')[1]
            labelIndices.append(self.labelDictionary[label])

        return np.asarray(labelIndices)


# dataset = np.array([
#     ['ST:food', 'I love to eat pasta'],
#     ['ST:clothes', 'My shirt is blue and my jacket is green'],
#     ['TH:device', 'iPhones are the best smarphones'],
#     ['TH:device', 'I use a macbook'],
#     ['TH:device', 'I dont like beats headphones']
# ])

# # print(dataset)

# le = ChildLabelEmbedder(dataset, 'first')

# # # print(le.getAllLabels())
# # # print(le.getParentLabels())
# # # print(le.getChildLabels())

# print(le.labelDictionary)
# print(le.embedLabels())
