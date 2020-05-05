from SupportClasses.DatasetLabels import DatasetLabels
from SupportClasses.LabelDictionary import LabelDictionary
import numpy as np


class DatasetParentLabels(DatasetLabels):
    def __init__(self, dataset, labelPosition='last'):
        # super
        DatasetLabels.__init__(self, dataset, labelPosition)

        # override
        self.allLabels = self.__getParentLabels()
        self.uniquelabels = self.__getUniquelabels()
        self.labelDictionary = self.__getParentLabelDictionary()

    # parent labels
    def __getParentLabels(self):
        parentlabels = []
        for item in self.allLabels:
            parentlabels.append(item.split(':')[0])
        return parentlabels

    def __getParentLabelDictionary(self):
        labelDict = LabelDictionary(np.unique(self.allLabels))
        return labelDict.getDictionary()

    def __getUniquelabels(self):
        return np.unique(self.allLabels)


# test
# dataset = np.array([['ST:food', 'pasta', 'chicken wings', 'rice'],
#                     ['ST:clothes', 'shirt', 'jeans', 'jacket'], ['TH:device', 'phone', 'laptop', 'headphones'],  ['TH:device', 'phone', 'laptop', 'headphones'], ['TH:gadgets', 'phone', 'laptop', 'headphones']])

# labels = DatasetParentLabels(dataset, labelPosition='first')
# print(labels.allLabels)
# print(labels.uniquelabels)
# print(labels.labelDictionary)
