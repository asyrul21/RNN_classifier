from DatasetLabels import DatasetLabels
from LabelDictionary import LabelDictionary
import numpy as np


class HierarchicalDatasetLabels(DatasetLabels):
    def __init__(self, dataset, labelPosition='last'):
        # super
        DatasetLabels.__init__(self, dataset, labelPosition)

        # parent labels
        self.parentLabels = self.__getParentLabels()
        self.parentLabelDictionary = self.__getParentLabelDictionary()

        # child labels
        self.childLabels = self.__getChildLabels()
        self.childLabelDictionary = self.__getChildLabelDictionary()

    # parent labels
    def __getParentLabels(self):
        parentlabels = []
        for item in self.allLabels:
            parentlabels.append(item.split(':')[0])
        return parentlabels

    def __getParentLabelDictionary(self):
        labelDict = LabelDictionary(np.unique(self.parentLabels))
        return labelDict.getDictionary()

    # child labels
    def __getChildLabels(self):
        childlabels = []
        for item in self.allLabels:
            childlabels.append(item.split(':')[1])
        return childlabels

    def __getChildLabelDictionary(self):
        labelDict = LabelDictionary(np.unique(self.childLabels))
        return labelDict.getDictionary()


# test
# dataset = np.array([['ST:food', 'pasta', 'chicken wings', 'rice'],
#                     ['ST:clothes', 'shirt', 'jeans', 'jacket'], ['TH:device', 'phone', 'laptop', 'headphones'],  ['TH:device', 'phone', 'laptop', 'headphones'], ['TH:gadgets', 'phone', 'laptop', 'headphones']])

# labels = HerarchicalDatasetLabels(dataset, labelPosition='first')
# print(labels.parentLabelDictionary)
# print(labels.childLabelDictionary)
# print(labels.labelDictionary)
