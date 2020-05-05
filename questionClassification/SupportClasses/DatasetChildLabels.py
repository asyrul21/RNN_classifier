from SupportClasses.DatasetLabels import DatasetLabels
from SupportClasses.LabelDictionary import LabelDictionary
import numpy as np


class DatasetChildLabels(DatasetLabels):
    def __init__(self, dataset, labelPosition='last'):
        # super
        DatasetLabels.__init__(self, dataset, labelPosition)

        # override
        self.allLabels = self.__getChildLabels()
        self.uniquelabels = self.__getUniquelabels()
        self.labelDictionary = self.__getChildLabelDictionary()

    # child labels
    def __getChildLabels(self):
        childlabels = []
        for item in self.allLabels:
            childlabels.append(item.split(':')[1])
        return childlabels

    def __getChildLabelDictionary(self):
        labelDict = LabelDictionary(np.unique(self.allLabels))
        return labelDict.getDictionary()

    def __getUniquelabels(self):
        return np.unique(self.allLabels)


# test
# dataset = np.array([['ST:food', 'pasta', 'chicken wings', 'rice'],
#                     ['ST:clothes', 'shirt', 'jeans', 'jacket'], ['TH:device', 'phone', 'laptop', 'headphones'],  ['TH:device', 'phone', 'laptop', 'headphones'], ['TH:gadgets', 'phone', 'laptop', 'headphones']])

# labels = DatasetChildLabels(dataset, labelPosition='first')
# print(labels.allLabels)
# print(labels.uniquelabels)
# print(labels.labelDictionary)
