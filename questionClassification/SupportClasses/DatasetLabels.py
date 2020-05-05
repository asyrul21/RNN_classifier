import numpy as np
from SupportClasses.LabelDictionary import LabelDictionary


class DatasetLabels:
    def __init__(self, dataset, labelPosition='last'):
        self.dataset = dataset
        self.labelPosition = labelPosition

        # all labels
        self.allLabels = self.__getAllLabels()
        self.uniquelabels = self.__getUniquelabels()
        self.labelDictionary = self.__getDictionary()

        # assert
        if (self.labelPosition != 'last' and self.labelPosition != 'first'):
            raise ValueError("Label position must either be first or last.")

    def __getAllLabels(self):
        labelIndex = 0
        allLabels = []

        for data in self.dataset:
            if(self.labelPosition == 'last'):
                labelIndex = len(data) - 1
            elif(self.labelPosition == 'first'):
                labelIndex = 0
            allLabels.append(data[labelIndex])

        return np.asarray(allLabels)

    def __getUniquelabels(self):
        return np.unique(self.allLabels)

    def __getDictionary(self):
        labelDict = LabelDictionary(self.uniquelabels)
        return labelDict.getDictionary()


# dataset = np.array([['ST:food', 'pasta', 'chicken wings', 'rice'],
#                     ['ST:clothes', 'shirt', 'jeans', 'jacket'], ['TH:device', 'phone', 'laptop', 'headphones'],  ['TH:device', 'phone', 'laptop', 'headphones'], ['TH:gadgets', 'phone', 'laptop', 'headphones']])

# labels = DatasetLabels(dataset, labelPosition='first')

# print(labels.labelDictionary)

# print(dataset)
