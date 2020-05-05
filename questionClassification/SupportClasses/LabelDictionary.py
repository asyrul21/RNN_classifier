import numpy as np


class LabelDictionary:
    def __init__(self, uniqueLabels):
        self.uniqueLabels = uniqueLabels
        self.dictionary = self.getDictionary()

    def getDictionary(self):
        # return dict(zip(np.arange(len(self.uniqueLabels)), self.uniqueLabels))
        return dict(zip(self.uniqueLabels, np.arange(len(self.uniqueLabels))))


# labels = np.array(['clothes', 'device', 'food'])
# labelDict = LabelDictionary(labels)

# print(labelDict.dictionary)
