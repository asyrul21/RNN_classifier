class TrainTestSplitter:
    def __init__(self, trainSplitPercentage, dataset, labels):
        self.trainSplitPercentage = trainSplitPercentage
        self.dataset = dataset
        self.labels = labels

        if(len(dataset) != len(labels)):
            raise ValueError("Dataset and label size must me the same.")

    def getTrainTestSplit(self):
        dataSize = len(self.dataset)

        trainSize = int(dataSize * self.trainSplitPercentage)
        # testSize = dataSize - trainSize

        trainData = self.dataset[:trainSize]
        trainLabels = self.labels[:trainSize]

        testData = self.dataset[trainSize:]
        testLabels = self.labels[trainSize:]

        return trainData, trainLabels, testData, testLabels
