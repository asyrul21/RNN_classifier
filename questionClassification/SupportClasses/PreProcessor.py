from SupportClasses.DatasetEmbedder import DatasetEmbedder
from SupportClasses.TrainTestSplitter import TrainTestSplitter


class PreProcessor:
    # def __init__(self, classifyToParent=False, classifyToChild=False):
    #     if(classifyToParent or classifyToChild):
    #         self.datasetEmbedder = self.__createDatasetEmbedder()
    #     else:
    #         self.datasetEmbedder = self.__createHierarchicalDatasetEmbedder()

    # __createDatasetEmbedder(self):
    #     return DatasetEmbedder()

    def preProcess(self, data, trainSplitPercentage=0.8, labelPosition='last', embeddingMode='none'):
        dataEmbedder = DatasetEmbedder(data, labelPosition, embeddingMode)

        # trainTestSplitter
        tts = TrainTestSplitter(trainSplitPercentage,
                                dataEmbedder.getPaddedData(),
                                dataEmbedder.getEmbeddedLabels())
        (trainData, trainLabels,
            testData, testLabels) = tts.getTrainTestSplit()

        # get label dictionary
        labelDictionary = dataEmbedder.getLabelDictionary()

        return trainData, trainLabels, testData, testLabels, labelDictionary
