from SupportClasses.IDataset import IDataset
from SupportClasses.PreProcessor import PreProcessor


class DatasetTxt(IDataset):
    def __init__(self, file, labelPosition):
        # labelPosition='last' or 'first'
        self.file = file
        self.labelPosition = labelPosition
        self.preProcessor = self.__createPreProcessor()
        self.formattedData = self._convertToFormatted()
        self.labelDictionary = {}

    def __createPreProcessor(self):
        return PreProcessor()

    def load(self, embeddingMode='none', trainSplitPercentage=0.8):
        (trainData, trainLabels,
            testData, testLabels, labelDictionary) = self.preProcessor.preProcess(
                self.formattedData, trainSplitPercentage, self.labelPosition, embeddingMode)

        self.labelDictionary = labelDictionary
        return trainData, trainLabels, testData, testLabels

    def _convertToFormatted(self):
        with open(self.file, 'r', encoding="ISO-8859-1") as in_file:
            stripped = (line.strip() for line in in_file)

            data = []
            for line in stripped:
                if(line):
                    # if(self.file.split('.')[len(self.file-1) == 'txt']):
                    frags = line.split(' ')
                    sentence = ' '.join(frags[1:])
                    dataLine = [frags[0], self.removeNonACII(sentence)]
                    data.append(dataLine)

            return data


# dtx = DatasetTxt('train_5500.txt')
# data = dtx.load()

# print(data)
