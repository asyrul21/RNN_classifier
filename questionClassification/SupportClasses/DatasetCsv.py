from SupportClasses.IDataset import IDataset
from SupportClasses.PreProcessor import PreProcessor
import csv


class DatasetCsv(IDataset):
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
        with open(self.file, 'r', encoding="ISO-8859-1") as csvFile:
            csv_reader = csv.reader(csvFile, delimiter=',')
            # header = next(csv_reader)

            data = []
            for row in csv_reader:
                data.append(row)

            return data


# dtx = DatasetTxt('train_5500.txt')
# data = dtx.load()

# print(data)


# for csv
#  with open('train_5500.csv', 'r') as read_obj:
#     csv_reader = reader(read_obj)
#     header = next(csv_reader)
#     # Check file as empty
#     if header != None:
#         # Iterate over each row after the header in the csv
#         for row in csv_reader:
#             allQuestions.append(row)
