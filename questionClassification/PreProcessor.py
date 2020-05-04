
from DatasetEmbedder import DatasetEmbedder
from TrainTestSplitter import TrainTestSplitter
import re


class PreProcessor:
    def __init__(self, file, trainSplitPercentage=0.8):
        self.file = file
        self.trainSplitPercentage = trainSplitPercentage  # 0.8

    def preProcess(self):
        data = self.__convertToFormatted()
        embeddedData = DatasetEmbedder(data, labelPosition='first')

        # trainTestSplitter
        tts = TrainTestSplitter(self.trainSplitPercentage, embeddedData.getPaddedData(
        ), embeddedData.getEmbeddedLabels())

        trainData, trainLabels, testData, testLabels = tts.getTrainTestSplit()

        return trainData, trainLabels, testData, testLabels

    def removeNonACII(self, sentence):
        return re.sub(r'[^\x00-\x7F]+', '', sentence)

    # convert data in text file or csv to arrays
    # something like
    # [
    #     ['ST:food', 'I love to eat pasta'],
    #     ['ST:clothes', 'My shirt is blue and my jacket is green'],
    #     ['TH:device', 'iPhones are the best smarphones'],
    #     ['TH:device', 'I use a macbook'],
    #     ['TH:device', 'I dont like beats headphones']
    # ]
    # for .txt
    def __convertToFormatted(self):
        with open(self.file, 'r', encoding="ISO-8859-1") as in_file:
            stripped = (line.strip() for line in in_file)

            data = []
            for line in stripped:
                if(line):
                    # if(self.file.split('.')[len(self.file-1) == 'txt']):
                    frags = line.split(' ')
                    # elif(self.file.split('.')[len(self.file-1) == 'txt']):
                    #     frags = line.split(',')
                    sentence = ' '.join(frags[1:])
                    dataLine = [frags[0], self.removeNonACII(sentence)]
                    data.append(dataLine)

            return data


# pp = PreProcessor()

# data = pp.preProcess('train_5500.txt')
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
