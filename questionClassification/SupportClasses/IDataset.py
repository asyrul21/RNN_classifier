from abc import ABC, abstractmethod
import re
import csv

# from DatasetTxt import DatasetTxt


class IDataset(ABC):
    @abstractmethod
    def __init__(self, file):
        self.file = file
        self.labelPosition = labelPosition
        self.preProcessor = self.__createPreProcessor()
        self.formattedData = self._convertToFormatted()
        self.labelDictionary = {}
        ...

    @abstractmethod
    def load(self):
        ...

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
    @abstractmethod
    def _convertToFormatted(self):
        ...

    @classmethod
    def removeNonACII(self, sentence):
        return re.sub(r'[^\x00-\x7F]+', '', sentence)

    # shared method
    def _saveToCsv(self, data, filename):
        csvFile = filename + '.csv'
        with open(csvFile, 'w', newline='') as myfile:
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)

            for line in data:
                wr.writerow(line)

        print("File saved successfully")

    # shared method
    def filterByParentClass(self, parentClass, save=False):
        res = []

        for data in self.formattedData:
            if(self.labelPosition == 'first'):
                labelIndex = 0
            else:
                labelIndex = len(data) - 1

            # get label and compare
            if(data[labelIndex].split(':')[0] == parentClass):
                res.append(data)
        if(save):
            self._saveToCsv(res, parentClass)
        return res


# ds = Dataset() #you will get error
