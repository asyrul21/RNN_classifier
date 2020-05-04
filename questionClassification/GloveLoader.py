import numpy as np


class GloveLoader:
    def __init__(self, file='glove.6B.100d.txt'):
        self.file = file
        self.gloveDict = {}

    # private
    def __readGloveTxt(self):
        with open(self.file, 'r', encoding="ISO-8859-1") as in_file:
            stripped = (line.strip() for line in in_file)

            dictionary = {}
            counter = 0
            print('Loading Glove word embeddings...')
            for line in stripped:
                if(line):
                    frags = line.split(' ')
                    word = frags[0]
                    vector = np.asarray(frags[1:])
                    dictionary.update({word: vector})

                    # for debugging
                    # counter += 1
                    # if(counter == 3):
                    #     break

            print('Glove embeddings loaded.')

            if(not self.gloveDict):
                self.gloveDict = dictionary

    # public
    def getGloveDictionary(self):
        if(not self.gloveDict):
            # print('Dictionary is currently empty')
            self.__readGloveTxt()
            return self.gloveDict
        else:
            # print('Dictionary has been loaded')
            return self.gloveDict


# gl = GloveLoader()

# # print(gl.gloveDict)

# print(gl.getGloveDictionary())
# print(gl.getGloveDictionary())
