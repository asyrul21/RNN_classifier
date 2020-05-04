from WordEmbedder import WordEmbedder
import numpy as np


class SentenceEmbedder:
    def __init__(self):
        self.wordEmbedder = self.__initWordEmbedder()

    def __initWordEmbedder(self):
        return WordEmbedder()

    def getVector(self, sentence):
        sentenceVectors = []
        for word in sentence.split(' '):
            # print('Word:', word)
            sentenceVectors.append(self.wordEmbedder.getVector(word))

        return np.asarray(sentenceVectors)


# sentence = 'I love to eat Pizza'
# sentence = 'Build me a workflow'
# sentence = 'My file size is 3GB'
# # sentence = 'I love to eat Pizza'

# print('sentence:', sentence)
# se = SentenceEmbedder()

# print(se.getVector(sentence))
