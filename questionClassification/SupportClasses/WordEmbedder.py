from SupportClasses.GloveLoader import GloveLoader
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class WordEmbedder:
    def __init__(self):
        self.gloveDictionary = self.__loadGlove()

    def __loadGlove(self):
        GL = GloveLoader()
        return GL.getGloveDictionary()

    def getVector(self, word):
        word = word.lower()
        if(word in self.gloveDictionary):
            return self.gloveDictionary[word].astype(np.float)
        else:  # if word not found
            subWordVectors = []
            for sub in word:
                subWordVectors.append(self.gloveDictionary[sub])

            # add all vector
            subWordSum = np.zeros(len(subWordVectors[0]))
            for vec in subWordVectors:
                # print(np.asarray(vec.astype(np.float)))
                subWordSum = np.add(
                    subWordSum, np.asarray(vec.astype(np.float)))

            return subWordSum


# word2GB = '3GB'
# we = WordEmbedder()

# vec_2GB = we.getVector('2GB')
# vec_3GB = we.getVector('3GB')

# # find cosine similarity difference
# v2GB = vec_2GB.reshape(1, len(vec_2GB))
# v3GB = vec_3GB.reshape(1, len(vec_3GB))
# cos_lib = cosine_similarity(v2GB, v3GB)

# print('2GB:')
# print(v2GB)
# print('3GB')
# print(v3GB)
# print('Cosine similarity:', cos_lib)

# arr1 = np.array([1, 2, 3, 4])
# arr2 = np.array([4, 3, 2, 1])

# print(np.add(arr1, arr2))  # [5,5,5,5]
