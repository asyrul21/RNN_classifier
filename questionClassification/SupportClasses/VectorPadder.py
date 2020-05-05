import numpy as np


class VectorPadder:
    def __init__(self, vectorList):
        self.vectors = vectorList
        self.maxLength = self.__getMaxVectorlength()

    def __getMaxVectorlength(self):
        maxLength = 0
        for v in self.vectors:
            if(len(v) > maxLength):
                maxLength = len(v)
        return maxLength

    def __padVector(self, vector):
        # amount of vectors to pad
        toPad = self.maxLength - len(vector)

        if(toPad > 0):
            # get dim of the vectors
            dim = len(vector[0])

            # create list of 0 vectors
            padding = []
            for i in range(0, toPad):
                padding.append(np.zeros(dim, dtype=float))

            # arr = np.zeros(toPad, dtype=float)
            return np.concatenate([np.asarray(padding), vector])
        else:
            return vector

    def padVectors(self):
        newVectors = []
        for v in self.vectors:
            newVectors.append(self.__padVector(v))
        return np.asarray(newVectors)


# tests
# vectors = np.array(
#     [np.array([1.0, 2.0, 3.0, 4.0]), np.array([5.0, 6.0, 7.0]), np.array([6.0, 7.0, 8.0, 9.0, 0.0])])
# vectors = np.array(
#     [
#         np.array([
#             np.array([1.0, 2.0, 3.0, 4.0]),
#             np.array([1.0, 2.0, 3.0, 4.0]),
#             np.array([1.0, 2.0, 3.0, 4.0]),
#             np.array([1.0, 2.0, 3.0, 4.0]),
#         ]),
#         np.array([
#             np.array([5.0, 6.0, 7.0, 8.0]),
#             np.array([5.0, 6.0, 7.0, 8.0]),
#             np.array([5.0, 6.0, 7.0, 8.0])
#         ]),
#         np.array([
#             np.array([9.0, 8.0, 7.0, 6.0]),
#             np.array([9.0, 8.0, 7.0, 6.0]),
#             np.array([9.0, 8.0, 7.0, 6.0]),
#             np.array([9.0, 8.0, 7.0, 6.0]),
#             np.array([9.0, 8.0, 7.0, 6.0])
#         ])
#     ])

# vd = VectorPadder(vectors)

# print(vd.padVectors())

# print(vectors)
