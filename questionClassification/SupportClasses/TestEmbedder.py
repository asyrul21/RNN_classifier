from SupportClasses.LabelEmbedder import LabelEmbedder
from SupportClasses.DatasetParentLabels import DatasetParentLabels
from SupportClasses.ILabel import ILabel
import numpy as np


class TestEmbedder(ILabel):
    def __init__(self):
        pass

    def hello(self):
        print('Hello World!')


# test
te = TestEmbedder()
te.hello()
