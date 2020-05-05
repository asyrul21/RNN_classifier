from abc import ABC, abstractmethod


class ILabel(ABC):
    @abstractmethod
    def __init__(self, dataset, labelPosition='last'):
        self.dataset = dataset
        self.labelPosition = labelPosition
        self.datasetLabels = self._createDatasetLabels()
        self.labelDictionary = ...

    @abstractmethod
    def _createDatasetLabels(self):
        ...

    @abstractmethod
    def getLabels(self):
        ...

    @abstractmethod
    def getLabelDictionay(self):
        ...

    @abstractmethod
    def embedLabels(self):
        ...


# il = ILabel()
