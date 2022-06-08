from abc import abstractmethod


class TrainerBase:
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def evaluate(self, **kwargs):
        NotImplemented()

    @abstractmethod
    def save_model(self, **kwargs):
        NotImplemented()

    @abstractmethod
    def train_one_epoch(self, **kwargs):
        NotImplemented()

    @abstractmethod
    def fit(self, **kwargs):
        NotImplemented()