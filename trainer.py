import abc

import pandas as pd


class Trainer(abc.ABC):
    @abc.abstractmethod
    def train_model(self, model, X_train: pd.DataFrame, Y_train: pd.DataFrame, **kwargs) -> [object, float]: pass


class TrainerImpl(Trainer):
    def train_model(self, ModelClass, X_train, Y_train, **kwargs):
        model = ModelClass(**kwargs)
        model.fit(X_train, Y_train)

        accuracy_score = round(model.score(X_train, Y_train) * 100, 2)

        return model, accuracy_score