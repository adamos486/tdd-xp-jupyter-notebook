import unittest

import pandas as pd
from assertpy import assert_that

from trainer import Trainer, TrainerImpl

# we are conforming to the relevant parts of the implicit interface of the ModelClass
class ModelStubSpy:
    fit_called = False

    def score(self, X_train: pd.DataFrame, Y_train: pd.DataFrame) -> float:
        return 0.31415926535

    def fit(self, X_train: pd.DataFrame, Y_train: pd.DataFrame):
        self.fit_called = True


class MyTestCase(unittest.TestCase):

    def test_train_model(self):
        trainer: Trainer = TrainerImpl()

        X_train = pd.DataFrame([], columns=['a', 'b'])
        Y_train = pd.DataFrame([], columns=['xor'])

        model, accuracy_score = trainer.train_model(ModelStubSpy, X_train, Y_train)

        assert_that(model).is_instance_of(ModelStubSpy)
        assert_that(model.called).is_true()
        assert_that(accuracy_score).is_equal_to(31.42)
