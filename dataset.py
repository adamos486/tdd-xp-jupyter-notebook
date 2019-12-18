from typing import List

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron, SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from preprocessor import Preprocessor
from trainer import Trainer


class DataSet:
    def __init__(self, paths: List[str], pre: Preprocessor, trainer: Trainer):
        self._df: pd.DataFrame = pd.concat(map(lambda path: pd.read_csv(path), paths))
        self._preprocessor: Preprocessor = pre
        self._trainer: Trainer = trainer

    def sample(self, frac: float) -> pd.DataFrame:
        self._df = self._df.sample(frac=frac)

    def create_age_bins(self, bin_count: int):
        self._df = self._preprocessor.create_age_bins(bin_count, self._df)

    def enumerate_month(self):
        self._df = self._preprocessor.enumerate_month(self._df)

    def enumerate_gender(self):
        self._df = self._preprocessor.enumerate_gender(self._df)

    def enumerate_user_type(self):
        self._df = self._preprocessor.enumerate_user_type(self._df)

    def drop_columns(self):
        self._df = self._preprocessor.drop_columns(self._df)

    def clean_start_station_ids(self):
        self._df = self._preprocessor.clean_start_station_ids(self._df)

    def train(self):
        train_df, test_df = train_test_split(self._df, test_size=0.1)

        X_train = train_df.drop("trip_duration_sec", axis=1)
        Y_train = train_df["trip_duration_sec"]
        X_test = test_df.drop("trip_duration_sec", axis=1)

        model,current_high = self._trainer.train_model(KNeighborsClassifier, X_train, Y_train)

        gaussian, acc_gaussian = self._trainer.train_model(GaussianNB, X_train, Y_train)
        if acc_gaussian > current_high:
            model = gaussian
            current_high = acc_gaussian

        perceptron, acc_perceptron = self._trainer.train_model(Perceptron, X_train, Y_train)
        if acc_perceptron > current_high:
            model = perceptron
            current_high = acc_perceptron

        sgd, acc_sgd = self._trainer.train_model(SGDClassifier, X_train, Y_train)
        if acc_sgd > current_high:
            model = sgd
            current_high = acc_sgd

        decision_tree, acc_decision_tree = self._trainer.train_model(DecisionTreeClassifier, X_train, Y_train)
        if acc_decision_tree > current_high:
            model = decision_tree
            current_high = acc_decision_tree

        random_forest, acc_random_forest = self._trainer.train_model(RandomForestClassifier, X_train, Y_train,
                                                                     n_estimators=100)
        if acc_random_forest > current_high:
            model = random_forest

        return model, X_test



