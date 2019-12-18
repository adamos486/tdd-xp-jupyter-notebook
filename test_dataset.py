import unittest

import pandas as pd
from assertpy import assert_that
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron, SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from dataset import DataSet
from preprocessor import Preprocessor
from trainer import Trainer


class TrainerStub(Trainer):
    map = {KNeighborsClassifier: 10,
           GaussianNB: 10,
           Perceptron: 10,
           SGDClassifier: 20,
           DecisionTreeClassifier: 10,
           RandomForestClassifier: 10}

    def train_model(self, model, X_train: pd.DataFrame, Y_train: pd.DataFrame, **kwargs) -> [object, float]:
        return model(), self.map[model]


class PreprocessorSpy(Preprocessor):
    dataframe: pd.DataFrame = pd.read_csv("./fixtures/four_records.csv")
    age_bins_called: bool = False
    enumerate_month_called: bool = False
    enumerate_gender_called: bool = False
    enumerate_user_type_called: bool = False
    drop_columns_called: bool = False
    clean_start_station_ids_called: bool = False

    def clean_start_station_ids(self, df: pd.DataFrame) -> pd.DataFrame:
        self.clean_start_station_ids_called = True
        return self.dataframe

    def drop_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        self.drop_columns_called = True
        return self.dataframe

    def enumerate_gender(self, df: pd.DataFrame) -> pd.DataFrame:
        self.enumerate_gender_called = True
        return self.dataframe

    def enumerate_month(self, df: pd.DataFrame):
        self.enumerate_month_called = True
        return self.dataframe

    def enumerate_user_type(self, df: pd.DataFrame) -> pd.DataFrame:
        self.enumerate_user_type_called = True
        return self.dataframe

    def create_age_bins(self, bin_count: int, df: pd.DataFrame) -> pd.DataFrame:
        self.age_bins_called = True
        return self.dataframe


class TestDataSet(unittest.TestCase):
    def test_constructor_reads_from_file_and_concats(self):
        one_result: pd.DataFrame = DataSet(["./fixtures/nine_records.csv"], PreprocessorSpy(), TrainerStub())._df

        assert_that(one_result).is_not_none()
        assert_that(len(one_result)).is_equal_to(9)

        two_results: pd.DataFrame = \
            DataSet(["./fixtures/nine_records.csv", "./fixtures/four_records.csv"], PreprocessorSpy(), TrainerStub()) \
                ._df

        assert_that(two_results).is_not_none()
        assert_that(len(two_results)).is_equal_to(13)

    def test_sample(self):
        dataset: DataSet = DataSet([
            "./fixtures/nine_records.csv"
        ], PreprocessorSpy(), TrainerStub())

        dataset.sample(0.22)

        assert_that(len(dataset._df)).is_equal_to(2)

    def test_create_age_bins(self):
        spy = PreprocessorSpy()
        data_set = DataSet(["./fixtures/nine_records.csv"], spy, TrainerStub())
        data_set.create_age_bins(3)

        assert_that(spy.age_bins_called).is_true()
        assert_that(data_set._df).is_length(4)

    def test_enumerate_month(self):
        spy = PreprocessorSpy()
        data_set = DataSet(["./fixtures/nine_records.csv"], spy, TrainerStub())
        data_set.enumerate_month()

        assert_that(spy.enumerate_month_called).is_true()
        assert_that(data_set._df).is_length(4)

    def test_enumerate_gender(self):
        spy = PreprocessorSpy()
        data_set = DataSet(["./fixtures/nine_records.csv"], spy, TrainerStub())
        data_set.enumerate_gender()

        assert_that(spy.enumerate_gender_called).is_true()
        assert_that(data_set._df).is_length(4)

    def test_enumerate_user_type(self):
        spy = PreprocessorSpy()
        data_set = DataSet(["./fixtures/nine_records.csv"], spy, TrainerStub())
        data_set.enumerate_user_type()

        assert_that(spy.enumerate_user_type_called).is_true()
        assert_that(data_set._df).is_length(4)

    def test_drop_columns(self):
        spy = PreprocessorSpy()
        data_set = DataSet(["./fixtures/nine_records.csv"], spy, TrainerStub())
        data_set.drop_columns()

        assert_that(spy.drop_columns_called).is_true()
        assert_that(data_set._df).is_length(4)

    def test_clean_start_station_ids(self):
        spy = PreprocessorSpy()
        data_set = DataSet(["./fixtures/nine_records.csv"], spy, TrainerStub())
        data_set.clean_start_station_ids()

        assert_that(spy.clean_start_station_ids_called).is_true()
        assert_that(data_set._df).is_length(4)

    def test_train(self):
        trainer = TrainerStub()

        data_set = DataSet(["./fixtures/nine_records.csv"], PreprocessorSpy(), trainer)
        result, testData = data_set.train()
        assert_that(result).is_instance_of(SGDClassifier)
        assert_that(testData).is_length(1)
