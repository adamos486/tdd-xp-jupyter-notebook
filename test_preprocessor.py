import unittest
import pandas as pd
from assertpy import assert_that

from dataset import DataSet
from preprocessor import Preprocessor, PreprocessorImpl


class TestPreprocessor(unittest.TestCase):
    def setUp(self):
        self.subject: Preprocessor = PreprocessorImpl()
        self.df: pd.DataFrame = pd.read_csv("./fixtures/nine_records.csv")

    def test_create_age_bins(self):
        result: pd.DataFrame = self.subject.create_age_bins(3, self.df)

        assert_that(result[result['age'] == 2]).is_length(2)
        assert_that(result[result['age'] == 1]).is_length(1)
        assert_that(result[result['age'] == 0]).is_length(6)

        assert_that(result).does_not_contain_key('member_birth_year')

    def test_enumerate_month(self):
        result: pd.DataFrame = self.subject.enumerate_month(self.df)
        assert_that(result[result['month'] == 1]).is_length(9)

    def test_enumerate_gender(self):
        result: pd.DataFrame = self.subject.enumerate_gender(self.df)
        assert_that(result[result['member_gender'] == 0]).is_length(7)
        assert_that(result[result['member_gender'] == 1]).is_length(1)
        assert_that(result[result['member_gender'] == 2]).is_length(1)

    def test_enumerate_user_type(self):
        result: pd.DataFrame = self.subject.enumerate_user_type(self.df)
        assert_that(result[result['user_type'] == 0]).is_length(6)
        assert_that(result[result['user_type'] == 1]).is_length(3)

    def test_drop_columns(self):
        result: pd.DataFrame = self.subject.drop_columns(self.df)

        assert_that(result).does_not_contain_key('start_station_name')
        assert_that(result).does_not_contain_key('end_station_id')
        assert_that(result).does_not_contain_key('end_station_name')
        assert_that(result).does_not_contain_key('bike_id')

    def test_clean_start_station_ids(self):
        result: pd.DataFrame = self.subject.clean_start_station_ids(self.df)

        assert_that(result).is_length(8)
