import abc

import pandas as pd

COLUMNS_TO_DROP = ['start_station_name', 'end_station_id', 'end_station_name', 'bike_id']
GENDER_MAP = {'Male': 0, 'Female': 1}
MONTH_MAP = {'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5}
USER_TYPE_MAP = {'Subscriber': 0, 'Customer': 1}


class Preprocessor(abc.ABC):
    @abc.abstractmethod
    def clean_start_station_ids(self, df: pd.DataFrame) -> pd.DataFrame: pass

    @abc.abstractmethod
    def create_age_bins(self, bin_count: int, df: pd.DataFrame) -> pd.DataFrame: pass

    @abc.abstractmethod
    def enumerate_month(self, df: pd.DataFrame) -> pd.DataFrame: pass

    @abc.abstractmethod
    def enumerate_gender(self, df: pd.DataFrame) -> pd.DataFrame: pass

    @abc.abstractmethod
    def enumerate_user_type(self, df: pd.DataFrame) -> pd.DataFrame: pass

    @abc.abstractmethod
    def drop_columns(self, df: pd.DataFrame) -> pd.DataFrame: pass


class PreprocessorImpl(Preprocessor):
    def clean_start_station_ids(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[-df['start_station_id'].isna()]

    def drop_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.drop(COLUMNS_TO_DROP, axis=1)

    def enumerate_user_type(self, df: pd.DataFrame) -> pd.DataFrame:
        df['user_type'] = df['user_type'].map(USER_TYPE_MAP)
        return df

    def enumerate_gender(self, df: pd.DataFrame) -> pd.DataFrame:
        df['member_gender'] = df['member_gender'].map(GENDER_MAP).fillna(2)
        return df

    def enumerate_month(self, df: pd.DataFrame) -> pd.DataFrame:
        df['month'] = df['month'].map(MONTH_MAP).fillna(6)
        return df

    def create_age_bins(self, bin_count: int, df: pd.DataFrame) -> pd.DataFrame:
        df['member_birth_year'] = df['member_birth_year'].fillna(df['member_birth_year'].median())
        df['age'] = pd.cut(df['member_birth_year'], bin_count, labels=range(bin_count - 1, -1, -1)).astype(int)
        df = df.drop('member_birth_year', axis=1)
        return df
