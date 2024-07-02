# -- ML main packages
from abc import ABC, abstractmethod
import pandas as pd


class AnomalyDetectionModel(ABC):

    def __init__(self, df: pd.DataFrame, dict_params: dict):
        self.df = df
        self.dict_params = dict_params

    @abstractmethod
    def fit_model(self, df_train: pd.DataFrame, dict_params: dict):
        pass

    @abstractmethod
    def get_anomalies(self):
        pass

    def plot(self):
        pass
