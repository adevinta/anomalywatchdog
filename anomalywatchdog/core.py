from anomalywatchdog.data_treatment.data_handler\
    import DataADHandler
from anomalywatchdog.data_treatment.input_checker\
    import InputChecker
from anomalywatchdog.data_treatment.model_updater\
    import ModelUpdater
from anomalywatchdog.engine.model_trainer\
    import ModelTrainer
from anomalywatchdog.engine.model_predictor\
    import ModelPredictor
from anomalywatchdog.anomaly_features.create_features\
    import create_features
import os.path as op
import yaml
from logging import Logger
import pandas as pd
from typing import Union, List
from pyspark.sql import DataFrame
import numpy as np


class AnomalyWatchdog:

    def __init__(
            self,
            df: Union[pd.DataFrame, DataFrame],
            column_date: str,
            column_target: str,
            granularity: str,
            columns_dimension: list[str] = [],
            start_date: Union[str, None] = None,
            end_date: Union[str, None] = None,
            models_to_use: List[str] = ['auto_arima', 'Prophet']
    ):
        # -- Initialize logs
        self.log = Logger(name="anomaly_detector")
        # -- Read Config
        current_path = op.dirname(__file__)
        with open(op.join(current_path, "config_ad.yaml"), 'rb') as fp:
            self.config = yaml.safe_load(fp)
        # -- Check inputs
        input_checker = InputChecker(
            df=df.copy(),
            column_date=column_date,
            column_target=column_target,
            columns_dimension=columns_dimension,
            granularity=granularity,
            models_to_use=models_to_use,
            start_date=start_date,
            end_date=end_date,
            config=self.config
        )
        # -- Update inputs
        self.df_input = input_checker.df.copy()
        self.column_target = input_checker.column_target
        self.columns_dimension = input_checker.columns_dimension
        self.config = input_checker.config
        self.granularity = granularity
        self.start_date = input_checker.start_date
        self.end_date = input_checker.end_date
        self.df_input.rename(
            columns={column_date: "date", column_target: "value"},
            inplace=True
        )
        # -- Initialize inputs
        self.max_date = self.df_input["date"].max()
        # -- Initialize output
        self.df_anomaly = pd.DataFrame()
        self.df_anomaly_dimension = pd.DataFrame()
        self.log.info(">> 1. Data Treatment")
        data_ad_handler = DataADHandler(
            df=self.df_input,
            granularity=self.granularity
        )
        df = data_ad_handler.df_grouped.copy()
        df = df[["date", "value"]].copy()
        self.log.info(">> 1.1 Get TS properties")
        ModelUpdater(
            df=df.copy(),
            granularity=self.granularity,
            config=self.config
        )
        self.log.info(">> 1.2 Get features")
        df = create_features(df=df.copy(), granularity=self.granularity)
        self.log.info(">> 2. Get global anomalies")
        self.df_anomaly = self.__detect_anomalies(
            df_handled=df.copy(),
            list_models=self.config['models_to_use']
        )
        self.log.info(">> 2. Get drilled anomalies")
        self.df_anomaly_dimension = self.__detect_granular_anomalies(
            df_predictions=self.df_anomaly.copy(),
            columns_dimension=self.columns_dimension,
            granularity=self.granularity
        )
        self.log.info(">> 3. Filter selected dates")
        print('print2')
        if self.start_date:
            self.df_anomaly = self.df_anomaly.loc[
                (self.df_anomaly['date'] >= self.start_date) &
                (self.df_anomaly['date'] <= self.end_date)
                ].copy()
            if len(self.df_anomaly_dimension) > 0:
                self.df_anomaly_dimension = self.df_anomaly_dimension.loc[
                    (self.df_anomaly_dimension['date'] >= self.start_date) &
                    (self.df_anomaly_dimension['date'] <= self.end_date)
                    ].copy()
        print(self.df_anomaly)
        print(self.df_anomaly_dimension)

    def __detect_anomalies(
            self,
            df_handled: pd.DataFrame,
            list_models: list[str]
    ) -> pd.DataFrame():
        self.log.info(">> 2.1. Train Models")
        model_trainer = ModelTrainer(
            model_names=list_models,
            df_train=df_handled.copy(),
            config=self.config
        )
        df_models_trained = model_trainer.train()
        self.log.info(">> 2.2. Predict Models")
        model_predictor = ModelPredictor(
            df_models_trained=df_models_trained)
        df_predictions = model_predictor.predict()
        return df_predictions

    def __detect_granular_anomalies(
            self,
            df_predictions: pd.DataFrame,
            columns_dimension: list,
            granularity: str
    ) -> pd.DataFrame():
        df_dimension = pd.DataFrame()
        if len(columns_dimension) > 0:
            list_df_dimension = []
            filtered_df = df_predictions.copy()
            if self.start_date:
                filtered_df = filtered_df.loc[
                    (filtered_df['date'] <= self.end_date) &
                    (filtered_df['date'] >= self.start_date)
                    ]
            if len(filtered_df["model"].unique()) > 0:
                for model in filtered_df["model"].unique():
                    is_anomaly_in_interval = (
                            filtered_df.loc[filtered_df['model'] == model,
                                            ['anomaly']].sum()
                            > 0
                    ).bool()
                    if is_anomaly_in_interval:
                        for column_dimension in columns_dimension:
                            list_dimension_value = [
                                dimension for dimension
                                in self.df_input[column_dimension].unique()
                                if dimension is not (None or np.nan)
                            ]
                            for dimension_value in list_dimension_value:
                                df_dimension = (
                                    self.df_input
                                    .loc[self.df_input[column_dimension]
                                         == dimension_value, ["date", "value"]]
                                    .reset_index(drop=True)
                                    .copy()
                                )
                                data_ad_handler = DataADHandler(
                                    df=df_dimension,
                                    granularity=granularity
                                )
                                df = data_ad_handler.df_grouped.copy()
                                df = create_features(df=df.copy(),
                                                     granularity=self.granularity)
                                df_predictions_element = self.__detect_anomalies(
                                    df_handled=df.copy(),
                                    list_models=[model]
                                )
                                print(df_predictions_element)
                                df_predictions_element['dimension'] = (
                                    column_dimension
                                )
                                df_predictions_element['dimension_value'] = (
                                    dimension_value
                                )
                                list_df_dimension.append(df_predictions_element)
                if len(list_df_dimension) > 0:
                    df_dimension = pd.concat(list_df_dimension)
                return df_dimension
            else:
                return df_dimension
        else:
            return df_dimension
