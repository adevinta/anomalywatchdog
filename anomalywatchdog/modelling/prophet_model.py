from anomalywatchdog.modelling.abstract_model \
    import AnomalyDetectionModel
import pandas as pd
from prophet import Prophet
import numpy as np
import tensorflow as tf
import random


def set_seed(seed=42):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)


set_seed(42)


class ProphetModel(AnomalyDetectionModel):
    def __init__(self,
                 *args,
                 **kwargs
                 ):
        super(ProphetModel, self).__init__(*args, **kwargs)
        # -- fit model
        self.model_fitted = self.fit_model(
            df_train=self.df,
            dict_params=self.dict_params
        )

    def fit_model(self, df_train: pd.DataFrame, dict_params: dict):
        # -- column names
        df_train.rename(columns={'date': 'ds', 'value': 'y'}, inplace=True)
        # -- Create tensor
        df_train['ds'] = pd.to_datetime(df_train['ds'])
        df_train['ds'] = df_train['ds'].dt.tz_localize(None)
        prophet = Prophet(
            seasonality_mode=dict_params["seasonality_mode"],
            interval_width=dict_params["interval_width"],
            changepoint_range=dict_params["changepoint_range"]
        )
        if dict_params['weekly_seasonality']:
            prophet.add_seasonality(
                name='weekly', period=7, fourier_order=3
            )
        if dict_params['monthly_seasonality']:
            prophet.add_seasonality(
                name='monthly', period=30.5, fourier_order=5
            )
        if dict_params['yearly_seasonality']:
            prophet.add_seasonality(
                name='yearly', period=365.25, fourier_order=3
            )
        if self.dict_params['features']['holidays']:
            prophet.add_country_holidays(country_name='Spain')
        return prophet.fit(df_train[["ds", "y"]])

    def get_anomalies(self):
        # -- Predict over train
        df_anomaly = self.model_fitted.predict(self.df)
        df_anomaly['fact'] = self.df['y'].reset_index(drop=True)
        df_anomaly = df_anomaly[
            ['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'fact']].copy()
        df_anomaly['anomaly'] = False
        df_anomaly.loc[
            (df_anomaly['fact'] > df_anomaly['yhat_upper'])
            | (df_anomaly['fact'] < df_anomaly['yhat_lower']),
            'anomaly'
        ] = True
        # -- Tune output
        df_anomaly.rename(columns={"ds": "date", "fact": "value"},
                          inplace=True)
        df_anomaly = df_anomaly[["date", "value", "anomaly"]].copy()
        df_anomaly['model'] = 'prophet'
        df_anomaly['date'] = pd.to_datetime(df_anomaly['date'])
        self.df = df_anomaly.copy()
