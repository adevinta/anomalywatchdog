from anomalywatchdog.modelling.abstract_model \
    import AnomalyDetectionModel
from anomalywatchdog.utils.create_fourier_terms \
    import create_fourier_terms
import pandas as pd
from pmdarima.arima import auto_arima
import numpy as np
import tensorflow as tf
import random


def set_seed(seed=42):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)


set_seed(42)


class AutoArimaModel(AnomalyDetectionModel):
    def __init__(self,
                 *args,
                 **kwargs
                 ):
        super(AutoArimaModel, self).__init__(*args, **kwargs)
        # -- fit model
        self.model_fitted = self.fit_model(
            df_train=self.df,
            dict_params=self.dict_params)

    def fit_model(self, df_train: pd.DataFrame, dict_params: dict):
        list_df_fourier = []
        if dict_params['weekly_seasonality']:
            list_df_fourier.append(create_fourier_terms(self.df, freq=7, K=3))
        if dict_params['monthly_seasonality']:
            list_df_fourier.append(
                create_fourier_terms(self.df, freq=30, K=3)
            )
        if dict_params['yearly_seasonality']:
            list_df_fourier.append(
                create_fourier_terms(self.df, freq=365, K=3)
            )
        if len(list_df_fourier) == 0:
            return auto_arima(
                df_train['value'], seasonal=False, suppress_warnings=True
            )
        else:
            df_exogenous = pd.concat(list_df_fourier, axis=1)
            if self.dict_params['features']['holidays']:
                df_exogenous = pd.concat(
                    [df_exogenous, df_train['holiday']], axis=1
                )
            return auto_arima(
                df_train['value'], seasonal=False, suppress_warnings=True,
                exogenous=df_exogenous
            )

    def get_anomalies(self):
        # -- Predict over train
        df_fit, df_confidence_intervals = (
            self.model_fitted.predict_in_sample(return_conf_int=True)
        )
        df_anomaly = pd.concat([
            self.df[["date", "value"]].copy(),
            df_fit,
            pd.DataFrame(df_confidence_intervals)
            ], axis=1)
        df_anomaly.columns = [
            'date', 'value', 'yhat', 'yhat_lower', 'yhat_upper'
        ]
        df_anomaly['anomaly'] = False
        df_anomaly.loc[
            (df_anomaly['value'] > df_anomaly['yhat_upper'])
            | (df_anomaly['value'] < df_anomaly['yhat_lower']),
            'anomaly'
        ] = True
        df_anomaly = df_anomaly[["date", "value", "anomaly"]].copy()
        df_anomaly['model'] = 'auto_arima'
        df_anomaly['date'] = pd.to_datetime(df_anomaly['date'])
        self.df = df_anomaly.copy()
