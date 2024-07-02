from anomalywatchdog.modelling.abstract_model import AnomalyDetectionModel
from anomalywatchdog.utils.create_sequences import create_sequences
from anomalywatchdog.utils.data_normalizer import DataNormalizer
import pandas as pd
import keras
from keras import layers
import numpy as np
import tensorflow as tf
import random


def set_seed(seed=42):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)


set_seed(42)


class AutoEncoderLSTMModel(AnomalyDetectionModel):
    def __init__(self, *args, **kwargs):
        super(AutoEncoderLSTMModel, self).__init__(*args, **kwargs)
        # -- Treat data
        self.df_train = self.df.copy()
        self.df_train.sort_values('date', inplace=True)
        self.df_train.set_index('date', inplace=True)
        # -- Normalize data
        self.df_train['value'] = DataNormalizer.normalization_standard(
            self.df_train[['value']].values
        )
        # -- Select holiday features
        if self.dict_params['features']['holidays']:
            self.df_train['holiday'] = DataNormalizer.normalization_standard(
                self.df_train[['holiday']].values
            )
        else:
            self.df_train = self.df_train.drop(['holiday'], axis=1)
        # -- fit model
        self.model_fitted = self.fit_model(
            df_train=self.df_train,
            dict_params=self.dict_params
        )

    def fit_model(self, df_train, dict_params: dict):
        # -- Create sequences of fit
        df_seq_values = create_sequences(
            values=self.df_train.values,
            time_steps=(
                self.dict_params["timesteps"][self.dict_params['granularity']]
            )
        )
        # -- AutoEncoder
        autoencoder_lstm = keras.Sequential(
            [
                layers.Input(
                    shape=(df_seq_values.shape[1], df_seq_values.shape[2])
                ),
                layers.LSTM(
                    units=128,
                    activation=dict_params["activation"],
                    return_sequences=True
                ),
                layers.Dropout(rate=0.2),
                layers.LSTM(
                    units=64,
                    activation=dict_params["activation"],
                    return_sequences=False
                ),
                layers.RepeatVector(df_seq_values.shape[1]),
                layers.LSTM(
                    units=64,
                    activation=dict_params["activation"],
                    return_sequences=True
                ),
                layers.Dropout(rate=0.2),
                layers.LSTM(
                    units=128,
                    activation=dict_params["activation"],
                    return_sequences=True
                ),
                layers.TimeDistributed(layers.Dense(df_seq_values.shape[2]))
            ]
        )
        autoencoder_lstm.compile(
            optimizer=keras.optimizers.Adam(
                learning_rate=dict_params["learning_rate"]
            ),
            loss=dict_params["loss"]
        )

        autoencoder_lstm.fit(
            df_seq_values,
            df_seq_values,
            epochs=dict_params["epochs"],
            batch_size=(
                dict_params["batch_size"][self.dict_params['granularity']]
            ),
            validation_split=dict_params["validation_split"],
            callbacks=[
                keras.callbacks.EarlyStopping(monitor="val_loss", patience=30,
                                              mode="min"),
                keras.callbacks.ReduceLROnPlateau(monitor="val_loss",
                                                  factor=0.5,
                                                  patience=15, mode="min")
            ],
        )
        return autoencoder_lstm

    def get_anomalies(self):
        # -- Create sequences of fit
        df_seq_values = create_sequences(
            values=self.df_train.values,
            time_steps=(
                self.dict_params["timesteps"][self.dict_params['granularity']]
            )
        )
        df_seq_index = create_sequences(
            values=self.df_train.index,
            time_steps=(
                self.dict_params["timesteps"][self.dict_params['granularity']]
            )
        )
        # -- Reconstruction of time series
        df_seq_pred = self.model_fitted.predict(df_seq_values)
        # -- Get actuals vs reconstruction
        df_anomaly = pd.DataFrame({
            'date': df_seq_index.flatten(),
            'actuals': df_seq_values[:, :, 0].flatten(),
            'reconstruction': df_seq_pred[:, :, 0].flatten(),
        })
        df_anomaly = df_anomaly.groupby('date')[
            ["actuals", "reconstruction"]].mean().reset_index()
        # -- Get absolute error
        df_anomaly['absolute_error'] = np.abs(
            df_anomaly['reconstruction'] - df_anomaly['actuals'])
        # -- Get threshold
        threshold = (
                df_anomaly['absolute_error'].quantile(q=0.75) *
                1.5
        )
        # -- Get anomalies
        df_anomaly['anomaly'] = False
        df_anomaly.loc[
            df_anomaly['absolute_error'] > threshold, 'anomaly'
        ] = True
        # -- Get output
        df_anomaly = pd.concat(
            [self.df[['date', 'value']], df_anomaly['anomaly']],
            axis=1)
        df_anomaly['date'] = pd.to_datetime(df_anomaly['date'])
        df_anomaly['model'] = 'autoencoder_lstm'
        self.df = df_anomaly.copy()
