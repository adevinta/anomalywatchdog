from anomalywatchdog.modelling.autoencoder_lstm_model \
    import AutoEncoderLSTMModel
import pandas as pd

DICT_PARAMS = {
    "granularity": 'M',
    "timesteps":
        {"D": 30,
         "W": 12,
         "M": 8
         },
    "activation": 'relu',
    "optimizer": 'adam',
    "loss": "mse",
    "epochs": 50,
    "batch_size":
        {"D": 32,
         "W": 16,
         "M": 12
         },
    "validation_split": 0.2,
    "learning_rate": 0.001,
    "quantile": 0.75,
    "quantile_multiplier": 1.5,
    "features":
        {"holidays": False}
}

def test_autoencoder_lstm_model():
    model = AutoEncoderLSTMModel(
        df=input_df_value(),
        dict_params=DICT_PARAMS)
    model.get_anomalies()
    pd.testing.assert_frame_equal(
        model.df,
        expected_df()
    )

def input_df_value():
    date_list = [
        "2021-01-01",
        "2021-02-01",
        "2021-03-01",
        "2021-04-01",
        "2021-05-01",
        "2021-06-01",
        "2021-07-01",
        "2021-08-01",
        "2021-09-01",
        "2021-10-01",
        "2021-11-01",
        "2021-12-01",
        "2022-01-01",
        "2022-02-01",
        "2022-03-01",
        "2022-04-01",
        "2022-05-01",
        "2022-06-01",
        "2022-07-01",
        "2022-08-01",
        "2022-09-01",
        "2022-10-01",
        "2022-11-01",
        "2022-12-01",
        "2023-01-01",
        "2023-02-01",
        "2023-03-01",
        "2023-04-01",
        "2023-05-01",
        "2023-06-01",
        "2023-07-01",
        "2023-08-01",
        "2023-09-01",
        "2023-10-01",
        "2023-11-01",
        "2023-12-01"
    ]
    value_list = [
        5830302.908,
        7400457.853,
        7449702.162,
        6480787.829,
        6995587.495,
        1675807.097,
        8372644.865,
        11324372.84,
        1799333.476,
        12702195.21,
        4509435.271,
        13316086.92,
        12826666.4,
        15473684.6,
        14899404.32,
        12463053.52,
        12991805.35,
        3016452.775,
        14652128.51,
        19318047.79,
        2998889.127,
        20724634.3,
        7215096.433,
        20925279.44,
        19823029.89,
        23546911.35,
        22349106.49,
        18445319.21,
        18988023.2,
        4357098.453,
        20931612.16,
        27311722.74,
        4198444.778,
        28747073.38,
        9920757.596,
        28534471.96,
    ]
    holiday_list = [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
    ]
    return pd.DataFrame(
        {'date': pd.to_datetime(date_list),
         'value': value_list,
         'holiday': holiday_list}
    )


def expected_df():
    df = input_df_value().copy()
    anomaly_list = [
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        True,
        False,
        False,
        True,
        False,
        True,
        False
    ]
    del df['holiday']
    df['anomaly'] = anomaly_list
    df['model'] = 'autoencoder_lstm'
    return df
