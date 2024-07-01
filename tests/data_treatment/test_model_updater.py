from loaders.package.a_data_treatment.model_updater\
    import ModelUpdater
import pandas as pd

GRANULARITY = 'M'
CONFIG = {
    "models_to_use":{
        "prophet",
        "auto_arima"
    },
    "ts_seasonality":{
        "weekly_seasonality": False,
        "monthly_seasonality": False,
        "yearly_seasonality": False,
    },
    "prophet":{
        "weekly_seasonality": False,
        "monthly_seasonality": False,
        "yearly_seasonality": False,
    },
    "auto_arima":{
        "weekly_seasonality": False,
        "monthly_seasonality": False,
        "yearly_seasonality": False
    }
}
CONFIG_EXPECTED = {
    "models_to_use":{
        "prophet",
        "auto_arima"
    },
    "ts_seasonality":{
        "weekly_seasonality": False,
        "monthly_seasonality": False,
        "yearly_seasonality": True,
    },
    "prophet":{
        "weekly_seasonality": False,
        "monthly_seasonality": False,
        "yearly_seasonality": True,
    },
    "auto_arima":{
        "weekly_seasonality": False,
        "monthly_seasonality": False,
        "yearly_seasonality": True
    }
}

def test_model_updater(
):

    model_updater = ModelUpdater(
            df=input_df_value(),
            granularity=GRANULARITY,
            config=CONFIG
        )
    pd.testing.assert_frame_equal(
        pd.DataFrame([model_updater.config == CONFIG_EXPECTED]),
        pd.DataFrame([True])
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
    return pd.DataFrame(
        {'date': pd.to_datetime(date_list), 'value': value_list}
    )
