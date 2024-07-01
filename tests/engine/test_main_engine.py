import pandas as pd
from loaders.package.c_engine.model_trainer \
    import ModelTrainer
from loaders.package.c_engine.model_predictor \
    import ModelPredictor

MODELS_TO_USE = ["prophet", "auto_arima"]
CONFIG = {
    "prophet":{
        "weekly_seasonality": False,
        "monthly_seasonality": False,
        "yearly_seasonality": False,
        "seasonality_mode": "additive",
        "interval_width": 0.95,
        "changepoint_range": 0.8,
        "features": {
            "holidays": False
        }
    },
    "auto_arima":{
        "weekly_seasonality": False,
        "monthly_seasonality": False,
        "yearly_seasonality": False,
        "features": {
            "holidays": False
        }
    }
}



def test_monthly_main_engine():
    """
    This function tests the following classes:
    - ModelFactory
    - ModelPredictor
    - ModelTrainer
    """
    # -- Get inputs
    model_trainer = ModelTrainer(
        model_names=MODELS_TO_USE,
        df_train=input_df_value(),
        config=CONFIG
    )
    df_models_trained = model_trainer.train()
    model_predictor = ModelPredictor(df_models_trained=df_models_trained)
    df_predictions = model_predictor.predict()
    # -- Tests
    pd.testing.assert_frame_equal(
        df_predictions,
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
    return pd.DataFrame(
        {'date': pd.to_datetime(date_list), 'value': value_list}
    )


def expected_df():
    df = pd.concat([input_df_value(), input_df_value()])
    model_list = (
            ["prophet"] * len(input_df_value()) +
            ["auto_arima"] * len(input_df_value())
    )
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
        True,
        True,
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
        True,
        False,
        False,
        False,
        False
    ]
    df['anomaly'] = anomaly_list
    df['model'] = model_list
    #df = df.reset_index(drop=True)
    return df



