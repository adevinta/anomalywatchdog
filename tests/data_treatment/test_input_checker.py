import pandas as pd
from anomalywatchdog.data_treatment.input_checker \
    import InputChecker


INPUT_CONFIG = {"models_to_use": ["Autoencoder_basic", "Prophet"]}
EXPECTED_CONFIG = {"models_to_use": ["autoencoder_basic", "prophet"]}


def test_input_checker():
    # -- Initialize input checker
    input_checker = InputChecker(
        df=input_df(),
        column_date='Date',
        column_target='Value',
        granularity='M',
        columns_dimension=['Col_dim'],
        models_to_use=INPUT_CONFIG["models_to_use"],
        config=INPUT_CONFIG
    )
    # -- Check df and columns
    pd.testing.assert_frame_equal(
        input_checker.df,
        expected_df()
    )
    # -- Check models_to_use and config
    pd.testing.assert_frame_equal(
        pd.DataFrame(input_checker.config),
        pd.DataFrame(EXPECTED_CONFIG)
    )


def input_df():
    date_list = [
        "2021-01-01",
        "2021-02-01",
        "2021-03-01",
        "2021-05-01",
        "2021-06-01",
    ]
    value_list = [
        5830302.908,
        2330528.002,
        7400157.853,
        2400457.853,
        3400257.853,
    ]
    dim_list = [
        "a",
        "a",
        "a",
        "a",
        "a",
    ]
    return pd.DataFrame(
        {'Date': pd.to_datetime(date_list),
         'Value': value_list,
         'Col_dim': dim_list}
    )

def expected_df():
    date_list = [
        "2021-01-01",
        "2021-02-01",
        "2021-03-01",
        "2021-05-01",
        "2021-06-01",
    ]
    value_list = [
        5830302.908,
        2330528.002,
        7400157.853,
        2400457.853,
        3400257.853,
    ]
    dim_list = [
        "a",
        "a",
        "a",
        "a",
        "a",
    ]
    return pd.DataFrame(
        {'date': pd.to_datetime(date_list),
         'value': value_list,
         'col_dim': dim_list}
    )