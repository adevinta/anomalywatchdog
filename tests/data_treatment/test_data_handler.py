import pandas as pd
from loaders.package.a_data_treatment.data_handler \
    import DataADHandler



def test_data_handler():
    # -- Test for monthly data
    transformed_data_monthly = DataADHandler(
        df = input_df_monthly(),
        granularity = 'M'
    )
    pd.testing.assert_frame_equal(
        transformed_data_monthly.df_grouped,
        expected_df_monthly()
    )
    # -- Test for weekly data
    transformed_data_weekly = DataADHandler(
        df = input_df_weekly(),
        granularity = 'W'
    )
    pd.testing.assert_frame_equal(
        transformed_data_weekly.df_grouped,
        expected_df_weekly()
    )
    # -- Test for daily data
    transformed_data_daily = DataADHandler(
        df = input_df_daily(),
        granularity = 'D'
    )
    pd.testing.assert_frame_equal(
        transformed_data_daily.df_grouped,
        expected_df_daily()
    )

def input_df_monthly():
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
    return pd.DataFrame(
        {'date': pd.to_datetime(date_list),
         'value': value_list}
    )

def expected_df_monthly():
    date_list = [
        "2021-01-01",
        "2021-02-01",
        "2021-03-01",
        "2021-04-01",
        "2021-05-01",
        "2021-06-01"
    ]
    value_list = [
        5830302.908,
        2330528.002,
        7400157.853,
        0.000,
        2400457.853,
        3400257.853
    ]
    return pd.DataFrame(
        {'date': pd.to_datetime(date_list, utc=True),
         'value': value_list}
    )

def input_df_weekly():
    date_list = [
        "2021-01-01",
        "2021-01-08",
        "2021-01-15",
        "2021-01-29",
        "2021-02-05",
    ]
    value_list = [
        5830302.908,
        2330528.002,
        7400157.853,
        2400457.853,
        3400257.853,
    ]
    return pd.DataFrame(
        {'date': pd.to_datetime(date_list),
         'value': value_list}
    )

def expected_df_weekly():
    date_list = [
        "2021-01-01",
        "2021-01-08",
        "2021-01-15",
        "2021-01-22",
        "2021-01-29",
        "2021-02-05",
    ]
    value_list = [
        5830302.908,
        2330528.002,
        7400157.853,
        0.000,
        2400457.853,
        3400257.853
    ]
    return pd.DataFrame(
        {'date': pd.to_datetime(date_list, utc=True),
         'value': value_list}
    )

def input_df_daily():
    date_list = [
        "2021-01-27",
        "2021-01-28",
        "2021-01-29",
        "2021-01-31",
        "2021-02-01",
    ]
    value_list = [
        5830302.908,
        2330528.002,
        7400157.853,
        2400457.853,
        3400257.853,
    ]
    return pd.DataFrame(
        {'date': pd.to_datetime(date_list),
         'value': value_list}
    )

def expected_df_daily():
    date_list = [
        "2021-01-27",
        "2021-01-28",
        "2021-01-29",
        "2021-01-30",
        "2021-01-31",
        "2021-02-01"
    ]
    value_list = [
        5830302.908,
        2330528.002,
        7400157.853,
        0.000,
        2400457.853,
        3400257.853
    ]
    return pd.DataFrame(
        {'date': pd.to_datetime(date_list, utc=True),
         'value': value_list}
    )