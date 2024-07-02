import pandas as pd
from anomalywatchdog.anomaly_features\
    .CalendarFeatureCreator import CalendarFeatureCreator


def create_features(
        df: pd.DataFrame,
        granularity: str
) -> pd.DataFrame:
    # -- Create calendar features
    calendar_feature_creator = CalendarFeatureCreator(df=df, country="Spain")
    calendar_feature_creator.add_holidays(granularity=granularity)
    df_feature = calendar_feature_creator.df
    # -- Return output
    return df_feature
