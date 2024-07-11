import pandas as pd
from datetime import date
from datetime import timedelta
import holidays


class CalendarFeatureCreator:

    def __init__(self, df: pd.DataFrame, country: str):
        # -- Main data
        self.df = df
        self.country = country
        # -- Initialize holidays
        self.df_holidays = self.__get_holidays()

    def __get_holidays(self):
        # -- Add Spain Holidays
        holidays = eval("holidays." + self.country + "()")
        # This is done because of a bug in the package holiday. Weird, yes.
        min_year = int(self.df['date'].astype(str).str[0:4].min())
        max_year = int(self.df['date'].astype(str).str[0:4].max()) + 1
        year_list = [i for i in range(min_year, max_year)]
        [date(year, 1, 1) in holidays for year in year_list]
        df_holidays = pd.DataFrame.from_dict(
            holidays,
            orient='index'
        ).reset_index()
        df_holidays["date"] = pd.to_datetime(df_holidays['index'])
        df_holidays['holiday'] = 1
        df_holidays = df_holidays[["date", "holiday"]].copy()
        return df_holidays

    def add_holidays(self, granularity):
        df_holiday_granular = self.df_holidays.copy()
        df_ts = self.df.copy()
        if granularity == "M":
            # -- Compute holidays by month
            df_holiday_granular['date'] = (
                df_holiday_granular["date"].astype(str).str[0:7]
            )
            df_holiday_granular = (
                df_holiday_granular
                .groupby('date')["holiday"].sum().reset_index()
            )
            df_holiday_granular['date'] = (
                pd.to_datetime(df_holiday_granular['date'] + '-01')
            )
        elif granularity == "W":
            # -- Compute holidays by week
            # ----- Create dummy key
            df_holiday_granular['key'] = 0
            df_ts['key'] = 0
            # ----- Create end date of week
            df_ts['date_max_week'] = df_ts['date'] + timedelta(days=6)
            # ----- Cartesian
            df_holiday_granular.rename(
                columns={'date': 'date_holidays'}, inplace=True
            )
            df_ts = df_ts.merge(
                df_holiday_granular,
                on='key',
                how='outer'
            )
            # ----- Filter holidays within each period of the week
            df_ts = df_ts.loc[
                (df_ts['date_holidays'] >= df_ts['date'])
                & (df_ts['date_holidays'] <= df_ts['date_max_week'])].copy()
            # ----- Aggregate holidays by week
            df_holiday_granular = (
                df_ts.groupby('date')['holiday'].sum().reset_index()
            )
        df_holiday_granular['date'] = pd.to_datetime(
            df_holiday_granular['date']
        )
        self.df = self.df.merge(
            df_holiday_granular,
            on='date',
            how='left'
        )[["date", "value", "holiday"]].fillna(0)
