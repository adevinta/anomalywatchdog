# -- Packages
import pandas as pd


class DataADHandler:
    """
    This class creates a data set that cleans the Data for Anomaly Detection
    """
    def __init__(
            self,
            df: pd.DataFrame,
            granularity: str = 'D'
    ):
        # -- Main data and parameters
        self.df = df
        self.granularity = granularity
        self.group_columns = ["date"]
        self.df_grouped = pd.DataFrame()
        # -- Methods
        self.__expand_dates()
        self.__group_by()
        self.__get_ordered_dataframe()

    def __expand_dates(self):
        # -- Get first date of month and max date to disaggregate
        min_date = self.df["date"].min()
        max_date = self.df["date"].max()
        if self.granularity == 'W':
            num_weeks = int((max_date - min_date).days / 7) + 1
            dates = pd.date_range(start=min_date,
                                  periods=num_weeks, freq="7D").tolist()
        elif self.granularity == 'M':
            dates = pd.date_range(start=min_date,
                                  end=max_date, freq="MS").tolist()
        else:
            dates = pd.date_range(start=min_date,
                                  end=max_date, freq="D").tolist()
        # -- dates df
        df_dates = pd.DataFrame(
            range(len(dates)),
            dates
        ).reset_index()
        df_dates.columns = ["date", 'index']
        df_dates.drop(['index'], axis=1, inplace=True)
        # -- Add values
        self.df = df_dates.merge(
            self.df,
            on=self.group_columns,
            how='left'
        )
        # -- set correct date format
        self.df["date"] = pd.to_datetime(self.df["date"])

    def __get_ordered_dataframe(self):
        self.df.sort_values(self.group_columns, inplace=True)

    def __group_by(self):
        self.df_grouped = (
            self.df
                .groupby(self.group_columns)["value"]
                .sum()
                .reset_index()
        )
