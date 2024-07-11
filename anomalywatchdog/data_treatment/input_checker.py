import pandas as pd
from pyspark.sql import DataFrame
from typing import Union, List


class InputChecker:

    def __init__(
            self,
            df: Union[pd.DataFrame, DataFrame],
            column_date: str,
            column_target: str,
            granularity: str,
            columns_dimension: List[str],
            models_to_use: List[str],
            config: dict,
            start_date: Union[str, None] = None,
            end_date: Union[str, None] = None
    ):
        # -- Main inputs
        self.df = df
        self.column_date = column_date
        self.column_target = column_target
        self.granularity = granularity
        self.columns_dimension = columns_dimension
        self.models_to_use = models_to_use
        self.start_date = start_date
        self.end_date = end_date
        self.config = config
        # -- Preliminary Checks
        self.__check_df_instance()
        self.__check_columns_in_dataframe()
        if len(self.columns_dimension) > 0:
            self.__check_column_list_types(self.columns_dimension)
        self.__check_column_str_type(self.column_target)
        self.__check_column_str_type(self.column_date)
        self.__enforce_lowercase()
        self.__check_granularity()
        self.start_date = self.__check_date(self.start_date)
        self.end_date = self.__check_date(self.end_date)
        self.__check_dates_consistency()

        if self.models_to_use:
            self.__check_column_list_types(self.models_to_use)
            self.__check_models_to_use()
            self.__update_config()

    def __check_df_instance(self):
        if (
            not isinstance(self.df, pd.DataFrame)
            and isinstance(self.df, DataFrame)
        ):
            self.df = self.df.toPandas()
        elif (
            not isinstance(self.df, pd.DataFrame)
            and not isinstance(self.df, DataFrame)
        ):
            error_string = (
                    f"Input df should be a pandas or Spark DataFrame."
            )
            raise TypeError(error_string)

    def __check_columns_in_dataframe(self):
        columns_to_check = [self.column_target, self.column_date]
        if self.columns_dimension is not None:
            columns_to_check += self.columns_dimension
        for column in columns_to_check:
            if column not in self.df.columns:
                error_string = (
                    f"Column name '{column}' does not exist in the input " +
                    "DataFrame. Columns available in the input DataFrame " +
                    f"are {self.df.columns}"
                )
                raise ValueError(error_string)

    @staticmethod
    def __check_column_list_types(list_name: List[str]):
        if not isinstance(list_name, list):
            error_string = (
                    f"Input parameter {list_name} is "
                    f"{type(list_name)}. " +
                    f"Expected input type is List[str]."
            )
            raise TypeError(error_string)
        for column in list_name:
            if not isinstance(column, str):
                error_string = (
                    f"Element of list {column} is a {type(column)}. " +
                    f"Expected elements of the list must be all str."
                )
                raise TypeError(error_string)

    @staticmethod
    def __check_column_str_type(column_name: str):
        if not isinstance(column_name, str):
            error_string = (
                    f"Input parameter {column_name} is "
                    f"{type(column_name)}. " +
                    f"Expected input type is str."
            )
            raise TypeError(error_string)

    def __enforce_lowercase(self):
        self.column_target = self.column_target.lower()
        self.column_date = self.column_date.lower()
        if self.columns_dimension is not None:
            self.columns_dimension = (
                [col.lower() for col in self.columns_dimension]
            )
        for col in self.df.columns:
            self.df.rename(columns={col: col.lower()}, inplace=True)
        if self.models_to_use:
            self.models_to_use = (
                [model.lower() for model in self.models_to_use]
            )

    def __check_granularity(self):
        granularity_list = ["D", "M", "W"]
        if not isinstance(self.granularity, str):
            error_string = (
                    "Input parameter granularity is " +
                    f"{type(self.granularity)}. " +
                    "Expected input type is str."
            )
            raise TypeError(error_string)
        if self.granularity not in granularity_list:
            error_string = (
                f"Parameter granularity must be one of {granularity_list}."
            )
            raise ValueError(error_string)

    def __check_models_to_use(self):
        models_to_use_list = [
                "autoencoder_basic",
                "autoencoder_lstm",
                "prophet",
                "auto_arima",
                "autoencoder_conv"
            ]
        for model in self.models_to_use:
            if model not in models_to_use_list:
                error_string = (
                    f"Model {model} is unknown. " +
                    f"Models available are {models_to_use_list}."
                )
                raise TypeError(error_string)

    def __update_config(self):
        if self.models_to_use:
            self.config['models_to_use'] = self.models_to_use

    @staticmethod
    def __check_date(date_string: Union[str, None]):
        if date_string:
            formatting_error = (
                f"Format for {date_string} not understood. "
                f"Accepted format is 'YYYY-MM-DD'"
                f"(e.g. 2021-03-28)."
            )
            try:
                return pd.to_datetime(
                    date_string,
                    format="%Y-%m-%d"
                )
            except:
                raise ValueError(formatting_error)

    def __check_dates_consistency(self):
        if not self.start_date and self.end_date:
            self.start_date = self.end_date
        if not self.end_date and self.start_date:
            self.end_date = self.start_date

        if self.start_date:
            if (pd.to_datetime(self.end_date, format="%Y-%m-%d")
                    < pd.to_datetime(self.start_date, format="%Y-%m-%d")):
                formatting_error = (
                    f"Value for end_date: {self.end_date} must be greater or "
                    f"equal than start_date: {self.start_date}."
                )
                raise ValueError(formatting_error)

