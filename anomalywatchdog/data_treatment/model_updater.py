import pandas as pd
from anomalywatchdog.utils.kruskal_test_seasonality \
    import kruskal_test_seasonality


class ModelUpdater:

    def __init__(
            self,
            df: pd.DataFrame,
            granularity: str,
            config: dict
    ):
        # -- Initialize inputs
        self.df = df
        self.config = config
        self.granularity = granularity
        # -- Update seasonality
        self.__update_ts_seasonality()
        if "prophet" in self.config['models_to_use']:
            self.__update_model_seasonality("prophet")
        if "auto_arima" in self.config['models_to_use']:
            self.__update_model_seasonality("auto_arima")
        # -- Update granularity
        if "autoencoder_lstm" in self.config['models_to_use']:
            self.__update_model_granularity("autoencoder_lstm")
        if "autoencoder_conv" in self.config['models_to_use']:
            self.__update_model_granularity("autoencoder_conv")

    def __update_ts_seasonality(self):
        df_ts = self.df.copy()
        dict_seasonality = {
            "yearly_seasonality": "year"
        }
        if self.granularity == 'D':
            dict_seasonality["weekly_seasonality"] = "week"
            dict_seasonality["monthly_seasonality"] = "month"
        elif self.granularity == 'W':
            dict_seasonality["monthly_seasonality"] = "month"
            self.config['ts_seasonality']['weekly_seasonality'] = False
        elif self.granularity == 'M':
            self.config['ts_seasonality']['monthly_seasonality'] = False
            self.config['ts_seasonality']['weekly_seasonality'] = False
        for key_seasonality in dict_seasonality.keys():
            df_ts['period'] = eval(
                "pd.to_datetime(df_ts['date']).dt." + dict_seasonality[
                    key_seasonality])
            self.config['ts_seasonality'][key_seasonality] = (
                kruskal_test_seasonality(
                    df=df_ts,
                    column_period='period',
                    column_value='value')
            )

    def __update_model_seasonality(self, model):
        list_seasonality = [
            "weekly_seasonality", "monthly_seasonality", "yearly_seasonality"
        ]
        for i_seasonality in list_seasonality:
            self.config[model][i_seasonality] = (
                self.config['ts_seasonality'][i_seasonality]
            )

    def __update_model_granularity(self, model):
        if model == 'autoencoder_conv':
            self.config[model]['granularity'] = self.granularity
        if model == 'autoencoder_lstm':
            self.config[model]['granularity'] = self.granularity
