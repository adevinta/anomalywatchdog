import pandas as pd


class ModelPredictor:
    def __init__(self, df_models_trained):
        self.df_models_trained = df_models_trained

    def predict(self) -> pd.DataFrame:
        list_output = []
        for model in self.df_models_trained['model'].unique():
            # -- Get anomalies
            model.get_anomalies()
            # -- Append
            list_output.append(model.df)
        # -- Convert to df
        df_output = pd.concat(list_output)
        return df_output
