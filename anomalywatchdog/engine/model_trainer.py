import pandas as pd
from anomalywatchdog.engine.model_factory \
    import ModelFactory


class ModelTrainer:
    def __init__(self, model_names, df_train, config):
        self.model_names = model_names
        self.df_train = df_train
        self.config = config

    @staticmethod
    def filter_config_by(config, model) -> dict:
        return config[model]

    def train(self) -> pd.DataFrame:
        list_output = []
        print(">> Training:")
        for model_name in self.model_names:
            dict_output_id = {}
            model = ModelFactory.get_model(
                model=model_name,
                df_train=self.df_train.copy(),
                dict_config=ModelTrainer.filter_config_by(
                    self.config,
                    model_name
                )
            )
            dict_output_id['model'] = model
            list_output.append(dict_output_id)
        return pd.DataFrame(list_output)
