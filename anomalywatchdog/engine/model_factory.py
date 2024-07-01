from loaders.package.b_modelling.autoencoder_lstm_model\
    import AutoEncoderLSTMModel
from loaders.package.b_modelling.autoencoder_conv_model\
    import AutoEncoderConvModel
from loaders.package.b_modelling.prophet_model\
    import ProphetModel
from loaders.package.b_modelling.auto_arima_model\
    import AutoArimaModel


class ModelFactory:
    @staticmethod
    def get_model(model, df_train, dict_config):
        if model == "autoencoder_lstm":
            return AutoEncoderLSTMModel(
                df=df_train.copy(),
                dict_params=dict_config
        )
        if model == "autoencoder_conv":
            return AutoEncoderConvModel(
                df=df_train.copy(),
                dict_params=dict_config
        )
        if model == "prophet":
            return ProphetModel(
                df=df_train.copy(),
                dict_params=dict_config
        )
        if model == "auto_arima":
            return AutoArimaModel(
                df=df_train.copy(),
                dict_params=dict_config
        )
