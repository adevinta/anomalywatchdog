# AnomalyWatchdog

AnomalyWatchdog detects outliers for time series using both statistical and 
machine learning approaches and showcase them. It works for both daily, weekly 
and monthly data.

If a time series split in different dimensions is provided, AnomalyWatchdog
first groups the data by the id provided and analyzes outliers at its highest
level. If an outlier is detected, it will analyze outliers at the different 
dimensions to detect the origin of the anomaly.

## Installation

```bash

pip install AnomalyWatchdog

```

## Quickstart

To detect anomalies in your data, you need to insert the following parameters
in the AnomalyWatchdog class as you can see below.

```{python, error=TRUE, include=TRUE}
from anomalywatchdog import AnomalyWatchdog

anomaly_watchdog = AnomalyWatchdog(
            df: Union[pd.DataFrame, DataFrame],
            column_date: str,
            column_target: str,
            granularity: str,
            columns_dimension: list[str] = None,
            start_date: Union[str, None] = None,
            end_date: Union[str, None] = None,
            models_to_use: List[str] = ['auto_arima', 'Prophet'],
        )
```

### Inputs
AnomalyWatchdog has the following inputs:
- df: pandas DataFrame or spark DataFrame that contains the required column_id, column_date, column_target and columns_dimension.
- column_date: String containing the column name of the time series dates. Values should be str in format YYYY-MM-DD (i.e. 2020-01-30).
- column_target: String containing the column name of the time series values. Values should be float or int.
- granularity: String containing the granularity of the time series data. Values available are "D" for daily, "M" for monthly and "W" for weekly data.
- columns_dimension: List of strings containing the column dimension names representing the disaggregation of the data if any.
- start_date: String containing the start date to return anomalies. Values should be str in format YYYY-MM-DD (i.e. 2020-01-30). If None, it returns all the history.
- end_date: String containing the end date to return anomalies. Values should be str in format YYYY-MM-DD (i.e. 2020-01-30). If None, it returns all the history.
- models_to_use: List of strings containing the models available. Models available are "autoencoder_basic", "autoencoder_lstm", "prophet" and "auto_arima". If non value is provided, AnomalyWatchdog performs with only "prophet" and "auto_arima".

### Outputs
AnomalyWatchdog has two outputs, one of which is only delivered if 
columns_dimension parameter is specified.

```{python, error=TRUE, include=TRUE}
# -- AnomalyWatchdog output for main time series
anomaly_watchdog.df_anomaly
# -- AnomalyWatchdog output for each of the dimensions (only if columns_dimension is specified)
anomaly_watchdog.df_anomaly_dimension
```
