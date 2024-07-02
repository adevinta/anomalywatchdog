import pandas as pd
from spicy import stats


def kruskal_test_seasonality(
        df: pd.DataFrame,
        column_period: str,
        column_value: str
):
    res = []
    for i in df[column_period].unique():
        res.append(df[df[column_period] == i][column_value].values)
    test_kruskal = stats.kruskal(*res)
    print(test_kruskal.pvalue)
    if test_kruskal.pvalue < 0.05:
        return True
    else:
        return False
