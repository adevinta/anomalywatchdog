import numpy as np
import pandas as pd


def create_fourier_terms(df, freq, K):
    t = np.arange(len(df))
    df_fourier_terms = pd.DataFrame()
    for k in range(1, K + 1):
        df_fourier_terms[f'sin_{freq}_{k}'] = np.sin(2 * np.pi * k * t / freq)
        df_fourier_terms[f'cos_{freq}_{k}'] = np.cos(2 * np.pi * k * t / freq)
    return df_fourier_terms
