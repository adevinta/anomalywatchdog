import numpy as np
import pandas as pd

def create_sequences(values:pd.Series, time_steps:int):
    output = []
    for i in range(len(values) - time_steps + 1):
        output.append(values[i: (i + time_steps)])
    return np.stack(output)