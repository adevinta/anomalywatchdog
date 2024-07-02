import numpy as np


def create_sequences(values: np.ndarray, time_steps: int):
    output = []
    for i in range(len(values) - time_steps + 1):
        output.append(values[i: (i + time_steps)])
    return np.stack(output)
