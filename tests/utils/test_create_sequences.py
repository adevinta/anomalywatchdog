import pandas as pd
import numpy as np
from anomalywatchdog.utils.create_sequences \
    import create_sequences


TIME_STEPS=2

def test_create_fourier_terms():
    # -- Initialize input checker
    array_sequence = create_sequences(
        values=input_df_value(), time_steps=TIME_STEPS
    )
    df_array_sequence = pd.DataFrame(
        array_sequence.reshape(-1, array_sequence.shape[-1]),
        columns=['value']
    )
    # -- Check df and columns
    pd.testing.assert_frame_equal(
        pd.DataFrame(df_array_sequence),
        pd.DataFrame(expected_df())
    )

def input_df_value():
    value_list = [
        5,
        10,
        15,
    ]
    return pd.DataFrame(
        {'value': value_list}
    )

def expected_df():
    expected_array = np.array([[[5], [10]], [[10], [15]]])
    expected_array = expected_array.reshape(-1, expected_array.shape[-1])
    return pd.DataFrame(expected_array, columns=['value'])