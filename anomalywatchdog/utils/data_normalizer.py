import numpy as np


class DataNormalizer:
    @staticmethod
    def normalization_standard(
            s
    ) -> np.ndarray:
        return (s - s.mean()) / s.std()
