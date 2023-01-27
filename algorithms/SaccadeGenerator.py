from typing import Dict, List, Tuple

import numpy as np
from tensorflow.keras.utils import Sequence

from util.math import get_random_pool_indices


class SaccadeGenerator(Sequence):
    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        saccade_size: Tuple[int, int],
        no_saccades: int = 5,
        batch_size: int = 128,
    ):

        self.size = (x[0].shape[0], x[0].shape[1])
        self.saccade_size = saccade_size
        self.x = x.copy()
        self.current = 0
        self.no_saccades = no_saccades
        self.y = y.copy()
        self.batch_size = batch_size

    """
    def __iter__(self):
        return self

    def __next__(self) -> List[Dict]:
        self.current += 1

        if self.current < self.x.shape[0]:
            saccades = self.generate_saccades(self.current)

            return saccades

        raise StopIteration
    """

    def __len__(self) -> int:
        return self.x.shape[0] // self.batch_size

    def __getitem__(self, index: int) -> Tuple:
        batch_x = []
        batch_y = []

        for i in range(self.batch_size):
            idx = index * self.batch_size + i

            saccades = self.generate_saccades(idx)
            x = self.to_timeseries_vector(saccades)

            batch_x.append(x)
            batch_y.append(self.y[idx])

        x = np.array(batch_x)
        y = np.array(batch_y)

        return x, y

    def to_vector(self, saccades: List[Dict]) -> np.ndarray:
        saccade_size = self.saccade_size[0] * self.saccade_size[1]
        location_size = 2
        total_location_size = location_size * self.no_saccades

        vector = np.zeros(
            shape=(self.no_saccades * saccade_size + total_location_size,)
        )

        for i, saccade in enumerate(saccades):
            start = i * saccade_size
            vector[start : start + saccade_size] = np.reshape(
                saccade["feature"], (saccade_size,)
            )
            vector[
                start + saccade_size : start + saccade_size + location_size
            ] = saccade["location"]

        return vector

    def to_timeseries_vector(self, saccades: List[Dict]) -> np.ndarray:
        saccade_size = self.saccade_size[0] * self.saccade_size[1]
        location_size = 2
        time_step_size = saccade_size + location_size

        vector = np.zeros(shape=(self.no_saccades, time_step_size))

        for i, saccade in enumerate(saccades):
            vector[i, :saccade_size] = np.reshape(saccade["feature"], (saccade_size,))
            vector[i, saccade_size : saccade_size + location_size] = saccade["location"]

        return vector

    def generate_saccades(self, idx: int) -> List[Dict]:
        """
        :return: [{location: <displacement vector>, feature: <feature_vector>}, ...]
        """

        saccades = []
        prev_center = None

        for _ in range(self.no_saccades):
            center, xs, ys = get_random_pool_indices(
                np.array(self.size), np.array(self.saccade_size)
            )

            features = self.x[idx, xs, ys]
            location = center
            displacement = np.array([0, 0])
            indices = np.array([xs, ys])

            if prev_center is not None:
                displacement = center - prev_center

            saccade = {
                "location": location,
                "feature": features,
                "displacement": displacement,
                "indices": indices,
                "data_index": idx,
            }

            saccades.append(saccade)

        return saccades
