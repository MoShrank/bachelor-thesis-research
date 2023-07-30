from typing import List, Tuple

import numpy as np


class CoordinateEncoder:
    def __init__(
        self,
        min_val: int,
        max_val: int,
        number_of_bits: int,
        number_of_active_bits: int,
    ):
        self.min_val = min_val
        self.max_val = max_val
        self.range = max_val - min_val
        self.number_of_bits = number_of_bits
        self.number_of_active_bits = number_of_active_bits
        self.buckets = number_of_bits - number_of_active_bits + 1
        self.bucket_size = self.range / self.buckets

    def get_sdr(self, val: int) -> np.ndarray:
        encoded_value = np.zeros(self.number_of_bits, dtype=int)

        bucket_starting_index = int((self.buckets * (val - self.min_val)) / self.range)
        bucket_ending_index = bucket_starting_index + self.number_of_active_bits

        encoded_value[bucket_starting_index:bucket_ending_index] = 1

        return encoded_value

    def encode(self, coordinates: List[Tuple[int, int]]) -> np.ndarray:
        encoded_coordinates = np.zeros(
            (len(coordinates), self.number_of_bits * 2), dtype=int
        )

        for idx, coordinate in enumerate(coordinates):
            x_sdr = self.get_sdr(coordinate[0])
            y_sdr = self.get_sdr(coordinate[1])

            sdr = np.concatenate((x_sdr, y_sdr))
            encoded_coordinates[idx] = sdr

        return encoded_coordinates
