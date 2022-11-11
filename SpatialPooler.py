from typing import Tuple

import numpy as np


class SpatialPooler:
    def __init__(
        self,
        input_dimension: Tuple[int],
        column_dimension: Tuple[int],
        connection_sparsity: float,
        permanence_threshold: float,
        stimulus_threshold: float,
        permanence_increment: float,
        permanence_decrement: float,
        column_sparsity: int,
        seed: int = 42,
    ):
        self.input_dimension = input_dimension
        self.column_dimension = column_dimension
        self.number_of_inputs = np.prod(input_dimension)
        self.number_of_columns = np.prod(column_dimension)

        self.synapse_connection_size = int(self.number_of_inputs * connection_sparsity)

        self.permanence_threshold = permanence_threshold
        self.stimulus_threshold = stimulus_threshold

        self.permanence_increment = permanence_increment
        self.permanence_decrement = permanence_decrement

        self.number_of_active_columns = int(self.number_of_columns * column_sparsity)

        self.potential_synapses = np.zeros(
            (self.number_of_columns, self.number_of_inputs), dtype=bool
        )

        self.permanences = np.zeros(
            (self.number_of_columns, self.number_of_inputs), dtype=float
        )

        self.boost_factors = np.ones(self.number_of_columns, dtype=float)

        np.random.seed(seed)

        self._initialize()

    def _initialize(self):
        for column in range(self.number_of_columns):
            random_indices = np.random.choice(
                self.number_of_inputs,
                size=self.synapse_connection_size,
                replace=False,
            )
            self.potential_synapses[column, random_indices] = True

        self.permanences = np.random.rand(self.number_of_columns, self.number_of_inputs)

    def calculate_overlap(self, input_vector: np.ndarray) -> np.ndarray:
        overlap = np.zeros(self.number_of_columns, dtype=int)
        input_flattened = input_vector.flatten()

        for column in range(self.number_of_columns):
            overlap[column] = np.sum(
                np.logical_and(
                    input_flattened, self.potential_synapses[column, :]
                ).astype(int)
            )
        return overlap

    def get_winning_columns(self, overlap: np.ndarray) -> np.ndarray:
        overlap[overlap < self.stimulus_threshold] = 0
        top_columns = np.argsort(overlap)[::-1][: self.number_of_active_columns]
        return top_columns

    def top_columns_to_sdr(self, top_columns: np.ndarray) -> np.ndarray:
        sdr = np.zeros(self.number_of_columns, dtype=bool)
        sdr[top_columns] = True
        return sdr

    def update_permanences(self, input_vector: np.ndarray, top_columns: np.ndarray):
        flattened_input = input_vector.flatten()
        for column in top_columns:
            for input_index in range(self.number_of_inputs):
                if flattened_input[input_index]:
                    self.permanences[column, input_index] += self.permanence_increment
                else:
                    self.permanences[column, input_index] -= self.permanence_decrement

                if self.permanences[column, input_index] > 1:
                    self.permanences[column, input_index] = 1
                elif self.permanences[column, input_index] < 0:
                    self.permanences[column, input_index] = 0

    def update_boost_factors(self):
        pass

    def save_state(self, path: str):
        np.savez(
            path,
            potential_synapses=self.potential_synapses,
            permanences=self.permanences,
        )

    def compute(self, input_vector: np.ndarray, learn: bool) -> np.ndarray:
        overlap = self.calculate_overlap(input_vector)
        winning_columns = self.get_winning_columns(overlap)
        if learn:
            self.update_permanences(input_vector, winning_columns)

        return winning_columns
