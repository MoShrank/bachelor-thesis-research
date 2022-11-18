from typing import Tuple

import numpy as np


class SpatialPooler:
    def __init__(
        self,
        input_dimension: Tuple[int],
        column_dimension: Tuple[int],
        # TODO rename connection sparsity
        connection_sparsity: float,
        permanence_threshold: float,
        stimulus_threshold: float,
        permanence_increment: float,
        permanence_decrement: float,
        column_sparsity: float,
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
        self.overlap_duty_cycles = np.zeros(self.number_of_columns, dtype=float)
        self.active_duty_cycle = np.zeros(self.number_of_columns, dtype=float)
        self.duty_cycle_period = 1000  # TODO find out why init with 1000
        self.iteration = 0

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
        overlap = np.zeros(self.number_of_columns, dtype=float)
        input_flattened = input_vector.flatten()

        for column in range(self.number_of_columns):
            overlap[column] = np.sum(
                np.logical_and(
                    input_flattened, self.potential_synapses[column, :]
                ).astype(float)
            )

        return overlap

    def boost_columns(self, overlap: np.ndarray) -> np.ndarray:
        boosted_overlap = overlap * self.boost_factors
        return boosted_overlap

    def get_winning_columns(self, overlap: np.ndarray) -> np.ndarray:
        top_columns = np.argsort(overlap)[::-1][: self.number_of_active_columns]
        top_columns = top_columns[overlap[top_columns] >= self.stimulus_threshold]

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

    def calculate_moving_average(
        self, duty_cycles: np.ndarray, period: int, new_value: np.ndarray
    ) -> np.ndarray:
        moving_average = ((period - 1) * duty_cycles + new_value) / (period)
        return moving_average

    def update_duty_cycles(
        self, overlap_columns: np.ndarray, active_columns: np.ndarray
    ):

        overlap = np.zeros(self.number_of_columns, dtype=float)
        overlap[overlap_columns > 0] = 1

        active = np.zeros(self.number_of_columns, dtype=float)
        active[active_columns] = 1

        period = self.duty_cycle_period
        if period > self.iteration:
            period = self.iteration

        self.overlap_duty_cycles = self.calculate_moving_average(
            self.overlap_duty_cycles, self.duty_cycle_period, overlap
        )
        self.active_duty_cycle = self.calculate_moving_average(
            self.active_duty_cycle, self.duty_cycle_period, active
        )

    def exp_boost_function(
        self, boost_strength: np.ndarray, duty_cycle: np.ndarray, target_density: float
    ):
        return np.exp((target_density - duty_cycle) * boost_strength)

    def update_boost_factors(self):
        target_density = self.number_of_active_columns / self.number_of_columns
        self.boost_factors = self.exp_boost_function(
            self.boost_factors, self.active_duty_cycle, target_density
        )

    def save_state(self, path: str):
        np.savez(
            path,
            potential_synapses=self.potential_synapses,
            permanences=self.permanences,
        )

    def load_state(self, path: str):
        state = np.load(path)
        self.potential_synapses = state["potential_synapses"]
        self.permanences = state["permanences"]

    def compute(self, input_vector: np.ndarray, learn: bool) -> np.ndarray:
        """
        Computes active columns for given input vector
        and applies boost and updates internal permanences if learning is on

        :param input_vector: input vector with shape <input_dimension>
        :param learn: if True, learning is applied

        :return: active columns with shape <number_of_columns>
        """

        assert (
            input_vector.shape == self.input_dimension
        ), f"Input vector has wrong shape. Expected {self.input_dimension}, got {input_vector.shape}"

        overlap = self.calculate_overlap(input_vector)

        if learn:
            overlap = self.boost_columns(overlap)

        winning_columns = self.get_winning_columns(overlap)

        if learn:
            self.update_permanences(input_vector, winning_columns)
            self.update_duty_cycles(overlap, winning_columns)
            # TODO bump weak columns
            self.update_boost_factors()

        self.iteration += 1

        return winning_columns
