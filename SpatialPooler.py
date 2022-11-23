import pickle
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
        potential_pool_radius: int,
        seed: int = 42,
    ):
        self.input_dimension = np.array(input_dimension)
        self.column_dimension = np.array(column_dimension)
        self.number_of_inputs = np.prod(input_dimension)
        self.number_of_columns = np.prod(column_dimension)

        self.synapse_connection_size = int(self.number_of_inputs * connection_sparsity)
        self.connection_sparsity = connection_sparsity
        self.potential_pool_radius = potential_pool_radius

        self.permanence_threshold = permanence_threshold
        self.stimulus_threshold = stimulus_threshold

        self.permanence_increment = permanence_increment
        self.permanence_decrement = permanence_decrement

        self.number_of_active_columns = int(self.number_of_columns * column_sparsity)

        self.potential_pools = np.zeros(
            (self.number_of_columns, self.number_of_inputs), dtype=bool
        )

        self.permanences = np.zeros(
            (self.number_of_columns, self.number_of_inputs), dtype=float
        )

        self.connected_synapses = np.zeros(
            (self.number_of_columns, self.number_of_inputs), dtype=bool
        )

        self.boost_factors = np.ones(self.number_of_columns, dtype=float)
        self.overlap_duty_cycles = np.zeros(self.number_of_columns, dtype=float)
        self.active_duty_cycle = np.zeros(self.number_of_columns, dtype=float)
        self.duty_cycle_period = 1000  # TODO find out why init with 1000
        self.iteration = 0

        np.random.seed(seed)

        self._initialize()

    def get_potential_pool_center(self, column_idx: int) -> int:
        """
        this function returns the index of the center of the potential pool
        given a column index. It distributes the potential pool evenly over the
        input space.
        TODO: find out how this exactly works and find better implementation
        """

        column_coords = np.array(np.unravel_index(column_idx, self.column_dimension))

        # does this normalise the coords?
        ratios = column_coords / self.column_dimension
        input_coords = ratios * self.input_dimension
        input_coords += 0.5 * self.input_dimension / self.column_dimension
        inputs_coords = input_coords.astype(int)
        input_index = np.ravel_multi_index(
            inputs_coords, self.input_dimension, mode="clip"
        )

        return input_index

    def get_neighborhood(self, center_index: int) -> np.ndarray:
        """
        returns the indices of the neighborhood of a given center index
        """

        center_coords = np.array(np.unravel_index(center_index, self.input_dimension))
        intervals = []
        for idx, dimension in enumerate(self.input_dimension):
            left = max(0, center_coords[idx] - self.potential_pool_radius)
            right = min(dimension, center_coords[idx] + self.potential_pool_radius)
            intervals.append(np.arange(left, right + 1))

        neighborhood = np.array(np.meshgrid(*intervals)).T.reshape(
            -1, len(self.input_dimension)
        )

        return neighborhood

    def get_potential_pool(self, column_idx: int) -> np.ndarray:
        center_index = self.get_potential_pool_center(column_idx)
        neighborhood = self.get_neighborhood(center_index)

        indices = np.ravel_multi_index(
            neighborhood.T, self.input_dimension, mode="clip"
        )

        return indices

    def _init_potential_pools(self):
        for column in range(self.number_of_columns):
            potential_pool = self.get_potential_pool(column)
            no_connections = int(len(potential_pool) * self.connection_sparsity + 0.5)

            connections = np.random.choice(
                potential_pool,
                size=no_connections,
                replace=False,
            )
            self.potential_pools[column, connections] = True

    def _init_permanences(self):
        self.permanences = np.random.uniform(
            0, 1, size=(self.number_of_columns, self.number_of_inputs)
        )

        self.permanences[self.potential_pools == False] = 0

    def _init_connected_synapses(self):
        self.connected_synapses = self.permanences >= self.permanence_threshold

    def _initialize(self):
        self._init_potential_pools()
        self._init_permanences()
        self._init_connected_synapses()

    def calculate_overlap(self, input_vector: np.ndarray) -> np.ndarray:
        overlap = np.zeros(self.number_of_columns, dtype=float)
        input_flattened = input_vector.flatten()

        for column in range(self.number_of_columns):
            overlap[column] = np.sum(
                np.logical_and(
                    input_flattened, self.connected_synapses[column, :]
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
        for column_idx in top_columns:
            for synapse_idx, connected in enumerate(
                self.potential_pools[column_idx, :]
            ):
                if not connected:
                    continue

                if flattened_input[synapse_idx]:
                    self.permanences[
                        column_idx, synapse_idx
                    ] += self.permanence_increment
                else:
                    self.permanences[
                        column_idx, synapse_idx
                    ] -= self.permanence_decrement

                if self.permanences[column_idx, synapse_idx] < 0:
                    self.permanences[column_idx, synapse_idx] = 0

                if self.permanences[column_idx, synapse_idx] > 1:
                    self.permanences[column_idx, synapse_idx] = 1

                self.connected_synapses[column_idx, synapse_idx] = (
                    self.permanences[column_idx, synapse_idx]
                    >= self.permanence_threshold
                )

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
        with open(path, "wb") as f:
            pickle.dump(self, f)

    def load_state(self, path: str):
        with open(path, "rb") as f:
            temp = pickle.load(f)
            self.__dict__.update(temp.__dict__)

    def compute(self, input_vector: np.ndarray, learn: bool) -> np.ndarray:
        """
        Computes active columns for given input vector
        and applies boost and updates internal permanences if learning is on

        :param input_vector: input vector with shape <input_dimension>
        :param learn: if True, learning is applied

        :return: active columns with shape <number_of_columns>
        """
        assert (
            np.array(input_vector.shape) == self.input_dimension
        ).all(), f"Input vector has wrong shape. Expected {self.input_dimension}, got {input_vector.shape}"

        overlap = self.calculate_overlap(input_vector)

        # if learn:
        #    overlap = self.boost_columns(overlap)

        winning_columns = self.get_winning_columns(overlap)

        if learn:
            self.update_permanences(input_vector, winning_columns)
            # self.update_duty_cycles(overlap, winning_columns)
            # TODO bump weak columns
            # self.update_boost_factors()

        self.iteration += 1

        return winning_columns
