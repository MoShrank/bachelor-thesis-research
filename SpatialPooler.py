import pickle
from typing import Any, Tuple

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
        column_sparsity: float,
        potential_pool_radius: int,
        boost_strength: float,
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

        self.column_sparsity = column_sparsity
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

        self.boost_strength = boost_strength
        self.boost_factors = np.ones(self.number_of_columns, dtype=float)
        self.overlap_duty_cycles = np.zeros(self.number_of_columns, dtype=float)
        self.active_duty_cycle = np.zeros(self.number_of_columns, dtype=float)
        self.duty_cycle_period = 1000
        self.iteration = 1

        np.random.seed(seed)

        self._initialize()

    def get_potential_pool_center(self, column_idx: int) -> int:
        """
        this function returns the index of the center of the potential pool
        given a column index. It distributes the potential pool evenly over the
        input space.

        :param column_idx: index of the column

        :return: index of the center of the potential pool
        """

        column_coords = np.array(np.unravel_index(column_idx, self.column_dimension))

        # does this normalise the coords?
        ratios = column_coords / self.column_dimension
        input_coords = ratios * self.input_dimension
        input_coords += 0.5 * self.input_dimension / self.column_dimension
        inputs_coords = input_coords.astype(int)

        center_index = np.ravel_multi_index(  # type: ignore
            inputs_coords, self.input_dimension, mode="clip"
        )

        return center_index

    def get_neighborhood(self, center_index: int) -> np.ndarray:
        """
        Returns the indices of the neighborhood of a given center index by
        taking all indices within a radius of <self.potential_pool_radius>

        :param center_index: index of the center

        :return: indices of the neighborhood
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
        """
        Returns the indices of the potential pool of a given column by
        calculating the center of the potential pool given a column index
        and then calculating the neighborhood of that center.

        :param column_idx: index of the column

        :return: indices of the potential pool
        """

        if self.potential_pool_radius == 0:
            indices = np.arange(self.number_of_inputs)

        else:
            center_index = self.get_potential_pool_center(column_idx)
            neighborhood = self.get_neighborhood(center_index)

            indices = np.ravel_multi_index(  # type: ignore
                neighborhood.T, self.input_dimension, mode="clip"
            )

        return indices

    def set_connected_synapses(self, permanences: np.ndarray):
        self.connected_synapses = permanences >= self.permanence_threshold

    def _init_potential_pools(self):
        """
        Initializes the potential pools of all columns.
        A potential pool is a set of input bits that a column can connect to.
        """

        for column in range(self.number_of_columns):
            potential_pool_indices = self.get_potential_pool(column)
            no_connections = int(
                len(potential_pool_indices) * self.connection_sparsity + 0.5
            )

            # get <no_connections> random indices from potential pool indices
            connections = np.random.choice(
                potential_pool_indices,
                size=no_connections,
                replace=False,
            )
            self.potential_pools[column, connections] = True

    def _init_permanences(self):
        for column in range(self.number_of_columns):
            potential_pool_size = np.sum(self.potential_pools[column, :])

            permanences = np.random.uniform(0, 1, potential_pool_size)

            self.permanences[column, self.potential_pools[column, :]] = permanences

    def _init_connected_synapses(self):
        self.set_connected_synapses(self.permanences)

    def _initialize(self):
        self._init_potential_pools()
        self._init_permanences()
        self._init_connected_synapses()

    def calculate_overlap(self, input_vector: np.ndarray) -> np.ndarray:
        """
        Calculates the overlap of the input vector with the columns.
        The overlap is defined as the number of on bits (1's) in the input vector
        that overlap with the connected synapses of a column.

        :param input_vector: the input vector

        :return: the overlap of the input vector with the columns shape: (number_of_columns,)
        """
        overlap = np.zeros(self.number_of_columns, dtype=float)
        input_flattened = input_vector.flatten()

        for column in range(self.number_of_columns):
            overlap[column] = np.sum(
                np.logical_and(
                    input_flattened, self.connected_synapses[column, :]
                ).astype(int)
            )

        return overlap

    def boost_columns(self, overlap: np.ndarray) -> np.ndarray:
        boosted_overlap = overlap * self.boost_factors
        return boosted_overlap

    def get_winning_columns(self, overlap: np.ndarray) -> np.ndarray:
        """
        Selects <number_of_active_columns> columns with the highest overlap
        which are above the stimulus threshold and returns their indices.

        :param overlap: the overlap of the input vector with the columns shape: (number_of_columns,)

        :return: the indices of the winning columns
        """

        # sort indices by overlap count in descending order and
        # select the first <number_of_active_columns> indices
        top_columns = np.argsort(overlap)[::-1][: self.number_of_active_columns]

        # select only the indices that are above the stimulus threshold
        winning_columns = top_columns[overlap[top_columns] >= self.stimulus_threshold]

        return winning_columns

    def top_columns_to_sdr(self, top_columns: np.ndarray) -> np.ndarray:
        """
        Converts the indices of the winning columns into an SDR.

        :param top_columns: the indices of the winning columns

        :return: SDR with shape (number_of_columns,)
        """

        sdr = np.zeros(self.number_of_columns, dtype=bool)
        sdr[top_columns] = True
        return sdr

    def update_permanences(self, input_vector: np.ndarray, top_columns: np.ndarray):
        flattened_input = input_vector.flatten()
        flipped_flattened_input = np.logical_not(flattened_input)

        for column_idx in top_columns:
            active = np.logical_and(
                flattened_input, self.potential_pools[column_idx, :]
            )

            inactive = np.logical_and(
                flipped_flattened_input, self.potential_pools[column_idx, :]
            )

            # increase active synapses
            self.permanences[column_idx, active == True] += self.permanence_increment

            # decrease inactive synapses
            self.permanences[column_idx, inactive == True] -= self.permanence_decrement

            # clip permanences to [0, 1] -> faster than np.clip
            self.permanences[column_idx, active > 1] = 1
            self.permanences[column_idx, inactive < 0] = 0

        self.set_connected_synapses(self.permanences)

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
            self.overlap_duty_cycles, period, overlap
        )
        self.active_duty_cycle = self.calculate_moving_average(
            self.active_duty_cycle, period, active
        )

    def exp_boost_function(
        self, boost_strength: np.ndarray, duty_cycle: np.ndarray, target_density: float
    ):
        return np.exp((target_density - duty_cycle) * boost_strength)

    def update_boost_factors(self):
        self.boost_factors = self.exp_boost_function(
            self.boost_strength, self.active_duty_cycle, self.column_sparsity
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
