from typing import Any, Dict, List, Tuple

import numpy as np


class GridCells:
    def __init__(
        self,
        no_modules: int,
        grid_cell_module_dimension: Tuple[int, int],
        orientation: float,
        scale: float,
        no_columns: int,
        no_cells_per_column: int,
        location_layer_threshold: float,
        seed: None | int = None,
    ):
        """
        :param: no_modules: number of grid cell modules
        :param: grid_cell_module_dimension: dimension of grid cell module should be a square
        :param: orientation: orientation of grid cell module - %TODO should be a vector for each module to have unique orientation
        :param: scale: scale of grid cell module - %TODO should be a vector for each module to have unique scale
        :param: no_columns: number of columns in input layer
        :param: no_cells_per_column: number of cells per column in input layer
        :param: location_layer_threshold: number of active dendritic segments required to activate a cell in sensory layer
        :param: seed: seed for random number generator
        """

        self.no_modules = no_modules
        self.grid_cell_module_dimension = np.array(grid_cell_module_dimension)

        assert grid_cell_module_dimension[0] == grid_cell_module_dimension[
            1
        ] and self.grid_cell_module_dimension.shape == (
            2,
        ), f"Grid cell module dimension should be a square. Received Shape: {self.grid_cell_module_dimension.shape}"
        self.no_cells_per_grid_cell_module = np.prod(grid_cell_module_dimension)

        assert (
            orientation >= 0 and orientation <= 360
        ), "Orientation should be in degrees"
        self.orientation = orientation

        assert scale > 0, "Scale should be positive"
        self.scale = scale
        self.phases_per_unit_distance = 1.0 / np.array([scale, scale])

        self.translation_matrix = np.array(
            [
                [math.cos(orientation), -math.sin(orientation)],
                [math.sin(orientation), math.cos(orientation)],
            ]
        )

        self.no_columns = no_columns
        self.no_cells_per_column = no_cells_per_column

        self.location_layer_threshold = location_layer_threshold

        if seed:
            np.random.seed(seed)

        # Learning State and Parameters
        self.dendritic_segments = np.zeros(
            (
                self.no_modules,
                self.no_cells_per_grid_cell_module,
                self.no_columns,
                self.no_cells_per_column,
            )
        )

        self.active_phases: List[Any] = []

    def get_random_phases(self) -> np.ndarray:
        """
        Returns a random phase for each grid cell module in shape (no_modules, 2)

        :return: random phases
        """

        active_phases = np.random.random((self.no_modules, 2))

        return active_phases

    def compute_active_cells(self, active_phases: np.ndarray) -> np.ndarray:
        """
        Computes the index of the active cells in the grid cell layer from the active phases

        :param: active_phases: active phases of grid cell modules in shape (no_modules, 2)
        """
        active_cell_coordinates = np.floor(
            active_phases * self.grid_cell_module_dimension
        ).astype(int)

        cells_for_active_phases = np.ravel_multi_index(
            active_cell_coordinates.T, self.grid_cell_module_dimension
        )

        return cells_for_active_phases

    def compute_movement(
        self, displacement: np.ndarray, active_phases: np.ndarray
    ) -> np.ndarray:
        """
        Computes the movement of previous phase to a new phase given a displacement vector

        :param: displacement: displacement vector in shape (2,)
        :param: active_phases: active phases of grid cell modules in shape (no_modules, 2)

        """

        # rotate and scale dsiplacement vector
        phase_displacement = (
            np.matmul(self.translation_matrix, displacement)
            * self.phases_per_unit_distance
        )

        active_phases = active_phases + phase_displacement

        active_phases = np.round(active_phases, decimals=9)
        active_phases = np.mod(active_phases, 1.0)

        return active_phases

    def reset_phases(self):
        self.active_phases = []

    def learn_object(self, object: List[Dict]):
        """
        :param: obj: [{location: <displacement vector>, feature: <feature_vector>}, ...]
        """

        self.active_phases.append(self.get_random_phases())

        for saccade in object:
            location = saccade["location"]
            feature = saccade["feature"]

            active_phases = self.active_phases[-1]
            if location:
                active_phases = self.compute_movement(location, active_phases)
                self.active_phases.append(active_phases)

            active_grid_cell_indices = self.compute_active_cells(
                active_phases
            )  # shape: (<no_modules>,)

            # get overlap of phases and dendritic distal segments to predict cells and
            # check if number of dendritic segments are above threshold for cell to be predicted
            predicted_cells = np.sum(
                self.dendritic_segments[:, active_grid_cell_indices], axis=(0, 1)
            )  # shape: (<no_columns>, <no_cells>)
            predicted_cells[predicted_cells < self.location_layer_threshold] = 0

            # get active columns from input feature
            active_columns = np.argwhere(feature)

            # set predicted cells in inactive columns to zero
            predicted_cells[~active_columns] = 0

            # choose active cells:
            # if active column has predicted cells, set cells to active -> copied from predicted_cells
            # if not either set all cells to be active or during learning choose one random cell
            active_cells = np.copy(predicted_cells)
            active_cells[active_cells > 0] = 1

            no_pred_cells_per_column = np.sum(
                predicted_cells, axis=1
            )  # shape: (<no_columns>,)

            column_indices_to_choose_cell = np.argwhere(
                active_cells[active_columns & (no_pred_cells_per_column == 0)]
            )

            no_columns_to_choose_cell = column_indices_to_choose_cell.shape[
                0
            ]  # number of columns to chose random cell for

            active_random_cells = np.random.randint(
                0, self.no_columns + 1, size=(no_columns_to_choose_cell)
            )

            active_cells[column_indices_to_choose_cell, active_random_cells] = 1

            # update location representation by calculating overlap whereas on active dendritic segment is enough
            # determine active grid cells from active cells in sensory layer
            active_cells_indices_sen_layer = np.argwhere(active_cells)
            # split into columns and cells
            active_cells_indices_separated = np.split(
                active_cells_indices_sen_layer, 2, axis=1
            )
            column_indices = active_cells_indices_separated[0].flatten()
            cell_indices = active_cells_indices_separated[1].flatten()

            # get indices of active grid cells
            active_grid_cell_indices = np.argwhere(
                self.dendritic_segments[:, :, column_indices, cell_indices]
            )

            # properly index active_grid_cell_indices by adding module dimension
            active_grid_cell_indices = np.insert(
                active_grid_cell_indices,
                1,
                np.arange(len(active_grid_cell_indices)),
                axis=1,
            )

            # merge active_grid_cell_indices with active_cells_indices_sen_layer into a tuple
            active_grid_cell_indices = np.concatenate(
                (active_grid_cell_indices, active_cells_indices_sen_layer), axis=1
            )

            # split into modules and cells
            active_grid_cell_indices_separated = np.split(
                active_grid_cell_indices, 2, axis=1
            )

            # form dendritic segments
            self.dendritic_segments[
                active_grid_cell_indices_separated[0].flatten(),
                active_grid_cell_indices_separated[1].flatten(),
                column_indices,
                cell_indices,
            ] = 1

        phases = self.active_phases.copy()
        self.reset_phases()

        return phases

    def infer(self):
        pass
