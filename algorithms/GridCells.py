import numpy as np
import math


class GridCells:
    def __init__(self, no_modules: int, no_cells: int, orientation: float):

        self.no_modules = no_modules
        self.no_cells = no_cells

        self.module_dim = int(np.sqrt(no_cells))

        self.translation_matrix = np.array([
            [math.cos(orientation), -math.sin(orientation)],
            [math.sin(orientation), math.cos(orientation)]
        ])
    
    def update_location(self, module,):