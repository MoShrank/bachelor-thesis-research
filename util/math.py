import numpy as np


def get_random_pool_indices(data_size: np.ndarray, pool_size: np.ndarray):
    size = data_size - pool_size
    total = np.prod(size)
    center = np.random.randint(total)

    center_coords = np.unravel_index(center, size)

    # get all points around center
    pool_indices = np.indices(pool_size).reshape((len(pool_size), -1)).T
    pool_indices += center_coords

    # split array into x and y coordinates
    x, y = np.split(pool_indices, 2, axis=1)

    return center_coords, x, y
