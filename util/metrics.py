import numpy as np


def calculate_overlap(input_one: np.ndarray, input_two: np.ndarray) -> np.ndarray:
    """
    calculate overlap of bits between two vectors
    """
    overlap = np.sum(
        np.logical_and(input_one.astype(bool), input_two.astype(bool)).astype(int)
    )
    return overlap


def get_similiraty(
    reference_vectors: np.ndarray, input_vector: np.ndarray
) -> np.ndarray:
    """
    calculate similiraty based on a set of reference vectors of a specific class
    """

    similiraty = np.zeros(reference_vectors.shape[0], dtype=float)
    for ref_idx in range(reference_vectors.shape[0]):
        similiraty[ref_idx] = calculate_overlap(
            reference_vectors[ref_idx, :], input_vector
        )

    return similiraty


def get_mean_similiraty(
    reference_vectors: np.ndarray, input_vector: np.ndarray
) -> np.ndarray:
    """
    calculate mean similiraty based on a set of reference vectors of a specific class
    """
    similiraty = get_similiraty(reference_vectors, input_vector)
    mean = np.mean(similiraty)
    return mean


def similiraty_to_percent(
    similiraty: float, sparsity: float, total_columns: int
) -> float:
    """
    convert mean similiraty to percent
    """
    assert 1 >= sparsity > 0, "sparsity must be in range (0, 1]"
    percent = similiraty / (total_columns * sparsity)
    return percent
