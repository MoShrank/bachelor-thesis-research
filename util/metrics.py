import numpy as np
from tqdm import tqdm

from SpatialPooler import SpatialPooler
from util.data import get_sp_sdr_test_set


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
) -> float:
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


def sp_stability(sp: SpatialPooler, x: np.ndarray, y: np.ndarray, random: bool):
    classes = np.unique(y)

    no_classes = classes.shape[0]

    # create 2D matrix for measuring stability for each class against each other class
    results = np.zeros((no_classes, no_classes), dtype=float)

    for ref_idx, class_value in tqdm(enumerate(classes)):
        reference_vectors, _ = get_sp_sdr_test_set(sp, x, y, class_value, random)

        for test_idx, test_class_value in enumerate(classes):
            # get reference vectors
            _, test_input = get_sp_sdr_test_set(sp, x, y, test_class_value, random)

            # calculate similiraty and convert to %
            mean_similiraty = get_mean_similiraty(reference_vectors, test_input)
            percent = similiraty_to_percent(
                # TODO sparsity and column dim should be inferred from sp
                mean_similiraty,
                0.2,
                46 * 46,
            )

            # store result
            results[ref_idx, test_idx] = percent

    return results
