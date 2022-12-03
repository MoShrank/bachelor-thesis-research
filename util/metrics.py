from itertools import product

import numpy as np
from scipy.integrate import quad
from tqdm import tqdm

from algorithms import SpatialPooler
from util.data import add_noise, get_sdrs


def calculate_entropy(sp: SpatialPooler, x: np.ndarray) -> float:
    # 1.
    # activation frequence for each minicolumn
    # by summing up activity for each input per column

    # 2.
    # binary entropy function on column activation frequency
    # where if activation frequency is 0 or 1, entropy is 0

    # 3.
    # sum up entropy for each column and calculate mean

    column_activity = np.zeros((x.shape[0], sp.number_of_columns), dtype="int32")

    for idx, input in enumerate(tqdm(x)):
        winning_columns = sp.compute(input, learn=False)
        column_activity[idx, winning_columns] = 1

    column_activity_frequency = np.sum(column_activity, axis=0) / x.shape[0]

    entropy = np.sum(
        [
            -p * np.log2(p) - (1 - p) * np.log2(1 - p)
            for p in column_activity_frequency
            if p != 0 and p != 1
        ]
    )

    mean = entropy / sp.number_of_columns

    return mean


def calc_noise_robustness(sp: SpatialPooler, inputs: np.ndarray) -> int:
    sdrs = np.zeros((inputs.shape[0], sp.number_of_columns), dtype=int)

    for idx, input_vector in enumerate(inputs):
        winning_columns = sp.compute(input_vector, learn=False)
        sdr = sp.top_columns_to_sdr(winning_columns)

        sdrs[idx, :] = sdr

    integrals = np.zeros(inputs.shape[0], dtype=float)

    def func(x, input, sdr):
        no_on_bits = np.sum(input)
        noisy_input = add_noise(input, x)
        noisy_winning_columns = sp.compute(noisy_input, learn=False)
        noisy_sdr = sp.top_columns_to_sdr(noisy_winning_columns)

        overlap = calculate_overlap(sdr, noisy_sdr)

        shared_bits = overlap / no_on_bits

        return shared_bits

    for idx, input, sdr in tqdm(zip(range(inputs.shape[0]), inputs, sdrs)):
        integrant = lambda x: func(x, input, sdr)

        integral, _ = quad(integrant, 0, 1)
        integrals[idx] = integral

    mean = np.mean(integrals)
    return mean


def calculate_overlap(input_one: np.ndarray, input_two: np.ndarray) -> np.ndarray:
    """
    Calculate overlap of bits between two vectors. The overlap is defined as
    the number of overlapping on bits.
    """
    overlap = np.sum(
        np.logical_and(input_one.astype(bool), input_two.astype(bool)).astype(int)
    )
    return overlap


def get_stability(input_one: np.ndarray, input_two: np.ndarray) -> float:
    """
    Calculate stability of two vectors. The stability is defined as:
    stability = L0_norm(overlap(input_one, input_two)) / L0_norm(input_one)
    """
    overlap = calculate_overlap(input_one, input_two)
    stability = overlap / np.sum(input_one)
    return stability


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


def get_sp_stability(sp: SpatialPooler, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Creates a 2D array of the mean similiraty of each class to all other classes.
    """
    classes = np.unique(y)

    no_classes = classes.shape[0]

    # create 2D matrix for measuring stability for each class against each other class
    results = np.zeros((no_classes, no_classes), dtype=float)

    class_data = {key: get_sdrs(sp, x[y == key][:10]) for key in classes}
    for ref_class_val, ref_class_sdrs in tqdm(class_data.items()):
        for class_val, ref_call_sdrs in class_data.items():
            combinations = product(ref_class_sdrs, ref_call_sdrs)

            stabilities = []
            for combination in combinations:
                stability = get_stability(combination[0], combination[1])
                stabilities.append(stability)
            mean = np.mean(stabilities)
            results[ref_class_val, class_val] = mean
    return results
