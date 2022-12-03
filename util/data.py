from typing import List, Tuple, cast

import numpy as np
import tensorflow as tf

from algorithms import SpatialPooler


def load_mnist() -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    return (x_train, y_train), (x_test, y_test)


def encode_data(*data: np.ndarray) -> Tuple[np.ndarray, ...] | np.ndarray:
    encoded_data: List[np.ndarray] = []

    for d in data:
        encoded = np.copy(d)
        encoded[encoded > 0] = 1
        encoded_data.append(encoded)

    if len(encoded_data) == 1:
        return encoded_data[0]
    else:
        return tuple(encoded_data)


def sample_class(
    x: np.ndarray, y: np.ndarray, class_value: int, no_samples: int, random: bool
) -> np.ndarray:
    """
    sample a specific class from a dataset
    """
    assert x.shape[0] == y.shape[0], "x and y must have same number of samples"
    assert no_samples <= x.shape[0], "no_samples must be smaller than x.shape[0]"

    # get indices of class
    class_indices = np.where(y == class_value)[0]

    # sample indices
    if random:
        sample_indices = np.random.choice(class_indices, no_samples, replace=False)
    else:
        sample_indices = class_indices[:no_samples]

    # get samples
    samples = x[sample_indices, :]

    return samples


def get_sdrs(sp: SpatialPooler, x: np.ndarray) -> np.ndarray:
    sdrs = np.zeros((x.shape[0], sp.number_of_columns), dtype=int)

    for idx, sample in enumerate(x):
        active_columns = sp.compute(sample, learn=False)
        sdr = sp.top_columns_to_sdr(active_columns)
        sdrs[idx] = sdr

    return sdrs


def get_sp_sdr_test_set(
    sp: SpatialPooler, x: np.ndarray, y: np.ndarray, class_value: int, random: bool
):
    """
    get a test set for the spatial pooler consisting of five references and one sample
    """
    sdrs = sdrs = np.zeros((sp.number_of_inputs, sp.number_of_columns))

    no_samples = 6

    samples = sample_class(x, y, class_value, no_samples, random)
    encoded_samples = encode_data(samples)
    encoded_samples = cast(np.ndarray, encoded_samples)

    sdrs = get_sdrs(sp, encoded_samples)

    return sdrs[:5], sdrs[5]


def add_noise(img: np.ndarray, k: float) -> np.ndarray:
    """
    Randomly flips k percent of on bits in an image to off and vice versa.
    """

    assert 1 >= k > 0, "k must be in range (0, 1]"

    flipped_img = np.copy(img.flatten())
    no_on_bits = (flipped_img == 1).sum()

    no_flip_bits = int(no_on_bits * k)

    # get indices of on and off bits
    on_indices = np.where(flipped_img == 1)[0]
    off_indices = np.where(flipped_img == 0)[0]

    # get indices of bits to flip
    bit_indices_flip_off = np.random.choice(on_indices, no_flip_bits, replace=False)
    bit_indices_flip_on = np.random.choice(off_indices, no_flip_bits, replace=False)

    # flip bits
    flipped_img[bit_indices_flip_off] = 0
    flipped_img[bit_indices_flip_on] = 1

    return flipped_img.reshape(img.shape)
