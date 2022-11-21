import numpy as np


def encode_img(x: np.ndarray) -> np.ndarray:
    encoded_x = np.copy(x)

    encoded_x[encoded_x > 0] = 1
    return encoded_x


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
