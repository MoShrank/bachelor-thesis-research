from typing import List

import matplotlib.pyplot as plt
import numpy as np


def plot_image(arr: np.ndarray):
    """
    plot image
    """
    assert arr.ndim == 2, f"input must be 2d array. Got {arr.ndim}d array"

    plt.imshow(arr)
    plt.axis("off")
    plt.show()


def plot_images(arr: np.ndarray):
    """
    plot images
    """
    assert arr.ndim == 3, f"input must be 3d array. Got {arr.ndim}d array"
    assert (
        arr.shape[1] == arr.shape[2]
    ), f"input must be square. Got {arr.shape[1]}x{arr.shape[2]}"

    num_images = arr.shape[0]

    _, axes = plt.subplots(1, num_images, figsize=(num_images * 2.5, 2))

    for idx in range(num_images):
        axes[idx].imshow(arr[idx, :, :])
        axes[idx].axis("off")

    plt.show()


def scatter_plot(xs: List, ys: List):
    # draw quadartic scatter plot with xlim and ylim and grid with grid lines per 1 unit
    plt.figure(figsize=(7, 7))
    plt.scatter(xs, ys)
    plt.xlim(0, 28)
    plt.ylim(0, 28)
    plt.grid(b=True, which="major", color="#666666", linestyle="-")
    plt.minorticks_on()
    plt.grid(b=True, which="minor", color="#999999", linestyle="-", alpha=0.2)
    plt.show()
