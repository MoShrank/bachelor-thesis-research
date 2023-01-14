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


def scatter_plot(xs: List, ys: List, lim=(0, 28), plot_arrows: bool = False):
    # draw quadartic scatter plot with xlim and ylim and grid with grid lines per 1 unit
    plt.figure(figsize=(7, 7))
    plt.scatter(xs, ys)
    plt.xlim(lim[0], lim[1])
    plt.ylim(lim[0], lim[1])
    plt.grid(b=True, which="major", color="#666666", linestyle="-")
    plt.minorticks_on()
    plt.grid(b=True, which="minor", color="#999999", linestyle="-", alpha=0.2)

    if plot_arrows:

        for i in range(len(xs) - 1):
            plt.arrow(
                xs[i],
                ys[i],
                xs[i + 1] - xs[i],
                ys[i + 1] - ys[i],
                head_width=0.3,
                head_length=0.3,
                length_includes_head=True,
                fc="k",
                ec="k",
            )

        plt.scatter(xs[0], ys[0], c="r")

    plt.show()
