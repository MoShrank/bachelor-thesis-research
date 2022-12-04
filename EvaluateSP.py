import json
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from algorithms import SpatialPooler
from util.metrics import (
    calc_noise_robustness,
    calculate_entropy,
    calculate_sparsity,
    get_sp_stability,
)

SP_RESULTS_DIR = os.path.join("data", "12-04-2022_15-51-52")


def load_json(path):
    with open(path, "r") as f:
        data = json.load(f)

    return data


def main():
    settings_path = os.path.join(SP_RESULTS_DIR, "settings.json")
    settings = load_json(settings_path)

    sp_args = settings["sp_arguments"]

    sp = SpatialPooler(**sp_args)

    x_test = np.load(os.path.join(SP_RESULTS_DIR, "x_test.npy"))
    x_test = x_test.reshape(x_test.shape[0], 784)
    y_test = np.load(os.path.join(SP_RESULTS_DIR, "y_test.npy"))

    results = pd.DataFrame(
        columns=[
            "entropy",
            "activation_freq",
            "sparsity",
            "stability",
            "noise_robustness",
        ]
    )

    for epoch in tqdm(range(settings["epochs"] + 1)):
        sp_path = os.path.join(SP_RESULTS_DIR, f"sp_epoch_{epoch}.pkl")
        sp.load_state(sp_path)

        entropy, activation_freq = calculate_entropy(sp, x_test)
        sparsity = calculate_sparsity(sp, x_test)
        stability = get_sp_stability(sp, x_test, y_test)
        noise_robustness = calc_noise_robustness(sp, x_test[:1])

        results.loc[epoch] = [
            entropy,
            activation_freq,
            sparsity,
            stability,
            noise_robustness,
        ]

    results.to_csv(os.path.join(SP_RESULTS_DIR, "results.csv"))


if __name__ == "__main__":
    main()
