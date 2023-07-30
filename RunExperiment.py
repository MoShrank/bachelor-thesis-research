import json
import os
from datetime import datetime

import numpy as np
from tqdm import tqdm

from algorithms import SpatialPooler
from util.data import encode_data, load_mnist

NO_EPOCHS = 20
COLUMN_DIM = (1024,)
BASE_PATH = os.path.join("experiments", "spatial_pooler")
description = """
Test of SP over several epochs with high boost strength and no topology.
"""
settings_path = os.path.join(BASE_PATH, "settings.json")


def load_json(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data


def save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def update_json(new_data, path):
    with open(path, "r") as f:
        data = json.load(f)

    updated_data = {**data, **new_data}

    save_json(updated_data, path)


def run_single_config(config, epochs, x_train):
    sp = SpatialPooler(
        **config,
    )

    for epoch in tqdm(range(NO_EPOCHS)):
        for x in tqdm(x_train, leave=False):
            x = x.flatten()
            sp.compute(x, learn=True)

        sp_path = os.path.join(BASE_PATH, f"sp_epoch_{epoch + 1}.pkl")
        sp.save_state(sp_path)

        update_json({"cur_epoch": epoch}, settings_path)


def main():
    settings = load_json(settings_path)
    configs = settings["configs"]

    (x_train, y_train), (x_test, y_test) = load_mnist()
    x_train, x_test = encode_data(x_train, x_test)

    # save data
    np.save(os.path.join(BASE_PATH, "x_train.npy"), x_train)
    np.save(os.path.join(BASE_PATH, "y_train.npy"), y_train)
    np.save(os.path.join(BASE_PATH, "x_test.npy"), x_test)
    np.save(os.path.join(BASE_PATH, "y_test.npy"), y_test)

    input_dimension = int(np.prod(x_train[0].shape))

    for config in configs:
        config["input_dimension"] = input_dimension
        run_single_config(config)

    settings = {
        "epochs": NO_EPOCHS,
        "cur_epoch": 0,
        "description": description,
        "sp_arguments": {**sp_args},
    }

    save_json(settings, settings_path)

    sp_path = os.path.join(BASE_PATH, f"sp_epoch_0.pkl")
    sp.save_state(sp_path)

    for epoch in tqdm(range(NO_EPOCHS)):
        for x in tqdm(x_train, leave=False):
            x = x.flatten()
            sp.compute(x, learn=True)

        sp_path = os.path.join(BASE_PATH, f"sp_epoch_{epoch + 1}.pkl")
        sp.save_state(sp_path)

        update_json({"cur_epoch": epoch}, settings_path)


if __name__ == "__main__":
    main()
