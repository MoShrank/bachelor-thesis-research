import json
import os
from datetime import datetime

import numpy as np
from tqdm import tqdm

from algorithms import SpatialPooler
from util.data import encode_data, load_mnist

NO_EPOCHS = 20
COLUMN_DIM = (1024,)
description = """
Test of SP over several epochs with high boost strength and no topology.
"""


def save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def update_json(new_data, path):
    with open(path, "r") as f:
        data = json.load(f)

    updated_data = {**data, **new_data}

    save_json(updated_data, path)


def main():

    now = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
    path = os.path.join("data", now)
    os.mkdir(path)

    settings_path = os.path.join(path, "settings.json")

    (x_train, y_train), (x_test, y_test) = load_mnist()
    x_train, x_test = encode_data(x_train, x_test)

    # save data
    np.save(os.path.join(path, "x_train.npy"), x_train)
    np.save(os.path.join(path, "y_train.npy"), y_train)
    np.save(os.path.join(path, "x_test.npy"), x_test)
    np.save(os.path.join(path, "y_test.npy"), y_test)

    input_dimension = x_train[0].shape

    sp_args = {
        "input_dimension": (int(np.prod(input_dimension)),),
        "column_dimension": COLUMN_DIM,
        "connection_sparsity": 0.75,
        "permanence_threshold": 0.5,
        "stimulus_threshold": 1,
        "permanence_increment": 0.1,
        "permanence_decrement": 0.02,
        "column_sparsity": 0.02,
        "potential_pool_radius": 2048,
        "boost_strength": 50,
    }

    sp = SpatialPooler(
        **sp_args,
    )

    settings = {
        "epochs": NO_EPOCHS,
        "cur_epoch": 0,
        "description": description,
        "sp_arguments": {**sp_args},
    }

    save_json(settings, settings_path)

    sp_path = os.path.join(path, f"sp_epoch_0.pkl")
    sp.save_state(sp_path)

    for epoch in tqdm(range(NO_EPOCHS)):
        for x in tqdm(x_train, leave=False):
            x = x.flatten()
            sp.compute(x, learn=True)

        sp_path = os.path.join(path, f"sp_epoch_{epoch + 1}.pkl")
        sp.save_state(sp_path)

        update_json({"cur_epoch": epoch}, settings_path)


if __name__ == "__main__":
    main()
