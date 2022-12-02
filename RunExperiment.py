import os
from datetime import datetime

from tqdm import tqdm

from SpatialPooler import SpatialPooler
from util.data import encode_data, load_mnist

NO_EPOCHS = 10
COLUMN_DIM = (45, 45)


def main():

    now = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
    path = os.path.join("data", now)
    os.mkdir(path)

    (x_train, y_train), (x_test, y_test) = load_mnist()
    x_train, x_test = encode_data(x_train, x_test)

    input_dimension = x_train[0].shape

    sp = SpatialPooler(
        input_dimension=input_dimension,
        column_dimension=COLUMN_DIM,
        connection_sparsity=0.7,
        permanence_threshold=0.5,
        stimulus_threshold=10,
        permanence_increment=0.1,
        permanence_decrement=0.02,
        column_sparsity=0.02,
        potential_pool_radius=8,
        boost_strength=10,
    )

    for _ in tqdm(range(NO_EPOCHS)):
        for x in tqdm(x_train, leave=False):
            sp.compute(x, learn=True)

        sp_path = os.path.join(path, "sp.pkl")
        sp.save_state(sp_path)


if __name__ == "__main__":
    main()
