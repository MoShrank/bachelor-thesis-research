import numpy as np

from SpatialPooler import SpatialPooler


def init_spatial_pooler(stimulus_treshold: int) -> SpatialPooler:
    return SpatialPooler(
        input_dimension=(20,),
        column_dimension=(40,),
        connection_sparsity=0.2,
        permanence_threshold=0.5,
        stimulus_threshold=stimulus_treshold,
        permanence_increment=0.1,
        permanence_decrement=0.01,
        column_sparsity=0.2,
    )


def test_get_winning_columns():
    sp = init_spatial_pooler(1)
    overlap = np.array([2, 2, 2, 2, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    # indices of columns with overlap > stimulus_thresholds
    # and max no_active_columns defined by column sparsity
    exp_winning_columns = np.array([0, 1, 2, 3, 4, 5, 6])

    winning_columns = sp.get_winning_columns(overlap)
    print("winning columns: ", winning_columns)

    np.testing.assert_array_equal(winning_columns, exp_winning_columns)
