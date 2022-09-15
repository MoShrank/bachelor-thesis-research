import tensorflow as tf

from layers.kWTA import KWTA


def test_kwta():
    input = tf.constant([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]], dtype=tf.float32)
    sparsity = 0.4

    expexted_output = tf.constant([[0, 0, 0, 0, 5, 6, 7, 8, 9, 10]], dtype=tf.float32)

    kwta = KWTA(sparsity=sparsity)

    output = kwta(input)

    assert tf.reduce_all(tf.equal(output, expexted_output))
