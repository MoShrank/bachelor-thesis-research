import tensorflow as tf


class KWTA(tf.keras.layers.Layer):
    """
    k-Winner-Take-All (k-WTA) layer for tensorflow models.
    It enforces sparsity by letting only the k largest values pass through.
    """
    def __init__(self, sparsity: float):
        """
        :param sparsity: sparsity of the layer (between 0 and 1)
        """
        super(KWTA, self).__init__()
        
        assert 1. >= sparsity >= 0., "sparsity must be a fraction"
        self.sparsity = sparsity
        
    def call(self, inputs):
        batch_size, feature_size = inputs.shape
        
        # calculate number of inactive neurons
        top_k = int(self.sparsity * feature_size)
        
        assert feature_size >= top_k, "the number of active neurons must be smaller or equal than the number of neurons"
        
        # negate top_k to get k lowest values
        lowest_k_indices = tf.math.top_k(-inputs, k=top_k, sorted=False, name=None).indices
        
        # total amount of indices to update
        total_size = batch_size * top_k
        
        # create tensor of indices to update
        batch_indices = tf.reshape(tf.repeat(tf.range(0, batch_size, dtype="int32"), top_k), [total_size, 1])
        lowest_k_indices = tf.reshape(lowest_k_indices, [total_size, 1])
        indices_flattened = tf.concat([batch_indices, lowest_k_indices], axis=1)
        
        updates = tf.zeros(total_size)

        out = tf.tensor_scatter_nd_update(
            inputs, indices_flattened, updates, name="out_tensor"
        )
                
        return out
