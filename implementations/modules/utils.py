import tensorflow as tf


def scaled_dot_product_attention(query, key, value):
    """
    Implements the scaled dot product attention operation.
    """
    # Dimension of the key vectors.
    dim_k = key.shape[-1]

    # Compute the attention scores.
    scores = tf.matmul(
        query,
        # Leaving the batch shape as the first dimension, it's ignored
        # in the matrix multiplication.
        tf.transpose(key, perm=(0, 2, 1))
    )

    # Compute the attention weights.
    weights = tf.math.softmax(
        scores / tf.sqrt(tf.cast(dim_k, tf.float32)),
        axis=-1
    )

    # Return a linear combination of the value vectors
    # with weights equal to the attention weights.
    return tf.matmul(weights, value)
