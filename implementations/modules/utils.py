import numpy as np
import tensorflow as tf


def scaled_dot_product_attention(query, key, value, return_weights=False):
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

    if return_weights:
        return weights

    # Return a linear combination of the value vectors
    # with weights equal to the attention weights.
    return tf.matmul(weights, value)


def masked_scaled_dot_product_attention(
        query,
        key,
        value,
        return_weights=False
    ):
    """
    Implements the scaled dot product attention operation with masking: the
    entries in the upper triangle (excluding the diagonal) of the attention
    scores matrix are set to -inf so that when softmax is applied the resulting
    attention weights are 0 and the attention with the corresponding tokens in
    the sequence is forced to be null (tokens are ignored).
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

    # Compute the mask.
    seq_len = key.shape[-2]

    mask = 1. - tf.linalg.band_part(
        tf.ones(shape=(seq_len, seq_len)),
        num_lower=-1,
        num_upper=0
    )

    # Apply the mask.
    scores = np.where(mask == 1, - mask * np.infty, scores)

    # Compute the attention weights.
    weights = tf.math.softmax(
        scores / tf.sqrt(tf.cast(dim_k, tf.float32)),
        axis=-1
    )

    if return_weights:
        return weights

    # Return a linear combination of the value vectors
    # with weights equal to the attention weights.
    return tf.matmul(weights, value)
