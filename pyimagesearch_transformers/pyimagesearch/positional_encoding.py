import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Embedding, Layer


def positional_encoding(length, depth):
    """
    Computes (sinusoidal) positional encoding.

    Parameters
    ----------
    length : int
        The number of positions to encode (i.e. maximum number of tokens in a
        sequence).
    depth : int
        The length of each encoding: each position will correspond to an array
        of length `depth`.
    """
    # Consider half the depth (one half for the sines and on half for the
    # cosines).
    depth = depth / 2.

    # Positions array (shape: (length, 1)).
    positions = np.arange(length)[:, np.newaxis]

    # Depths array (shape: (1, depth)).
    depths = np.arange(depth)[np.newaxis, :] / depth

    # Build the arguments of the sines and cosines.
    angleRates = 1. / (1e4 ** depths)
    angleRads = positions * angleRates

    # Build the positional encodings.
    posEncoding = np.concatenate(
        [np.sin(angleRads), np.cos(angleRads)],
        axis=-1
    )

    return tf.cast(posEncoding, dtype=tf.float32)


class PositionalEmbedding(Layer):
    """
    Layer sublcass performing word embedding, including positional encoding.
    Sequence of operations, schematically:
        1. A Keras `Embedding` layer computes a higher-dimensional embedding
           for the vectorized text. The input tensor to the layer has shape
           (n_sentences, max_length) and since the layer computes an embedding
           for each token the output has shape (n_sentences, max_length,
           dModel), where `dModel` is the dimension of the model, i.e. the
           dimension of each token's representation (embedding).
        2. The positional encoding of each token is computed. This has the same
           shape as the output of the embedding layer.
        3. The embedding and the positional encoding are added up to integrate
           the information they carry.
    """
    def __init__(self, vocabSize, dModel, maximumPositionEncoding, **kwargs):
        """
        Parameters
        ----------
        vocabSize : int
            Vocabulary size of the target or source dataset.
        dModel : int
            Dimension of the transformer model.
        maximumPositionEncoding : int
            Maximum length of a sentence in the dataset.
        """
        # Call the parent class' constructor first.
        super().__init__(**kwargs)

        # An embedding layer. Maps tensors with positive integer components
        # into output tensor of the given size. The expected input is exectly
        # of the format outputted by tf.keras.layers.TextVectorization.
        self.embedding = Embedding(
            input_dim=vocabSize,
            output_dim=dModel,
            mask_zero=True
        )

        # Compute the positional encoding for every possible token position in
        # a sequence (up to the maximal sequence length considered).
        self.posEncoding = positional_encoding(
            length=maximumPositionEncoding,
            depth=dModel
        )

        self.dModel = dModel

    def compute_mask(self, *args, **kwargs):
        """
        Returns the padding mask for the input.
        """
        return self.embedding.compute_mask(*args, **kwargs)

    def call(self, x):
        """
        Parameters
        ----------
        x : tf.Tensor
            Input sequence.
        """
        # Compute the sequence lenght from the shape of the input tensor.
        seqLen = tf.shape(x)[1]

        # Compute the embedding of the input tensor.
        x = self.embedding(x)

        # Rescale the embeddings multiplying by the square root of the model's
        # dimension.
        x *= tf.math.sqrt(tf.cast(self.dModel, tf.float32))

        # Add the positional encoding and the scaled embeddings.
        # Note: adding a new empty axis to the left is needed to have tensors
        #       that can be broadcast one into the other when added up.
        # Note: we select the encoding up to `seqLen` right because that's the
        #       number of tokens we have (shape of the embedding).
        x += self.posEncoding[tf.newaxis, :seqLen, :]

        return x
