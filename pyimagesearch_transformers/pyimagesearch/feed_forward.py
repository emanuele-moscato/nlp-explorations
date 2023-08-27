from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Add, Dense, Dropout, Layer, LayerNormalization)


class FeedForward(Layer):
    """
    Layer implementing the feed-forward neural network used both in a
    transformer's encoder and decoder. The structure is that of a
    fully-connected layer with dropout regularization with the additional
    feature of a skip connection layer followed by layer normalization.
    """
    def __init__(self, dff, dModel, dropoutRate=0.1, **kwargs):
        """
        Parameters
        ----------
        dff : int
            Number of units in the intermediate layer of the FFN.
        dModel : int
            Dimension of the model (size of the token's representation).
        dropoutRate : float
            Rate of dropout for the regularization.
        """
        super().__init__()

        # Initialize a fully-connected sequential model with dropout
        # regularization.
        self.seq = Sequential([
            Dense(units=dff, activation='relu'),
            Dense(units=dModel),
            Dropout(rate=dropoutRate)
        ])

        # Initialize skip connection operation (addition between the input and
        # output tensors, component by component).
        self.add = Add()

        # Initialize the layer normalization operation.
        self.layernorm = LayerNormalization()

    def call(self, x):
        # Add the input x with the output of the sequential model comoputed on
        # x itself (skip connection).
        x = self.add([x, self.seq(x)])

        # Normalize the result with layer normalization.
        x = self.layernorm(x)

        return x
