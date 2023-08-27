import tensorflow as tf
from tensorflow.keras.layers import Dropout, Layer
# Note: relative imports for running with scripts.
# from .attention import GlobalSelfAttention
# from .feed_forward import FeedForward
# from .positional_encoding import PositionalEmbedding

# Imports for running via a notebook in the notebooks/ directory.
from attention import GlobalSelfAttention
from feed_forward import FeedForward
from positional_encoding import PositionalEmbedding


class EncoderLayer(Layer):
    """
    Class implementing the one encoder layer of a transformer model. The
    encoder layer is composed of a sequence two sublayers:
        1. An attention layer, as defined in the attention module.
        2. A feed-forward layer, as defined in the feed_forward module.
    This is NOT the full encoder, but just one of the components of the stack
    of EncoderLayer layers it contains (see the Encoder class below for the
    implementation of the full encoder architecture).
    """
    def __init__(self, dModel, numHeads, dff, dropOutRate=0.1, **kwargs):
        """
        Initializes the sub-layers.

        Parameters
        ----------
        dModel : int
            Dimension of the transformer module.
        numHeads : int
            Number of heads of the multi-head attention module.
        dff : int
            Size of the intermediate dimension of the feed-forward neural
            network.
        dropOutRate : float
            Rate of dropout in the corresponding layer.
        """
        # Call the parent class (Layer) constructor method.
        super().__init__()

        # Define the attention layer.
        # Note: the numHeads option is passed as a kwarg to the constructor of
        #       GlobalSelfAttention, which is in fact the constructor of
        #       the BaseAttention class it is a subclass of. numHeads ends up
        #       being passed to the constructor of Keras' MultiHeadAttention
        #       layer. That's how the number of heads is defined.
        self.globalSelfAttention = GlobalSelfAttention(
            num_heads=numHeads,
            key_dim=dModel // numHeads,
            dropout=dropOutRate
        )

        # Initialize the feed-forward layer.
        self.ffn = FeedForward(dff=dff, dModel=dModel, dropoutRate=dropOutRate)

    def call(self, x):
        """
        Computes the output applying first the attention and then the
        feed-forward layer.
        """
        x = self.globalSelfAttention(x)

        x = self.ffn(x)

        return x


class Encoder(Layer):
    """
    Class implementing the full encoder architecture. The full encoder is given
    by the following sequence of operations:
        1. A PositionalEmbedding layer, accepting the vectorized text as its
           inputs.
        2. A Dropout layer for regularization.
        3. A sequence of numLayers EncoderLayer layers, acting in a chain one
           after the other.
    """
    def __init__(
        self,
        numLayers,
        dModel,
        numHeads,
        sourceVocabSize,
        maximumPositionEncoding,
        dff,
        dropOutRate=0.1,
        **kwargs
    ):
        """
        Parameters
        ----------
        numLayers : int
            Number of EncoderLayer layers in the encoder.
        dModel : int
            Dimension of the transformer module.
        numHeads : int
            Number of heads of the multi-head attention module.
        sourceVocabSize :
            Size of the source vocabulary.
        maximumPositionEncoding :
            Maximum number of tokens in a sentence from the source dataset.
        dff : int
            Size of the intermediate dimension of the feed-forward neural
            network.
        dropOutRate : float
            Rate of dropout in the corresponding layer.
        """
        # Call the parent class' (Layer) constructor.
        super().__init__(**kwargs)

        self.dModel = dModel
        self.numLayers = numLayers

        # Initialize a PositionalEmbedding layer.
        self.positionalEmbedding = PositionalEmbedding(
            vocabSize=sourceVocabSize,
            dModel=dModel,
            maximumPositionEncoding=maximumPositionEncoding
        )

        # Initialize a stack of numLayers EncoderLayer layers.
        self.encoderLayers = [
            EncoderLayer(
                dModel=dModel,
                dff=dff,
                numHeads=numHeads,
                dropOutRate=dropOutRate
            )
            for _ in range(numLayers)
        ]

        # Initialize a dropout layer.
        self.dropout = Dropout(rate=dropOutRate)

    def call(self, x):
        x = self.positionalEmbedding(x)

        x = self.dropout(x)

        # Input is passed through the stack of EncoderLayer layers
        # sequentially.
        for encoderLayer in self.encoderLayers:
            x = encoderLayer(x=x)

        return x
