import tensorflow as tf
from tensorflow.keras.layers import Dropout, Layer
# Use these imports when working from a script.
from .attention import CausalSelfAttention, CrossAttention
from .feed_forward import FeedForward
from .positional_encoding import PositionalEmbedding
# Use these imports when working from a notebook in the notebooks/ directory.
# from attention import CausalSelfAttention, CrossAttention
# from feed_forward import FeedForward
# from positional_encoding import PositionalEmbedding


class DecoderLayer(Layer):
    """
    Custom layer implementing one decoder layer. The full decoder architecture
    is made up of a stack (a sequence) of these decoder layers. Each decoder
    layer in turns is a sequence of sub-layers:
        1. A causal self-attention layer, which is the masked one accepting
           the inputs to the decoder.
        2. A cross-attention layer that integrates the output of the previous
           layer with the output of the full encoder.
        3. A feed-forward layer.
    This whole structure is repeated in a sequence numLayers times to give the
    full decoder architecture: this class does NOT represent the full decoder.
    """
    def __init__(self, dModel, numHeads, dff, dropOutRate=0.1, **kwargs):
        """
        Parameters
        ----------
        dModel : int
            Dimension of the model (i.e. of the representation of the tokens).
        numHeads : int
            Number of heads in the MHA layers.
        dff : int
            Dimension of the intermediate layer in the feed-forward NN part.
        dropOutRate : float
            Rate of dropout for the dropout regularization layer.
        """
        # First, call the constructor of the parent class (Layer).
        super().__init__(**kwargs)

        # Initialize causal self-attention layer (i.e. with causal masking:
        # this is the masked multi-head attention layer).
        self.causalSelfAttention = CausalSelfAttention(
            num_heads=numHeads,
            key_dim=dModel // numHeads,
            dropout=dropOutRate
        )

        # Initialize a cross-attention layer: this is the second MHA layer in
        # the decoder.
        self.crossAttention = CrossAttention(
            num_heads=numHeads,
            key_dim=dModel // numHeads,
            dropout=dropOutRate
        )

        # Initialize the feed-forward neural network that's part of the
        # decoder.
        self.ffn = FeedForward(
            dff=dff,
            dModel=dModel,
            dropoutRate=dropOutRate
        )

    def call(self, x, context):
        x = self.causalSelfAttention(x)
        x = self.crossAttention(x=x, context=context)

        # Get the attention scores from the MHA layer (the non-masked one).
        self.lastAttentionScores = self.crossAttention.lastAttentionScores

        x = self.ffn(x)

        return x


class Decoder(Layer):
    """
    Custom layer implementing the full decoder architecture. The layer is
    composed of a sequence of sub-layers:
        1. A PositionalEmbedding layer that computes an embedding with
           positional encoding of the vectorized target sentences (after
           masking).
        2. A dropout regularization acting on the output of the above layer.
        3. A stack of numLayers DecoderLayer objects working sequentially.
    The attention scores considered are the ones coming from the last layer in
    the stack.
    """
    def __init__(
        self,
        numLayers,
        dModel,
        numHeads,
        targetVocabSize,
        maximumPositionEncoding,
        dff,
        dropOutRate=0.1,
        **kwargs
    ):
        """
        Parameters
        ----------
        numLayers : int
            The number of DecoderLayer objects in the full decoder
            architecture.
        dModel : int
            Dimension of the model (i.e. of the representation of the tokens).
        numHeads : int
            Number of heads in the MHA layers.
        targetVocabSize : int
            Size of the target language vocabulary.
        maximumPositionEncoding : int
            Maximum length of a sentence (sequence of tokens).
        dff : int
            Dimension of the intermediate layer in the feed-forward NN part.
        dropOutRate : float
            Rate of dropout for the dropout regularization layer.
        """
        # Execute the constructor of the parent class (Layer).
        super().__init__(**kwargs)

        self.dModel = dModel
        self.numLayers = numLayers

        # Initialize a positional embedding layer. The input to the decoder
        # is just given by the vectorized target sentences (masked).
        self.positionalEmbedding = PositionalEmbedding(
            vocabSize=targetVocabSize,
            dModel=dModel,
            maximumPositionEncoding=maximumPositionEncoding
        )

        # Stack (sequence) of DecoderLayer objects. Data will pass through
        # the sequence from the first to the last layer.
        self.decoderLayers = [
            DecoderLayer(
                dModel=dModel,
                dff=dff,
                numHeads=numHeads,
                dropOutRate=dropOutRate
            )
            for _ in range(numLayers)
        ]

        # Initialize dropout regularization layer.
        self.dropout = Dropout(rate=dropOutRate)

    def call(self, x, context):
        # First, the vectorized target tokens are passed through positional
        # embedding.
        x = self.positionalEmbedding(x)

        # Dropout regularization.
        x = self.dropout(x)

        for decoderLayer in self.decoderLayers:
            x = decoderLayer(x=x, context=context)

        # Get the last attention score from the last decoder layer used (the
        # last in the sequence).
        self.lastAttentionScores = self.decoderLayers[-1].lastAttentionScores

        return x
