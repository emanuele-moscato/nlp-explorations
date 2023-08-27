import tensorflow as tf
from tensorflow.keras.layers import Add, Layer, LayerNormalization, MultiHeadAttention

class BaseAttention(Layer):
    """
    A subclass of Keras' `Layer` representing the base attention layer. All
    other attention layers will be subclasses of this one.
    """
    def __init__(self, **kwargs):
        # Run the parent class' constructor.
        super().__init__()

        # The layer has other Keras layers as attributes. The kwargs to the
        # constructor are used to initialized the multi-head attention layer
        # only.
        self.mha = MultiHeadAttention(**kwargs)
        self.layernorm = LayerNormalization()
        self.add = Add()


class CrossAttention(BaseAttention):
    """
    Layer object implementing cross-attention, in particular multi-head
    attention is used, following by a skip connection and a layer normalization
    operation.

    In cross-attention, we start from an a query and in input matrix and we
    project the latter into a key and a value matrix. Multi-head attention then
    works the following way:
        1. Attention (similarity) scores are computed between the query and the
           key via dot products.
        2. Attention weights are computed applying the softmax function to the
           attention scores.
        3. Output is computed by matri multiplication between the matrix of
           attention weights and the value matrix.
    """
    def call(self, x, context):
        """
        Parameters
        ----------
        x : tf.Tensor
            The query in the attention mechanism.
        context : tf.Tensor
            The input matrix in the attention mechanism, from which the key and
            value tensors are built.

        Returns
        -------
        x : tf.Tensor
            The modified query, after undergoing the skip connection mechanism
            to include the output of the MHA layer and layer normalization.
        """
        # Apply multi-head attention.
        (attentionOutputs, attentionScores) = self.mha(
            query=x,
            key=context,
            value=context,
            return_attention_scores=True
        )

        # Store the (last) attention scores for later visualization.
        self.lastAttentionScores = attentionScores

        # Skip connection: add the input and the output. `x` will now contain
        # information on the output of the MHA layer: it will be outputted in
        # the end.
        # Note: this simply adds up the tensors in the input list (which must
        #       have the same shape), component by component.
        x = self.add([x, attentionOutputs])

        # Applies layer normalization to the output.
        x = self.layernorm(x)

        return x


class GlobalSelfAttention(BaseAttention):
    """
    Layer implementing self-attention: we start from the input matrix only and
    the query, key and value matrices are all built from that.
    """
    def call(self, x):
        """
        Parameters
        ----------
        x : tf.Tensor
            The input matrix.
        """
        # Apply self multi-head attention to the input matrix.
        attentionOutputs = self.mha(
            query=x,
            key=x,
            value=x
        )

        # Skip connection: add the input and the output.
        x = self.add([x, attentionOutputs])

        # Applies layer normalization to the output.
        x = self.layernorm(x)

        return x


class CausalSelfAttention(BaseAttention):
    """
    Layer implementing self-attention with the addition of causal masking
    (look-ahead mask). This is the masked multi-head attention (MMHA) layer
    used in the decoder, enforcing that the decoder doesn't see tokens that
    are later in the sequence than the one it's trying to predict.
    """
    def call(self, x):
        """
        """
        attentionOutputs = self.mha(
            query=x,
            key=x,
            value=x,
            use_causal_mask=True
        )

        # Skip connection: add the input and the output.
        x = self.add([x, attentionOutputs])

        # Applies layer normalization to the output.
        x = self.layernorm(x)

        return x
