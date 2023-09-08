import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, LayerNormalization
from utils import (scaled_dot_product_attention,
    masked_scaled_dot_product_attention)
from encoder import FeedForward, Embeddings


class MaskedAttentionHead(Layer):
    """
    Implementation of a masked single-head self-attention layer. This is
    analogous to the non-masked version, but uses the masked scaled dot-product
    attention procedure.
    """
    def __init__(self, embed_dim, head_dim):
        """
        Initializes the layers that project the input to the
        AttentionHead layer onto the corresponding query (q),
        key (k) and value (v) vectors.

        Parameters
        ----------
        embed_dim : int
            Size of the input embeddings. This is established by
            the particular token embedding chosen.
        head_dim : int
            Size of the output of the AttentionHead layer. In
            multi-head attention this will be < embed_dim, and
            the full dimension of the embeddings is recovered
            when the outputs of each head are concatenated back
            together.
        """
        # Execute the parent class' constructor.
        super().__init__()

        # Initialize the query, key and value weighting matrices.
        self.q = Dense(units=head_dim)
        self.k = Dense(units=head_dim)
        self.v = Dense(units=head_dim)

    def call(self, hidden_state):
        """
        Forward pass of the layer. The query, key and value
        vectors are computed applying the q, k and v layers
        to the input.

        Input shape: (batch_shape, seq_len, embed_dim)
        Ouput shape: (batch_shape, seq_len, head_dim)
        """
        attn_outputs = masked_scaled_dot_product_attention(
            self.q(hidden_state),
            self.k(hidden_state),
            self.v(hidden_state)
        )

        return attn_outputs


class MaskedMultiHeadAttention(Layer):
    """
    Implementation of a masked multi-head self-attention layer. This is
    analogous to the non-masked multi-head attention layer, but this time the
    heads are `MaskedAttentionHead` layers.
    """
    def __init__(self, config):
        """
        Initializes a list of masked single-head self-attention layers and
        the final dense (fully-connected) layer.
        """
        super().__init__()

        embed_dim = config.hidden_size
        num_heads = config.num_attention_heads
        head_dim = embed_dim // num_heads

        # Initialize a list of attention heads.
        self.heads = [
            MaskedAttentionHead(embed_dim=embed_dim, head_dim=head_dim)
            for _ in range(num_heads)
        ]

        # Initialize the final dense layer.
        self.output_linear = Dense(units=embed_dim)

    def call(self, hidden_state):
        """
        Forward pass: the input is passed through each head
        independently, then the outputs are concatenated back
        together and passed through the final dense layer.

        Input shape: (batch_shape, seq_len, hidden_dim)
        Output layer: (batch_shape, seq_len, hidden_dim)
        """
        # Pass the input through each head and concatenate
        # the outputs.
        x = tf.concat(
            [h(hidden_state) for h in self.heads],
            axis=-1
        )

        # Pass the concatenated outputs through the final
        # linear layer.
        x = self.output_linear(x)

        return x


class EncoderDecoderAttentionHead(Layer):
    """
    Implementation of an encoder-decoder single-head attention layer. In the
    encoder-decoder version, attention scores are computed using the hidden
    states coming from the previous layers in the decoder and the key and
    value vectors coming from the encoder. In particular, attention scores are
    computed between the decoder's hidden states and the encoder's key vectors,
    which can have different sizes, resulting in a rectangular rather than
    square scores matrix.
    """
    def __init__(self, head_dim):
        """
        Initializes the dense layer that project the query, key and value
        vectors to the dimension of the head.
        """
        super().__init__()

        # Initialize the query, key and value weighting matrices.
        self.q = Dense(units=head_dim)
        self.k = Dense(units=head_dim)
        self.v = Dense(units=head_dim)

    def call(self, decoder_hidden_state, encoder_k, encoder_v):
        """
        Forward pass. The layer is passed
          * the hidden state from the decoder, used as the query vectors,
          * the key vectors coming from the encoder,
          * the value vectors coming from the encoder.
        All of them E
        """
        attn_outputs = scaled_dot_product_attention(
            self.q(decoder_hidden_state),
            self.k(encoder_k),
            self.v(encoder_v)
        )

        return attn_outputs


class EncoderDecoderMultiHeadAttention(Layer):
    """
    Implementation of an encoder-decoder multi-head attention layer.
    """
    def __init__(self, config):
        """
        Initializes a list of encoder-decoder single-head attention layers and
        the final dense (fully-connected) layer. Analogous to the multi-head
        self-attention layer used in the encoder, but this time each head
        computes the attention weights between the decoder hidden states and
        the encoder key vectors to produce a linear combination of the encoder
        value vectors.
        """
        super().__init__()

        embed_dim = config.hidden_size
        num_heads = config.num_attention_heads
        head_dim = embed_dim // num_heads

        # Initialize a list of attention heads.
        self.heads = [
            EncoderDecoderAttentionHead(head_dim=head_dim)
            for _ in range(num_heads)
        ]

        # Initialize the final dense layer.
        self.output_linear = Dense(units=embed_dim)

    def call(self, decoder_hidden_state, encoder_k, encoder_v):
        """
        Forward pass: the input is passed through each head
        independently, then the outputs are concatenated back
        together and passed through the final dense layer.

        Input shape: (batch_shape, decoder_seq_len, hidden_dim)
        Output layer: (batch_shape, decoder_seq_len, hidden_dim)
        """
        # Pass the input through each head and concatenate
        # the outputs.
        x = tf.concat(
            [h(decoder_hidden_state, encoder_k, encoder_v)
             for h in self.heads],
            axis=-1
        )

        # Pass the concatenated outputs through the final
        # linear layer.
        x = self.output_linear(x)

        return x


class TransformerDecoderLayer(Layer):
    """
    Implementation of a transformer decoder layer. The full transformer will
    comprise a stack (sequence) of these layers.
    """
    def __init__(self, config):
        """
        Initializes inner layers.
        """
        super().__init__()

        self.layer_norm_1 = LayerNormalization()
        self.layer_norm_2 = LayerNormalization()
        self.layer_norm_3 = LayerNormalization()
        self.masked_attention = MaskedMultiHeadAttention(config=config)
        self.encoder_decoder_attention = EncoderDecoderMultiHeadAttention(
            config=config)
        self.feed_forward = FeedForward(config=config)

    def call(self, x, encoder_k, encoder_v):
        """
        Forward pass. The specific architecture (how many layer normalization
        and skip connections steps are there and where) is taken from:
            https://pyimagesearch.com/2022/09/05/a-deep-dive-into-transformers-with-tensorflow-and-keras-part-1/
        """
        # Masked multi-head attention with a skip connection "around it".
        x = x + self.masked_attention(x)

        # Layer normalization.
        x = self.layer_norm_1(x)

        # Encoder-decoder multi-head attention with a skip connection "around
        # it".
        x = x + self.encoder_decoder_attention(x, encoder_k, encoder_v)

        # Layer normalization.
        x = self.layer_norm_2(x)

        # FNN layer with a skip connection around it.
        x = x + self.feed_forward(x)

        # Layer normalization.
        x = self.layer_norm_3(x)

        return x


class TransformerDecoder(Layer):
    """
    Class instantiating a full transformer decoder object as a Keras `Layer`.
    """
    def __init__(self, config):
        """
        Initializes the two parts a full transformer decoder is made of:
          1. An embedding layer that produces the hidden state (token +
             positional embeddings) given the input token IDs.
          2. A stack (sequence) of `TransformerDecoderLayer` objects,
             traversed one after the other.
        """
        super().__init__()

        # Instantiate the embeddings layer.
        self.embeddings = Embeddings(config)

        # Instantiate the stack of encoder layers.
        self.layers = [
            TransformerDecoderLayer(config=config)
            for _ in range(config.num_hidden_layers)
        ]

    def call(self, inputs):
        """
        Forward pass of the decoder. Steps:
          1. Embeddings are computed.
          2. Data goes through the stack of `TransformerDecoderLayer` layers.

        Parameters
        ----------
        input_ids : tf.Tensor
            Tensor of shape (batch_size, decoder_seq_len) containing the
            numerical encodings (IDs) of the tokens in the input sequences of
            the decoder.
        encoder_k : tf.Tensor
            Tensor of shape (batch_size, encoder_seq_len, encoder_hidden_dim)
            representing the key vectors coming from the encoder. The batch
            size needs to be the same as that of the decoder's input but the
            hidden dimension could in principle be different as the Dense
            layers projecting the tensors into the head dimension can take that
            into account.
        encoder_v : tf.Tensor
            Tensor of shape (batch_size, encoder_seq_len, encoder_hidden_dim)
            representing the value vectors coming from the encoder. The batch
            size needs to be the same as that of the decoder's input but the
            hidden dimension could in principle be different as the Dense
            layers projecting the tensors into the head dimension can take that
            into account.

        Returns
        -------
        x : tf.Tensor
            Tensor of shape (batch_size, decoder_seq_len, encoder_hidden_dim)
            representing the linear combination of the value vectors coming
            from the encoder with weights given by the weights computed in the
            encoder-decoder attention part (multiple times, once for each
            TransformerDecoderLayer in the stack).
        """
        # Unpack the `inputs` list.
        input_ids, encoder_k, encoder_v = inputs

        # Compute the numerical embeddings of the input.
        x = self.embeddings(input_ids)

        # Act with all the encoder layers in the stack
        # sequentially.
        for layer in self.layers:
            x = layer(x, encoder_k, encoder_v)

        return x
