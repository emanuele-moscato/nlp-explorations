import tensorflow as tf
from tensorflow.keras.layers import (
    Layer, Embedding, LayerNormalization, Dropout, Dense)
from utils import scaled_dot_product_attention


class Embeddings(Layer):
    """
    Class implementing the embedding layer, with output
    embeddings incorporating both information from token
    and from position embeddings.
    """
    def __init__(self, config):
        """
        Initializes the inner layers.
        """
        super().__init__()

        # Initialize all the inner layers.
        # Simple dense embedding of the numerical tokens.
        self.token_embeddings = Embedding(
            input_dim=config.vocab_size,
            output_dim=config.hidden_size
        )

        # For the positional embedding, we use the Keras
        # `Embedding` layer again, this time with an input
        # dimension equal to the maximum positional embedding
        # (the maximum index of a token within a sequence, closely
        # related to the maximum length of a sequence) rather
        # than to the size of the vocabulary.
        self.position_embeddings = Embedding(
            input_dim=config.max_position_embeddings,
            output_dim=config.hidden_size
        )

        self.layer_norm = LayerNormalization(epsilon=1e-12)

        self.dropout = Dropout(rate=config.dropout)

    def call(self, input_ids):
        """
        Forward pass of the embedding layer.
        Input: token IDs.
        Output: embeddings.

        Token and posiiton embeddings are generated for the
        input IDs and then added up. The resulting embeddings
        are then normalized with layer normalization and regularized
        with a dropout layer.
        """
        # Get the sequence length.
        seq_length = input_ids.shape[1]

        # Get all the position IDs as the rage from 0 to seq_length - 1.
        position_ids = tf.range(
            seq_length,
            dtype=tf.int64
        )

        # Create token embeddings.
        token_embeddings = self.token_embeddings(input_ids)

        # Create position embeddings.
        position_embeddings = (
            self.position_embeddings(position_ids)[tf.newaxis, ...])

        # Combine the information in token and position embeddings
        # by adding them up. The shape of `position_embeddings` is
        # broadcast to that of `token_embeddings`.
        embeddings = token_embeddings + position_embeddings

        # Normalize the combined embeddings.
        embeddings = self.layer_norm(embeddings)

        # Dropout regularization on the embeddings.
        embeddings = self.dropout(embeddings)

        return embeddings


class AttentionHead(Layer):
    """
    Implementation of a single-head self-attention layer.
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
        attn_outputs = scaled_dot_product_attention(
            self.q(hidden_state),
            self.k(hidden_state),
            self.v(hidden_state)
        )

        return attn_outputs


class MultiHeadAttention(Layer):
    """
    Implementation of a multi-head self-attention layer.
    """
    def __init__(self, config):
        """
        Initializes a list of single-head self-attention layers and
        the final dense (fully-connected) layer.
        """
        super().__init__()

        embed_dim = config.hidden_size
        num_heads = config.num_attention_heads
        head_dim = embed_dim // num_heads

        # Initialize a list of attention heads.
        self.heads = [
            AttentionHead(embed_dim=embed_dim, head_dim=head_dim)
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


class FeedForward(Layer):
    """
    Implementation of the feed-forward layer to be added
    after the MHA layers (both in the encoder and in the
    decoder part of a transformer).
    """
    def __init__(self, config):
        """
        Initializes the inner layers and activation function.
        Rule of thumb for the intermediate size: 4 * [hidden_size].
        """
        super().__init__()

        self.linear_1 = Dense(
            units=config.hidden_dim,
            activation='gelu'
        )
        self.linear_2 = Dense(units=config.hidden_size)

        self.dropout = Dropout(rate=config.dropout)

    def call(self, x):
        """
        Forward pass for the layer. Sequentially, the input passes
        through:
          1. A dense layer with GELU activation function.
          2. Another dense layer with identity activation.
          3. A dropout regularization layer.

        Input shape: (batch_size, seq_len, hidden_dim)
        Output shape: (batch_size, seq_len, hidden_dim)

        Note: by default, the Dense layers act on the last (right-most)
              dimension of an input tensor, leaving any other dimension
              untouched - which is exactly what we want to process
              each embedding independently from the others.
        """
        x = self.linear_1(x)
        x = self.linear_2(x)
        x = self.dropout(x)

        return x


class TransformerEncoderLayer(Layer):
    """
    Class implementing one full encoder layer of a transformer.
    The full transformer will comprise a stack of these layers.
    """
    def __init__(self, config):
        """
        Initializes all the inner layers.
        """
        super().__init__()

        # Initialize all the layers.
        self.layer_norm_1 = LayerNormalization()
        self.layer_norm_2 = LayerNormalization()
        self.attention = MultiHeadAttention(config)
        self.feed_forward = FeedForward(config)

    def call(self, x):
        """
        Forward pass: the input passes through the MHA and FFN
        layers, with layer normalization and skip connections
        in between. Pre-layer normalization is used.
        """
        # Compute the normalized input.
        # Note: we don't put this back into the x variable
        #       because we need to have them both for skip
        #       connections.
        hidden_state = self.layer_norm_1(x)

        # Skip connection: the input to the first layer normalization
        # is added to the output obtained from the MHA layer acting
        # on the normalized input.
        x = x + self.attention(hidden_state)

        # Same as before, but with the FFN layer instead of the MHA
        # one.
        # Note: we could have avoided defining the additional `hidden_state`
        #       variable above by doing as we are doing here.
        x = x + self.feed_forward(self.layer_norm_2(x))

        return x


class TransformerEncoder(Layer):
    """
    Class implementing the full encoder part of a transformer
    model.
    """
    def __init__(self, config):
        """
        Instantiates all the inner layers. In particular, sequentially:
          1. The embedding layer to go from token IDs to dense numerical
             embeddings.
          2. A stack (sequence) of TransformerEncoderLayer layers.
        """
        super().__init__()

        # Instantiate the embeddings layer.
        self.embeddings = Embeddings(config)

        # Instantiate the stack of encoder layers.
        self.layers = [
            TransformerEncoderLayer(config=config)
            for _ in range(config.num_hidden_layers)
        ]

    def call(self, x):
        """
        Forward pass of the encoder. First, embeddings are computed,
        then the stack of TransformerEncoderLayer layers acts on the
        embeddings.
        """
        # Compute the numerical embeddings of the input.
        x = self.embeddings(x)

        # Act with all the encoder layers in the stack
        # sequentially.
        for layer in self.layers:
            x = layer(x)

        return x
