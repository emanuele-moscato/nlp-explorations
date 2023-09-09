from tensorflow.keras.layers import Layer, Dense
from encoder import TransformerEncoder
from decoder import TransformerDecoder


class Transformer(Layer):
    """
    Class implementing a full (encoder-decoder) transformer as a Keras `Layer`
    object.
    """
    def __init__(self, config):
        """
        Instantiates all the inner layers. The dense layers are used to
        project the hidden states outputted by the encoder into key and value
        vectors.
        """
        super().__init__()

        self.encoder = TransformerEncoder(config=config)
        self.decoder = TransformerDecoder(config=config)

        # For simplicity, we decompose the hiddens states from the encoder into
        # key and value vectors with size equal to the hidden dimension.
        self.encoder_k = Dense(units=config.hidden_dim)
        self.encoder_v = Dense(units=config.hidden_dim)

    def call(self, inputs):
        """
        Forward pass for the transformer. The inputs are unpacked into the
        inputs to the encoder and those to the decoder. Steps:
          1. The encoder produces its output hidden states.
          2. The hidden states are projected into key and value vectors by
             dense layers (the projections are trainable!).
          3. The decoder input and the key and value vectors from the encoder
             are fed to the decoder.
          4. The output from the decoder is returned.
        """
        # Unpack the input for the encoder and the decoder.
        encoder_input, decoder_input = inputs

        # Compute the encoder's output (hiddens states).
        encoder_hidden_states = self.encoder(encoder_input)

        # Project the encoder's hidden states onto query and value vectors.
        encoder_k = self.encoder_k(encoder_hidden_states)
        encoder_v = self.encoder_v(encoder_hidden_states)

        # Pass the decoder input and the key and value vectors obtaines from
        # the hidden states from the encoder to the decoder.
        output = self.decoder([decoder_input, encoder_k, encoder_v])

        return output
