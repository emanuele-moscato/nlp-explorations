from tensorflow.keras.layers import Layer, Dropout, Dense
from encoder import TransformerEncoder


class TransformerForSequenceClassification(Layer):
    """
    Encoder-only model for text classification. It's composed of a body given
    by an encoder and a head given by a dense layer.
    """
    def __init__(self, config):
        """
        Initializes the inner layers.

        Note: because there's no softmax activation function for the head, for
              each input sample the model returns the unnormalized logits for
              each ouput class.
        """
        super().__init__()

        # Encoder part: extracts the hidden state from the token IDs. This is
        # the body of the model (which is usually pre-trained separately from
        # the head).
        self.encoder = TransformerEncoder(config=config)

        # Dorpout layer for regularization.
        self.dropout = Dropout(rate=config.hidden_dropout_prob)

        # Dense layer: this is the head of the model, performing the
        # classification task with the hidden state provided by the body as its
        # input. This part is trained on the specific task.
        self.classifier = Dense(units=config.num_labels)

    def call(self, x):
        """
        Forward pass for the model. Note that we only use the hidden state of
        the start-of-sequence token (which, sequence by sequence, contains
        information about its context).

        Parameters
        ----------
        x : tf.Tensor
            Tensor of shape (batch_size, seq_len) containing the token IDs for
            the input sequences.

        Returns
        -------
        x : tf.Tensor
            Tensor of shape (batch_size, num_classes) containing the
            unnormalized logits for each sample in the batch.
        """
        # For each sequence in the batch, produce the hidden state of the
        # start-of-sequence ([CLS]) token only.
        x = self.encoder(x)[:, 0, :]

        # Dropout regularization.
        x = self.dropout(x)

        # Classification.
        x = self.classifier(x)

        return x
