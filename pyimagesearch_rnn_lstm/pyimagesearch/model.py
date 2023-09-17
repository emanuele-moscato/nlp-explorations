from tensorflow.keras import layers
from tensorflow import keras


def get_rnn_model(vocabSize):
    """
    Give the size of the vocabulary, returns a Keras `Model` object
    implementing a recurrent NN.
    """
    # Create an input layer (placeholder for the input of the model).
    inputs = keras.Input(shape=(None,), dtype="int32")

    # Embed tokens using the `Embedding` layer. This maps tokens (positive
    # integers) to dense vectors of the specified size.
    x = layers.Embedding(input_dim=vocabSize, output_dim=128, mask_zero=True)(inputs)

    # Apply dropout regularization.
    x = layers.Dropout(0.2)(x)

    # Stack of `SimpleRNN` layers. Activating the `return_sequences` option
    # makes the layer return the entire sequence of outputs generated for
    # each sample, with an output shape (batch_shape, timesteps, units) (while
    # normally the output shape would be just (batch_shape, units)).
    # Note: by default, `SimpleRNN` layers have a tanh activation.
    x = layers.SimpleRNN(units=64, return_sequences=True)(x)
    x = layers.SimpleRNN(units=64, return_sequences=True)(x)
    x = layers.SimpleRNN(units=64)(x)

    # Add a classifier head build out of `Dense` layers with dropout
    # regularization.
    x = layers.Dense(units=64, activation='relu')(x)
    x = layers.Dense(units=32, activation='relu')(x)
    x = layers.Dropout(0.2)(x)

    # Output layer, with sigmoid activation (so the output can be interpreted)
    # as a probability over two classes.
    outputs = layers.Dense(units=1, activation='sigmoid')(x)

    # Build the `Model` object.
    model = keras.Model(inputs=inputs, outputs=outputs, name='RNN')

    return model


def get_lstm_model(vocabSize):
    """
    Returns a Keras `Model` object implementing an LSTM neural network with
    a classifier head.
    """
    # `Input` object (placeholder for the input tensor to build the model).
    inputs = keras.Input(shape=(None,), dtype='int32')

    # Token embedding and dropout regularization.
    x = layers.Embedding(vocabSize, 128, mask_zero=True)(inputs)
    x = layers.Dropout(0.2)(x)

    # Stack (sequence) of LSTM layers.
    x = layers.LSTM(64, return_sequences=True)(x)
    x = layers.LSTM(64, return_sequences=True)(x)
    x = layers.LSTM(64)(x)

    # Add a classifier head (fully-connected NN) with dropout regularization.
    x = layers.Dense(units=64, activation='relu')(x)
    x = layers.Dense(units=32, activation='relu')(x)
    x = layers.Dropout(0.2)(x)

    # Output layer with sigmoid activation for binary classification.
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = keras.Model(
        inputs=inputs,
        outputs=outputs,
        name='LSTM'
    )

    return model
