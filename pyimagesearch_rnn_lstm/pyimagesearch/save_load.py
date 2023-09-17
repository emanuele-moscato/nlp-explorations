from tensorflow.keras.layers import TextVectorization
import tensorflow as tf
import pickle


def save_vectorizer(vectorizer, name):
    """
    Saves the weight from the vectorization layer to a specified pickle file.
    """
    # Save the weights of the vectorization layer.
    with open(f'{name}.pkl', 'wb') as f:
        pickle.dump({'weights': vectorizer.get_weights()}, f)


def load_vectorizer(name, maxTokens, outputLength, standardize=None):
    """
    Loads a saved `TextVectorization` layer object. Since only the weights of
    the vectorizer are saved (see the `save_vectorizer` function), we first
    load the weights, then instantiate a new `TextVectorization` object
    (initialized with dummy data), whose weights are then set to be the loaded
    ones.

    Parameters
    ----------
    name : str
        File path for the saved `TextVectorization` layer weights.
    maxTokens : int
        Maximum number of tokens in the vocabulary (i.e. the vocabulary size).
    outputLength : int
        Maximum length of the output sequences (with padding, all sequences
        have a length equal to the maximal one).
    standardize : callable (default: None)
        Standardization function to apply.

    Returns
    -------
    newVectorizer : tf.keras.layers.TextVectorization
        Instance of a `TextVectorization` layer from Keras, initialized with
        the loaded weights.
    """
    # Load a saved vectorizer object from disk.
    with open(f'{name}.pkl', 'rb') as f:
        fromDisk = pickle.load(f)

    # Instantiate a new vectorizer object.
    newVectorizer = TextVectorization(
        max_tokens=maxTokens,
        output_mode='int',
        output_sequence_length=outputLength,
        standardize=standardize
    )

    # Initialize the new vectorizer by calling its `adapt` method on some
    # dummy data.
    newVectorizer.adapt(tf.data.Dataset.from_tensor_slices(['xxz']))

    # Set the the weights of the new vectorizer with the values of the loaded
    # one. This effectively reproduces the old vectorizer.
    newVectorizer.set_weights(fromDisk['weights'])

    return newVectorizer
