import tensorflow as tf


def tokenize_data(data):
    """
    Given the input data (text), returns its tokenization.
    """
    # Convert data into tokens.
    tokenizedText = tf.keras.preprocessing.text.text_to_word_sequence(
        input_text=data
    )

    # Build the vocabulary taking the set of tokenized words (since it's a set,
    # each is taken only once).
    vocab = sorted(set(tokenizedText))

    # Size of the tokenized text.
    tokenizedTextSize = len(tokenizedText)

    return (vocab, tokenizedTextSize, tokenizedText)
