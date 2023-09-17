import tensorflow as tf
import string
import re


def custom_standardization(inputData):
    """
    Standardizes the text by
      1. Turning everything to lowercase.
      2. Eliminating HTML breakpoints.
      3. Eliminating punctuation.
    """
    # Turn text data all to lowercase.
    lowercase = tf.strings.lower(inputData)

    # Replace the HTML breakpoints in the text with a blank space.
    strippedHtml = tf.strings.regex_replace(lowercase, '<br />', ' ')

    # Replace punctuation with a blank space.
    strippedPunctuation = tf.strings.regex_replace(
        strippedHtml,
        f'[{re.escape(string.punctuation)}]',
        ' '
    )

    return strippedPunctuation
