import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import StringLookup

class Translator(tf.Module):
    """
    Subclass of tf.Module providing an API for inference (translation
    operation) by wrapping a Transformer object.

    Note: a `tf.Module` object is a container for Tensorflow Variables, Modules
          and functions. Subclassing `tf.Module` allows to recognize all the
          objects defined therein as belonging the the same namespace (the
          Module itself), which is useful to keep track of them (e.g.
          trainable variables etc.).
    """
    def __init__(
        self,
        sourceTextProcessor,
        targetTextProcessor,
        transformer,
        maxLength
    ):
        """
        sourceTextProcessor : ?
            Processor (vectorizer?) for the source text.
        targetTextProcessor : ?
            Processor (vectorizer?) for the target text.
        transformer : transformer.Transformer
            Trained transformer model.
        maxLength : int
            Maximum lenght of the translated sentence.
        """
        # Initialize source and target text processors (vectorizers?).
        self.sourceTextProcessor = sourceTextProcessor
        self.targetTextProcessor = targetTextProcessor

        # Initialize StringLookup object: this will be used to go from the
        # predicted tokens' indices to the corresponding words in the target
        # vocabulary.
        self.targetStringFromIndex = StringLookup(
            vocabulary=targetTextProcessor.get_vocabulary(),
            mask_token='',
            invert=True
        )

        # Initialize the Transformer object: this is the trained trasformer
        # model that is used by the translator.
        self.transformer = transformer

        # Maximum length of the translated sentences.
        self.maxLength = maxLength

    def tokens_to_text(self, resultTokens):
        """
        Given the predicted sequences of tokens IDs (in a tensor), returns the
        corresponding sentences in natural language (target language).
        """
        # Perform the lookup operation to go from the predicted tokens to the
        # strings in the target vocabulary.
        resultTextTokens = self.targetStringFromIndex(resultTokens)

        # Join the resulting strings into sentences.
        resultText = tf.strings.reduce_join(
            input=resultTextTokens,
            axis=1,
            separator=' '
        )

        # Strip the strings in the tensor of leading and trailing whitespaces.
        resultText = tf.strings.strip(resultText)

        return resultText

    @tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.string)])
    def __call__(self, sentence):
        """
        Implements the sequence of operations executed when the Translator
        object is called (on an input sentence to translate). Logic: the input
        sentence is passed to the transformer within an iteration over the
        predicted tokens in the target language (which are also passed to the
        transformers as they are produced: they are the input to the decoder).
        The iteration goes on until either the maximum length of the target
        sentence is reached or the end token in the target language is
        predicted. Predicted tokens are accumulated in a `TensorArray` object.
        The predicted sequence of token IDs is then turned into text and
        returned.

        Parameters
        ----------
        sentence : tf.Tensor (dtype: tf.string)
            Input sentence in the source language to translate

        Returns
        -------
        text : tf.Tensor (dtype: tf.string) (guess)
            The predicted sentence in the target language.
        """
        # Vectorize the source sentence.
        sentence = self.sourceTextProcessor(sentence[tf.newaxis])

        encoderInput = sentence

        # Initialize the target sentence: when called on an empty sentence the
        # targetTextProcessor inserts only the start and end tokens.
        # Note: the final 0 index is for the only batch.
        startEnd = self.targetTextProcessor([''])[0]

        # Take the start and end tokens in the initialized target sentence
        # individually.
        startToken = startEnd[0][tf.newaxis]
        endToken = startEnd[1][tf.newaxis]

        # Initialize the output array by putting the target sentence start
        # token in it.
        # Note: a `tf.TensorArray` object is a dynamical array that can be
        #       populated iteratively.
        outputArray = tf.TensorArray(dtype=tf.int64, size=0, dynamic_size=True)
        outputArray = outputArray.write(index=0, value=startToken)

        # Perform up to `maxLength` iterations to generate the output token
        # IDs.
        for i in tf.range(self.maxLength):
            # Transpose the output array stack (?).
            output = tf.transpose(outputArray.stack())

            # Generate predictions. The transformer is passed the source
            # sentence, but also its own output, iteration after iteration. The
            # prediction for each token is a collection of predicted
            # probabilities over the target vocabulary (with one-hot encoding).
            # Note: the `predictions` tensor has
            #       (batch_size, ?, vocabulary_size).
            predictions = self.transformer(
                [encoderInput, output],
                training=False
            )
            predictions = predictions[:, -1, :]

            # Extract the predicted token ID in the target language by taking
            # the argmax of the probabilities for each predicted token.
            predictedId = tf.argmax(predictions, axis=-1)

            # Write the predicted token ID to the output array.
            outputArray = outputArray.write(i+1, predictedId[0])

            # If the model predicts the end token, the sentence is over and we
            # exit from the loop.
            if predictedId == endToken:
                break

        # Transpose the output array back to a more useful shape for text
        # manipulation.
        output = tf.transpose(outputArray.stack())

        # Extract the text in the target language from the output array (which
        # only contains the predicted token IDs).
        text = self.tokens_to_text(output)

        return text
