import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.metrics import Mean
# Use these imports when working from a script.
# from .decoder import Decoder
# from .encoder import Encoder
# Use these imports when working from a notebook in the notebooks/ directory.
from decoder import Decoder
from encoder import Encoder



class Transformer(Model):
    """
    Subclass of keras.Model implementing the full transformer. A transformer
    is composed of an Encoder that computes embeddings for the source text, and
    a Decoder that computes an output based on the target text and the output
    of the Encoder. The output of the Decoder is then passed to a Dense layer
    that output the logits that will be passed to the softmax function.
    """
    def __init__(
        self,
        encNumLayers,
        decNumLayers,
        dModel,
        numHeads,
        dff,
        sourceVocabSize,
        targetVocabSize,
        maximumPositionEncoding,
        dropOutRate=0.1,
        **kwargs
    ):
        """
        Parameters
        ----------
        encNumLayers : int
            Number of EncoderLayer stacked in the encoder architecture.
        decNumLayers : int
            Number of DecoderLayer stacked in the decoder architecture.
        dModel : int
            Dimension of the model (i.e. size of the representation of the
            tokens).
        numHeads : int
            Number of heads in the MHA layers.
        dff : int
            Intermediate dimension of the FFN layers.
        sourceVocabSize : int
            Size of the source vocabulary.
        targetVocabSize : int
            Size of the target vocabulary.
        maximumPositionEncoding : int
            Maximum length of a sentence (sequence of tokens).
        dropOutRate : float (default: 0.1)
            Dropout rate for dropout regularization.
        """
        # Call the constructor of the parent class.
        super().__init__(**kwargs)

        # Initialize the encoder.
        self.encoder = Encoder(
            numLayers=encNumLayers,
            dModel=dModel,
            numHeads=numHeads,
            sourceVocabSize=sourceVocabSize,
            maximumPositionEncoding=maximumPositionEncoding,
            dff=dff,
            dropOutRate=dropOutRate
        )

        # Initialize the decoder.
        self.decoder = Decoder(
            numLayers=decNumLayers,
            dModel=dModel,
            numHeads=numHeads,
            targetVocabSize=targetVocabSize,
            maximumPositionEncoding=maximumPositionEncoding,
            dff=dff,
            dropOutRate=dropOutRate
        )

        # Final layer of the transformer (a Dense layer whose output is then
        # passed to softmax.
        # Note: we use one-hot encoding, so the last layer should output a
        #       number of logits for each sample equal to the size of the
        #       target vocabulary.
        self.finallayer = Dense(units=targetVocabSize)

    def call(self, inputs):
        # Unpack the input into source and target.
        (source, target) = inputs

        # Compute the output from the encoder.
        encoderOutput = self.encoder(x=source)

        # Compute the output from the decoder.
        # Note: the decoder has two inputs - the target and the ouptut of the
        #       encoder.
        decoderOutput = self.decoder(x=target, context=encoderOutput)

        # The final layer of the transformer outputs the logits.
        logits = self.finallayer(decoderOutput)

        # Drop the Keras mask the logits might have so the losses/metrics are
        # scaled the correct way (?).
        try:
            del logits._keras_mask
        except AttributeError:
            pass

        return logits
