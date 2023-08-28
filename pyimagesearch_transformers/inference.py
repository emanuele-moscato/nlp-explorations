import tensorflow_text as tf_text
import tensorflow as tf
import argparse


# Define the command line arguments for the script.
ap = argparse.ArgumentParser()

ap.add_argument(
    '-s',
    '--sentence',
    required=True,
    help='Input English sentence'
)

# Parse the arguments.
args = vars(ap.parse_args())

# Put the `sentence` argument into a tensor: this is the input sentence that
# will be translated.
sourceText = tf.constant(args['sentence'])

# Load the Translator model.
print('[INFO] loading the translator model from disk...')

translator = tf.saved_model.load('models')

# Perform inference.
print('[INFO] translating English sentence to French...')

result = translator(sentence=sourceText)

translatedText = result.numpy()[0].decode()

print('[INFO] English sentence:', args['sentence'])
print('[INFO] French translation:', translatedText)
