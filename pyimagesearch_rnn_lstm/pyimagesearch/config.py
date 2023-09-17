import os


# Path to the dataset.
DATASET_PATH = './data/'

# Batch size for training.
BATCH_SIZE = 512

# Buffer size for shuffling the Tensorflow `Dataset` object.
BUFFER_SIZE = 100

VOCAB_SIZE = 50000

# Maximum number of tokens for each sequence.
MAX_SEQUENCE_LENGTH = 100

# Learning rate for the optimizer.
LR = 0.1

# Number of training epochs.
EPOCHS = 10  # 500

# Output directory,
OUTPUT_PATH = './output/'

# Paths for the various objects to save at the enc of the training phase.
RNN_PLOT = os.path.join(OUTPUT_PATH, 'rnn_plot.png')
LSTM_PLOT = os.path.join(OUTPUT_PATH, 'lstm_plot.png')
RNN_MODEL_PATH = os.path.join(OUTPUT_PATH, 'rnn_model/')
LSTM_MODEL_PATH = os.path.join(OUTPUT_PATH, 'lstm_model/')
TEXT_VEC_PATH = os.path.join(OUTPUT_PATH, 'vectorizer')
