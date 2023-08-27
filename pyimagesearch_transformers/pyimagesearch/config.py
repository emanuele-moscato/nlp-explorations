# Dataset filename.
DATA_FNAME = "fra.txt"

# Batch size for training.
BATCH_SIZE = 512

# Sizes of the source and target vocabularies.
SOURCE_VOCAB_SIZE = 15000
TARGET_VOCAB_SIZE = 15000

# Maximum position of a token (word), both in the source and target dataset.
MAX_POS_ENCODING = 2048

# Number of encorder and decoder layers in the transformer.
ENCODER_NUM_LAYERS = 6
DECODER_NUM_LAYERS = 6

# Dimension of the model. Note: model dimension is static and can therefore
# be parametrized.
D_MODEL = 512

# Number of units in the feed-forward neural network.
DFF = 2048

# Number of heads in the attention layers.
NUM_HEADS = 8

# Dropout rate (for dropout regularization layers).
DROP_RATE = 0.1

# Number of epochs used in training.
EPOCHS = 25

# Name of the output directory.
OUTPUT_DIR = 'output'
