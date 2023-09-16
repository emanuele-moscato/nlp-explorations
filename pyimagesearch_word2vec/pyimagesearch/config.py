import os


# Size of the embedding vectors.
EMBEDDING_SIZE = 10

# Window size for context words and number of iterations (training epochs).
WINDOW_SIZE = 5
ITERATIONS = 1000

# Path to the output directory.
OUTPUT_PATH = 'outputs'

# Paths to the skipgram outputs.
SKIPGRAM_LOSS = os.path.join(OUTPUT_PATH, 'loss_skipgram.png')
SKIPGRAM_TSNE = os.path.join(OUTPUT_PATH, 'tsne_skipgram.png')

# Paths to the CBOW outputs.
CBOW_LOSS = os.path.join(OUTPUT_PATH, 'loss_cbow.png')
CBOW_TSNE = os.path.join(OUTPUT_PATH, 'tsne_cbow.png')
