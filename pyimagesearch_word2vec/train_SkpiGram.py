import os
import tensorflow as tf
from pyimagesearch import config
from pyimagesearch.create_vocabulary import tokenize_data
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import numpy as np


# Set Tensorflow's random seed.
tf.random.set_seed(42)


# Read data.
print('[INFO] Reading the data from disk...')

with open('./data.txt', 'r') as filePointer:
    lines = filePointer.readlines()

textData = ''.join(lines)

# Tokenize text and get the vocabulary.
vocab, tokenizedTextSize, tokenizedText = tokenize_data(textData)

# Map the vocabulary words to progressively increasing indices, then build the
# inverse map (from indices to words) as well.
vocabToIndex = {uniqueWord: index for index, uniqueWord in enumerate(vocab)}

indexToVocab = np.array(vocab)

# Convert tokens into the corresponding integers.
textAsInt = np.array([vocabToIndex[word] for word in tokenizedText])

# Create the embedding matrices as Tensorflow Variable tensors (so they can be)
# trained. Each matrix has shape: (n_tokens, embedding_size) (one row for each
# token in the tokenized text - we don't use unique words as we want to keep
# track of the context). Entries are initialized randomly sampling from a
# uniform distribution on the [0, 1) interval.
contextVectorMatrix = tf.Variable(
    np.random.rand(tokenizedTextSize, config.EMBEDDING_SIZE)
)
centerVectorMatrix = tf.Variable(
    np.random.rand(tokenizedTextSize, config.EMBEDDING_SIZE)
)

# Initialize an optimizer.
optimizer = tf.optimizers.Adam()

# Initialize an empty list in which to store each epoch's loss value.
lossList = list()

# Training: loop over the epochs.
print('[INFO] Start SkipGram training...')

for iter in tqdm(range(config.ITERATIONS)):
    # Initialize the loss per epoch.
    lossPerEpoch = 0.

    # Loop over each window.
    for start in range(tokenizedTextSize - config.WINDOW_SIZE):
        # Compute the integer encodings correpsonding to the words in the
        # window.
        indices = textAsInt[start:start + config.WINDOW_SIZE]

        # For each window, compute the loss and proceed with a gradient descent
        # step.
        with tf.GradientTape() as tape:
            # Initialize the context loss.
            loss = 0

            # Take the embedding vector form the center embedding matrix
            # correpsonding to the center vector.
            centerVector = centerVectorMatrix[
                # index of the center vector.
                indices[config.WINDOW_SIZE // 2],
                :
            ]

            # Multiply the context embedding matrix and the center vector.
            # Resulting shape: (n_tokens, 1) (ready to be passed to softmax to
            # produce probability over the tokens.
            output = tf.matmul(
                # Shape: (n_tokens, embedding_size).
                contextVectorMatrix,
                # Shape after expanding dimensions: (embedding_size, 1).
                tf.expand_dims(centerVector, 1)
            )

            # Apply the softmax function to get probabilities over the tokens.
            softmaxOutput = tf.nn.softmax(output, axis=0)

            # Loop over the integer token encodings in the window.
            for count, index in enumerate(indices):
                # For each context token (conut different from the center of
                # the window) compute the loss (actually the negative loss)
                # and add it to the total loss for the current window.
                # Note: in fact, as we want the probabilities for the correct
                #       context words to get as close as possible to 1, this
                #       quantity would have to be maximized.
                if count != config.WINDOW_SIZE // 2:
                    loss += softmaxOutput[index]

            # Compute the negative log of the total loss: this is what we
            # actually have to minimize.
            logLoss = -tf.math.log(loss)

        # Add the negative log loss computed for the current window to the
        # total loss for the whole epoch.
        lossPerEpoch += logLoss.numpy()

        # Compute the gradient.
        grad = tape.gradient(
            logLoss, [contextVectorMatrix, centerVectorMatrix]
        )

        # Apply the gradients to the embedding matrices (gradient descent
        # step).
        optimizer.apply_gradients(
            zip(grad, [contextVectorMatrix, centerVectorMatrix])
        )

    # Append the log loss for the current epoch to the loss list.
    lossList.append(lossPerEpoch)

# Create the output directory if it doesn't exist already.
if not os.path.exists(config.OUTPUT_PATH):
    os.makedirs(config.OUTPUT_PATH)

# Plot the log loss along the epochs.
print('[INFO] Plotting loss...')

plt.plot(lossList)

plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.savefig(config.SKIPGRAM_LOSS, dpi=400, bbox_inches='tight')

# Apply dimensional reduction to visualize the embedding vectors in 2
# dimensions.
tsneEmbed = TSNE(n_components=2).fit_transform(centerVectorMatrix.numpy())
tsneDecode = TSNE(n_components=2).fit_transform(contextVectorMatrix.numpy())

# Plot the first 50 dimensionally reduced embedding vectors of center tokens.
print('[INFO] Plotting TSNE embeddings...')

indexCount = 0

plt.figure(figsize=(25, 5))

for (word, embedding) in tsneEmbed[:50]:
    plt.scatter(word, embedding)

    plt.annotate(indexToVocab[indexCount], (word, embedding))

    indexCount += 1

plt.savefig(config.SKIPGRAM_TSNE, dpi=400, bbox_inches='tight')
