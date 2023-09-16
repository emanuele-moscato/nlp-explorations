import os
import tensorflow as tf
from pyimagesearch import config
from pyimagesearch.create_vocabulary import tokenize_data
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import numpy as np


# Set random seed in Tensorflow for reproducibility.
tf.random.set_seed(42)


# Read data.
print('[INFO] Reading data from disk...')

with open('data.txt', 'r') as filePointer:
    lines = filePointer.readlines()

textData = ''.join(lines)

# Tokenize data and get the vocabulary.
(vocab, tokenizedTextSize, tokenizedText) = tokenize_data(textData)

# Map each word in the vocabulary to a progressively increasing index, then
# build the inverse map (from index to word) as an array.
vocabToIndex = {
    uniqueWord: index for (index, uniqueWord) in enumerate(vocab)
}

indexToVocab = np.array(vocab)

# Using the mapping to indices above, convert the token into integers.
textAsInt = np.array([vocabToIndex[word] for word in tokenizedText])

# Create representation matrices as Tensorflow Variable tensors (so their
# components can be modified, i.e. learned). The component values are
# initialized randomly sampling from a uniform distribution over [0, 1).
# Shape: (n_tokens, embedding_size).
# Note: the number of rows of these matrices equals the number of tokens in the
#       text, NOT IN THE VOCABULARY, as we want to extract vectors that are
#       context-dependent.
contextVectorMatrix = tf.Variable(
    np.random.rand(tokenizedTextSize, config.EMBEDDING_SIZE)
)

centerVectorMatrix = tf.Variable(
    np.random.rand(tokenizedTextSize, config.EMBEDDING_SIZE)
)

# Instantiate the optimizer.
optimizer = tf.optimizers.Adam()

lossList = list()

# Iterate over the epochs.
print('[INFO] Starting CBOW training')

for iter in tqdm(range(config.ITERATIONS)):
    # Initialize the loss per epoch.
    lossPerEpoch = 0.

    # Create the window for the prediction of the center vector. Idea: we keep
    # the context and ask the model to predict the center word. We loop over
    # all the possible initial positions of the window.
    for start in range(tokenizedTextSize - config.WINDOW_SIZE):
        # Create the indices for the window.
        indices = textAsInt[start:start + config.WINDOW_SIZE]

        # Initialize gradient tape: we write the training loop explicitly.
        with tf.GradientTape() as tape:
            # Initialize the context vector.
            combinedContext = 0

            # Loop over the indices in the window and take the representation
            # for all the context words from the embedding matrix (note: we
            # skip the center work we have to predict).
            for count, index in enumerate(indices):
                if count != config.WINDOW_SIZE // 2:
                    # Extract the embedding of the word and accumulate it into
                    # `combinedContext`.
                    combinedContext += contextVectorMatrix[index, :]

            # Rescale the resulting vector by the window size to standardize
            # it.
            combinedContext /= (config.WINDOW_SIZE - 1)

            # Compute the predictions for the center word embedding.
            # Resulting shape: (n_tokens, 1) (indeed this is then passed to
            # softmax to compute the probabilities over the tokens).
            output = tf.matmul(
                # Shape: (n_tokens, embedding_size).
                centerVectorMatrix,
                # Shape with expanded dimension: (embedding_size, 1).
                tf.expand_dims(combinedContext, 1)
            )

            # Apply softmax to get probabilities over the tokens.
            softOut = tf.nn.softmax(output, axis=0)

            # Get the predicted probability for the center word.
            # Note: this is NOT actually the loss we want to minimize, in fact
            #       we'd like this probability to be as close as possible to 1.
            loss = softOut[indices[config.WINDOW_SIZE // 2]]

            # Compute the negative log loss: this is the object we'll want to
            # minimize.
            logLoss = - tf.math.log(loss)

        # Update the loss per epoch and apply the gradient to the embedding
        # matrices. The loss per epoch is the sum of the losses computed for
        # each center word in that epoch.
        lossPerEpoch += logLoss.numpy()

        grad = tape.gradient(
            logLoss, [contextVectorMatrix, centerVectorMatrix]
        )

        optimizer.apply_gradients(
            zip(grad, [contextVectorMatrix, centerVectorMatrix])
        )

    # Append the loss from the current epoch to the loss list.
    lossList.append(lossPerEpoch)

# Create output directory.
if not os.path.exists(config.OUTPUT_PATH):
    os.makedirs(config.OUTPUT_PATH)

# Plot loss for evaluation.
print('[INFO] Plotting loss...')

plt.plot(lossList)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig(config.CBOW_LOSS, dpi=400, bbox_inches='tight')

# Dimensional reduction to 2 dimensions (for plotting) on the representation
# matrices (each row represents a word embedding vector).
tsneEmbed = (
    TSNE(n_components=2).fit_transform(centerVectorMatrix.numpy())
)

tsneDecode = (
    TSNE(n_components=2).fit_transform(contextVectorMatrix.numpy())
)

# Plot the reduced vectors for the first 100 words.
print('[INFO] Plotting TSNE embeddings...')

indexCount = 0

plt.figure(figsize=(25, 5))

for (word, embedding) in tsneDecode[:50]:
    plt.scatter(word, embedding)

    plt.annotate(indexToVocab[indexCount], (word, embedding))

    indexCount += 1

plt.savefig(config.CBOW_TSNE, dpi=400, bbox_inches='tight')
