import os
import tensorflow as tf
from pyimagesearch.standardization import custom_standardization
from pyimagesearch.plot import plot_loss_accuracy
from pyimagesearch.save_load import save_vectorizer
from pyimagesearch.dataset import get_imdb_dataset
from pyimagesearch.model import get_rnn_model
# from pyimagesearch.model import get_lstm_model
from pyimagesearch import config
from tensorflow.keras import layers
from tensorflow import keras


# Set random seed for Tensorflow.
tf.keras.utils.set_random_seed(42)


# Get the IMDB dataset.
print('[INFO] Getting the IMDB dataset...')

trainDs, valDs = get_imdb_dataset(
    folderName=config.DATASET_PATH,
    batchSize=config.BATCH_SIZE,
    bufferSize=config.BUFFER_SIZE,
    autotune=tf.data.AUTOTUNE,
    test=False
)

# Initialize the text vectorization layer.
vectorizeLayer = layers.TextVectorization(
    max_tokens=config.VOCAB_SIZE,
    output_mode='int',
    output_sequence_length=config.MAX_SEQUENCE_LENGTH,
    standardize=custom_standardization
)

# Adapt the text vectorization layer on the training data.
# Fetch the text only from the training dataset (which also contains the
# labels).
trainText = trainDs.map(lambda text, label: text)

vectorizeLayer.adapt(trainText)

# Vectorize the training and validation datasets.
trainDs = trainDs.map(lambda text, label: (vectorizeLayer(text), label))
valDs = valDs.map(lambda text, label: (vectorizeLayer(text), label))

# Instantiate the RNN model and compile it.
print('[INFO] Building the RNN model...')

modelRNN = get_rnn_model(vocabSize=config.VOCAB_SIZE)

modelRNN.compile(
    metrics=['accuracy'],
    optimizer=keras.optimizers.Adam(learning_rate=config.LR),
    loss=keras.losses.BinaryCrossentropy(from_logits=False)
)

# Train the RNN model.
print('[INFO] Training the RNN model...')

historyRNN = modelRNN.fit(
    trainDs,
    epochs=config.EPOCHS,
    validation_data=valDs
)

# Write output.
# Create output directory.
if not os.path.exists(config.OUTPUT_PATH):
    os.makedirs(config.OUTPUT_PATH)

# Save the loss and accuracy plots.
plot_loss_accuracy(history=historyRNN.history, filepath=config.RNN_PLOT)
# plot_loss_accuracy(history=historyLSTM.history, filepath=config.LSTM_PLOT)

# Save the trained models to disk.
print(f'[INFO] Saving the RNN model to {config.RNN_MODEL_PATH}...')

keras.models.save_model(
    model=modelRNN,
    filepath=config.RNN_MODEL_PATH,
    include_optimizer=False
)

# print(f'[INFO] Saving the LSTM model to {config.LSTM_MODEL_PATH}...')

# keras.models.save_model(
#     model=modelLSTM,
#     filepath=config.LSTM_MODEL_PATH,
#     include_optimizer=False
# )

# Save the text vectorization layer to disk.
save_vectorizer(vectorizer=vectorizeLayer, name=config.TEXT_VEC_PATH)
