import tensorflow as tf
from tensorflow import keras
from pyimagesearch.standardization import custom_standardization
from pyimagesearch.save_load import load_vectorizer
from pyimagesearch.dataset import get_imdb_dataset
from pyimagesearch import config


# Load the pre-trained RNN and LSTM models and compile them.
print('[INFO] Loading the pre-trained RNN model...')

modelRNN = keras.models.load_model(filepath=config.RNN_MODEL_PATH)

modelRNN.compile(
    optimizer='adam',
    metrics=['accuracy'],
    loss=keras.losses.BinaryCrossentropy(from_logits=False)
)

# print('[INFO] Loading the pre-trained LSTM model...')

# modelLSTM = keras.models.load_model(filepath=config.LSTM_MODEL_PATH)

# modelLSTM.compile(
#     optimizer='adam',
#     metrics=['accuracy'],
#     loss=keras.losses.BinaryCrossentropy(from_logits=False)
# )

# Load the IMDB Reviews test dataset.
print('[INFO] Getting the IMDB test dataset...')

testDs = get_imdb_dataset(
    folderName=config.DATASET_PATH,
    batchSize=config.BATCH_SIZE,
    bufferSize=config.BUFFER_SIZE,
    autotune=tf.data.AUTOTUNE,
    test=True
)

# Load the pre-trained text vectorization layer.
vectorizeLayer = load_vectorizer(
    name=config.TEXT_VEC_PATH,
    maxTokens=config.VOCAB_SIZE,
    outputLength=config.MAX_SEQUENCE_LENGTH,
    standardize=custom_standardization
)

# Vectorize the test dataset.
testDs = testDs.map(lambda text, label: (vectorizeLayer(text), label))

# Evaluate the trained models.
print('[INFO] Test evaluation for the RNN model:')

testLoss, testAccuracy = modelRNN.evaluate(testDs)

print(f'\n[INFO] test loss: {testLoss:0.2f}')
print(f'\n[INFO] test accuracy: {testAccuracy * 100:0.2f}%')
