import sys
import tensorflow as tf
from pyimagesearch.loss_accuracy import masked_accuracy, masked_loss
from pyimagesearch.translate import Translator
from tensorflow.keras.layers import TextVectorization
from tensorflow.keras.optimizers import Adam
from pyimagesearch import config
from pyimagesearch.dataset import (load_data, make_dataset, splitting_dataset,
    tf_lower_and_split_punct)
from pyimagesearch.rate_schedule import CustomSchedule
from pyimagesearch.transformer import Transformer

# Set random seed for reproducibility.
tf.keras.utils.set_random_seed(42)

# Load data.
print(f'[INFO] loading dta from {config.DATA_FNAME}...')

(source, target) = load_data(fname=config.DATA_FNAME)

# Split data into training, validation and test sets.
print(f'[INFO] splitting the dataset into train, val and test...')

(train, val, test) = splitting_dataset(source=source, target=target)

# TEST
train = (train[0][:5], train[1][:5])

# Source text processing (adapted on the training set).
print(f'[INFO] adapting the source text processor on the source dataset...')

sourceTextProcessor = TextVectorization(
    standardize=tf_lower_and_split_punct, max_tokens=config.SOURCE_VOCAB_SIZE
)
sourceTextProcessor.adapt(train[0])

# Target text processing (adapted on the training set).
print(f'[INFO] adapting the target text processor on the target dataset...')

targetTextProcessor = TextVectorization(
    standardize=tf_lower_and_split_punct, max_tokens=config.TARGET_VOCAB_SIZE
)
targetTextProcessor.adapt(train[1])

# Build the TensorFlow datasets for the train, validation and test sets.
print(f'[INFO] building TensorFlow Data input pipeline...')

trainDs = make_dataset(
    splits=train,
    batchSize=config.BATCH_SIZE,
    train=True,
    sourceTextProcessor=sourceTextProcessor,
    targetTextProcessor=targetTextProcessor
)

valDs = make_dataset(
    splits=val,
    batchSize=config.BATCH_SIZE,
    train=False,
    sourceTextProcessor=sourceTextProcessor,
    targetTextProcessor=targetTextProcessor
)

testDs = make_dataset(
    splits=test,
    batchSize=config.BATCH_SIZE,
    train=False,
    sourceTextProcessor=sourceTextProcessor,
    targetTextProcessor=targetTextProcessor
)

# Build the transformer model.
print(f'[INFO] building the transformer model...')

transformerModel = Transformer(
    encNumLayers=config.ENCODER_NUM_LAYERS,
    decNumLayers=config.DECODER_NUM_LAYERS,
    dModel=config.D_MODEL,
    numHeads=config.NUM_HEADS,
    dff=config.DFF,
    sourceVocabSize=config.SOURCE_VOCAB_SIZE,
    targetVocabSize=config.TARGET_VOCAB_SIZE,
    maximumPositionEncoding=config.MAX_POS_ENCODING,
    dropOutRate=config.DROP_RATE
)

# Compile the model.
print(f'[INFO] compiling the transformer model...')

# Instantiate the learning rate based on the custom schedule.
learningRate = CustomSchedule(dModel=config.D_MODEL)

# Instantiate the Adam optimizer.
optimizer = Adam(
    learning_rate=learningRate,
    beta_1=0.9,
    beta_2=0.98,
    epsilon=1e-9
)

transformerModel.compile(
    loss=masked_loss,
    optimizer=optimizer,
    metrics=[masked_accuracy]
)

# Train the model on the training dataset.
transformerModel.fit(
    trainDs,
    epochs=config.EPOCHS,
    validation_data=valDs
)

# Instantiate a Translator object containing the trained transformer model so
# we can then serialize it to reload it  later to translate new sentences (i.e.
# perform inference).
translator = Translator(
    sourceTextProcessor=sourceTextProcessor,
    targetTextProcessor=targetTextProcessor,
    transformer=transformerModel,
    maxLength=50
)

print('[INFO] serialize the inference translator to disk...')

tf.saved_model.save(
    obj=translator,
    export_dir='models'
)
