import random
import tensorflow as tf
import tensorflow_text as tf_text


# Module-level autotune.
_AUTO = tf.data.AUTOTUNE


def load_data(fname):
    """
    Loads the data contained in the file `fname`.
    """
    with open(fname, 'r', encoding='utf-8') as textFile:
        # Read text lines and put them into a list.
        lines = textFile.readlines()

        # Each line is a pair of tab-separated sentences, the first in one
        # language and the second in the other one.
        pairs = [line.split('\t')[:-1] for line in lines]

        # Random shuffling of the pairs to forget about the initial ordering.
        random.shuffle(pairs)

        # Separate the pairs into different lists (with analogous indexing).
        source = [src for src, _ in pairs]
        target = [trgt for _, trgt in pairs]

    return (source, target)


def splitting_dataset(source, target):
    """
    Splits the dataset into, training, validation and test datasets. Relative
    sizes are hard-coded:
        * Training data: 80%
        * Validation data: 10%
        * Test data: 10%
    """
    # Compute the size of the training and validation datasets.
    trainSize = int(len(source) * 0.8)
    valSize = int(len(source) * 0.1)

    # Split the input data into training, validation and test data.
    (trainSource, trainTarget) = (source[:trainSize], target[:trainSize])
    (valSource, valTarget) = (
        source[trainSize:trainSize + valSize],
        target[trainSize:trainSize + valSize]
    )
    (testSource, testTarget) = (
        source[trainSize + valSize:],
        target[trainSize + valSize:]
    )

    return (
        (trainSource, trainTarget),
        (valSource, valTarget),
        (testSource, testTarget)
    )


def make_dataset(
    splits, batchSize, sourceTextProcessor, targetTextProcessor, train=False
):
    """
    Builds a Tensorflow Dataset object starting from the data.
    The `sourceTextProcessor` and `targetTextProcessor` arguments are meant
    to be some variation of Keras' TextVectorization layer: they vectorize the
    text and generate the tokens with their IDs.
    """
    (source, target) = splits

    # Create the Tensorflow Dataset object.
    dataset = tf.data.Dataset.from_tensor_slices((source, target))

    def prepare_batch(source, target):
        """
        Prepares a batch in the dataset. This function will be applied batch
        by batch when building the dataset. The `(source, targetInput)` pair
        are the inputs to the model, while `targetOutput` contains the targets.
        This format is needed in training when calling the model's `fit`
        method.
        """
        source = sourceTextProcessor(source)
        targetBuffer = targetTextProcessor(target)

        # Slicing: the input will NOT contain the last token while the output
        # will contain from the second token to the end. This is basically the
        # right shift of the tokens that allows the model not to see what it
        # has to predict beforehand.
        targetInput = targetBuffer[:, :-1]
        targetOutput = targetBuffer[:, 1:]

        return (source, targetInput), targetOutput

    if train:
        # If we are in the training phase, shuffle the dataset, batch it,
        # prepare the batches with the function defined above and pre-fetch
        # the dataset.
        dataset = (
            dataset
            .shuffle(dataset.cardinality().numpy())
            .batch(batchSize)
            .map(prepare_batch, _AUTO)
            .prefetch(_AUTO)
        )
    else:
        # If we aren't in the training phase, we don't perform any shuffling.
        dataset = (
            dataset
            .batch(batchSize)
            .map(prepare_batch, _AUTO)
            .prefetch(_AUTO)
        )

    return dataset


def tf_lower_and_split_punct(text):
    """
    Manipulates a single sentence (e.g. by putting it in lower case, adding
    start and end tokens etc.).
    """
    # Split accented characters.
    text = tf_text.normalize_utf8(text, 'NFKD')

    # Put the whole text into lower case.
    text = tf.strings.lower(text)

    # Eliminate anything that is not an aphabetic character or a selected
    # punctuation symbol.
    text = tf.strings.regex_replace(text, '[^ a-z.?!,]', '')


    # Add spaces around punctuation.
    text = tf.strings.regex_replace(text, '[.?!,]', r'\0')

    # Strip whitespaces.
    text = tf.strings.strip(text)

    # Add `[START]` and `[END]` tokens at the start and at the end of the
    # sentence respectively.
    text = tf.strings.join(['[START]', text, '[END]'], separator=' ')

    return text
