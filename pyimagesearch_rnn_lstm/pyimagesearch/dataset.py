import tensorflow_datasets as tfds


def get_imdb_dataset(
    folderName,
    batchSize,
    bufferSize,
    autotune,
    test=False
):
    """
    Loads the IMDB Reviews datasets, distinguishing training from testing.
    """
    if test:
        # Load the test dataset.
        testDs = tfds.load(
            name='imdb_reviews',
            data_dir=folderName,
            as_supervised=True,
            shuffle_files=True,
            split='test'
        )

        # Divide the dataset into batches and pre-fetch it.
        testDs = testDs.batch(batchSize).prefetch(autotune)

        return testDs
    else:
        # Load the training and validation datasets.
        trainDs, valDs = tfds.load(
            name='imdb_reviews',
            data_dir=folderName,
            as_supervised=True,
            shuffle_files=True,
            split=['train[:90%]', 'train[90%:]']
        )

        # Take the training and validation datasets, suffle the samples,
        # divide them into batches and pre-fetch them.
        trainDs = (
            trainDs
            .shuffle(bufferSize)
            .batch(batchSize)
            .prefetch(autotune)
        )

        valDs = (
            valDs
            .shuffle(bufferSize)
            .batch(batchSize)
            .prefetch(autotune)
        )

        return trainDs, valDs
