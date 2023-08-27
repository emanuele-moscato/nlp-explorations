import tensorflow as tf
from tensorflow.keras.losses import SparseCategoricalCrossentropy


def masked_loss(label, prediction):
    """
    Computes the masked loss, i.e. the loss evaluated on the subset of the
    samples selected by the mask. The loss is averaged over the samples it's
    computed on (i.e. the batch excluding the samples filtered out by the
    mask).

    Parameters
    ----------
    label : tf.Tensor (guess)
        True labels for the samples.
    prediction : tf.Tensor (guess)
        Predicted logits (probabilities) for the samples.
    """
    # Define a boolean mask. This is the right syntax to mask the contributions
    # from the samples whose label is not 0.
    mask = (label != 0)

    # Instantiate a loss object.
    lossObject = SparseCategoricalCrossentropy(
        # Categorical cross-entropy is computed from the logits
        # (probabilities), rather than from each sample's predicted label.
        from_logits=True,
        reduction='none'
    )
    # Compute the loss on all the samples in the batch: this is a tensor with
    # shape (n_samples_in_batch).
    loss = lossObject(label, prediction)

    # Cast the mask for the loss to the same dtype as the loss itself.
    mask = tf.cast(mask, dtype=loss.dtype)

    # Mask the loss: all the entries in the loss tensor corresponding to
    # samples that are filtered out by the mask are set to zero so they don't
    # contribute to the average.
    loss *= mask

    # Compute the average of the loss values over the samples they were
    # computed over.
    loss = tf.reduce_sum(loss) / tf.reduce_sum(mask)

    return loss


def masked_accuracy(label, prediction):
    """
    Parameters
    ----------
    label : tf.Tensor (guess)
        True labels for the samples.
    prediction : tf.Tensor (guess)
        Predicted logits (probabilities) for the samples.
    """
    # Same mask as in the loss.
    mask = (label != 0)

    # The prediction is given by the logits (probabilities): to compute the
    # accuracy we consider as the predicted labels the one with the highest
    # predicted probability (as usual). This is the predicted label.
    prediction = tf.argmax(prediction, axis=2)

    # Cast the true labels to the same dtype as the prediction's.
    label = tf.cast(label, dtype=prediction.dtype)

    # Compute the matching labels.
    match = (label == prediction)

    # Mask the matches to only the samples selected by the mask.
    match = (match & mask)

    # Cast the matches and the mask to the correct dtype.
    match = tf.cast(match, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)

    # Up to now the matches are computed sample by sample for each sample in
    # the batch selected by the mask. Now we take the average of these values.
    match = tf.reduce_sum(match) / tf.reduce_sum(mask)

    return match
