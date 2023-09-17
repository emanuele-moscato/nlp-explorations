import matplotlib.pyplot as plt


def plot_loss_accuracy(history, filepath):
    """
    Plots the loss and accuracy (training and validation) and saves the plot
    to file.
    """
    plt.style.use('ggplot')

    fig, axs = plt.subplots(2, 1)

    axs[0].plot(history['loss'], label='train_loss')
    axs[0].plot(history['val_loss'], label='val_loss')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].legend()

    axs[1].plot(history['accuracy'], label='train_accuracy')
    axs[1].plot(history['val_accuracy'], label='val_accuracy')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Accuracy')
    axs[1].legend()

    plt.savefig(filepath, dpi=400, bbox_inches='tight')
