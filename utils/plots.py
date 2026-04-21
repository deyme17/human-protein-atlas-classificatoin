import matplotlib.pyplot as plt


def visualize_training(history: dict[str, list[float]], save: bool = False) -> None:
    fig, ax = plt.subplots(2, 1, figsize=(10, 6))
    ax[0].plot(history['train_loss'], label='Train Loss')
    ax[0].plot(history['valid_loss'], label='Valid Loss')
    ax[0].set_title('Losses')
    ax[0].legend()
    ax[1].plot(history['f1_macro'], label='F1-Macro')
    ax[1].plot(history['f1_micro'], label='F1-Micro')
    ax[1].plot(history['f1_samples'], label='F1-Samples')
    ax[1].set_title('F1 Scores')
    ax[1].legend()
    plt.tight_layout()
    if save:
        fig.savefig("training_plot.png")
        plt.close(fig)
    else:
        plt.show()