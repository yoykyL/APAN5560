"""Small utilities: device selection and plotting."""
import matplotlib.pyplot as plt
import torch


def get_device():
    """Use the GPU if available (CUDA, or MPS on Apple Silicon), else the CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def plot_history(history, path="training_history.png"):
    """Plot train/validation loss and accuracy per epoch and save to `path`."""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(epochs, history["train_loss"], label="train")
    ax1.plot(epochs, history["val_loss"], label="validation")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss"); ax1.legend()
    ax2.plot(epochs, history["train_acc"], label="train")
    ax2.plot(epochs, history["val_acc"], label="validation")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy (%)"); ax2.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)

def plot_gan_losses(history, path="gan_losses.png"):
    """Plot the discriminator (D) and generator (G) losses and save to `path`."""
    fig = plt.figure()
    plt.plot(history["step"], history["d_loss"], label="Discriminator (D) loss")
    plt.plot(history["step"], history["g_loss"], label="Generator (G) loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()
    fig.savefig(path)
    plt.close(fig)