"""Saving and loading training checkpoints."""
import os

import torch


def save_checkpoint(model, optimizer, epoch, loss, accuracy,
                    checkpoint_dir="checkpoints", filename=None):
    """Save model + optimizer state, epoch number and metrics.

    By default the file is named model_epoch_XXX.pth; pass `filename` to choose
    another name (e.g. "best_model.pth"). Returns the path of the saved file.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    if filename is None:
        filename = f"model_epoch_{epoch:03d}.pth"
    path = os.path.join(checkpoint_dir, filename)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        # Optimizer state (e.g. Adam's running averages) is needed to resume
        # training exactly where it stopped, not just the model weights.
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "accuracy": accuracy,
    }, path)
    return path


def load_checkpoint(model, optimizer, checkpoint_path, device):
    """Load a checkpoint into `model` (and `optimizer`, if given).

    Pass optimizer=None when you only need the model for predictions.
    Returns (epoch, loss, accuracy).
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["epoch"], checkpoint["loss"], checkpoint["accuracy"]
