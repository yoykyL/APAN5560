"""
Assignment 2 - Part 2: train the CNN on CIFAR-10 using helper_lib.

Saves a checkpoint every epoch to checkpoints/, and the best model
(highest validation accuracy) to checkpoints/best_model.pth, used by main.py.

Run:  uv run python train_cnn.py
"""
import torch.nn as nn
import torch.optim as optim

from helper_lib.checkpoints import load_checkpoint
from helper_lib.data_loader import get_data_loader, get_train_val_loaders
from helper_lib.evaluator import evaluate_model
from helper_lib.model import get_model
from helper_lib.trainer import train_model
from helper_lib.utils import get_device, plot_history

device = get_device()
print("Using device:", device)

# Load data: 45,000 train / 5,000 validation images, plus the 10,000 test images
train_loader, val_loader = get_train_val_loaders("data", batch_size=64)
test_loader = get_data_loader("data", batch_size=64, train=False)

# Initialize model and training components
model = get_model("CNN")
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Option 1: Train from scratch with checkpoint saving
trained_model = train_model(
    model, train_loader, val_loader, criterion, optimizer,
    device=device, epochs=10, checkpoint_dir="checkpoints",
)

# Option 2: Resume training from a checkpoint (comment out Option 1 first)
# start_epoch, _, _ = load_checkpoint(model, optimizer, "checkpoints/model_epoch_005.pth", device)
# trained_model = train_model(model, train_loader, val_loader, criterion, optimizer,
#                             device=device, epochs=5, start_epoch=start_epoch)

plot_history(trained_model.history, "training_history.png")

# Evaluate the best model (highest validation accuracy) on the test set
best_epoch, _, best_val_acc = load_checkpoint(model, None, "checkpoints/best_model.pth", device)
avg_loss, accuracy = evaluate_model(model, test_loader, criterion, device)
print(f"Best model from epoch {best_epoch} (val acc {best_val_acc:.2f}%)")
print(f"Test Loss: {avg_loss:.4f}  Test Accuracy: {accuracy:.2f}%")
