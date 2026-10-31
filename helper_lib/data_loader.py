"""Data loading: image preprocessing and CIFAR-10 data loaders."""
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from .model import IMAGE_SIZE


def get_transform():
    """Preprocessing shared by training and the API.

    CIFAR-10 images are 32x32; resize to 64x64 to match the assignment architecture.
    """
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


def get_data_loader(data_dir, batch_size=32, train=True):
    """Return a DataLoader for the CIFAR-10 train split (train=True) or test split.

    data_dir is the folder CIFAR-10 is downloaded to / read from (e.g. "data").
    """
    dataset = datasets.CIFAR10(root=data_dir, train=train, download=True,
                               transform=get_transform())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=train)
    return loader


def get_train_val_loaders(data_dir, batch_size=32, val_size=5000, seed=42):
    """Split the CIFAR-10 training set into train and validation loaders.

    CIFAR-10 has no separate validation set, so we hold out `val_size` training
    images. The test set stays untouched for the final evaluation.
    """
    dataset = datasets.CIFAR10(root=data_dir, train=True, download=True,
                               transform=get_transform())
    generator = torch.Generator().manual_seed(seed)  # same split every run
    indices = torch.randperm(len(dataset), generator=generator).tolist()
    train_set = Subset(dataset, indices[val_size:])
    val_set = Subset(dataset, indices[:val_size])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader
