"""
Model definitions: FCNN, CNN (Assignment 2 architecture) and EnhancedCNN.

All models take B x 3 x 64 x 64 images and output 10 class scores.

CNN shapes (B x C x H x W), using  out = (W + 2*P - (K - 1) - 1) / S + 1
  conv1 (16, k3, s1, p1) -> B x 16 x 64 x 64
  pool  (k2, s2)         -> B x 16 x 32 x 32
  conv2 (32, k3, s1, p1) -> B x 32 x 32 x 32
  pool  (k2, s2)         -> B x 32 x 16 x 16
  flatten                -> B x 8192
  fc1                    -> B x 100
  fc2                    -> B x 10
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGE_SIZE = 64  # the assignment specifies a 64 x 64 x 3 input

CLASSES = ("airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck")


class FCNN(nn.Module):
    """Fully connected network: flattens the image and uses only linear layers."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(3 * IMAGE_SIZE * IMAGE_SIZE, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.flatten(x)        # B x 12288
        x = F.relu(self.fc1(x))    # B x 512
        x = F.relu(self.fc2(x))    # B x 256
        return self.fc3(x)         # B x 10


class SimpleCNN(nn.Module):
    """Assignment 2 - Part 1 CNN architecture."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # after two poolings: 64 -> 32 -> 16, with 32 channels
        self.fc1 = nn.Linear(32 * 16 * 16, 100)
        self.fc2 = nn.Linear(100, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # B x 16 x 32 x 32
        x = self.pool(F.relu(self.conv2(x)))   # B x 32 x 16 x 16
        x = torch.flatten(x, 1)                # B x 8192
        x = F.relu(self.fc1(x))                # B x 100
        x = self.fc2(x)                        # B x 10
        return x


class EnhancedCNN(nn.Module):
    """Deeper CNN with batch normalization and dropout."""

    def __init__(self, num_classes: int = 10):
        super().__init__()

        def block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )

        self.features = nn.Sequential(
            block(3, 32),     # B x 32 x 32 x 32
            block(32, 64),    # B x 64 x 16 x 16
            block(64, 128),   # B x 128 x 8 x 8
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


MODELS = {"FCNN": FCNN, "CNN": SimpleCNN, "EnhancedCNN": EnhancedCNN}


def get_model(model_name):
    """Return a new model: one of "FCNN", "CNN", "EnhancedCNN"."""
    if model_name not in MODELS:
        raise ValueError(f"Unknown model '{model_name}'. Choose from: {list(MODELS)}")
    model = MODELS[model_name]()
    return model


if __name__ == "__main__":
    # Quick shape check, layer by layer:  uv run python -m helper_lib.model
    model = SimpleCNN()
    x = torch.randn(4, 3, IMAGE_SIZE, IMAGE_SIZE)
    x = model.pool(F.relu(model.conv1(x))); print("after conv1+pool:", x.shape)
    x = model.pool(F.relu(model.conv2(x))); print("after conv2+pool:", x.shape)
    x = torch.flatten(x, 1);                print("after flatten:   ", x.shape)
    x = F.relu(model.fc1(x));               print("after fc1:       ", x.shape)
    x = model.fc2(x);                       print("after fc2:       ", x.shape)
