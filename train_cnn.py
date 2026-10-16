"""
Assignment 2 - Part 2: train the CNN on CIFAR-10 (same loop as Practical 3).
Saves weights to cnn_cifar10.pth, used by main.py.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from tqdm import tqdm

from cnn_model import SimpleCNN, IMAGE_SIZE

# ---- Device (same as the notebook) ----
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print("Using device:", device)

# ---- Data ----
# CIFAR-10 images are 32x32; resize to 64x64 to match the assignment architecture.
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

train_set = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
test_set = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=False)

print("Length of the train dataset:", len(train_set))
images, labels = next(iter(train_loader))
print("Shape of the first batch:", images.shape)  # [32, 3, 64, 64]

# ---- Model, loss, optimizer ----
model = SimpleCNN().to(device)
print(model)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ---- Training ----
num_epochs = 10
losses, steps = [], []

model.train()
for epoch in range(num_epochs):
    running_loss = 0.0
    loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")
    for i, (inputs, labels) in enumerate(loop):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 100 == 99:
            losses.append(running_loss / (i + 1))
            steps.append(epoch + (i + 1) / len(train_loader))
            loop.set_postfix(loss=running_loss / (i + 1))

print("Finished Training")

plt.plot(steps, losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig("training_loss.png")

# ---- Evaluation ----
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        predicted = model(inputs).argmax(dim=1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print(f"Accuracy of the network on the 10000 test images: {100 * correct / total:.2f}%")

# ---- Save for the API ----
torch.save(model.state_dict(), "cnn_cifar10.pth")
print("Saved model to cnn_cifar10.pth")
