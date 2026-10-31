import os

import torch
import torch.nn as nn
import torch.optim as optim

from helper_lib.data_loader import get_mnist_loader
from helper_lib.generator import generate_samples
from helper_lib.model import get_model
from helper_lib.trainer import train_gan
from helper_lib.utils import get_device, plot_gan_losses

device = get_device()
print("Using device:", device)

# Load data: 60,000 MNIST digits, scaled to [-1, 1]
data_loader = get_mnist_loader("data", batch_size=64)

# Initialize the GAN, the loss, and one optimizer per network
model = get_model("GAN")
criterion = nn.BCELoss()  # the discriminator outputs a real/fake probability
opt_gen = optim.Adam(model.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
opt_disc = optim.Adam(model.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Train
trained_model = train_gan(model, data_loader, criterion,
                          optimizer=(opt_gen, opt_disc), device=device, epochs=20)

plot_gan_losses(trained_model.history, "gan_losses.png")

# Save only the generator: it is all the API needs to create digits
os.makedirs("checkpoints", exist_ok=True)
torch.save(trained_model.generator.state_dict(), "checkpoints/gan_generator.pth")
print("Saved generator to checkpoints/gan_generator.pth")

# Show some generated digits (close the window to finish)
generate_samples(trained_model, device, num_samples=20, save_path="gan_samples.png")