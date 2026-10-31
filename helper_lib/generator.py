"""Generate images with a trained GAN."""
import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid


def sample_images(model, device, num_samples=10):
    """Return `num_samples` generated images (N x 1 x 28 x 28), pixels in [0, 1]."""
    gen = model.generator.to(device)
    gen.eval()  # use BatchNorm's learned statistics, not the batch's
    with torch.no_grad():
        noise = torch.randn(num_samples, model.z_dim, device=device)  # latent points
        images = gen(noise).cpu()
    return (images + 1) / 2  # undo the [-1, 1] normalization


def generate_samples(model, device, num_samples=10, save_path=None):
    """Generate `num_samples` points in the latent space, run the generator to
    construct the images, and plot the samples on a grid."""
    images = sample_images(model, device, num_samples)
    grid = make_grid(images, nrow=min(num_samples, 10), padding=2)
    plt.figure(figsize=(min(num_samples, 10), (num_samples + 9) // 10 + 0.5))
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis("off")
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()