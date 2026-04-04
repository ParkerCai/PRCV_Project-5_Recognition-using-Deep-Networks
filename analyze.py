"""
Parker Cai
Jenny Nguyen
March 28, 2026

CS 5330 - Project 5: Recognition using Deep Networks
Task 2: Examine the trained network's first convolutional layer
"""

import sys
import cv2
import torch
import torchvision
import matplotlib.pyplot as plt

from train import Network


def main(argv):
    """Load trained model, print it, visualize conv1 filters."""

    # Load trained model
    model = Network()
    model.load_state_dict(torch.load("results/mnist_model.pth", weights_only=True))
    model.eval()

    # Print the full model structure
    print(model)

    # Task 2A: Analyze the first layer

    with torch.no_grad():
        # Get the weights of the first convolutional layer
        # Shape: [10, 1, 5, 5] -> 10 filters, 1 input channel, 5x5 kernel
        # .detach() disconnects from the computation graph so numpy/matplotlib can use it
        weights = model.conv1.weight.detach()

    print(f"\nconv1 weight shape: {weights.shape}")
    print(f"\nconv1 weights:\n{weights}")

    # Visualize the 10 filters in a 3x4 grid
    fig, axes = plt.subplots(3, 4, figsize=(8, 6))
    for i, ax in enumerate(axes.flat):
        if i < 10:
            # Each filter is shape [1, 5, 5], squeeze to [5, 5] for display
            ax.imshow(weights[i].squeeze(), cmap="viridis")  # or "gray" for grayscale
            ax.set_title(f"Filter {i}")
        ax.axis("off")
    fig.suptitle("Conv1 Filters (10 learned 5x5 filters)")
    plt.tight_layout()
    plt.savefig("results/conv1_filters.png")
    plt.show()

    # Task 2B: Show the effect of the filters

    # Load the first training example (no normalization -- we want the raw image for filter2D)
    train_set = torchvision.datasets.MNIST(
        root='./data', train=True, download=True,
        transform=torchvision.transforms.ToTensor()
    )
    first_image = train_set[0][0].squeeze().numpy()  # shape: (28, 28)

    # Apply each of the 10 conv1 filters using OpenCV's filter2D
    with torch.no_grad():
        filters = model.conv1.weight.detach().numpy()  # shape: (10, 1, 5, 5)

    fig, axes = plt.subplots(3, 4, figsize=(10, 8))
    # Show the original image in the first cell
    axes.flat[0].imshow(first_image, cmap='gray')
    axes.flat[0].set_title('Original')
    axes.flat[0].axis('off')

    # Apply and plot each filter
    for i in range(10):
        ax = axes.flat[i + 1]
        kernel = filters[i].squeeze()  # shape: (5, 5)
        filtered = cv2.filter2D(first_image, -1, kernel)
        ax.imshow(filtered, cmap='gray')
        ax.set_title(f'Filter {i}')
        ax.axis('off')

    # Last cell is empty
    axes.flat[11].axis('off')

    fig.suptitle('Effect of Conv1 Filters on First Training Image')
    plt.tight_layout()
    plt.savefig('results/filtered_images.png')
    plt.show()


if __name__ == "__main__":
    main(sys.argv)
