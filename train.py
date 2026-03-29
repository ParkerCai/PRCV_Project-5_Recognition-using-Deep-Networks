"""
Parker Cai
Jenny Nguyen
March 28, 2026

CS 5330 - Project 5: Recognition using Deep Networks
Train a CNN on MNIST digit data
"""

import sys
import torch
import torchvision
import matplotlib.pyplot as plt


def main(argv):
    """
    Main function to load MNIST data and plot the first 6 examples from the test set.
    """

    # Task 1A: Load MNIST data

    # Normalize to match standard MNIST preprocessing (mean=0.1307, std=0.3081)
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    # Download and load training set (60k images, each 28x28 labeled digits)
    train_set = torchvision.datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)

    # Download and load test set (10k images), no shuffle so same outputs each time
    test_set = torchvision.datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1000, shuffle=False)

    # Plot the first 6 examples from the test set
    fig, axes = plt.subplots(2, 3, figsize=(8, 5))
    for i, ax in enumerate(axes.flat):
        image, label = test_set[i]
        ax.imshow(image.squeeze(), cmap="gray")
        ax.set_title(f"Label: {label}")
        ax.axis("off")
    fig.suptitle("First 6 MNIST Test Examples")
    plt.tight_layout()
    plt.savefig("results/first_six_test_examples.png")
    plt.show()

    print(f"Training set size: {len(train_set)}")
    print(f"Test set size:     {len(test_set)}")
    print(f"Image shape:       {train_set[0][0].shape}")


if __name__ == "__main__":
    main(sys.argv)
