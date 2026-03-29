"""
Parker Cai
Jenny Nguyen
March 28, 2026

CS 5330 - Project 5: Recognition using Deep Networks
Train a CNN on MNIST digit data
Referenced from: https://nextjournal.com/gkoehler/pytorch-mnist
"""

import sys
import torch
import torch.nn as nn  # neural network layers
import torch.nn.functional as F  # convolutional layers and max pooling
import torchvision  # MNIST dataset
import matplotlib.pyplot as plt


class Network(nn.Module):
    """CNN for MNIST digit recognition.

    Architecture:
    Input: 28x28x1 (grayscale image), 784 pixels input
        conv1 (1->10, 5x5) ->
        maxpool(2x2) + relu ->
        conv2 (10->20, 5x5) ->
        dropout(0.5) ->
        maxpool(2x2) + relu ->
        flatten -> fully connected Linear layer fc1 (320->50) + relu ->
        fully connected Linear layer fc2 (50->10) + log_softmax -> output
    Output: log-probability distribution over 10 digit classes
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.dropout = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        """Computes a forward pass through the network.

        Input: x with shape (batch, 1, 28, 28)
        Output: log-probability distribution over 10 digit classes
        """
        # 28x28 -> conv1 -> 24x24 -> maxpool -> 12x12
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        # 12x12 -> conv2 -> 8x8 -> dropout -> maxpool -> 4x4
        x = F.relu(F.max_pool2d(self.dropout(self.conv2(x)), 2))
        # 20 channels * 4 * 4 = 320
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.log_softmax(self.fc2(x), dim=1)
        return x  # output


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
    # Save the plot to the results directory
    plt.savefig("results/first_six_test_examples.png")
    plt.show()

    print(f"Training set size: {len(train_set)}")
    print(f"Test set size:     {len(test_set)}")
    print(f"Image shape:       {train_set[0][0].shape}")


if __name__ == "__main__":
    main(sys.argv)
