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
import torch.optim as optim  # optimizer for training
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


def train_epoch(
    model, train_loader, optimizer, epoch, train_losses, train_counter, log_interval=10
):
    """Train the model for one epoch with per-batch logging."""
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()  # reset gradients
        output = model(data)
        loss = F.nll_loss(output, target)  # negative log likelihood loss
        loss.backward()  # backpropagate the loss
        optimizer.step()  # update the weights using SGD
        if batch_idx % log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )
            train_losses.append(loss.item())
            train_counter.append(
                (batch_idx * 64) + ((epoch - 1) * len(train_loader.dataset))
            )


def test_epoch(model, test_loader, test_losses):
    """Evaluate the model on a dataset. Appends avg loss and returns accuracy."""
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction="sum").item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    accuracy = 100.0 * correct / len(test_loader.dataset)
    print(
        "\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss, correct, len(test_loader.dataset), accuracy
        )
    )
    return accuracy


def main(argv):
    """
    Main function to load MNIST data and plot the first 6 examples from the test set.
    Build network, train for 5 epochs, save model.
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
    # 2x3 grid of subplots 2d array axes
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

    # Task 1B: Build the network
    model = Network()
    print(model)

    # Task 1C: Train the model
    n_epochs = 5
    learning_rate = 0.01
    momentum = 0.5
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    # Initialize lists to store loss and counter values
    train_losses = []
    train_counter = []
    test_losses = []
    test_counter = [i * len(train_set) for i in range(n_epochs + 1)]

    # Evaluate once before training to see baseline with random weights
    test_epoch(model, test_loader, test_losses)
    # Train
    for epoch in range(1, n_epochs + 1):
        train_epoch(model, train_loader, optimizer, epoch, train_losses, train_counter)
        test_epoch(model, test_loader, test_losses)

    # Plot training curve (per-batch train loss + per-epoch test loss)
    fig = plt.figure()
    plt.plot(train_counter, train_losses, color="blue")
    plt.scatter(test_counter, test_losses, color="red")
    plt.legend(["Train Loss", "Test Loss"], loc="upper right")
    plt.xlabel("Number of training examples seen")
    plt.ylabel("Negative log likelihood loss")
    plt.tight_layout()
    plt.savefig("results/training_curves.png")
    plt.show()


if __name__ == "__main__":
    main(sys.argv)
