"""
Parker Cai
Jenny Nguyen
March 28, 2026

CS 5330 - Project 5: Recognition using Deep Networks
Load trained CNN and evaluate on MNIST test set and handwritten digits
"""

import sys
import torch
import torchvision
import matplotlib.pyplot as plt

from train import Network


def main(argv):
    """Load saved model, run on first 10 test examples, plot first 9 with predictions."""

    # Load the MNIST test set (same transform as training)
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    test_set = torchvision.datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )

    # Task 1E: Read the network and run it on the test set

    # Load trained model weights
    model = Network()
    model.load_state_dict(torch.load("results/mnist_model.pth", weights_only=True))
    model.eval()  # set to evaluation mode, each value * (1 - dropout rate)
    # so the same pattern will generate the same output each time. 

    # Run on the first 10 test examples
    print("Output values (log-probabilities), predicted label, and correct label:")
    print("-" * 80)
    for i in range(10):
        image, label = test_set[i]
        with torch.no_grad():
            output = model(image.unsqueeze(0))  # add batch dimension: (1,1,28,28)
        values = output.squeeze()  # remove batch dimension
        pred = values.argmax().item()

        # Print all 10 output values to 2 decimal places
        values_str = ", ".join(f"{v:.2f}" for v in values)
        print(f"Example {i}: [{values_str}]  Predicted: {pred}  Correct: {label}")

    # Plot the first 9 test digits in a 3x3 grid with predictions
    fig, axes = plt.subplots(3, 3, figsize=(8, 8))
    for i, ax in enumerate(axes.flat):
        image, label = test_set[i]
        with torch.no_grad():
            output = model(image.unsqueeze(0))
        pred = output.argmax(dim=1).item()

        ax.imshow(image.squeeze(), cmap="gray")
        ax.set_title(f"Pred: {pred} (Label: {label})")
        ax.axis("off")
    fig.suptitle("First 9 Test Digits with Predictions")
    plt.tight_layout()
    plt.savefig("results/first_nine_predictions.png")
    plt.show()


if __name__ == "__main__":
    main(sys.argv)
