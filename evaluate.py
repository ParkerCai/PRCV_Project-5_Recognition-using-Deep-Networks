"""
Parker Cai
Jenny Nguyen
March 28, 2026

CS 5330 - Project 5: Recognition using Deep Networks
Load trained CNN and evaluate on MNIST test set and handwritten digits
"""

import sys
import os
import torch
import torchvision
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
from PIL import Image

from train import Network


# load and preprocess a handwritten digit image
# had to invert the image because our photos are black digit on white
# but MNIST is the opposite (white on black)
def load_handwritten_digit(path):
    img = Image.open(path).convert("L")
    img = img.resize((28, 28), Image.LANCZOS)
    tensor = TF.to_tensor(img)
    tensor = 1.0 - tensor 
    tensor = TF.normalize(tensor, (0.1307,), (0.3081,))
    return tensor

 # goes through the handwritten digit folder and tests each one
def evaluate_handwritten(model, digits_dir):
    digit_files = []
    for d in range(10):
        found = False
        for ext in ["png", "jpg", "jpeg"]:
            path = os.path.join(digits_dir, f"{d}.{ext}")
            if os.path.exists(path):
                digit_files.append((d, path))
                found = True
                break
        # just skip if we dont have that digit

    if len(digit_files) == 0:
        print("no images found in", digits_dir)
        return

    print("\nHandwritten digit results:")

    images = []
    true_labels = []
    pred_labels = []
    correct = 0

    for true_label, path in digit_files:
        tensor = load_handwritten_digit(path)
        with torch.no_grad():
            output = model(tensor.unsqueeze(0))
        pred = output.argmax(dim=1).item()
        if pred == true_label:
            print(f"  {true_label} -> predicted {pred}  (correct)")
            correct += 1
        else:
            print(f"  {true_label} -> predicted {pred}  (wrong)")
        images.append(tensor.squeeze())
        true_labels.append(true_label)
        pred_labels.append(pred)

    print(f"\ngot {correct} out of {len(digit_files)} right")

    # plot them in a grid
    cols = 5
    rows = (len(images) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(10, 4 * rows))
    axes = axes.flat  # makes it easier to loop through
    for i in range(len(axes)):
        ax = axes[i]  # not sure if there's a cleaner way to do this
        if i < len(images):
            ax.imshow(images[i], cmap="gray")
            if pred_labels[i] == true_labels[i]:
                color = "green"
            else:
                color = "red"
            ax.set_title(f"Pred: {pred_labels[i]}  True: {true_labels[i]}", color=color)
            ax.axis("off")
        else:
            ax.axis("off")
    plt.suptitle(f"Handwritten Digits ({correct}/{len(digit_files)} correct)")
    plt.tight_layout()
    plt.savefig("results/handwritten_predictions.png")
    plt.show()

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

    # task 1F - test on our handwritten digits
    if len(argv) > 1:
        digits_dir = argv[1]
    else:
        digits_dir = "handwritten_digits"
    evaluate_handwritten(model, digits_dir)

if __name__ == "__main__":
    main(sys.argv)
