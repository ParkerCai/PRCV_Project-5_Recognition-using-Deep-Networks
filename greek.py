"""
Parker Cai
Jenny Nguyen
March 28, 2026

CS 5330 - Project 5: Recognition using Deep Networks
Transfer learning on Greek letters using the pretrained MNIST CNN
"""
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
from train import Network


# transforms greek images to look like MNIST
# expects subfolders named alpha, beta, gamma (ImageFolder handles the labels)
class GreekTransform:
    def __init__(self):
        pass

    def __call__(self, x):
        x = torchvision.transforms.functional.rgb_to_grayscale(x)
        x = torchvision.transforms.functional.affine(x, 0, (0, 0), 36/128, 0)
        x = torchvision.transforms.functional.center_crop(x, (28, 28))
        x = torchvision.transforms.functional.invert(x)
        return x


# loads the greek letter dataset from a folder
def load_greek_data(path):
    greek_train = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(
            path,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.Resize((128, 128)),
                torchvision.transforms.ToTensor(),
                GreekTransform(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,))
            ])
        ),
        batch_size=5,
        shuffle=True
    )
    return greek_train


# loads the pretrained MNIST model and freezes everything except the last layer
def build_greek_model():
    model = Network()
    model.load_state_dict(torch.load("results/mnist_model.pth", weights_only=True))

    for param in model.parameters():
        param.requires_grad = False

    # swap out the last layer for one with 3 outputs instead of 10
    model.fc2 = nn.Linear(50, 3)
    print(model)
    return model

# load and test our own handwritten greek letters
def evaluate_own_greek(model, own_path):
    class_names = ["alpha", "beta", "gamma"]

    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((128, 128)),
        torchvision.transforms.ToTensor(),
        GreekTransform(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ])

    dataset = torchvision.datasets.ImageFolder(own_path, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    print("\nOwn greek letter results:")
    print("-" * 40)

    images, true_labels, pred_labels = [], [], []
    correct = 0

    model.eval()
    with torch.no_grad():
        for data, target in loader:
            output = model(data)
            pred = output.argmax(dim=1).item()
            true = target.item()
            mark = "correct" if pred == true else "wrong"
            print(f"  {class_names[true]} -> predicted {class_names[pred]}  ({mark})")
            if pred == true:
                correct += 1
            images.append(data.squeeze())
            true_labels.append(true)
            pred_labels.append(pred)

    print(f"\n{correct}/{len(dataset)} correct")

    cols = 5
    rows = (len(images) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows))
    for i, ax in enumerate(axes.flat):
        if i < len(images):
            ax.imshow(images[i], cmap="gray")
            color = "green" if pred_labels[i] == true_labels[i] else "red"
            ax.set_title(
                f"Pred: {class_names[pred_labels[i]]}\nTrue: {class_names[true_labels[i]]}",
                color=color, fontsize=9
            )
            ax.axis("off")
        else:
            ax.axis("off")
    plt.suptitle(f"Own Greek Letters — {correct}/{len(dataset)} correct")
    plt.tight_layout()
    plt.savefig("results/own_greek_predictions.png")
    plt.show()

def main(argv):
    greek_path = argv[1] if len(argv) > 1 else "greek_train"

    train_loader = load_greek_data(greek_path)
    model = build_greek_model()
 
    optimizer = torch.optim.Adam(model.fc2.parameters(), lr=0.005)

    losses = []
    n_epochs = 100

    for epoch in range(1, n_epochs + 1):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print(f"epoch {epoch} loss: {losses[-1]:.4f}")

        # check accuracy on training set
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in train_loader:
                output = model(data)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += len(target)
        accuracy = 100.0 * correct / total
        print(f"  accuracy: {correct}/{total} ({accuracy:.0f}%)")

        if accuracy >= 100.0:
            print(f"hit 100% at epoch {epoch}, stopping early")
            break

    plt.figure()
    plt.plot(losses, color="blue")
    plt.xlabel("Batch")
    plt.ylabel("Negative Log Likelihood Loss")
    plt.title("Greek Letter Training Loss")
    plt.tight_layout()
    plt.savefig("results/greek_training_loss.png")
    plt.show()

    torch.save(model.state_dict(), "results/greek_model.pth")
    print("saved model")

    # test on our own handwritten greek letters
    evaluate_own_greek(model, "handwritten_greeks")


if __name__ == "__main__":
    main(sys.argv)