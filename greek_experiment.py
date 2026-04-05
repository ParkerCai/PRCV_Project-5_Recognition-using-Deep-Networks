"""
Parker Cai
Jenny Nguyen
March 28, 2026

CS 5330 - Project 5: Recognition using Deep Networks
Extension: test different learning rates, batch sizes, and epoch counts on greek letter transfer learning
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
import csv
import os

from train import Network


# same transform as greek.py
class GreekTransform:
    def __init__(self):
        pass

    def __call__(self, x):
        x = torchvision.transforms.functional.rgb_to_grayscale(x)
        x = torchvision.transforms.functional.affine(x, 0, (0, 0), 36/128, 0)
        x = torchvision.transforms.functional.center_crop(x, (28, 28))
        x = torchvision.transforms.functional.invert(x)
        return x


# load greek letter images with a given batch size
def load_greek_data(path, batch_size=5):
    return torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(
            path,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.Resize((128, 128)),
                torchvision.transforms.ToTensor(),
                GreekTransform(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,))
            ])
        ),
        batch_size=batch_size,
        shuffle=True
    )


# load pretrained MNIST model, freeze weights, swap last layer for n_classes outputs
def build_model(n_classes=6):
    model = Network()
    model.load_state_dict(torch.load("results/mnist_model.pth", weights_only=True))
    for param in model.parameters():
        param.requires_grad = False
    model.fc2 = nn.Linear(50, n_classes)
    return model


# one epoch of training
def train_epoch(model, loader, optimizer):
    model.train()
    total_loss = 0
    for data, target in loader:
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


# check accuracy on training set
def get_accuracy(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in loader:
            pred = model(data).argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += len(target)
    return 100.0 * correct / total


# run one experiment with given hyperparams, return final accuracy and epoch where 100% was hit
def run_experiment(greek_path, lr, batch_size, n_epochs, n_classes=6):
    loader = load_greek_data(greek_path, batch_size=batch_size)
    model = build_model(n_classes)
    optimizer = torch.optim.Adam(model.fc2.parameters(), lr=lr)

    epochs_to_100 = None
    final_acc = 0

    for epoch in range(1, n_epochs + 1):
        train_epoch(model, loader, optimizer)
        acc = get_accuracy(model, loader)
        if acc >= 100.0 and epochs_to_100 is None:
            epochs_to_100 = epoch
            break
        final_acc = acc

    if epochs_to_100 is not None:
        final_acc = 100.0

    if epochs_to_100:
        print(f"  lr={lr}  batch={batch_size}  epochs={n_epochs}  -> {final_acc:.1f}%  (hit 100% at epoch {epochs_to_100})")
    else:
        print(f"  lr={lr}  batch={batch_size}  epochs={n_epochs}  -> {final_acc:.1f}%  (never hit 100%)")

    return final_acc, epochs_to_100


def main(argv):
    os.makedirs("results", exist_ok=True)
    greek_path = argv[1] if len(argv) > 1 else "greek_train"
    results = []

    default_lr = 0.005
    default_bs = 5
    default_epochs = 50
    n_classes = 6

    # round 1: vary learning rate, hold batch and epochs fixed
    print("\nRound 1: varying learning rate")
    for lr in [0.0001, 0.001, 0.005, 0.01]:
        acc, ep = run_experiment(greek_path, lr, default_bs, default_epochs, n_classes)
        results.append({"lr": lr, "batch_size": default_bs, "n_epochs": default_epochs,
                        "final_acc": acc, "epochs_to_100": ep})

    best_lr = max(results, key=lambda x: x["final_acc"])["lr"]
    print(f"  best lr: {best_lr}")

    # round 2: vary batch size, hold best lr and epochs fixed
    print("\nRound 2: varying batch size")
    for bs in [3, 5, 9]:
        acc, ep = run_experiment(greek_path, best_lr, bs, default_epochs, n_classes)
        results.append({"lr": best_lr, "batch_size": bs, "n_epochs": default_epochs,
                        "final_acc": acc, "epochs_to_100": ep})

    best_bs = max([r for r in results if r["lr"] == best_lr], key=lambda x: x["final_acc"])["batch_size"]
    print(f"  best batch size: {best_bs}")

    # round 3: vary number of epochs with best lr and batch size
    print("\nRound 3: varying number of epochs")
    for ep in [10, 25, 50]:
        acc, epochs_to_100 = run_experiment(greek_path, best_lr, best_bs, ep, n_classes)
        results.append({"lr": best_lr, "batch_size": best_bs, "n_epochs": ep,
                        "final_acc": acc, "epochs_to_100": epochs_to_100})

    # save to csv
    with open("results/greek_experiment_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["lr", "batch_size", "n_epochs", "final_acc", "epochs_to_100"])
        writer.writeheader()
        writer.writerows(results)
    print(f"\nsaved {len(results)} experiments to results/greek_experiment_results.csv")

    # plots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    r1 = sorted([(r["lr"], r["final_acc"]) for r in results
                 if r["batch_size"] == default_bs and r["n_epochs"] == default_epochs])
    axes[0].plot([str(x[0]) for x in r1], [x[1] for x in r1], marker="o", color="blue")
    axes[0].set_xlabel("Learning Rate")
    axes[0].set_ylabel("Final Accuracy (%)")
    axes[0].set_title("Accuracy vs Learning Rate")
    axes[0].set_ylim(0, 105)

    r2 = sorted([(r["batch_size"], r["epochs_to_100"] or r["n_epochs"]) for r in results
                 if r["lr"] == best_lr and r["n_epochs"] == default_epochs])
    axes[1].bar([str(x[0]) for x in r2], [x[1] for x in r2], color="green")
    axes[1].set_xlabel("Batch Size")
    axes[1].set_ylabel("Epochs to 100%")
    axes[1].set_title("Convergence Speed vs Batch Size")

    r3 = sorted([(r["n_epochs"], r["epochs_to_100"] or r["n_epochs"]) for r in results
                 if r["lr"] == best_lr and r["batch_size"] == best_bs])
    axes[2].bar([str(x[0]) for x in r3], [x[1] for x in r3], color="orange")
    axes[2].set_xlabel("Max Epochs")
    axes[2].set_ylabel("Epochs to 100%")
    axes[2].set_title("Convergence Speed vs Num Epochs")

    plt.suptitle("Greek Letter Transfer Learning Experiment")
    plt.tight_layout()
    plt.savefig("results/greek_experiment_results.png")
    plt.show()

    best = max(results, key=lambda x: x["final_acc"])
    print(f"\nbest: lr={best['lr']}, batch={best['batch_size']}, epochs={best['n_epochs']} -> {best['final_acc']:.1f}%")


if __name__ == "__main__":
    main(sys.argv)