"""
Parker Cai
Jenny Nguyen
March 28, 2026

CS 5330 - Project 5: Recognition using Deep Networks
Task 5: Experiment with trying different filter/pool sizes to see what works best
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt
import csv
import os


# same as the original network but we made filter size, num filters, pool size
# all configurable so we can test different combos
class FlexNetwork(nn.Module):
    def __init__(self, filter_size=5, num_filters=(10, 20), pool_size=2):
        super().__init__()
        self.pool_size = pool_size
        f1, f2 = num_filters

        self.conv1 = nn.Conv2d(1, f1, kernel_size=filter_size)
        self.conv2 = nn.Conv2d(f1, f2, kernel_size=filter_size)
        self.dropout = nn.Dropout2d(p=0.5)

        # manually compute the flattened size after the two conv+pool layers
        s = (28 - filter_size + 1) // pool_size
        s = (s - filter_size + 1) // pool_size
        self.flat_size = f2 * s * s

        self.fc1 = nn.Linear(self.flat_size, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), self.pool_size))
        x = F.relu(F.max_pool2d(self.dropout(self.conv2(x)), self.pool_size))
        x = x.view(-1, self.flat_size)
        x = F.relu(self.fc1(x))
        x = F.log_softmax(self.fc2(x), dim=1)
        return x


def load_data(batch_size=64):
    # using same normalization values from class slides
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('./data', train=True, download=True, transform=transform),
        batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('./data', train=False, download=True, transform=transform),
        batch_size=1000, shuffle=False
    )
    return train_loader, test_loader


# trains one epoch
def train_epoch(model, train_loader, optimizer):
    model.train()
    for data, target in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()


def evaluate(model, test_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            pred = model(data).argmax(dim=1)
            correct += pred.eq(target).sum().item()
    return 100.0 * correct / len(test_loader.dataset)


def run_experiment(filter_size, num_filters, pool_size, train_loader, test_loader, n_epochs=5):
    # skip configs where the spatial size goes to 0 or negative after convs
    # not sure why pool=4 kept crashing so we just do 2 and 3 for now
    if filter_size not in [3, 5, 7] or pool_size not in [2, 3]:
        return None

    s = (28 - filter_size + 1) // pool_size
    s = (s - filter_size + 1) // pool_size
    if s <= 0:
        print(f"  skipping invalid config: filter={filter_size}, pool={pool_size} (spatial size collapses)")
        return None

    model = FlexNetwork(filter_size=filter_size, num_filters=num_filters, pool_size=pool_size)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    for _ in range(n_epochs):
        train_epoch(model, train_loader, optimizer)

    acc = evaluate(model, test_loader)
    print(f"  filter={filter_size}x{filter_size}  filters={num_filters}  pool={pool_size}x{pool_size}  -> {acc:.2f}%")
    return acc


def main(argv):
    os.makedirs("results", exist_ok=True)
    train_loader, test_loader = load_data()
    n_epochs = 5
    results = []

    # round 1: vary filter size, keep num_filters and pool fixed
    print("\nRound 1: varying conv filter size")
    fixed_filters = (10, 20)
    fixed_pool = 2
    for fs in [3, 5, 7]:
        acc = run_experiment(fs, fixed_filters, fixed_pool, train_loader, test_loader, n_epochs)
        if acc is not None:
            results.append({"filter_size": fs, "num_filters": str(fixed_filters), "pool_size": fixed_pool, "accuracy": acc})

    r1 = [r for r in results if r["pool_size"] == 2 and r["num_filters"] == str(fixed_filters)]
    best_fs = max(r1, key=lambda x: x["accuracy"])["filter_size"]
    print(f"  best filter size so far: {best_fs}x{best_fs}")

    # round 2: now vary num filters, hold best filter size from round 1
    print("\nRound 2: varying number of filters")
    for nf in [(8, 16), (10, 20), (16, 32), (32, 64)]:
        acc = run_experiment(best_fs, nf, fixed_pool, train_loader, test_loader, n_epochs)
        if acc is not None:
            results.append({"filter_size": best_fs, "num_filters": str(nf), "pool_size": fixed_pool, "accuracy": acc})

    r2 = [r for r in results if r["filter_size"] == best_fs and r["pool_size"] == 2]
    best_nf_str = max(r2, key=lambda x: x["accuracy"])["num_filters"]
    best_nf = eval(best_nf_str)
    print(f"  best num filters so far: {best_nf}")

    # round 3: vary pool size
    print("\nRound 3: varying pool size")
    for ps in [2, 3]:
        acc = run_experiment(best_fs, best_nf, ps, train_loader, test_loader, n_epochs)
        if acc is not None:
            results.append({"filter_size": best_fs, "num_filters": str(best_nf), "pool_size": ps, "accuracy": acc})

    r3 = [r for r in results if r["filter_size"] == best_fs and r["num_filters"] == str(best_nf)]
    best_ps = max(r3, key=lambda x: x["accuracy"])["pool_size"]
    print(f"  best pool size: {best_ps}x{best_ps}")

    # round 4: re-check filter size now that we have better num_filters and pool_size
    # wanted to see if the best filter size changes with different hyperparams
    print("\nRound 4: second pass on filter size")
    for fs in [3, 5, 7]:
        acc = run_experiment(fs, best_nf, best_ps, train_loader, test_loader, n_epochs)
        if acc is not None:
            results.append({"filter_size": fs, "num_filters": str(best_nf), "pool_size": best_ps, "accuracy": acc})

    # round 5: re-check num filters with the final pool and filter size
    print("\nRound 5: second pass on num filters")
    for nf in [(8, 16), (10, 20), (16, 32), (32, 64)]:
        acc = run_experiment(best_fs, nf, best_ps, train_loader, test_loader, n_epochs)
        if acc is not None:
            results.append({"filter_size": best_fs, "num_filters": str(nf), "pool_size": best_ps, "accuracy": acc})

    # write results out to csv
    with open("results/experiment_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filter_size", "num_filters", "pool_size", "accuracy"])
        writer.writeheader()
        writer.writerows(results)
    print(f"\nsaved {len(results)} experiments to results/experiment_results.csv")

    # plots - one per hyperparameter
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    fs_accs = sorted([(r["filter_size"], r["accuracy"]) for r in results
                      if r["num_filters"] == str(fixed_filters) and r["pool_size"] == 2])
    axes[0].plot([x[0] for x in fs_accs], [x[1] for x in fs_accs], marker="o", color="blue")
    axes[0].set_xlabel("Conv Filter Size")
    axes[0].set_ylabel("Test Accuracy (%)")
    axes[0].set_title("Accuracy vs Filter Size")
    axes[0].set_xticks([3, 5, 7])
    axes[0].set_ylim(97, 99)

    nf_accs = [(r["num_filters"], r["accuracy"]) for r in results
               if r["filter_size"] == best_fs and r["pool_size"] == 2]
    axes[1].bar([x[0] for x in nf_accs], [x[1] for x in nf_accs], color="green")
    axes[1].set_xlabel("Num Filters (conv1, conv2)")
    axes[1].set_ylabel("Test Accuracy (%)")
    axes[1].set_title("Accuracy vs Num Filters")
    axes[1].tick_params(axis="x", labelrotation=15)
    axes[1].set_ylim(97, 99)

    ps_accs = sorted([(r["pool_size"], r["accuracy"]) for r in results
                      if r["filter_size"] == best_fs and r["num_filters"] == str(best_nf)])
    axes[2].bar([str(x[0]) + "x" + str(x[0]) for x in ps_accs], [x[1] for x in ps_accs], color="orange")
    axes[2].set_xlabel("Pool Size")
    axes[2].set_ylabel("Test Accuracy (%)")
    axes[2].set_title("Accuracy vs Pool Size")
    axes[2].set_ylim(97, 99)

    plt.suptitle("Architecture Search Results")
    plt.tight_layout()
    plt.savefig("results/experiment_results.png")
    plt.show()

    best = max(results, key=lambda x: x["accuracy"])
    print(f"\nbest overall: filter={best['filter_size']}x{best['filter_size']}, "
          f"filters={best['num_filters']}, pool={best['pool_size']}x{best['pool_size']} "
          f"-> {best['accuracy']:.2f}%")


if __name__ == "__main__":
    main(sys.argv)