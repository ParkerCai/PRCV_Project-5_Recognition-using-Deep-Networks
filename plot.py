import torchvision
import matplotlib.pyplot as plt

test_set = torchvision.datasets.MNIST("./data", train=False, download=True)

# Plot first example
plt.imshow(test_set[0][0], cmap="gray")
plt.axis("off")
plt.savefig("results/mnist_7.png", bbox_inches="tight", pad_inches=0)
