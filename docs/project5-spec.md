# Project 5: Recognition using Deep Networks

## Overview

This project is about learning how to build, train, analyze, and modify a deep network for a recognition task. We will be using the MNIST digit recognition data set, primarily because it is simple enough to build and train a network without a GPU, but also because it is challenging enough to provide a good example of what deep networks can do.

As this is the last defined project. The last task of this project is to propose and design your final project. I would suggest thinking about this earlier than later and discussing it with me in office hours.

## Tasks

### Task 1: Build and train a network to recognize digits

The first task is to build and train a network to do digit recognition using the MNIST data base, then save the network to a file so it can be re-used for the later tasks. My strong recommendation is to use the python package pyTorch (torch) and the associated torchvision package.

- [PyTorch Home Page](https://pytorch.org)
- [pyTorch Tutorial (Fashion MNIST)](https://pytorch.org/tutorials/) - use MNIST digit data set instead
- [MNIST digits tutorial](https://nextjournal.com/gkoehler/pytorch-mnist)

**Code structure required:**

```python
# Your name here and a short header

# import statements
import sys

# class definitions
class MyNetwork(nn.Module):
    def __init__(self):
        pass

    # computes a forward pass for the network
    # methods need a summary comment
    def forward(self, x):
        return x

# useful functions with a comment for each function
def train_network(arguments):
    return

# main function (yes, it needs a comment too)
def main(argv):
    # handle any command line arguments in argv
    # main function code
    return

if __name__ == "__main__":
    main(sys.argv)
```

**Do not use a Jupyter notebook for this assignment.**

#### A. Get the MNIST digit data set

- Training set: 60k 28x28 labeled digits
- Test set: 10k 28x28 labeled digits
- Import via `torchvision.datasets.MNIST`
- Plot the first six example digits of the test set (don't shuffle test set)
- **Report:** Include a plot of the first six example digits from the test set

#### B. Build a CNN network model

Create a network with the following layers:
1. A convolution layer with 10 5x5 filters
2. A max pooling layer with a 2x2 window and a ReLU function applied
3. A convolution layer with 20 5x5 filters
4. A dropout layer with a 0.5 dropout rate (50%)
5. A max pooling layer with a 2x2 window and a ReLU function applied
6. A flattening operation followed by a fully connected Linear layer with 50 nodes and a ReLU function on the output
7. A final fully connected Linear layer with 10 nodes and the log_softmax function applied to the output

**Report:** Put a diagram of your network. A photo of a hand-drawn network is fine.

#### C. Train the model

- Train for at least five epochs, one epoch at a time
- Evaluate on both training and test sets after each epoch
- Pick a batch size of your choice
- **Report:** Plot of training and testing error/accuracy

#### D. Save the network to a file

#### E. Read the network and run it on the test set

In a separate python file:

- Read the network and run on first 10 test examples
- Set network to evaluation mode (`network.eval()`)
- Print the 10 network output values (2 decimal places), index of max, and correct label
- Plot the first 9 digits as a 3x3 grid with predictions above each
- **Report:** Table/screenshot of printed values + plot of first 9 digits

#### F. Test the network on new inputs

- Write out digits [0-9] in your own handwriting on white paper (use thick lines/marker)
- Take a picture, crop each digit to its own square image
- Convert to greyscale, resize to 28x28
- Match intensities to MNIST data (check if digits are white-on-black or black-on-white, may need to invert)
- Run through the network
- **Report:** Show digits and their classified results

### Task 2: Examine your network

Separate code file. Read in trained network and print the model.

#### A. Analyze the first layer

- Get weights of first layer (`model.conv1.weight`)
- Shape should be [10, 1, 5, 5] (10 filters, 1 channel, 5x5)
- Print filter weights and shape
- Visualize the ten filters using pyplot (3x4 grid)

#### B. Show the effect of the filters

- Use OpenCV's `filter2D` to apply 10 filters to first training example
- Plot 10 filtered images
- Use `with torch.no_grad():` when working with weights
- **Report:** Include plot and note whether results make sense given the filters

### Task 3: Transfer Learning on Greek Letters

Re-use MNIST network to recognize alpha, beta, and gamma.

Steps:
1. Generate the MNIST network (import from task 1)
2. 
3. Read existing model and load pre-trained weights
4. Freeze all network weights
5. Replace last layer with new Linear layer with 3 nodes

```python
# freezes the parameters for the whole network
for param in network.parameters():
    param.requires_grad = False
```

Greek letter transform:

```python
class GreekTransform:
    def __init__(self):
        pass

    def __call__(self, x):
        x = torchvision.transforms.functional.rgb_to_grayscale(x)
        x = torchvision.transforms.functional.affine(x, 0, (0,0), 36/128, 0)
        x = torchvision.transforms.functional.center_crop(x, (28, 28))
        return torchvision.transforms.functional.invert(x)
```

DataLoader:

```python
greek_train = torch.utils.data.DataLoader(
    torchvision.datasets.ImageFolder(training_set_path,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            GreekTransform(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,))
        ])),
    batch_size=5,
    shuffle=True)
```

- How many epochs to (almost) perfectly identify the 27 examples?
- Take pictures of your own alpha, beta, gamma symbols (thick lines, ~128x128)
- **Report:** Training error plot, modified network printout, results on additional data

### Task 4: Re-implement the network using transformer layers

Four things needed:

1. **Create tokens from image** by dividing into patches (can overlap), each patch through a linear layer
2. **Create the transformer encoder** - define parameters and number of layers
3. **Generate a single token** from encoder output (average all tokens OR add CLS token)
4. **Update classification layer** - one linear layer + ReLU (+ optional dropout) + final linear layer with output classes

Template available for NetTransformer class (forward method needs implementation).

**Report:** Results of running the transformer model with default settings.

### Task 5: Design your own experiment

Evaluate effect of changing network architecture. Pick at least three dimensions:

Potential dimensions:

- Number of convolution or transformer layers
- Convolution stride or transformer patch stride
- Size of convolution filters or transformer patches
- Number of convolution filters or transformer heads
- Number of hidden nodes in Dense layer or transformer dimension/MLP size
- Dropout rates
- Additional/removed dropout layers
- Pooling layer filter sizes (CNN only)
- Number/location of pooling layers (CNN only)
- Activation function for each layer
- Number of epochs
- Batch size

**A. Plan:** Evaluate 50-100 network variations. Use linear search strategy (hold two constant, optimize third, round-robin). Automate the process.

**B. Predict:** Hypothesize how the network will behave along each dimension before running.

**C. Execute:** Run evaluation and report results.

**D. Report:** Hypotheses, evaluation results, discussion of whether results supported hypotheses.

## Extensions

- Evaluate more dimensions on task 3
- Try more greek letters than alpha, beta, gamma
- Explore a different CV task with available data
- Load a pre-trained PyTorch network and evaluate its first couple of convolutional layers as in task 2
- Replace first layer with custom filter bank (e.g. Gabor filters), retrain rest, holding the first layer constant. How does it do?
- More sophisticated patch embedding or classification layers for Transformer
- Build a live video digit recognition application (w/ GPU)

## Report Requirements

- Short project description (200 words or less)
- Required images with short descriptions
- Extension descriptions and example images
- Short reflection of what you learned
- Acknowledgement of materials/people consulted

## Submission

Submit: `.py` files, PDF report, and `readme.txt` (or `readme.md`).
