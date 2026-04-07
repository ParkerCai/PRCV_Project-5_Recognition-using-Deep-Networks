"""
Parker Cai
Jenny Nguyen
March 28, 2026

CS 5330 - Project 5: Recognition using Deep Networks
Task 4: Re-implement the network using transformer layers
Template from: Bruce A. Maxwell and Andy Zhao, Spring 2026
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt


class NetConfig:
    def __init__(
        self,
        name="vit_base",
        dataset="mnist",
        patch_size=4,
        stride=2,
        embed_dim=48,  # D: token dimension (token size)
        depth=4,
        num_heads=8,
        mlp_dim=128,
        dropout=0.1,
        use_cls_token=False,
        epochs=15,
        batch_size=64,
        lr=1e-3,
        weight_decay=1e-4,
        seed=0,
        optimizer="adamw",
        device=None,
    ):

        # data set fixed attributes
        self.image_size = 28
        self.in_channels = 1
        self.num_classes = 10

        # variable things
        self.name = name
        self.dataset = dataset
        self.patch_size = patch_size
        self.stride = stride
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.dropout = dropout
        self.use_cls_token = use_cls_token
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.seed = seed
        self.optimizer = optimizer

        # auto-detect device if not specified
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        s = "Name,Dataset,PatchSize,Stride,Dim,Depth,Heads,MLPDim,Dropout,CLS,Epochs,Batch,LR,Decay,Seed,Optimizer,TestAcc,BestEpoch\n"
        s += "%s,%s,%d,%d,%d,%d,%d,%d,%0.2f,%s,%d,%d,%f,%f,%d,%s," % (
            self.name,
            self.dataset,
            self.patch_size,
            self.stride,
            self.embed_dim,
            self.depth,
            self.num_heads,
            self.mlp_dim,
            self.dropout,
            self.use_cls_token,
            self.epochs,
            self.batch_size,
            self.lr,
            self.weight_decay,
            self.seed,
            self.optimizer,
        )
        self.config_string = s

        return


# Patch Embedding class
#
# A Vision Transformer splits the image into small patches, then turns
# each patch into a token embedding.
class PatchEmbedding(nn.Module):
    """
    Converts an image into a sequence of patch embeddings.

    Input:
        x of shape (B, C, H, W)

    Output:
        tokens of shape (B, N, D)

    where:
        B = batch size
        N = number of patches (tokens)
        D = embedding dimension
    """

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        stride: int,
        in_channels: int,
        embed_dim: int,
    ) -> None:
        super().__init__()

        self.image_size = image_size
        self.patch_size = patch_size
        self.stride = stride
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        # - non-overlapping patches  (stride == patch_size)
        # - overlapping patches      (stride < patch_size)
        self.unfold = nn.Unfold(
            kernel_size=patch_size,
            stride=stride,
        )

        # Each extracted patch is flattened into one vector
        self.patch_dim = in_channels * patch_size * patch_size

        # After flattening a patch, project it into embedding space.
        self.proj = nn.Linear(self.patch_dim, self.embed_dim)

        # Precompute how many patches will be produced for this image setup
        self.num_patches = self._compute_num_patches()

    def _compute_num_patches(self) -> int:
        """
        Compute how many patches are extracted in total.

        Number of positions along one spatial dimension:
            ((image_size - patch_size) // stride) + 1

        Since the image is square and the patch is square, total patches are:
            positions_per_dim * positions_per_dim
        """
        positions_per_dim = ((self.image_size - self.patch_size) // self.stride) + 1
        return positions_per_dim * positions_per_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract patches and convert them to embeddings.

        Input:
            x shape = (B, C, H, W)

        Output:
            x shape = (B, N, D)
        """
        # Step 1: extract patches using nn.Unfold, the shape becomes (B, patch_dim, N)
        #   patch_dim = flattened size of one patch
        #   N = number of extracted patches
        x = self.unfold(x)

        # Step 2: move dimensions so each patch becomes one row/token.
        # Shape becomes: (B, N, patch_dim)
        x = x.transpose(1, 2)

        # Step 3: project each flattened patch into embedding space.
        # Shape becomes: (B, N, embed_dim)
        x = self.proj(x)

        return x


# The Transformer Network class
#
# network structure
#
# Patch embedding layer
# dropout
# Transformer layer (with dropout)
# Transformer layer (with dropout)
# Transformer layer (with dropout)
# Token averaging
# Linear layer w/GELU and dropout (FC layer = D^2 * 2)
# Fully connected output layer 10 nodes: softmax output
class NetTransformer(nn.Module):
    # the init method defines the layers of the network
    def __init__(self, config):

        # create all of the layers that have to store information
        super(NetTransformer, self).__init__()

        # make the patch embedding layer
        self.patch_embed = PatchEmbedding(
            image_size=config.image_size,
            patch_size=config.patch_size,
            stride=config.stride,
            in_channels=config.in_channels,
            embed_dim=config.embed_dim,
        )

        # how many tokens are there?
        num_tokens = self.patch_embed.num_patches
        print("Number of tokens: %d" % (num_tokens))

        # does it use a classifier token or a global average token?
        self.use_cls_token = config.use_cls_token

        # if it uses a classifier node, create a source for the node
        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
            total_tokens = num_tokens + 1
        else:  # no CLS token
            self.cls_token = None
            total_tokens = num_tokens

        # need to include a learned positional embedding, one for each token
        self.pos_embed = nn.Parameter(torch.zeros(1, total_tokens, config.embed_dim))
        self.pos_dropout = nn.Dropout(config.dropout)

        # Use the Torch Transformer Encoder Layer
        # transformer layer includes
        # multi-head self attention
        # feedforward network
        # layer normalization
        # residual connections
        # dropout
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.embed_dim,
            nhead=config.num_heads,
            dim_feedforward=config.mlp_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        # Create a stack of transformer layers to build an encoder
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=config.depth,
        )

        # final normalization layer prior to classification
        self.norm = nn.LayerNorm(config.embed_dim)

        # linear layer for classification (48->128->10)
        self.classifier = nn.Sequential(
            nn.Linear(config.embed_dim, config.mlp_dim),  # FC layer 1
            nn.GELU(),
            # nn.Dropout(config.dropout),  # optional
            nn.Linear(config.mlp_dim, config.num_classes),  # FC layer 2
        )

        return

    def _init_parameters(self) -> None:
        """
        initialize special parameters
        - positional embedding
        - optional CLS token
        """
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        if self.cls_token is not None:
            nn.init.trunc_normal_(self.cls_token, std=0.02)

    # execute a forward pass
    """
    Input x: (B, 1, 28, 28)

    Output: logits: (B, num_classes)
    """

    def forward(self, x):
        # execute the patch embedding layer
        x = self.patch_embed(x)

        # get the batch size (0 dimension of x)
        batch_size = x.size(0)

        # add the optional CLS token to the set
        if self.use_cls_token and self.cls_token is not None:
            cls_token = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat([cls_token, x], dim=1)

        # add the learnable positional embedding to each token
        x = x + self.pos_embed

        # run the dropout layer right after the patch embedding
        x = self.pos_dropout(x)

        # run the transformer encoder
        x = self.encoder(x)

        # either pool the tokens or use the cls token (first token)
        if self.use_cls_token:
            x = x[:, 0]  # classify based on the cls token
        else:
            x = x.mean(dim=1)  # classify using the mean token vector
            # (better than cls token approach according to Prof. Maxwell)

        # final normalization of the token to classify
        x = self.norm(x)

        # call the classification MLP
        x = self.classifier(x)

        # return the softmax of the output layer
        return F.log_softmax(x, dim=1)


def train_epoch(
    model,
    train_loader,
    optimizer,
    epoch,
    train_losses,
    train_counter,
    device,
    log_interval=10,
):
    """Train the transformer for one epoch with per-batch logging."""
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
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
                (batch_idx * len(data)) + ((epoch - 1) * len(train_loader.dataset))
            )


def test_epoch(model, test_loader, test_losses, device):
    """Evaluate the transformer. Appends avg loss and returns accuracy."""
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
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
    """Load MNIST, build transformer, train, evaluate, and plot results."""

    config = NetConfig()
    device = torch.device(config.device)
    print(f"Using device: {device}")

    torch.manual_seed(config.seed)

    # Load MNIST data
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    train_set = torchvision.datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=config.batch_size, shuffle=True
    )

    test_set = torchvision.datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1000, shuffle=False)

    # Build the transformer model
    model = NetTransformer(config).to(device)
    model._init_parameters()
    print(model)

    # Set up optimizer (AdamW as specified in config)
    if config.optimizer == "adamw":
        optimizer = optim.AdamW(
            model.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )
    else:
        optimizer = optim.SGD(model.parameters(), lr=config.lr, momentum=0.9)

    # Training loop
    train_losses = []
    train_counter = []
    test_losses = []
    test_counter = [i * len(train_set) for i in range(config.epochs + 1)]

    # Baseline evaluation before training
    test_epoch(model, test_loader, test_losses, device)

    best_acc = 0
    best_epoch = 0
    for epoch in range(1, config.epochs + 1):
        train_epoch(
            model, train_loader, optimizer, epoch, train_losses, train_counter, device
        )
        acc = test_epoch(model, test_loader, test_losses, device)
        if acc > best_acc:
            best_acc = acc
            best_epoch = epoch

    print(f"Best test accuracy: {best_acc:.2f}% at epoch {best_epoch}")
    print(config.config_string + f"{best_acc:.2f},{best_epoch}")

    # Plot training curves
    plt.figure()
    plt.plot(train_counter, train_losses, color="blue", zorder=1)
    plt.scatter(test_counter, test_losses, color="red", zorder=2)
    plt.legend(["Train Loss", "Test Loss"], loc="upper right")
    plt.xlabel("Number of training examples seen")
    plt.ylabel("Negative log likelihood loss")
    plt.title("Transformer Training Curves")
    plt.tight_layout()
    plt.savefig("results/transformer_training_curves.png")
    plt.show()

    # Save the model
    torch.save(model.state_dict(), "results/transformer_model.pth")
    print("Model saved to results/transformer_model.pth")


if __name__ == "__main__":
    main(sys.argv)
