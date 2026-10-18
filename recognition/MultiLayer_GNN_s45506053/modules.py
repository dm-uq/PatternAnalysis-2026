"""
modules.py
Author: Derek McKinney (dm-uq), SN: 45506053

Contains neural network modules used by the project experiments, including
the feature-only MLP baseline and graph neural network (GNN) models.

References:

    PyTorch "Build the Neural Network" tutorial [1]:
    https://docs.pytorch.org/tutorials/beginner/basics/
        buildmodel_tutorial.html

"""
from torch import Tensor, nn


# REF: Adapted from the standard PyTorch `nn.Module` pattern demonstrated in
#  the PyTorch "Build the Neural Network" tutorial [1].
class MLPNetwork(nn.Module):
    """
    Feature-only MLP baseline for node classification, with configurable
    node-feature dimensions and dropout parameter.
    """
    def __init__(
            self,
            in_channels: int,
            hidden_channels: int,
            out_channels: int,
            dropout: float = 0.5  # library default
    ) -> None:
        """
        Initialise the MLP network.

        Args:
            in_channels: Number of input node features.
            hidden_channels: Number of units in the hidden layer.
            out_channels: Number of output classes.
            dropout: Probability of an element being zeroed during dropout.

        """
        super().__init__()

        self.lin1 = nn.Linear(in_channels, hidden_channels)
        self.act1 = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.lin2 = nn.Linear(hidden_channels, out_channels)

    def forward(self, x: Tensor) -> Tensor:
        """
        Perform a forward pass through the network.

        Args:
            x: Node-feature tensor of shape `(num_nodes, in_channels)`.

        Returns:
            Raw class logits of shape `(num_nodes, out_channels)`.

        """
        x = self.lin1(x)
        x = self.act1(x)
        x = self.dropout(x)
        return self.lin2(x)
