"""
modules.py
Author: Derek McKinney (dm-uq), SN: 45506053

Contains neural network modules used by the project experiments, including
the feature-only MLP baseline and graph neural network (GNN) models.

References:

    PyTorch "Build the Neural Network" tutorial [1]:
    https://docs.pytorch.org/tutorials/beginner/basics/
        buildmodel_tutorial.html

    PyG GCNConv documentation [2]:
    https://pytorch-geometric.readthedocs.io/en/latest/generated/
        torch_geometric.nn.conv.GCNConv.html

"""
from torch import Tensor, nn
from torch_geometric.nn.conv import GCNConv


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


# REF: Adapted from the PyG GCNConv API demonstrated in [2].
class GCNNetwork(nn.Module):
    """
    Two-layer GCN for node classification, with configurable node-feature
    dimensions and dropout parameter.
    """

    def __init__(
            self,
            in_channels: int,
            hidden_channels: int,
            out_channels: int,
            dropout: float = 0.5  # library default
    ) -> None:
        """
        Initialise the GCN network.

        Args:
            in_channels: Number of input node features.
            hidden_channels: Number of units in the hidden layer.
            out_channels: Number of output classes.
            dropout: Probability of an element being zeroed during dropout.

        """
        super().__init__()

        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.act1 = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(
            self,
            x: Tensor,
            edge_index: Tensor
    ) -> Tensor:
        """
        Perform a forward pass through the network.

        Args:
            x: Node-feature tensor of shape `(num_nodes, in_channels)`.
            edge_index: Graph connectivity tensor of shape `(2, num_edges)`.

        Returns:
            Raw class logits of shape `(num_nodes, out_channels)`.

        """
        x = self.conv1(x, edge_index)
        x = self.act1(x)
        x = self.dropout(x)
        return self.conv2(x, edge_index)


# REF: Adapted from the PyG GCNConv API demonstrated in [2].
class GCNNetwork3Layer(nn.Module):
    """
    Three-layer GCN for node classification, with configurable node-feature
    dimensions and dropout parameter.
    """

    def __init__(
            self,
            in_channels: int,
            hidden_channels: int,
            out_channels: int,
            dropout: float = 0.5  # library default
    ) -> None:
        """
        Initialise the 3-Layer GCN network.

        Args:
            in_channels: Number of input node features.
            hidden_channels: Number of units in the hidden layers.
            out_channels: Number of output classes.
            dropout: Probability of an element being zeroed during dropout.

        """
        super().__init__()

        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.act1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.act2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.conv3 = GCNConv(hidden_channels, out_channels)

    def forward(
            self,
            x: Tensor,
            edge_index: Tensor
    ) -> Tensor:
        """
        Perform a forward pass through the network.

        Args:
            x: Node-feature tensor of shape `(num_nodes, in_channels)`.
            edge_index: Graph connectivity tensor of shape `(2, num_edges)`.

        Returns:
            Raw class logits of shape `(num_nodes, out_channels)`.

        """
        x = self.conv1(x, edge_index)
        x = self.act1(x)
        x = self.dropout1(x)

        x = self.conv2(x, edge_index)
        x = self.act2(x)
        x = self.dropout2(x)

        return self.conv3(x, edge_index)


# REF: Adapted from the PyG GCNConv API demonstrated in [2].
class GCNNetwork4Layer(nn.Module):
    """
    Four-layer GCN for node classification, with configurable node-feature
    dimensions and dropout parameter.
    """

    def __init__(
            self,
            in_channels: int,
            hidden_channels: int,
            out_channels: int,
            dropout: float = 0.5  # library default
    ) -> None:
        """
        Initialise the 4-Layer GCN network.

        Args:
            in_channels: Number of input node features.
            hidden_channels: Number of units in the hidden layers.
            out_channels: Number of output classes.
            dropout: Probability of an element being zeroed during dropout.

        """
        super().__init__()

        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.act1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.act2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.act3 = nn.ReLU()
        self.dropout3 = nn.Dropout(dropout)

        self.conv4 = GCNConv(hidden_channels, out_channels)

    def forward(
            self,
            x: Tensor,
            edge_index: Tensor
    ) -> Tensor:
        """
        Perform a forward pass through the network.

        Args:
            x: Node-feature tensor of shape `(num_nodes, in_channels)`.
            edge_index: Graph connectivity tensor of shape `(2, num_edges)`.

        Returns:
            Raw class logits of shape `(num_nodes, out_channels)`.

        """
        x = self.conv1(x, edge_index)
        x = self.act1(x)
        x = self.dropout1(x)

        x = self.conv2(x, edge_index)
        x = self.act2(x)
        x = self.dropout2(x)

        x = self.conv3(x, edge_index)
        x = self.act3(x)
        x = self.dropout3(x)

        return self.conv4(x, edge_index)


# REF: Adapted from the PyG GCNConv API demonstrated in [2].
class GCNNetwork5Layer(nn.Module):
    """
    Five-layer GCN for node classification, with configurable node-feature
    dimensions and dropout parameter.
    """

    def __init__(
            self,
            in_channels: int,
            hidden_channels: int,
            out_channels: int,
            dropout: float = 0.5  # library default
    ) -> None:
        """
        Initialise the 5-Layer GCN network.

        Args:
            in_channels: Number of input node features.
            hidden_channels: Number of units in the hidden layers.
            out_channels: Number of output classes.
            dropout: Probability of an element being zeroed during dropout.

        """
        super().__init__()

        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.act1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.act2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.act3 = nn.ReLU()
        self.dropout3 = nn.Dropout(dropout)

        self.conv4 = GCNConv(hidden_channels, hidden_channels)
        self.act4 = nn.ReLU()
        self.dropout4 = nn.Dropout(dropout)

        self.conv5 = GCNConv(hidden_channels, out_channels)

    def forward(
            self,
            x: Tensor,
            edge_index: Tensor
    ) -> Tensor:
        """
        Perform a forward pass through the network.

        Args:
            x: Node-feature tensor of shape `(num_nodes, in_channels)`.
            edge_index: Graph connectivity tensor of shape `(2, num_edges)`.

        Returns:
            Raw class logits of shape `(num_nodes, out_channels)`.

        """
        x = self.conv1(x, edge_index)
        x = self.act1(x)
        x = self.dropout1(x)

        x = self.conv2(x, edge_index)
        x = self.act2(x)
        x = self.dropout2(x)

        x = self.conv3(x, edge_index)
        x = self.act3(x)
        x = self.dropout3(x)

        x = self.conv4(x, edge_index)
        x = self.act4(x)
        x = self.dropout4(x)

        return self.conv5(x, edge_index)


class GCNNetwork4LayerELU(nn.Module):
    def __init__(
            self,
            in_channels: int,
            hidden_channels: int,
            out_channels: int,
            dropout: float = 0.5,
    ) -> None:
        """
        Initialise a 4-Layer GCN network with ELU activation functions.
        Uses default alpha=1 parameter value.

        Args:
            in_channels: Number of input node features.
            hidden_channels: Number of units in the hidden layers.
            out_channels: Number of output classes.
            dropout: Probability of an element being zeroed during dropout.

        """
        super().__init__()

        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.act1 = nn.ELU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.act2 = nn.ELU()
        self.dropout2 = nn.Dropout(dropout)

        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.act3 = nn.ELU()
        self.dropout3 = nn.Dropout(dropout)

        self.conv4 = GCNConv(hidden_channels, out_channels)

    def forward(
            self,
            x: Tensor,
            edge_index: Tensor,
    ) -> Tensor:
        """
        Perform a forward pass through the network.

        Args:
            x: Node-feature tensor of shape `(num_nodes, in_channels)`.
            edge_index: Graph connectivity tensor of shape `(2, num_edges)`.

        Returns:
            Raw class logits of shape `(num_nodes, out_channels)`.

        """
        x = self.conv1(x, edge_index)
        x = self.act1(x)
        x = self.dropout1(x)

        x = self.conv2(x, edge_index)
        x = self.act2(x)
        x = self.dropout2(x)

        x = self.conv3(x, edge_index)
        x = self.act3(x)
        x = self.dropout3(x)

        return self.conv4(x, edge_index)


class GCNNetwork4LayerLReLU(nn.Module):
    def __init__(
            self,
            in_channels: int,
            hidden_channels: int,
            out_channels: int,
            dropout: float = 0.5,
    ) -> None:
        """
        Initialise a 4-Layer GCN network with Leaky ReLU activation functions.
        Uses default negative_slope=1e-2 parameter value.

        Args:
            in_channels: Number of input node features.
            hidden_channels: Number of units in the hidden layers.
            out_channels: Number of output classes.
            dropout: Probability of an element being zeroed during dropout.

        """
        super().__init__()

        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.act1 = nn.LeakyReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.act2 = nn.LeakyReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.act3 = nn.LeakyReLU()
        self.dropout3 = nn.Dropout(dropout)

        self.conv4 = GCNConv(hidden_channels, out_channels)

    def forward(
            self,
            x: Tensor,
            edge_index: Tensor,
    ) -> Tensor:
        """
        Perform a forward pass through the network.

        Args:
            x: Node-feature tensor of shape `(num_nodes, in_channels)`.
            edge_index: Graph connectivity tensor of shape `(2, num_edges)`.

        Returns:
            Raw class logits of shape `(num_nodes, out_channels)`.

        """
        x = self.conv1(x, edge_index)
        x = self.act1(x)
        x = self.dropout1(x)

        x = self.conv2(x, edge_index)
        x = self.act2(x)
        x = self.dropout2(x)

        x = self.conv3(x, edge_index)
        x = self.act3(x)
        x = self.dropout3(x)

        return self.conv4(x, edge_index)
