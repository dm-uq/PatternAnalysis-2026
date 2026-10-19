"""
train.py
Author: Derek McKinney (dm-uq), SN: 45506053

Contains training and evaluation utilities used by the project experiments.

References: None

"""
from collections.abc import Callable

import torch
from torch import Tensor
from torch.optim import Optimizer

from modules import MLPNetwork, GCNNetwork


def train_mlp_epoch(
        model: MLPNetwork,
        features: Tensor,
        labels: Tensor,
        train_mask: Tensor,
        criterion: Callable[[Tensor, Tensor], Tensor],
        optimizer: Optimizer,
) -> float:
    """
    Train an MLP for one epoch using the selected training nodes.

    Args:
        model: MLP model to train.
        features: Node-feature tensor.
        labels: Node class-label tensor.
        train_mask: Boolean mask identifying training nodes.
        criterion: Loss function used for optimisation.
        optimizer: Optimizer used to update model parameters.

    Returns:
        Training loss for the epoch.

    """
    model.train()
    optimizer.zero_grad()  # clear gradients from the previous step

    logits = model(features)  # perform a forward pass

    # compute the loss using only the training nodes
    loss = criterion(
        logits[train_mask],
        labels[train_mask],
    )

    loss.backward()  # compute gradients via backpropagation
    optimizer.step()  # update model parameters

    return float(loss.item())


@torch.no_grad()
def evaluate_mlp(
        model: MLPNetwork,
        features: Tensor,
        labels: Tensor,
        mask: Tensor,
        criterion: Callable[[Tensor, Tensor], Tensor],
) -> tuple[float, float]:
    """
    Evaluate an MLP on the nodes selected by a mask.

    Args:
        model: MLP model to evaluate.
        features: Node-feature tensor.
        labels: Node class-label tensor.
        mask: Boolean mask identifying nodes to evaluate.
        criterion: Loss function used for evaluation.

    Returns:
        Evaluation loss and classification accuracy.

    """
    model.eval()  # switch to evaluation mode

    logits = model(features)  # perform a forward pass

    # select only the nodes included in the evaluation split
    masked_logits = logits[mask]
    masked_labels = labels[mask]

    loss = criterion(
        masked_logits,
        masked_labels,
    )

    # select the highest-scoring class for each node
    predictions = masked_logits.argmax(dim=1)

    # calculate the proportion of correctly classified nodes
    accuracy = (
            predictions == masked_labels
    ).float().mean()

    return float(loss.item()), float(accuracy.item())


def train_gcn_epoch(
        model: GCNNetwork,
        features: Tensor,
        edge_index: Tensor,
        labels: Tensor,
        train_mask: Tensor,
        criterion: Callable[[Tensor, Tensor], Tensor],
        optimizer: Optimizer,
) -> float:
    """
    Train a GCN for one epoch using the selected training nodes.

    Args:
        model: GCN model to train.
        features: Node-feature tensor.
        edge_index: Edge index tensor.
        labels: Node class-label tensor.
        train_mask: Boolean mask identifying training nodes.
        criterion: Loss function used for optimisation.
        optimizer: Optimizer used to update model parameters.

    Returns:
        Training loss for the epoch.

    """
    model.train()
    optimizer.zero_grad()  # clear gradients from the previous step

    logits = model(features, edge_index)  # perform a forward pass

    # compute the loss using only the training nodes
    loss = criterion(
        logits[train_mask],
        labels[train_mask],
    )

    loss.backward()  # compute gradients via backpropagation
    optimizer.step()  # update model parameters

    return float(loss.item())


@torch.no_grad()
def evaluate_gcn(
        model: GCNNetwork,
        features: Tensor,
        edge_index: Tensor,
        labels: Tensor,
        mask: Tensor,
        criterion: Callable[[Tensor, Tensor], Tensor],
) -> tuple[float, float]:
    """
    Evaluate a GCN on the nodes selected by a mask.

    Args:
        model: GCN model to evaluate.
        features: Node-feature tensor.
        edge_index: Edge index tensor.
        labels: Node class-label tensor.
        mask: Boolean mask identifying nodes to evaluate.
        criterion: Loss function used for evaluation.

    Returns:
        Evaluation loss and classification accuracy.

    """
    model.eval()  # switch to evaluation mode

    logits = model(features, edge_index)  # perform a forward pass

    # select only the nodes included in the evaluation split
    masked_logits = logits[mask]
    masked_labels = labels[mask]

    loss = criterion(
        masked_logits,
        masked_labels,
    )

    # select the highest-scoring class for each node
    predictions = masked_logits.argmax(dim=1)

    # calculate the proportion of correctly classified nodes
    accuracy = (
            predictions == masked_labels
    ).float().mean()

    return float(loss.item()), float(accuracy.item())
