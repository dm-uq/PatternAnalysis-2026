"""
predict.py
Author: Derek McKinney (dm-uq), SN: 45506053

Contains prediction and evaluation utilities used to conduct project analysis.

Frozen Models:

    MLP:
        layers: 3
        hidden: 64
        dropout: 0.5
        activation: ReLU
        weight_decay: n/a for inference
        checkpoint: mlp_final.pt

    GCN:
        layers: 4
        hidden: 64
        dropout: 0.55
        activation: ReLU
        weight_decay: n/a for inference
        checkpoint: gcn_final.pt

References: None

"""

# IMPORTS

from pathlib import Path
from typing import Final, Literal, TypeAlias, cast

import torch

from torch_geometric.data import Data
from torch_geometric.datasets import FacebookPagePage

from dataset import add_stratified_masks, load_facebook_data
from modules import (
    MLPNetwork3Layer,
    GCNNetwork4Layer,
)

# CONSTANTS

ModelName: TypeAlias = Literal["mlp", "gcn"]

PredictionModel: TypeAlias = (MLPNetwork3Layer | GCNNetwork4Layer)

PROJECT_DIR: Final[Path] = Path(__file__).resolve().parent
CHECKPOINTS_DIR: Final[Path] = PROJECT_DIR / "checkpoints"

MLP_CHECKPOINT: Final[Path] = CHECKPOINTS_DIR / "mlp_final.pt"
GCN_CHECKPOINT: Final[Path] = CHECKPOINTS_DIR / "gcn_final.pt"


# device setup
def select_device(
        requested_device: str | torch.device | None = None,
) -> torch.device:
    """
    Select the device used for model inference.

    Args:
        requested_device: Optional device requested for inference.

    Returns:
        requested device, or CUDA when available, otherwise CPU.

    """
    if requested_device is not None:
        return torch.device(requested_device)

    return torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )


def load_prediction_data(
        train_pct: int = 60,
        val_pct: int = 20,
        test_pct: int = 20,
        seed: int = 808,
) -> tuple[FacebookPagePage, Data]:
    """
    Load the Facebook Page-Page data and reconstruct the project data splits.

    Args:
        train_pct: Percentage of nodes assigned to training.
        val_pct: Percentage of nodes assigned to validation.
        test_pct: Percentage of nodes assigned to testing.
        seed: Random seed used for the stratified splits.

    Returns:
        Loaded dataset and graph data containing the reproducible split masks.

    """
    # load the Facebook Page-Page dataset
    dataset = load_facebook_data()

    # retrieve the graph data from the single-graph dataset
    data = cast(  # narrow the Typing scope to satisfy checker
        Data,
        dataset[0],
    )

    # reconstruct the stratified train, validation, and test masks
    data = add_stratified_masks(
        data,
        train_pct=train_pct,
        val_pct=val_pct,
        test_pct=test_pct,
        seed=seed,
    )

    return dataset, data


def load_mlp(
        checkpoint_path: Path,
        in_channels: int,
        out_channels: int,
        device: torch.device,
) -> MLPNetwork3Layer:
    """
    Load the selected final MLP model and trained weights.

    Args:
        checkpoint_path: Path to the saved MLP model state.
        in_channels: Number of input node features.
        out_channels: Number of output classes.
        device: Device on which to load the model.

    Returns:
        Loaded MLP model in evaluation mode.

    """
    # construct the selected final MLP architecture
    model = MLPNetwork3Layer(
        in_channels=in_channels,
        hidden_channels=64,
        out_channels=out_channels,
        dropout=0.5,
    ).to(device)

    # load the selected trained model state
    state_dict = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(state_dict)
    model.eval()

    return model


def load_gcn(
        checkpoint_path: Path,
        in_channels: int,
        out_channels: int,
        device: torch.device,
) -> GCNNetwork4Layer:
    """
    Load the selected final GCN model and trained weights.

    Args:
        checkpoint_path: Path to the saved GCN model state.
        in_channels: Number of input node features.
        out_channels: Number of output classes.
        device: Device on which to load the model.

    Returns:
        Loaded GCN model in evaluation mode.

    """
    # construct the selected final GCN architecture
    model = GCNNetwork4Layer(
        in_channels=in_channels,
        hidden_channels=64,
        out_channels=out_channels,
        dropout=0.55,
    ).to(device)

    # load the selected trained model state
    state_dict = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(state_dict)
    model.eval()

    return model


def load_model(
        model_name: ModelName,
        in_channels: int,
        out_channels: int,
        checkpoint_path: str | Path | None = None,
        requested_device: str | torch.device | None = None,
) -> tuple[PredictionModel, torch.device]:
    """
    Load a selected trained model for prediction.

    The model type determines the final architecture and default checkpoint.
    A checkpoint path or device may be provided to override the defaults.

    Args:
        model_name: Model to load - accepts either `"mlp"` or `"gcn"`.
        in_channels: Number of input node features.
        out_channels: Number of output classes.
        checkpoint_path: Optional path overriding the default model checkpoint.
        requested_device: Optional device on which to perform inference.

    Returns:
        A loaded model in evaluation mode, and its selected inference device.

    """
    # select the requested device, otherwise prefer CUDA when available
    device = select_device(requested_device)

    # select the model-specific default checkpoint unless overridden
    if model_name == "mlp":
        selected_checkpoint = (
            Path(checkpoint_path)
            if checkpoint_path is not None
            else MLP_CHECKPOINT
        )

    elif model_name == "gcn":
        selected_checkpoint = (
            Path(checkpoint_path)
            if checkpoint_path is not None
            else GCN_CHECKPOINT
        )

    else:
        raise ValueError(
            f"Unknown model '{model_name}'. Expected 'mlp' or 'gcn'."
        )

    # ensure selected checkpoint is available prior to model construction
    if not selected_checkpoint.is_file():
        raise FileNotFoundError(
            f"Model checkpoint not found: {selected_checkpoint}"
        )

    # perform model construction and state loading using the appropriate loader
    if model_name == "mlp":
        model = load_mlp(
            checkpoint_path=selected_checkpoint,
            in_channels=in_channels,
            out_channels=out_channels,
            device=device,
        )
    else:
        model = load_gcn(
            checkpoint_path=selected_checkpoint,
            in_channels=in_channels,
            out_channels=out_channels,
            device=device,
        )

    return model, device


# run inference TODO

def main():
    """
    Main function
    :return: None
    """
    dataset, data = load_prediction_data()

    mlp, mlp_device = load_model(
        model_name="mlp",
        in_channels=data.num_node_features,
        out_channels=dataset.num_classes,
    )

    gcn, gcn_device = load_model(
        model_name="gcn",
        in_channels=data.num_node_features,
        out_channels=dataset.num_classes,
    )

    print(f"\nDataset: {dataset}\n")
    print(f"Data: {data}\n")
    print(f"MLP Model: {mlp}\n")
    print(f"GCN Model: {gcn}\n")
    print(f"Selected MLP Device: {mlp_device.type}")
    print(f"Selected GCN Device: {gcn_device.type}\n")


if __name__ == "__main__":
    main()
