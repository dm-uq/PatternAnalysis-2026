"""
dataset.py
Author: Derek McKinney (dm-uq), SN: 45506053

Contains utilities for loading the FacebookPagePage dataset and for
constructing reproducible train/validation/test node masks used by the
project experiments.

References:

    Facebook Page-Page dataset:
    https://github.com/gayanku/facebook_page-page_dataset

    PyG FacebookPagePage:
    https://pytorch-geometric.readthedocs.io/en/latest/generated/
    torch_geometric.datasets.FacebookPagePage.html

"""
from pathlib import Path
from typing import Final

from torch_geometric.datasets import FacebookPagePage

DEFAULT_DATA_ROOT: Final[Path] = Path("data/FacebookPagePage")

# mapping from original dataset numerical class labels to descriptive labels
CLASS_NAMES: Final[dict[int, str]] = {
    0: "tvshow",
    1: "government",
    2: "company",
    3: "politician",
}


def load_facebook_data(
        root: str | Path = DEFAULT_DATA_ROOT,
) -> FacebookPagePage:
    """
    Load the Facebook Page-Page dataset using PyTorch Geometric.

    Args:
        root: Directory containing the raw and processed dataset files.

    Returns:
        The PyG FacebookPagePage dataset.
    """
    return FacebookPagePage(root=str(root))
