"""
Dataset utilities and data loading functions.

This module contains functions for loading and managing datasets, including
path resolution and data splitting utilities.
"""

from pathlib import Path
from typing import List, Tuple, Union


def get_data_paths(
    data_paths: Union[Path, List[Path]],
    task1: bool = True,
    debug: bool = False
) -> Tuple[List[str], List[str], List[str]]:
    """
    Get image, CT, and mask paths from the specified data directories.

    Args:
        data_paths: Path or list of paths to data directories
        task1: If True, use MR images; if False, use CBCT images
        debug: If True, limit to 5 samples per data path for debugging

    Returns:
        Tuple containing:
            - image_paths: List of image file paths (MR or CBCT)
            - cts_paths: List of CT file paths
            - masks_paths: List of mask file paths
    """
    if not isinstance(data_paths, list):
        data_paths = [data_paths]

    image_paths, cts_paths, masks_paths = [], [], []

    for data_path_item in data_paths:
        if not data_path_item.exists():
            print(f'Warning: Path {data_path_item} does not exist. Skipping.')
            continue

        subdirs = sorted([d for d in data_path_item.iterdir() if d.is_dir()])

        for i, subdirectory in enumerate(subdirs):
            if "overview" in str(subdirectory):
                continue
            if debug and len(image_paths) >= 5 * len(data_paths):
                # Adjust debug limit if multiple main paths
                break

            image_name = "mr.mha" if task1 else "cbct.mha"

            image_paths.append(str(Path(subdirectory / image_name)))
            cts_paths.append(str(Path(subdirectory / "ct.mha")))
            masks_paths.append(str(Path(subdirectory / "mask.mha")))

        if debug and len(image_paths) >= 5 * len(data_paths):
            break

    return image_paths, cts_paths, masks_paths

def split_data(
    data: List[dict],
    train_split: float = 0.8,
    val_split: float = 0.1,
    test_split: float = 0.1,
    random_seed: int = 42
) -> Tuple[List[dict], List[dict], List[dict]]:
    """
    Split data into training, validation, and test sets.

    Args:
        data: List of data dictionaries
        train_split: Fraction of data to use for training
        val_split: Fraction of data to use for validation
        test_split: Fraction of data to use for testing
        random_seed: Random seed for reproducibility

    Returns:
        Tuple containing (train_data, val_data, test_data)
    """
    import random

    # Validate splits
    total = train_split + val_split + test_split
    if not abs(total - 1.0) < 1e-6:
        raise ValueError(f"train_split + val_split + test_split must sum to 1. Got {total}")

    # Set random seed for reproducibility
    random.seed(random_seed)

    # Shuffle data
    shuffled = data.copy()
    random.shuffle(shuffled)
    n = len(shuffled)

    # Compute split indices
    train_end = int(n * train_split)
    val_end = train_end + int(n * val_split)

    # Slice data
    train_data = shuffled[:train_end]
    val_data = shuffled[train_end:val_end]
    test_data = shuffled[val_end:]

    return train_data, val_data, test_data
