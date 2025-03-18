from pathlib import Path
from constants import PATCH_SIZE
from monai.config import print_config
from monai.transforms import (Compose, LoadImaged, RandCropByPosNegLabeld, ToTensord)

def get_data_paths(data_path):
    if not data_path.exists():
        raise FileNotFoundError(f'Path {data_path} does not exist.')

    mris_paths, cts_paths, masks_paths = [], [], []

    for subdirectory in data_path.iterdir():
        if "overview" in str(subdirectory):
            continue

        mris_paths.append(str(Path(subdirectory / "mr.nii.gz")))
        cts_paths.append(str(Path(subdirectory / "ct.nii.gz")))
        masks_paths.append(str(Path(subdirectory / "mask.nii.gz")))

    return mris_paths, cts_paths, masks_paths

def get_transforms():
    return Compose([
        LoadImaged(keys=("image", "label"), image_only=True, ensure_channel_first=True),
        RandCropByPosNegLabeld(
            keys=("image", "label"),
            label_key="label",
            spatial_size=PATCH_SIZE,
            pos=1,
            neg=1,
            num_samples=4,
        ),
        ToTensord(keys=["image", "label"])
    ])

