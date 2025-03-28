from pathlib import Path
from constants import PATCH_SIZE, TASK1
from monai.transforms import (Compose, LoadImaged, RandCropByPosNegLabeld, ToTensord, SpatialPadd)

def get_data_paths(data_path, debug=False):
    if not data_path.exists():
        raise FileNotFoundError(f'Path {data_path} does not exist.')

    image_paths, cts_paths, masks_paths = [], [], []

    subdirs = sorted([d for d in data_path.iterdir() if d.is_dir()])

    for i, subdirectory in enumerate(subdirs):
        if "overview" in str(subdirectory):
            continue
        if debug and i > 5:
            break

        image_name = "mr.mha" if TASK1 else "cbct.mha"

        image_paths.append(str(Path(subdirectory / image_name)))
        cts_paths.append(str(Path(subdirectory / "ct.mha")))
        masks_paths.append(str(Path(subdirectory / "mask.mha")))

    return image_paths, cts_paths, masks_paths

def get_train_transforms():
    return Compose([
        LoadImaged(keys=("image", "label"), image_only=True, ensure_channel_first=True),
        SpatialPadd(keys=("image", "label"), method="symmetric", spatial_size=PATCH_SIZE),
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

def get_val_transforms():
    return Compose([
        LoadImaged(keys=("image", "label"), image_only=True, ensure_channel_first=True),
        SpatialPadd(keys=("image", "label"), method="symmetric", spatial_size=PATCH_SIZE),
        ToTensord(keys=["image", "label"])
    ])
