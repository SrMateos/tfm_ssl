import time
from ast import Num
from pathlib import Path

import numpy as np
from monai.transforms import (Compose, CropForegroundd, LoadImaged,
                              RandAdjustContrastd, RandAffined,
                              RandCropByPosNegLabeld, RandFlipd,
                              RandGaussianNoised, RandSpatialCropd,
                              ScaleIntensityRanged, SpatialPadd, ToTensord,
                              EnsureTyped)

from constants import A_MAX_HU, A_MIN_HU


def get_data_paths(data_paths, task1=True, debug=False):
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
            if debug and len(image_paths) >= 5 * len(data_paths): # Adjust debug limit if multiple main paths
                break

            image_name = "mr.mha" if task1 else "cbct.mha"

            image_paths.append(str(Path(subdirectory / image_name)))
            cts_paths.append(str(Path(subdirectory / "ct.mha")))
            masks_paths.append(str(Path(subdirectory / "mask.mha")))

        if debug and len(image_paths) >= 5 * len(data_paths):
            break

    return image_paths, cts_paths, masks_paths

def get_vae_train_transforms(patch_size=(64,)*3):
    return Compose([
        LoadImaged(keys=("image"), image_only=True, ensure_channel_first=True),
        SpatialPadd(keys=("image"), method="symmetric", spatial_size=patch_size),
        ScaleIntensityRanged(
            keys=("image"), a_min=A_MIN_HU, a_max=A_MAX_HU, # A broad range for different organ types https://radiopaedia.org/articles/windowing-ct
            b_min=0.0, b_max=1.0, clip=True
        ),
        RandSpatialCropd(
            keys=("image"),
            roi_size=patch_size,
            random_size=False,
        ),
        RandFlipd(keys="image", prob=0.2, spatial_axis=0),
        RandFlipd(keys="image", prob=0.2, spatial_axis=1),
        RandFlipd(keys="image", prob=0.2, spatial_axis=2),
        RandAffined(
            keys="image",
            prob=0.5,
            rotate_range=(np.pi/12, np.pi/12, np.pi/12), # Rotaciones leves
            scale_range=(0.1, 0.1, 0.1), # Escalados leves
            mode=('bilinear'), # Interpolación para la imagen
            padding_mode='border'
        ),
        RandAdjustContrastd(keys="image", prob=0.2, gamma=(0.7, 1.3)),
        RandGaussianNoised(keys="image", prob=0.2, mean=0.0, std=0.05 * (1.0 - 0.0)), # Ajustar std al rango normalizado
        EnsureTyped(keys="image", dtype=np.float32),
    ])

def get_vae_val_transforms(patch_size=(64,)*3):
    return Compose([
        LoadImaged(keys=("image"), image_only=True, ensure_channel_first=True),
        ScaleIntensityRanged(
            keys=("image"), a_min=A_MIN_HU, a_max=A_MAX_HU, # A broad range for different organ types https://radiopaedia.org/articles/windowing-ct
            b_min=0.0, b_max=1.0, clip=True
        ),
        SpatialPadd(keys=("image"), method="symmetric", spatial_size=patch_size),
    ])

def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        hours, minutes, seconds = map(int, time.gmtime(end_time - start_time))
        print(f"Function '{func.__name__}' took {hours}h {minutes}m {seconds}s")
        return result
    return wrapper
