import time
from pathlib import Path

from monai.transforms import (Compose, LoadImaged, RandCropByPosNegLabeld,
                              ScaleIntensityRanged, SpatialPadd, ToTensord)

from constants import PATCH_SIZE, TASK1


def get_data_paths(data_paths, debug=False):
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

            image_name = "mr.mha" if TASK1 else "cbct.mha"

            image_paths.append(str(Path(subdirectory / image_name)))
            cts_paths.append(str(Path(subdirectory / "ct.mha")))
            masks_paths.append(str(Path(subdirectory / "mask.mha")))

        if debug and len(image_paths) >= 5 * len(data_paths):
            break


    return image_paths, cts_paths, masks_paths

def get_train_transforms():
    return Compose([
        LoadImaged(keys=("image", "label"), image_only=True, ensure_channel_first=True),
        SpatialPadd(keys=("image", "label"), method="symmetric", spatial_size=PATCH_SIZE),
        ScaleIntensityRanged(
            keys=["image"], a_min=-1400, a_max=1400,
            b_min=0.0, b_max=1.0, clip=True
        ),
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
        ScaleIntensityRanged(
            keys=["image"], a_min=-1400, a_max=1400,
            b_min=0.0, b_max=1.0, clip=True
        ),
        ToTensord(keys=["image", "label"])
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
