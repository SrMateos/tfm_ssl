"""
Data transformations for VAE training and validation.

This module contains all the data augmentation and preprocessing transformations
used for training and validation of the variational autoencoder.
"""

from matplotlib import transforms
import numpy as np
from monai.transforms import (Compose, CropForegroundd, LoadImaged,
                              RandAdjustContrastd, RandAffined,
                              RandCropByPosNegLabeld, RandFlipd,
                              RandGaussianNoised, RandSpatialCropd,
                              ScaleIntensityRanged, SpatialPadd, ToTensord,
                              EnsureTyped, Invertd, Lambdad)

from src.constants import A_MAX_HU, A_MIN_HU

def unnormalize(x):
    return x * (A_MAX_HU - A_MIN_HU) + A_MIN_HU

def get_vae_train_transforms(patch_size=(64,)*3):
    """
    Get training transforms for VAE model.

    Args:
        patch_size (tuple): Size of the patches to extract (default: (64, 64, 64))

    Returns:
        Compose: MONAI compose object with training transforms
    """
    return Compose([
        LoadImaged(keys=("image", "mask"), image_only=True, ensure_channel_first=True),
        SpatialPadd(keys=("image", "mask"), method="symmetric", spatial_size=patch_size),
        ScaleIntensityRanged(
            keys=("image"),
            a_min=A_MIN_HU, # A broad range for different organ types https://radiopaedia.org/articles/windowing-ct
            a_max=A_MAX_HU,
            b_min=0.0,
            b_max=1.0,
            clip=True
        ),
        RandSpatialCropd(
            keys=("image", "mask"),
            roi_size=patch_size,
            random_size=False,
        ),
        RandFlipd(keys=("image", "mask"), prob=0.2, spatial_axis=0),
        RandFlipd(keys=("image", "mask"), prob=0.2, spatial_axis=1),
        RandFlipd(keys=("image", "mask"), prob=0.2, spatial_axis=2),
        RandAffined(
            keys=("image", "mask"),
            prob=0.5,
            rotate_range=(np.pi/12, np.pi/12, np.pi/12),  # Slight rotations
            scale_range=(0.1, 0.1, 0.1),  # Slight scaling
            mode=('bilinear', "nearest"),  # Interpolation for image
            padding_mode='border'
        ),
        RandAdjustContrastd(keys="image", prob=0.2, gamma=(0.7, 1.3)),
        RandGaussianNoised(
            keys="image",
            prob=0.2,
            mean=0.0,
            std=0.05 * (1.0 - 0.0)  # Adjust std to normalized range
        ),
        ScaleIntensityRanged(
            keys="image",
            a_min=0.0, a_max=1.0,   # rango de entrada esperado
            b_min=0.0, b_max=1.0,   # identidad
            clip=True               # recorta fuera de [0,1]
        ),
        EnsureTyped(keys=("image", "mask"), dtype=np.float32),
    ])


def get_vae_val_transforms(patch_size=(64,)*3):
    """
    Get validation transforms for VAE model.

    Args:
        patch_size (tuple): Size of the patches to extract (default: (64, 64, 64))

    Returns:
        Compose: MONAI compose object with validation transforms
    """
    return Compose([
        LoadImaged(keys=("image", "mask"), image_only=False, ensure_channel_first=True),
        ScaleIntensityRanged(
            keys=("image"),
            a_min=A_MIN_HU, # A broad range for different organ types https://radiopaedia.org/articles/windowing-ct
            a_max=A_MAX_HU,
            b_min=0.0,
            b_max=1.0,
            clip=True
        ),
        SpatialPadd(keys=("image", "mask"), method="symmetric", spatial_size=patch_size),
    ])


def get_vae_post_transforms(val_tf):
    """
    Deshace SpatialPadd y ScaleIntensityRanged sobre:
        "pred"  reconstrucción del AE
        "image" opcional: GT en HU
    """
    return Compose([
        # --- reconstrucción -------------------------------------------------
        Invertd(
            keys=["pred"],              # qué invertir
            transform=val_tf,           # pipeline directo
            orig_keys=["image"],        # de dónde copiar meta
            meta_keys=["image_meta_dict"],
            nearest_interp=False,       # bilinear
        ),
        # --- ground-truth (si la quieres sin normalizar) --------------------
        Invertd(
            keys=["image"],
            transform=val_tf,
            orig_keys=["image"],
            meta_keys=["image_meta_dict"],
            nearest_interp=False,
        ),
        Lambdad(
            keys=["pred", "image"],
            func=unnormalize,
        ),
        # --- asegúrate de que son tensores y tienen las claves correctas ----
        EnsureTyped(keys=["pred", "image", "mask"])
    ])
