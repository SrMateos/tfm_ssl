# Self-Supervised Learning for Medical Image Translation
This project implements self-supervised learning techniques for medical image translation tasks, specifically focused on CBCT (Cone Beam Computed Tomography) to CT (Computed Tomography) conversion. The framework utilizes the MONAI toolkit, which is built on PyTorch and specialized for medical imaging applications.

## Project Structure
The project is organized as follows:
```plaintext
tfm_ssl/
├── constants.py           # Global constants and configuration
├── metrics/               # Image evaluation metrics
│   ├── __init__.py
│   └── image_metrics.py   # MAE, PSNR, SSIM metrics
├── losses/                # Custom loss functions
│   ├── __init__.py
│   └── combined_loss.py   # Combined L1 and SSIM loss
├── utils/                 # Utility functions
│   ├── __init__.py
│   └── utils.py           # Data loading and transforms
├── tests/                 # Unit tests
│   └── test_image_metrics.py
├── ssl_train.py           # Self-supervised learning training script
├── transfer_learning.py   # Transfer learning implementation
└── README.md
```

## Data Preparation
The project expects data in a specific structure:

Each subject should have a directory containing:
cbct.nii.gz - The CBCT image
ct.nii.gz - The corresponding CT image (ground truth)
mask.nii.gz - Segmentation mask (if applicable)

Learn more about where to download the data from the [Synthrad2023 Challenge](https://synthrad2023.grand-challenge.org/Data/).

## Usage
Self-Supervised Learning
Train the self-supervised model:

```bash
python ssl_train.py
```

This trains a SwinUNETR model using self-supervised contrastive learning techniques.

Transfer Learning
After self-supervised training, fine-tune using transfer learning:

```bash
python transfer_learning.py
```

This loads the pre-trained model and fine-tunes it for the CBCT-to-CT translation task.

