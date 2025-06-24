import logging
import os
import time

import matplotlib.pyplot as plt
import mlflow
import torch
from monai.data import CacheDataset, DataLoader, pad_list_data_collate
from monai.inferers import sliding_window_inference
from monai.losses import PatchAdversarialLoss, PerceptualLoss
from monai.networks.nets import AutoencoderKL, PatchDiscriminator
from monai.utils import set_determinism
from torch.nn import L1Loss
from tqdm import tqdm

from config import ConfigParser
from constants import ALL_TASK1, TASK1_HN
from utils.mlflow_utils import log_config, setup_mlflow
from utils.utils import (get_data_paths, get_vae_train_transforms,
                         get_vae_val_transforms)

set_determinism(42)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def kl_loss(z_mu, z_sigma):
    '''
        kl_loss = 0.5 * sum(z_mu^2 + z_sigma^2 - log(z_sigma^2) - 1)
    '''
    kl_loss = 0.5 * torch.sum(
        z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1,
        dim=list(range(1, len(z_sigma.shape))),
    )
    return torch.sum(kl_loss) / kl_loss.shape[0]


def train_generator_step(autoencoder, discriminator, images, l1_loss, kl_loss_fn, loss_perceptual, adv_loss,
                        optimizer_g, kl_weight, perceptual_weight, adv_weight, epoch, warmup_epochs):
    """Entrena el generador (autoencoder) por un paso"""
    optimizer_g.zero_grad(set_to_none=True)

    reconstruction, z_mu, z_sigma = autoencoder(images)

    # Calculate individual losses for the generator
    loss_recon = l1_loss(reconstruction, images)
    loss_kl_val = kl_loss_fn(z_mu, z_sigma)
    loss_perc = loss_perceptual(reconstruction.float(), images.float())

    # Combine base losses
    loss_g_base = loss_recon + kl_weight * loss_kl_val + perceptual_weight * loss_perc

    # Add adversarial loss after warmup
    loss_g_adv = 0.0
    if epoch >= warmup_epochs:
        logits_fake = discriminator(reconstruction.contiguous().float())[-1]
        loss_g_adv = adv_loss(logits_fake, target_is_real=True, for_discriminator=False)
        loss_g = loss_g_base + adv_weight * loss_g_adv
    else:
        loss_g = loss_g_base

    loss_g.backward()
    optimizer_g.step()

    return {
        'total': loss_g.item(),
        'reconstruction': loss_recon.item(),
        'kl': loss_kl_val.item(),
        'perceptual': loss_perc.item(),
        'adversarial': loss_g_adv.item() if isinstance(loss_g_adv, torch.Tensor) else loss_g_adv,
        'reconstruction_tensor': reconstruction  # Needed for discriminator training
    }


def train_discriminator_step(discriminator, reconstruction, images, adv_loss, optimizer_d,
                           adv_weight, epoch, warmup_epochs):
    """Entrena el discriminador por un paso"""
    if epoch < warmup_epochs:
        return {'total': 0.0, 'fake': 0.0, 'real': 0.0}

    optimizer_d.zero_grad(set_to_none=True)

    # Detach reconstruction to avoid backprop through generator
    logits_fake = discriminator(reconstruction.contiguous().detach())[-1]
    loss_d_fake = adv_loss(logits_fake, target_is_real=False, for_discriminator=True)

    logits_real = discriminator(images.contiguous().detach())[-1]
    loss_d_real = adv_loss(logits_real, target_is_real=True, for_discriminator=True)

    discriminator_loss = (loss_d_fake + loss_d_real) * 0.5
    loss_d = adv_weight * discriminator_loss

    loss_d.backward()
    optimizer_d.step()

    return {
        'total': loss_d.item(),
        'fake': loss_d_fake.item(),
        'real': loss_d_real.item()
    }


def log_validation_images(val_images, val_outputs, epoch):
    """Log example validation images to MLflow"""
    img_slice = val_images[0, 0, :, :, val_images.shape[-1] // 2].cpu().numpy()
    recon_slice = val_outputs[0, 0, :, :, val_outputs.shape[-1] // 2].cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(img_slice, cmap='gray')
    axes[0].set_title('Original Image Slice')
    axes[0].axis('off')
    axes[1].imshow(recon_slice, cmap='gray')
    axes[1].set_title('Reconstruction Slice')
    axes[1].axis('off')
    plt.tight_layout()
    mlflow.log_figure(fig, f"validation_image_epoch_{epoch+1}.png")
    plt.close(fig)


def validate_epoch(autoencoder, val_loader, l1_loss, patch_size, sw_batch_size, overlap, mode, device, config, epoch):
    """Ejecuta validación para una época"""
    autoencoder.eval()
    val_step = 0
    val_recon_loss = 0

    val_bar = tqdm(total=len(val_loader), desc="Validation", unit="step", leave=False)

    with torch.no_grad():
        for val_batch in val_loader:
            val_step += 1
            val_images = val_batch["image"].to(device)
            val_outputs = sliding_window_inference(
                inputs=val_images,
                roi_size=patch_size,
                sw_batch_size=sw_batch_size,
                predictor=autoencoder.reconstruct,
                overlap=overlap,
                mode=mode,
                device=device,
            )

            # Calculate validation reconstruction loss
            batch_recon_loss = l1_loss(val_outputs, val_images)
            val_recon_loss += batch_recon_loss.item()

            val_bar.update(1)

    val_bar.close()

    # Calculate average validation metrics
    avg_val_recon_loss = val_recon_loss / val_step

    # Log validation metrics to MLFlow
    val_metrics = {
        "val_recon_loss": avg_val_recon_loss,
    }
    mlflow.log_metrics(val_metrics, step=epoch)

    # Print validation summary
    tqdm.write(f"  Validation - Avg Recon Loss: {avg_val_recon_loss:.4f}")

    # Log validation images if enabled
    if config["mlflow"].get("log_images", False) and val_step > 0:
        log_validation_images(val_images, val_outputs, epoch)

    return avg_val_recon_loss


def save_best_models(autoencoder, discriminator, best_val_metric, epoch):
    """Guarda los mejores modelos"""
    mlflow.log_metric("best_val_recon_loss_epoch", epoch)
    mlflow.log_metric("best_val_recon_loss", best_val_metric)

    # Save both models
    best_autoencoder_path = "best_autoencoder_model.pth"
    best_discriminator_path = "best_discriminator_model.pth"
    torch.save(autoencoder.state_dict(), best_autoencoder_path)
    torch.save(discriminator.state_dict(), best_discriminator_path)
    mlflow.log_artifact(best_autoencoder_path, artifact_path="models")
    mlflow.log_artifact(best_discriminator_path, artifact_path="models")
    tqdm.write(f"New best model saved with validation recon loss: {best_val_metric:.4f}")


def main():
    # Config
    config_parser = ConfigParser()
    config = config_parser.parse()

    # Setup MLFlow
    run = setup_mlflow(config, experiment_name="SSL_Training")
    log_config(config)

    # Data settings from config
    patch_size = tuple(config["data"].get("patch_size", [64, 64, 64]))
    train_val_split = config["data"].get("train_val_split", 0.8)
    task1 = config["data"].get("task1", True)

    debug_mode = bool(config.get("debug", False)) # Get debug mode from global config

    # Training settings from config
    batch_size = config["training"].get("batch_size", 1)
    num_epochs = config["training"].get("epochs", 200)
    learning_rate = float(config["training"].get("learning_rate", 1e-4)) # Use learning rate from config
    adv_weight = float(config["training"].get("adv_weight", 0.01))
    perceptual_weight = float(config["training"].get("perceptual_weight", 0.001))
    kl_weight = float(config["training"].get("kl_weight", 1e-3))
    autoencoder_warm_up_n_epochs = config["training"].get("warmup_epochs", 5) # Warmup epochs from config
    val_interval = config["training"].get("val_interval", 5) # Validation interval from config
    save_interval = config["training"].get("save_interval", 20) # Model save interval

    # Inference settings from config
    sw_batch_size = config["inference"].get("sw_batch_size", 1)
    overlap = config["inference"].get("overlap", 0.5)
    mode = config["inference"].get("mode", "gaussian")

    # Log hyperparameters explicitly
    mlflow.log_params({
        "patch_size": str(patch_size),
        "train_val_split": train_val_split,
        "debug_mode": debug_mode,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "learning_rate": learning_rate,
        "adv_weight": adv_weight,
        "perceptual_weight": perceptual_weight,
        "kl_weight": kl_weight,
        "warmup_epochs": autoencoder_warm_up_n_epochs,
        "val_interval": val_interval,
        "save_interval": save_interval,
        "sw_batch_size": sw_batch_size,
        "overlap": overlap,
        "inference_mode": mode,
        "random_seed": 42
    })

    # Load data
    logging.info(f"Loading data from {TASK1_HN}...")
    # image_paths, labels_paths, masks_paths = get_data_paths(TASK1_HN, debug=debug_mode) # Use debug_mode from config
    # data = [{"image": img, "label": label, "mask": mask} for img, label, mask in zip(image_paths, labels_paths, masks_paths)]
    _, cts_paths, masks_paths = get_data_paths(ALL_TASK1, task1=task1, debug=debug_mode)
    data = [{"image": ct, "mask": mask} for ct, mask in zip(cts_paths, masks_paths)]
    logging.info(f"First data sample: {data[0]}")
    train_data_split = int(len(data) * train_val_split)
    train_data = data[:train_data_split]
    val_data = data[train_data_split:]

    # Train
    train_ds = CacheDataset(data=train_data, transform=get_vae_train_transforms(patch_size=patch_size))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)

    # Validation
    val_ds = CacheDataset(data=val_data, transform=get_vae_val_transforms(patch_size=patch_size))
    val_loader = DataLoader(val_ds, batch_size=2, shuffle=False, num_workers=2, collate_fn=pad_list_data_collate)

    # Log dataset sizes
    mlflow.log_param("dataset_size", len(data))
    mlflow.log_param("train_dataset_size", len(train_data))
    mlflow.log_param("val_dataset_size", len(val_data))

    # Model Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mlflow.log_param("device", device.type)

    # Use model parameters from config
    model_config = config.get("model", {})
    autoencoder = AutoencoderKL(
        spatial_dims     = 3,
        in_channels      = model_config.get("in_channels", 1),
        out_channels     = model_config.get("out_channels", 1),
        channels         = tuple(model_config.get("channels", (32, 64, 64))),
        latent_channels  = model_config.get("latent_channels", 3),
        num_res_blocks   = model_config.get("num_res_blocks", 1),
        norm_num_groups  = model_config.get("norm_num_groups", 16),
        attention_levels = tuple(model_config.get("attention_levels", (False, False, True))),
    ).to(device)
    #autoencoder.compile()

    # Use discriminator parameters from config or defaults
    discriminator_config = config.get("discriminator", {})
    discriminator = PatchDiscriminator(
        spatial_dims = 3,
        num_layers_d = discriminator_config.get("num_layers_d", 3),
        channels     = discriminator_config.get("channels", 32),
        in_channels  = model_config.get("in_channels", 1), # autoencoder input
        out_channels = discriminator_config.get("out_channels", 1),
        norm         = discriminator_config.get("norm", "INSTANCE")
    ).to(device)
    #discriminator.compile()

    # Log model architecture summary ( TODO: maybe logging as text artifact for full details)
    mlflow.log_param("autoencoder_name", "AutoencoderKL")
    mlflow.log_param("autoencoder_architecture_summary", str(autoencoder)) # Basic summary
    mlflow.log_param("discriminator_name", "PatchDiscriminator")
    mlflow.log_param("discriminator_architecture_summary", str(discriminator)) # Basic summary

    # Losses
    l1_loss = L1Loss() # L1 loss for reconstruction
    l1_loss.to(device)
    adv_loss = PatchAdversarialLoss(criterion="least_squares") # Adversarial loss
    adv_loss.to(device)
    loss_perceptual = PerceptualLoss(
        spatial_dims  = 3,
        network_type  = "radimagenet_resnet50",
        is_fake_3d    = True,
        fake_3d_ratio = 0.2
    ) # Perceptual loss
    loss_perceptual.to(device)

    # Optimizers - Use learning_rate from config
    optimizer_g = torch.optim.AdamW(params=autoencoder.parameters(), lr=learning_rate)
    optimizer_d = torch.optim.AdamW(params=discriminator.parameters(), lr=learning_rate)

    # Reduce on plateau scheduler for the generator
    scheduler_g = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_g, mode='min', factor=0.01, patience=2, min_lr=1e-7
    )
    # Reduce on plateau scheduler for the discriminator
    scheduler_d = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_d, mode='min', factor=0.001, patience=3, min_lr=1e-7
    )

    # Training loop variables
    epoch_times = []
    best_val_metric = float('inf') # Using a metric (e.g., L1 loss) for best model saving

    # --- Training Loop ---
    for epoch in range(num_epochs):
        autoencoder.train()
        discriminator.train()

        epoch_start = time.time()
        epoch_g_loss = 0
        epoch_d_loss = 0
        epoch_recon_loss = 0
        epoch_kl_loss = 0
        epoch_perceptual_loss = 0
        epoch_adv_g_loss = 0
        epoch_adv_d_fake_loss = 0
        epoch_adv_d_real_loss = 0
        step = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="step", leave=True)

        for batch_data in progress_bar:
            step += 1
            images = batch_data["image"].to(device)

            # --- Train Generator (Autoencoder) ---
            g_losses = train_generator_step(
                autoencoder, discriminator, images, l1_loss, kl_loss, loss_perceptual, adv_loss,
                optimizer_g, kl_weight, perceptual_weight, adv_weight, epoch, autoencoder_warm_up_n_epochs
            )

            # --- Train Discriminator ---
            d_losses = train_discriminator_step(
                discriminator, g_losses['reconstruction_tensor'], images, adv_loss, optimizer_d,
                adv_weight, epoch, autoencoder_warm_up_n_epochs
            )

            # Accumulate generator losses for logging
            epoch_g_loss += g_losses['total']
            epoch_recon_loss += g_losses['reconstruction']
            epoch_kl_loss += g_losses['kl']
            epoch_perceptual_loss += g_losses['perceptual']
            epoch_adv_g_loss += g_losses['adversarial']

            # Accumulate discriminator losses for logging
            epoch_d_loss += d_losses['total']
            epoch_adv_d_fake_loss += d_losses['fake']
            epoch_adv_d_real_loss += d_losses['real']

            # Update progress bar
            progress_bar.set_postfix({
                "G_loss": f"{g_losses['total']:.4f}",
                "D_loss": f"{d_losses['total']:.4f}",
                "Recon": f"{g_losses['reconstruction']:.4f}",
                "KL": f"{g_losses['kl']:.4f}"
            })

        # --- End of Epoch ---
        progress_bar.close()
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)

        # Calculate average losses for the epoch
        avg_epoch_g_loss = epoch_g_loss / step
        avg_epoch_d_loss = epoch_d_loss / step
        avg_epoch_recon_loss = epoch_recon_loss / step
        avg_epoch_kl_loss = epoch_kl_loss / step
        avg_epoch_perceptual_loss = epoch_perceptual_loss / step
        avg_epoch_adv_g_loss = epoch_adv_g_loss / step if epoch >= autoencoder_warm_up_n_epochs else 0
        avg_epoch_adv_d_fake_loss = epoch_adv_d_fake_loss / step if epoch >= autoencoder_warm_up_n_epochs else 0
        avg_epoch_adv_d_real_loss = epoch_adv_d_real_loss / step if epoch >= autoencoder_warm_up_n_epochs else 0

        # Log average training losses to MLFlow
        train_metrics = {
            "train_generator_loss": avg_epoch_g_loss,
            "train_discriminator_loss": avg_epoch_d_loss,
            "train_recon_loss": avg_epoch_recon_loss,
            "train_kl_loss": avg_epoch_kl_loss,
            "train_perceptual_loss": avg_epoch_perceptual_loss,
            "train_adv_generator_loss": avg_epoch_adv_g_loss,
            "train_adv_discriminator_fake_loss": avg_epoch_adv_d_fake_loss,
            "train_adv_discriminator_real_loss": avg_epoch_adv_d_real_loss,
            "epoch_time_seconds": epoch_time,
        }
        mlflow.log_metrics(train_metrics, step=epoch)

        tqdm.write(f"Epoch {epoch + 1}/{num_epochs} Summary:")
        tqdm.write(f"  Avg Train G Loss: {avg_epoch_g_loss:.4f}, Avg Train D Loss: {avg_epoch_d_loss:.4f}")
        tqdm.write(f"  Avg Recon Loss: {avg_epoch_recon_loss:.4f}, Avg KL Loss: {avg_epoch_kl_loss:.4f}")
        tqdm.write(f"  Time: {epoch_time:.2f}s")

        # --- Validation ---
        if (epoch + 1) % val_interval == 0:
            avg_val_recon_loss = validate_epoch(
                autoencoder, val_loader, l1_loss, patch_size, sw_batch_size,
                overlap, mode, device, config, epoch
            )

            scheduler_d.step(avg_val_recon_loss)  # Update discriminator scheduler
            scheduler_g.step(avg_val_recon_loss)  # Update generator scheduler

            # Save best model if needed
            if avg_val_recon_loss < best_val_metric:
                best_val_metric = avg_val_recon_loss
                save_best_models(autoencoder, discriminator, best_val_metric, epoch)

        # Save model checkpoint periodically
        if (epoch + 1) % save_interval == 0:
            checkpoint_dir = "checkpoints"
            os.makedirs(checkpoint_dir, exist_ok=True)
            autoencoder_ckpt_path = os.path.join(checkpoint_dir, f"autoencoder_epoch_{epoch+1}.pth")
            discriminator_ckpt_path = os.path.join(checkpoint_dir, f"discriminator_epoch_{epoch+1}.pth")
            torch.save(autoencoder.state_dict(), autoencoder_ckpt_path)
            torch.save(discriminator.state_dict(), discriminator_ckpt_path)
            # Log checkpoints as artifacts
            mlflow.log_artifact(autoencoder_ckpt_path, artifact_path="checkpoints")
            mlflow.log_artifact(discriminator_ckpt_path, artifact_path="checkpoints")
            tqdm.write(f"Checkpoint saved at epoch {epoch+1}")

    # --- End of Training ---
    # Save final models
    final_autoencoder_path = "final_autoencoder_model.pth"
    final_discriminator_path = "final_discriminator_model.pth"
    torch.save(autoencoder.state_dict(), final_autoencoder_path)
    torch.save(discriminator.state_dict(), final_discriminator_path)
    mlflow.log_artifact(final_autoencoder_path, artifact_path="models")
    mlflow.log_artifact(final_discriminator_path, artifact_path="models")

    # Log models using MLFlow's PyTorch flavor (optional, but recommended)
    # This allows easier loading later using mlflow.pytorch.load_model
    mlflow.pytorch.log_model(autoencoder, "autoencoder_model")
    mlflow.pytorch.log_model(discriminator, "discriminator_model")

    # End the MLflow run
    mlflow.end_run()
    print("Training finished.")

if __name__ == '__main__':
    main()
