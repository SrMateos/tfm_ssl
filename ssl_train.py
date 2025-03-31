import os
import time

import matplotlib.pyplot as plt
import mlflow
import torch
from monai.data import DataLoader, Dataset
from monai.inferers import sliding_window_inference
from monai.losses import ContrastiveLoss
from monai.networks.nets import SwinUNETR
from monai.utils import set_determinism
from torch.nn import L1Loss
from tqdm import tqdm

from config import ConfigParser
from constants import DATA_PATH, PATCH_SIZE
from utils.mlflow_utils import log_config, setup_mlflow
from utils.utils import (get_data_paths, get_train_transforms,
                         get_val_transforms)

set_determinism(42)

def main():
    # Config
    config_parser = ConfigParser()
    config = config_parser.parse()

    # Setup MLFlow
    run = setup_mlflow(config, experiment_name="SSL_Training")
    log_config(config)

    # Load data
    image_paths, _, masks_paths = get_data_paths(DATA_PATH, debug=bool(config["debug"]))
    data = [{"image": img, "label": label} for img, label in zip(image_paths, masks_paths)]


    # Train
    train_data_split = int(len(data) * 0.8)
    batch_size = config["training"].get("batch_size", 1)

    train_data = data[:train_data_split]
    train_ds = Dataset(data=train_data, transform=get_train_transforms())
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    # Validation
    val_data = data[train_data_split:]
    val_ds = Dataset(data=val_data, transform=get_val_transforms())
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=True)

    # Log dataset sizes
    mlflow.log_param("dataset_size", len(data))
    mlflow.log_param("train_dataset_size", len(train_data))
    mlflow.log_param("val_dataset_size", len(val_data))

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SwinUNETR(
        img_size=PATCH_SIZE,
        in_channels=1,
        out_channels=1,
        feature_size=48
    ).to(device)

    # Log model architecture summary
    mlflow.log_param("model_name", "SwinUNETR")

    # Losses
    recon_loss = L1Loss()
    contrastive_loss = ContrastiveLoss(temperature=0.05)

    # Log loss configuration
    mlflow.log_param("contrastive_loss_temperature", 0.05)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), float(config["training"]["learning_rate"]))
    num_epochs = config["training"].get("epochs", 200)
    losses = []

    epoch_times = []

    for epoch in range(num_epochs):
        model.train()
        epoch_start = time.time()
        epoch_loss = 0
        val_loss = 0

        progress_bar = tqdm(total=len(train_loader), desc=f"Epoch {epoch + 1}/{num_epochs}", unit="step", leave=True)

        for batch_data in train_loader:
            images = batch_data["image"].to(device)

            optimizer.zero_grad()
            outputs_v1 = model(images)
            outputs_v2 = model(images)

            flat_out_v1 = outputs_v1.flatten(start_dim=1, end_dim=-1)
            flat_out_v2 = outputs_v2.flatten(start_dim=1, end_dim=-1)

            r_loss = recon_loss(outputs_v1, images)
            c_loss = contrastive_loss(flat_out_v1, flat_out_v2)

            loss = r_loss + c_loss * r_loss

            loss.backward()
            optimizer.step()

            step_loss = loss.item()
            epoch_loss += step_loss

            progress_bar.set_postfix(train_step_loss=f"{step_loss:.4f}")
            progress_bar.update(1)

        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        avg_epoch_time = sum(epoch_times) / len(epoch_times)
        remaining_time = avg_epoch_time * (num_epochs - (epoch + 1))
        avg_loss = epoch_loss / len(train_loader)

        # Validation
        model.eval()
        val_bar = tqdm(total=len(val_loader), desc="Validation", unit="step", leave=False)

        with torch.no_grad():
            for val_data in val_loader:
                val_images = val_data["image"].to(device)

                sw_batch_size = config["inference"].get("sw_batch_size", 1)
                overlap = config["inference"].get("overlap", 0.5)
                mode = config["inference"].get("mode", "gaussian")

                val_outputs = sliding_window_inference(
                    inputs=val_images,
                    roi_size=PATCH_SIZE,
                    sw_batch_size=sw_batch_size,
                    predictor=model,
                    overlap=overlap,
                    mode=mode,
                )

                batch_loss = recon_loss(val_outputs, val_images)
                val_loss += batch_loss
                val_bar.set_postfix(val_step_loss=f"{batch_loss:.4f}")
                val_bar.update(1)

            val_loss /= len(val_loader)
            val_bar.close()

        losses.append(avg_loss)
        progress_bar.close()

        # Log metrics to MLFlow
        mlflow.log_metrics({
            "train_loss": avg_loss,
            "validation_loss": val_loss,
            "epoch_time": epoch_time,
        }, step=epoch)

        # Save model checkpoint
        if (epoch + 1) % 10 == 0:  # Save every 10 epochs
            checkpoint_path = f"checkpoints/epoch_{epoch+1}.pth"
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), checkpoint_path)
            mlflow.log_artifact(checkpoint_path)

        tqdm.write(f"Epoch {epoch + 1}/{num_epochs} - Loss: {avg_loss:.4f} - Val Loss: {val_loss:.4f} - Time: {epoch_time:.2f}s - ETA: {remaining_time:.2f}s")

    # Save final model
    final_model_path = "ssl_model.pth"
    torch.save(model.state_dict(), final_model_path)
    mlflow.log_artifact(final_model_path)

    # Create and save loss plot
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plot_path = "training_loss.png"
    plt.savefig(plot_path)
    mlflow.log_artifact(plot_path)

    # Log model to MLFlow
    mlflow.pytorch.log_model(model, "model")

    # End the run
    mlflow.end_run()

if __name__ == '__main__':
    main()
