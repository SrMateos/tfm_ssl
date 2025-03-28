import time
from configparser import ConfigParser

import matplotlib.pyplot as plt
import torch
from flask import Config
from monai.data import DataLoader, Dataset
from monai.inferers import sliding_window_inference
from monai.losses import ContrastiveLoss
from monai.networks.nets import SwinUNETR
from monai.utils import set_determinism
from torch.nn import L1Loss
from tqdm import tqdm

from config import config_parser
from constants import DATA_PATH, PATCH_SIZE
from utils.utils import (get_data_paths, get_train_transforms,
                         get_val_transforms)

set_determinism(42)

def main():
    # Config
    config_p = config_parser.ConfigParser()
    config = config_p.parse()

    # Load data
    image_paths, _, masks_paths = get_data_paths(DATA_PATH, debug=config["debug"])
    data = [{"image": img, "label": label} for img, label in zip(image_paths, masks_paths)]

    # Train
    train_data_split = int(len(data) * 0.8)
    batch_size = config["training"]["batch"] if config["training"]["batch"] else 1

    train_data = data[:train_data_split]
    train_ds = Dataset(data=train_data, transform=get_train_transforms())
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    # Validation
    val_data = data[train_data_split:]
    val_ds = Dataset(data=val_data, transform=get_val_transforms())
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=True)

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SwinUNETR(
        img_size=PATCH_SIZE,
        in_channels=1,
        out_channels=1,
        feature_size=48
    ).to(device)

    # Losses
    recon_loss = L1Loss()
    contrastive_loss = ContrastiveLoss(temperature=0.05)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), config["training"]["learning_rate"])
    num_epochs = config["training"]["num_epochs"] if config["num_epochs"] else 10
    losses = []
    writer = torch.utils.tensorboard.SummaryWriter()

    epoch_times = []

    # Bucle de entrenamiento con tqdm a nivel de step
    for epoch in range(num_epochs):
        model.train()
        epoch_start = time.time()
        epoch_loss = 0
        val_loss = 0

        # Barra de progreso para el entrenamiento (nivel step)
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

            # Actualizar la barra dinámicamente con la pérdida actualizada
            progress_bar.set_postfix(train_step_loss=f"{step_loss:.4f}")
            progress_bar.update(1)


        # Medición de tiempo del epoch
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)

        # Calcular ETA basado en tiempos anteriores
        avg_epoch_time = sum(epoch_times) / len(epoch_times)
        remaining_time = avg_epoch_time * (num_epochs - (epoch + 1))

        # Calcular la pérdida media del epoch
        avg_loss = epoch_loss / len(train_loader)

        # Validación con tqdm
        model.eval()

        val_bar = tqdm(total=len(val_loader), desc="Validation", unit="step", leave=False)

        with torch.no_grad():
            for val_data in val_loader:
                val_images = val_data["image"].to(device)

                sw_batch_size = 1
                overlap = 0.5

                val_outputs = sliding_window_inference(
                    inputs=val_images,
                    roi_size=PATCH_SIZE,
                    sw_batch_size=sw_batch_size,
                    predictor=model,
                    overlap=overlap,
                    mode="gaussian"
                )

                batch_loss = recon_loss(val_outputs, val_images)
                val_loss += batch_loss
                val_bar.set_postfix(val_step_loss=f"{batch_loss:.4f}")
                val_bar.update(1)

            val_loss /= len(val_loader)
            val_bar.set_postfix(val_loss=f"{val_loss:.4f}")
            val_bar.close()  # Cerrar la barra de validación cuando termine

        # Guardar pérdidas
        losses.append(avg_loss)

        # Registro en TensorBoard
        writer.add_scalar('Loss/train', avg_loss, epoch)
        writer.add_scalar('Loss/validation', val_loss, epoch)

        progress_bar.close()  # Cerrar la barra de entrenamiento cuando termine el epoch

        # 🔹 Mostrar información del epoch sin interferir con la barra
        tqdm.write(f"Epoch {epoch + 1}/{num_epochs} - Loss: {avg_loss:.4f} - Val Loss: {val_loss:.4f} - Time: {epoch_time:.2f}s - ETA: {remaining_time:.2f}s")


    torch.save(model.state_dict(), "ssl_model.pth")

    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('training_loss.png')


if __name__ == '__main__':
    main()
