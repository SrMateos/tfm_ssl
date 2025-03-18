import time

from pkg_resources import ensure_directory
from sklearn.calibration import Hidden
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils.utils import get_data_paths, get_transforms
from constants import DATA_PATH_TASK1_PELVIS, PATCH_SIZE

from torch.nn import L1Loss
from monai.utils import set_determinism, first
from monai.networks.nets import ViTAutoEnc
from monai.losses import ContrastiveLoss
from monai.data import DataLoader, Dataset

from monai.networks.nets import SwinUNETR
set_determinism(42)

def main():
    mris_paths, _, masks_paths = get_data_paths(DATA_PATH_TASK1_PELVIS)

    data = [{"image": img, "label": label} for img, label in zip(mris_paths, masks_paths)]
    train_data_split = int(len(data)*0.8)
    train_data = data[:train_data_split]
    train_ds = Dataset(data=train_data, transform=get_transforms())
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)

    val_data = data[train_data_split:]
    val_ds = Dataset(data=val_data, transform=get_transforms())
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=True)

    # Train objects
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SwinUNETR(
        img_size=PATCH_SIZE,
        in_channels=1,
        out_channels=1,
        feature_size=48
    ).to(device)

    recon_loss = L1Loss()
    contrastive_loss = ContrastiveLoss(temperature=0.05)

    optimizer = torch.optim.Adam(model.parameters(), 1e-4)
    num_epochs = 100
    losses = []
    writer = torch.utils.tensorboard.SummaryWriter()

    epoch_times = []

    # Bucle de entrenamiento con tqdm a nivel de step
    for epoch in range(num_epochs):
        model.train()
        epoch_start = time.time()
        epoch_loss = 0

        # Barra de progreso para el entrenamiento (nivel step)
        progress_bar = tqdm(total=len(train_loader), desc=f"Epoch {epoch + 1}/{num_epochs}", unit="step", leave=True)

        for batch_data in train_loader:
            images = batch_data["image"].to(device)

            optimizer.zero_grad()
            outputs_v1 = model(images)
            outputs_v2 = model(images)

            flat_out_v1 = outputs_v1.flatten(start_dim=1, end_dim=-1)
            flat_out_v2 = outputs_v2.flatten(start_dim=1, end_dim=-1)

            loss = recon_loss(outputs_v1, images) + contrastive_loss(flat_out_v1, flat_out_v2)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # Calcular la pérdida promedio del epoch en tiempo real
            avg_loss = epoch_loss / (progress_bar.n + 1)

            # 🔹 Actualizar la barra dinámicamente con la pérdida actualizada
            progress_bar.set_postfix(train_loss=f"{avg_loss:.4f}")
            progress_bar.update(1)


        # Medición de tiempo del epoch
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)

        # Calcular ETA basado en tiempos anteriores
        avg_epoch_time = sum(epoch_times) / len(epoch_times)
        remaining_time = avg_epoch_time * (num_epochs - (epoch + 1))

        # Validación con tqdm
        model.eval()
        val_loss = 0

        val_bar = tqdm(total=len(val_loader), desc="Validation", unit="step", leave=False)

        with torch.no_grad():
            for val_data in val_loader:
                val_images = val_data["image"].to(device)
                val_outputs = model(val_images)
                batch_loss = recon_loss(val_outputs, val_images)
                val_loss += batch_loss
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
