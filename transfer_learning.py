import torch
import torch.nn as nn
import torch.optim as optim
from monai.networks.nets import SwinUNETR
from monai.data import DataLoader, Dataset
from monai.transforms import Compose, LoadImaged, RandCropByPosNegLabeld, ToTensord
from utils import get_data_paths, get_transforms
from constants import DATA_PATH_TASK1_PELVIS, PATCH_SIZE
from pathlib import Path

# Cargar datos
DATA_PATH_TASK1_PELVIS = Path('data/Task1/pelvis')
mris_paths, cts_paths = get_data_paths(DATA_PATH_TASK1_PELVIS)

data = [{"image": img, "label": label} for img, label in zip(mris_paths, cts_paths)]
train_data_split = int(len(data) * 0.8)

train_data = data[:train_data_split]
train_ds = Dataset(data=train_data, transform=get_transforms())
train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)

val_data = data[train_data_split:]
val_ds = Dataset(data=val_data, transform=get_transforms())
val_loader = DataLoader(val_ds, batch_size=1, shuffle=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pretrained_model = SwinUNETR(
    img_size=PATCH_SIZE,
    in_channels=1,
    out_channels=1,
    feature_size=48
).to(device)

pretrained_model.load_state_dict(torch.load("ssl_model.pth"))
encoder = pretrained_model.swinViT  # Enconder

# Definir el nuevo modelo usando el encoder preentrenado
class MRI2CTModel(nn.Module):
    def __init__(self, encoder):
        super(MRI2CTModel, self).__init__()
        self.encoder = encoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(768, 384, kernel_size=2, stride=2),  # Upsample
            nn.ReLU(),
            nn.ConvTranspose3d(384, 192, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose3d(192, 96, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv3d(96, 1, kernel_size=1)  # Output 1 canal (CT)
        )

    def forward(self, x):
        x = self.encoder(x)  # Pasar por el encoder preentrenado
        x = self.decoder(x)  # Pasar por el nuevo decoder
        return x

# Instanciar el nuevo modelo
model = MRI2CTModel(encoder).to(device)

# Definir función de pérdida y optimizador
criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Entrenamiento
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for batch in train_loader:
        images = batch["image"].to(device)
        targets = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

torch.save(model.state_dict(), "mri_to_ct_model.pth")
print("Modelo de Transfer Learning guardado correctamente.")
