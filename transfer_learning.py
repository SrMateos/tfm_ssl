import torch
import torch.nn as nn
import torch.optim as optim
from monai.networks.nets import SwinUNETR
from monai.data import DataLoader, Dataset
from utils.utils import get_data_paths, get_train_transforms, get_val_transforms
from constants import (DATA_PATH, PATCH_SIZE, OUTPUT_MODEL_PATH, DEBUG, MODEL_PATH)
from tqdm import tqdm
from monai.inferers import sliding_window_inference

def main():
    # Set reproducibility
    torch.manual_seed(42)

    # Load images
    image_paths, cts_paths, _ = get_data_paths(DATA_PATH, debug=DEBUG)

    # Create dataset dictionary
    data = [{"image": img, "label": label} for img, label in zip(image_paths, cts_paths)]
    train_data_split = int(len(data) * 0.8)

    # Split and create dataloaders
    train_data = data[:train_data_split]
    train_ds = Dataset(data=train_data, transform=get_train_transforms())
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)

    val_data = data[train_data_split:]
    val_ds = Dataset(data=val_data, transform=get_val_transforms())
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

   # Load pretrained model
    model = SwinUNETR(
        img_size=PATCH_SIZE,
        in_channels=1,
        out_channels=1,
        feature_size=48
    ).to(device)

    # Load weights from self-supervised training
    model.load_state_dict(torch.load(MODEL_PATH))

    # Freeze the encoder (swinViT) part
    for param in model.swinViT.parameters():
        param.requires_grad = False

    criterion = nn.L1Loss()

    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Setup TensorBoard
    writer = torch.utils.tensorboard.SummaryWriter()

    # Training variables
    num_epochs = 2
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0

        # Training progress bar
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False)

        for batch in train_bar:
            images = batch["image"].to(device)
            targets = batch["label"].to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            current_loss = loss.item()
            epoch_loss += current_loss
            train_bar.set_postfix(loss=f"{current_loss:.4f}")

        # Calculate average training loss
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation phase
        model.eval()
        val_loss = 0

        val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Valid]", leave=False)

        with torch.no_grad():
            for batch in val_bar:
                images = batch["image"].to(device)
                targets = batch["label"].to(device)

                sw_batch_size = 1
                overlap = 0.5

                outputs = sliding_window_inference(
                    inputs=images,
                    roi_size=PATCH_SIZE,
                    sw_batch_size=sw_batch_size,
                    predictor=model,
                    overlap=overlap,
                    device=device,
                    mode="gaussian"
                )

                loss = criterion(outputs, targets)

                current_val_loss = loss.item()
                val_loss += current_val_loss
                val_bar.set_postfix(loss=f"{current_val_loss:.4f}")

        # Calculate average validation loss
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        # Log to TensorBoard
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('Loss/validation', avg_val_loss, epoch)

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), OUTPUT_MODEL_PATH)
            print(f"✅ Epoch {epoch+1}: New best model saved with validation loss: {best_val_loss:.4f}")

        # Print epoch summary
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")


    print(f"Training completed. Best model saved with validation loss: {best_val_loss:.4f}")

if __name__ == '__main__':
    main()
