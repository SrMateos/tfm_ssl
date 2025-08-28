# --- LIBRERÍAS NECESARIAS ---
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid, save_image

# --- ASUMPCIONES SOBRE SU CÓDIGO ---
# (Las mismas que en el script anterior)
# from mi_proyecto.datasets import MiDatasetDePrueba
# from mi_proyecto.transforms import get_vae_val_transforms
# from mi_proyecto.modelos import AutoencoderKL

# --- CONFIGURACIÓN (MODIFICAR ESTOS VALORES) ---
# ---------------------------------------------------
CHECKPOINT_PATH = "ruta/a/su/modelo_entrenado.pth"
DATA_DIR = "ruta/a/sus/datos_de_prueba"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Índices de las dos imágenes del dataset que se van a interpolar
IMAGE_A_IDX = 0
IMAGE_B_IDX = 10

NUM_INTERPOLATION_STEPS = 10 # Número de imágenes intermedias a generar
# ---------------------------------------------------


def interpolate_in_latent_space(model, dataset, idx1, idx2, steps, device):
    """
    Realiza una interpolación lineal entre los vectores latentes de dos imágenes.
    """
    model.eval()

    # 1. Obtener y pre-procesar las dos imágenes de origen
    with torch.no_grad():
        # Asumo que el dataset devuelve un diccionario al acceder por índice
        img_a_data = dataset[idx1]
        img_b_data = dataset[idx2]

        # Asumo que las transformaciones se aplican al acceder al dataset
        # y que la imagen ya es un tensor
        img_a = img_a_data["image"].unsqueeze(0).to(device) # Añadir dimensión de batch
        img_b = img_b_data["image"].unsqueeze(0).to(device)

        print(f"Interpolando entre la imagen {idx1} y la imagen {idx2}...")

        # 2. Codificar ambas imágenes para obtener sus representaciones latentes (mu)
        z_mu_a, _ = model.encode(img_a)
        z_mu_b, _ = model.encode(img_b)

        # 3. Realizar la interpolación lineal
        interpolated_latents = []
        for i in range(steps):
            alpha = i / (steps - 1)
            # torch.lerp es la función de interpolación lineal: (1-alpha)*start + alpha*end
            z_interp = torch.lerp(z_mu_a, z_mu_b, alpha)
            interpolated_latents.append(z_interp)

        # 4. Decodificar cada vector latente interpolado
        reconstructed_images = []
        for z in interpolated_latents:
            decoded_img = model.decode(z)
            # Se toma un corte central del volumen 3D para la visualización en 2D
            central_slice_idx = decoded_img.shape[3] // 2
            slice_2d = decoded_img[0, 0, :, :, central_slice_idx].cpu()
            reconstructed_images.append(slice_2d)

    return reconstructed_images

if __name__ == "__main__":
    # 1. Cargar el modelo
    # -------------------
    aekl_model = AutoencoderKL(
        spatial_dims=3, in_channels=1, out_channels=1, channels=(32, 64, 64),
        latent_channels=3, num_res_blocks=1, norm_num_groups=16,
        attention_levels=(False, False, True),
    ).to(DEVICE)

    aekl_model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    print("Modelo cargado correctamente.")

    # 2. Cargar el dataset de prueba
    # ------------------------------
    # Reemplace esto con la carga de su conjunto de datos y transformaciones
    # val_transforms = get_vae_val_transforms(...)
    # test_dataset = MiDatasetDePrueba(root_dir=DATA_DIR, transform=val_transforms)
    # print(f"Datos de prueba cargados: {len(test_dataset)} muestras.")

    # --- EJEMPLO CON DATOS FALSOS (BORRAR CUANDO INTEGRE SU DATALOADER) ---
    from monai.data import create_test_image_3d
    # Creamos un dataset con transformaciones "al vuelo"
    class FakeDataset(torch.utils.data.Dataset):
        def __init__(self, count):
            self.count = count
        def __len__(self):
            return self.count
        def __getitem__(self, idx):
            img, _ = create_test_image_3d(64, 64, 64, num_seg_classes=1)
            return {"image": img, "label": 0}
    test_dataset = FakeDataset(count=20)
    # --------------------------------------------------------------------

    # 3. Realizar interpolación y guardar el resultado
    # ------------------------------------------------
    interpolated_slices = interpolate_in_latent_space(
        model=aekl_model,
        dataset=test_dataset,
        idx1=IMAGE_A_IDX,
        idx2=IMAGE_B_IDX,
        steps=NUM_INTERPOLATION_STEPS,
        device=DEVICE
    )

    # 4. Guardar la secuencia de imágenes como una única grilla
    # make_grid espera tensores en el formato [C, H, W], añadimos una dimensión de canal
    grid_tensor = torch.stack(interpolated_slices).unsqueeze(1)

    grid = make_grid(grid_tensor, nrow=NUM_INTERPOLATION_STEPS, padding=2, normalize=True)

    save_image(grid, "latent_space_interpolation.png")
    print("Imagen de interpolación guardada en 'latent_space_interpolation.png'")

    # Opcional: mostrar la imagen con matplotlib
    plt.figure(figsize=(15, 3))
    plt.imshow(grid.permute(1, 2, 0), cmap='gray')
    plt.title(f"Interpolación de {NUM_INTERPOLATION_STEPS} pasos en el Espacio Latente")
    plt.axis('off')
    plt.show()
