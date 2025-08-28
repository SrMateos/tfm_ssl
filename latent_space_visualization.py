# --- LIBRERÍAS NECESARIAS ---
# pip install torch monai numpy scikit-learn umap-learn matplotlib
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap # UMAP es a menudo más rápido y preserva mejor la estructura global que t-SNE

# --- ASUMPCIONES SOBRE SU CÓDIGO ---
# Asumo que tiene los siguientes elementos definidos en otros ficheros y puede importarlos:
# 1. Su clase de Dataset de PyTorch (p. ej., `MiDatasetDePrueba`)
# 2. Su pipeline de transformaciones de validación (p. ej., `get_vae_val_transforms`)
# 3. La definición de su modelo AutoencoderKL (p. ej., `AutoencoderKL`)
from torch.utils.data import DataLoader
# Reemplace estas importaciones con las suyas
# from mi_proyecto.datasets import MiDatasetDePrueba, ANATOMY_MAP
# from mi_proyecto.transforms import get_vae_val_transforms
# from mi_proyecto.modelos import AutoencoderKL

# --- CONFIGURACIÓN (MODIFICAR ESTOS VALORES) ---
# ---------------------------------------------------
CHECKPOINT_PATH = "ruta/a/su/modelo_entrenado.pth"
DATA_DIR = "ruta/a/sus/datos_de_prueba"
BATCH_SIZE = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_SAMPLES_TO_VISUALIZE = 500 # Usar un subconjunto para agilizar la visualización

# Mapa de anatomías para la leyenda del gráfico (ajústelo a sus etiquetas)
ANATOMY_MAP = {
    0: {"name": "Abdomen", "color": "blue"},
    1: {"name": "Cabeza y Cuello", "color": "red"},
    2: {"name": "Tórax", "color": "green"},
}
# ---------------------------------------------------

def get_latent_space_embeddings(model, dataloader, device, max_samples):
    """
    Pasa los datos a través del codificador para obtener los vectores latentes.
    """
    model.eval()
    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if len(all_labels) >= max_samples:
                print(f"Alcanzado el límite de {max_samples} muestras para visualización.")
                break

            # Asumo que el dataloader devuelve un diccionario
            images = batch["image"].to(device)
            labels = batch["label"] # Asumo que 'label' es un tensor de índices (0, 1, 2)

            # Obtenemos solo la media (mu) de la distribución latente
            z_mu, _ = model.encode(images)

            # Aplanamos el mapa de características espacial a un único vector por imagen
            # z_mu tiene forma [B, C, D, H, W], lo aplanamos a [B, C*D*H*W]
            flattened_mu = z_mu.view(z_mu.size(0), -1)

            all_embeddings.append(flattened_mu.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

            print(f"Procesado batch {i+1}/{len(dataloader)}...")

    # Concatenamos los resultados de todos los lotes
    embeddings_np = np.concatenate(all_embeddings, axis=0)
    labels_np = np.concatenate(all_labels, axis=0)

    return embeddings_np, labels_np

def visualize_embeddings(embeddings, labels, method='umap'):
    """
    Aplica t-SNE o UMAP y crea el gráfico de dispersión.
    """
    print(f"\nRealizando reducción de dimensionalidad con {method.upper()}...")
    if method == 'tsne':
        reducer = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
    else: # UMAP es la opción recomendada
        reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)

    # Proyectamos los vectores de alta dimensionalidad a 2D
    embeddings_2d = reducer.fit_transform(embeddings)

    # Creamos el gráfico
    plt.figure(figsize=(12, 10))
    for label_idx, info in ANATOMY_MAP.items():
        # Seleccionamos los puntos que corresponden a cada anatomía
        points = embeddings_2d[labels == label_idx]
        plt.scatter(
            points[:, 0],
            points[:, 1],
            c=info["color"],
            label=info["name"],
            alpha=0.7,
            s=15 # Tamaño del punto
        )

    plt.title(f"Visualización del Espacio Latente con {method.upper()}")
    plt.xlabel("Componente 1")
    plt.ylabel("Componente 2")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(f"latent_space_{method}.png")
    print(f"Gráfico guardado en 'latent_space_{method}.png'")
    plt.show()

if __name__ == "__main__":
    # 1. Cargar el modelo
    # -------------------
    # Reemplace esto con la instanciación de su modelo
    aekl_model = AutoencoderKL(
        spatial_dims=3, in_channels=1, out_channels=1, channels=(32, 64, 64),
        latent_channels=3, num_res_blocks=1, norm_num_groups=16,
        attention_levels=(False, False, True),
    ).to(DEVICE)

    aekl_model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    print("Modelo cargado correctamente.")

    # 2. Preparar el DataLoader de prueba
    # -----------------------------------
    # Reemplace esto con la carga de su conjunto de datos
    # val_transforms = get_vae_val_transforms(...)
    # test_dataset = MiDatasetDePrueba(root_dir=DATA_DIR, transform=val_transforms)
    # test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    # print(f"Datos de prueba cargados: {len(test_dataset)} muestras.")

    # --- EJEMPLO CON DATOS FALSOS (BORRAR CUANDO INTEGRE SU DATALOADER) ---
    from monai.data import create_test_image_3d
    fake_data = [{"image": create_test_image_3d(64, 64, 64, num_seg_classes=1)[0].unsqueeze(0), "label": np.random.randint(0,3)} for _ in range(200)]
    test_loader = DataLoader(fake_data, batch_size=BATCH_SIZE)
    # --------------------------------------------------------------------

    # 3. Generar y visualizar embeddings
    # ----------------------------------
    latent_embeddings, latent_labels = get_latent_space_embeddings(
        model=aekl_model,
        dataloader=test_loader,
        device=DEVICE,
        max_samples=NUM_SAMPLES_TO_VISUALIZE
    )

    visualize_embeddings(latent_embeddings, latent_labels, method='umap')
