# make_grid.py
import argparse
import re
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.image as mpimg

def natural_key(p: Path):
    m = re.search(r"(\d+)(?=\D*$)", p.stem)  # último número del stem
    return int(m.group(1)) if m else 0

def main():
    parser = argparse.ArgumentParser(description="Monta una figura 3x4 con 10 PNGs y leyenda global.")
    parser.add_argument("--dir", type=str, default=".", help="Directorio con latent_tsne_*.png")
    parser.add_argument("--pattern", type=str, default="latent_tsne_*.png", help="Patrón de búsqueda de imágenes")
    parser.add_argument("--out", type=str, default="latent_tsne_grid.png", help="Ruta de salida")
    parser.add_argument("--dpi", type=int, default=300, help="DPI de guardado")
    parser.add_argument("--width", type=float, default=12.0, help="Ancho de la figura (pulgadas)")
    parser.add_argument("--height", type=float, default=9.0, help="Alto de la figura (pulgadas)")
    parser.add_argument("--show_panel_labels", action="store_true", help="Añade etiquetas (a)…(j)")
    args = parser.parse_args()

    img_dir = Path(args.dir)
    files = sorted(img_dir.glob(args.pattern), key=natural_key)

    if len(files) < 10:
        raise FileNotFoundError(f"Se esperaban 10 imágenes; encontradas {len(files)} con patrón {args.pattern} en {img_dir}")

    # Posiciones (fila, col) para 10 imágenes en una rejilla 3x4
    # Filas 0-1: cuatro columnas (0..3). Fila 2: columnas centrales 1 y 2.
    positions = [(0, c) for c in range(4)] + [(1, c) for c in range(4)] + [(2, 1), (2, 2)]

    fig = plt.figure(figsize=(12, 9), constrained_layout=True)
    gs = fig.add_gridspec(3, 4)

    # Colores y leyenda global (coinciden con tu código de generación)
    colors = {"AB": "tab:blue", "HN": "tab:orange", "TH": "tab:green", "UNK": "tab:gray"}
    legend_elems = [
        Line2D([0], [0], marker="o", linestyle="None", markersize=6, color=colors["AB"], label="AB"),
        Line2D([0], [0], marker="o", linestyle="None", markersize=6, color=colors["HN"], label="HN"),
        Line2D([0], [0], marker="o", linestyle="None", markersize=6, color=colors["TH"], label="TH"),
    ]

    # Colocación de imágenes
    panel_labels = list("abcdefghij")
    for idx, (fpath, (r, c)) in enumerate(zip(files[:10], positions)):
        ax = fig.add_subplot(gs[r, c])
        img = mpimg.imread(fpath)
        ax.imshow(img)
        ax.set_axis_off()
        ax.set_aspect("auto")
        if args.show_panel_labels:
            ax.text(0.02, 0.98, f"({panel_labels[idx]})", transform=ax.transAxes,
                    va="top", ha="left", fontsize=10, bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"))

    # Apagar los ejes vacíos de la tercera fila (columnas 0 y 3)
    ax_empty_left = fig.add_subplot(gs[2, 0]); ax_empty_left.set_axis_off()
    ax_empty_right = fig.add_subplot(gs[2, 3]); ax_empty_right.set_axis_off()

    # Leyenda global centrada debajo
    ax_legend = fig.add_subplot(gs[2, 3])
    ax_legend.set_axis_off()

    colors = {"AB": "tab:blue", "HN": "tab:orange", "TH": "tab:green"}
    # using global import of Line2D
    legend_elems = [
        Line2D([0], [0], marker="o", linestyle="None", markersize=6, color=colors["HN"], label="HN"),
        Line2D([0], [0], marker="o", linestyle="None", markersize=6, color=colors["TH"], label="TH"),
        Line2D([0], [0], marker="o", linestyle="None", markersize=6, color=colors["AB"], label="AB"),
    ]

    # Coloca la leyenda centrada dentro de esa celda
    ax_legend.legend(
        handles=legend_elems,
        title="Anatomía",
        loc="center",          # centrada en el axes
        ncol=1,                # formato vertical (cabe mejor en 1 columna)
        frameon=False,
        handletextpad=0.6,
        labelspacing=0.6,
        borderaxespad=0.0,
    )
    # Ajuste de márgenes para dejar hueco a la leyenda
    plt.subplots_adjust(bottom=0.15)

    fig.savefig(args.out, dpi=300)
    print(f"Figura guardada en: {args.out}")

if __name__ == "__main__":
    main()
