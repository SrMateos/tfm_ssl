import re
import statistics
import numpy as np

def parse_metrics_from_text(text):
    """
    Parsea las métricas del texto y devuelve listas con los valores
    """
    # Patrón regex para capturar los valores
    pattern = r'Test Step \d+ - MAE: ([\d.]+), PSNR: ([\d.]+), MS-SSIM: ([\d.]+)'

    matches = re.findall(pattern, text)

    mae_values = [float(match[0]) for match in matches]
    psnr_values = [float(match[1]) for match in matches]
    ms_ssim_values = [float(match[2]) for match in matches]

    return mae_values, psnr_values, ms_ssim_values

def calculate_statistics(values, metric_name):
    """
    Calcula media, desviación típica, min, max para una lista de valores
    """
    if not values:
        return None

    mean_val = statistics.mean(values)
    std_val = statistics.stdev(values) if len(values) > 1 else 0
    min_val = min(values)
    max_val = max(values)

    return {
        'metric': metric_name,
        'count': len(values),
        'mean': mean_val,
        'std': std_val,
        'min': min_val,
        'max': max_val
    }

def print_statistics(stats):
    """
    Imprime las estadísticas de forma bonita
    """
    print(f"\n{'='*60}")
    print(f"ESTADÍSTICAS PARA {stats['metric']}")
    print(f"{'='*60}")
    print(f"Número de muestras: {stats['count']}")
    print(f"Media:              {stats['mean']:.4f}")
    print(f"Desviación típica:  {stats['std']:.4f}")
    print(f"Mínimo:             {stats['min']:.4f}")
    print(f"Máximo:             {stats['max']:.4f}")

def main():
    print("Calculadora de Estadísticas para Métricas de Test")
    print("-" * 50)

    # Opción 1: Leer desde archivo
    print("¿Cómo quieres introducir los datos?")
    print("1. Desde un archivo de texto")
    print("2. Pegando el texto directamente")

    choice = input("Elige una opción (1 o 2): ").strip()

    if choice == "1":
        filename = input("Introduce el nombre del archivo (ej: datos.txt): ").strip()
        try:
            with open(filename, 'r', encoding='utf-8') as file:
                text_data = file.read()
        except FileNotFoundError:
            print(f"Error: No se encontró el archivo '{filename}'")
            return
        except Exception as e:
            print(f"Error al leer el archivo: {e}")
            return

    elif choice == "2":
        print("\nPega aquí el texto con las métricas (presiona Enter dos veces para terminar):")
        lines = []
        while True:
            line = input()
            if line == "" and len(lines) > 0 and lines[-1] == "":
                break
            lines.append(line)
        text_data = "\n".join(lines)

    else:
        print("Opción no válida")
        return

    # Parsear las métricas
    mae_values, psnr_values, ms_ssim_values = parse_metrics_from_text(text_data)

    if not mae_values:
        print("No se encontraron datos válidos en el texto proporcionado.")
        print("Asegúrate de que el formato sea: 'Test Step X - MAE: Y, PSNR: Z, MS-SSIM: W'")
        return

    # Calcular estadísticas
    mae_stats = calculate_statistics(mae_values, "MAE (Mean Absolute Error)")
    psnr_stats = calculate_statistics(psnr_values, "PSNR (Peak Signal-to-Noise Ratio)")
    ms_ssim_stats = calculate_statistics(ms_ssim_values, "MS-SSIM (Multi-Scale SSIM)")

    # Mostrar resultados
    print_statistics(mae_stats)
    print_statistics(psnr_stats)
    print_statistics(ms_ssim_stats)

    # Resumen comparativo
    print(f"\n{'='*60}")
    print("RESUMEN COMPARATIVO")
    print(f"{'='*60}")
    print(f"{'Métrica':<25} {'Media':<12} {'Desv. Típica':<12} {'CV (%)':<10}")
    print("-" * 60)

    for stats in [mae_stats, psnr_stats, ms_ssim_stats]:
        cv = (stats['std'] / stats['mean'] * 100) if stats['mean'] != 0 else 0
        metric_short = stats['metric'].split('(')[0].strip()
        print(f"{metric_short:<25} {stats['mean']:<12.4f} {stats['std']:<12.4f} {cv:<10.2f}")

    print(f"\nCV = Coeficiente de Variación (Desv. Típica / Media * 100)")

if __name__ == "__main__":
    main()
