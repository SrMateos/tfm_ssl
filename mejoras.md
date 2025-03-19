# Mejoras propuestas para el código
- Hacer uso de datasets personalizados de Monai
- Meter soporte de archivos de configuración en formato YAML
- Añadir una exploración de datos al uso
- Meter uso de MLFlow para el registro y seguimiento de experimentos
```python
import mlflow

mlflow.start_run()
mlflow.log_param("learning_rate", lr)
# Train model
mlflow.log_metric("val_loss", val_loss)
mlflow.log_artifacts("./models")
mlflow.end_run()
```

- Meter un sistema de métricas
```python
# metrics/medical.py
def calculate_metrics(pred, target):
    return {
        "mse": mean_squared_error(pred, target),
        "psnr": peak_signal_noise_ratio(pred, target),
        "ssim": structural_similarity(pred, target),
        "mae": mean_absolute_error(pred, target),
    }
```
- Podría ser interesante crear clases para el uso de funciones de pérdida compuestas.
```python
# losses/combined_loss.py
class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.ssim = SSIMLoss()
        self.alpha = alpha
        self.beta = beta

    def forward(self, pred, target):
        return self.alpha * self.l1(pred, target) + self.beta * self.ssim(pred, target)
```
