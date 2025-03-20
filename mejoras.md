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
