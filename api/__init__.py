"""
API REST para detección de EPP en minería chilena.

Este módulo expone endpoints para inferencia de YOLOv8 optimizado
para detección de cascos, chalecos reflectantes y zapatos de seguridad
en condiciones de faenas mineras.

Endpoints principales:
- /health: Health check para Kubernetes probes
- /predict: Detección de EPP en imágenes
- /metrics: Métricas de performance del sistema
- /info: Información del modelo y configuración

Autor: Bastián Berríos
Proyecto: epp-detector
"""

__version__ = "0.1.0"
__author__ = "Bastián Berríos"
__description__ = "API REST para detección de EPP en minería chilena"

# Classes detectable by the model (aligned with Roboflow dataset)
# NOTE: Dataset uses English class names (hardhat, head, person)
EPP_CLASSES = {
    0: "hardhat",
    1: "head",  # head without hardhat (non-compliant)
    2: "person",
}

# Traducción de clases al español para usuarios finales chilenos
# Mapping: class_name_en -> class_name_es
EPP_CLASSES_ES = {
    "hardhat": "Casco de seguridad",
    "head": "Cabeza sin casco",  # Violación de seguridad
    "person": "Persona",
}

# Colores para visualización (formato BGR para OpenCV)
CLASS_COLORS = {
    "hardhat": (0, 255, 0),      # Verde
    "head": (255, 0, 0),          # Rojo (violación)
    "person": (0, 191, 255),      # Naranja
}

# Configuración de detección
# NOTA: Para obtener valores de configuración, importar directamente desde api.config:
#   from api.config import settings
#   threshold = settings.confidence_threshold
