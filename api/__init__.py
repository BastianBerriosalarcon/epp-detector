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
# NOTE: Dataset uses English class names in alphabetical order
# Source: Roboflow Construction Site Safety dataset v27
EPP_CLASSES = {
    0: "hardhat",           # Casco de seguridad (EPP obligatorio DS 132)
    1: "mask",              # Mascarilla (no relevante para minería chilena)
    2: "no_hardhat",        # Sin casco (VIOLACIÓN DS 132)
    3: "no_mask",           # Sin mascarilla (no relevante)
    4: "no_safety_vest",    # Sin chaleco (VIOLACIÓN DS 132)
    5: "person",            # Persona
    6: "safety_cone",       # Cono de seguridad (contexto)
    7: "safety_vest",       # Chaleco reflectante (EPP obligatorio DS 132)
    8: "machinery",         # Maquinaria (contexto)
    9: "vehicle",           # Vehículo (contexto)
}

# Traducción de clases al español para usuarios finales chilenos
# Mapping: class_name_en -> class_name_es
EPP_CLASSES_ES = {
    "hardhat": "Casco de seguridad",
    "mask": "Mascarilla",
    "no_hardhat": "Sin casco",  # Violación de seguridad
    "no_mask": "Sin mascarilla",
    "no_safety_vest": "Sin chaleco",  # Violación de seguridad
    "person": "Persona",
    "safety_cone": "Cono de seguridad",
    "safety_vest": "Chaleco reflectante",
    "machinery": "Maquinaria",
    "vehicle": "Vehículo",
}

# Colores para visualización (formato BGR para OpenCV)
CLASS_COLORS = {
    "hardhat": (0, 255, 255),        # Amarillo (BGR) - EPP
    "mask": (255, 200, 0),           # Cyan (BGR)
    "no_hardhat": (0, 0, 255),       # Rojo (BGR) - VIOLACIÓN
    "no_mask": (0, 100, 255),        # Naranja oscuro (BGR)
    "no_safety_vest": (0, 0, 200),   # Rojo oscuro (BGR) - VIOLACIÓN
    "person": (0, 255, 0),           # Verde (BGR)
    "safety_cone": (0, 200, 255),    # Amarillo naranja (BGR)
    "safety_vest": (0, 165, 255),    # Naranja (BGR) - EPP
    "machinery": (255, 255, 0),      # Cyan claro (BGR)
    "vehicle": (255, 0, 255),        # Magenta (BGR)
}

# Configuración de detección
# NOTA: Para obtener valores de configuración, importar directamente desde api.config:
#   from api.config import settings
#   threshold = settings.confidence_threshold
