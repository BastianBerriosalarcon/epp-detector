"""
Utilidades para la API de detección de EPP.

Funciones helper para validación de inputs, preprocessing,
y formateo de resultados.
"""

import io
from typing import Dict, List, Any, Tuple
from pathlib import Path

from fastapi import UploadFile, HTTPException, status

from api import EPP_CLASSES_ES, CLASS_COLORS

# TODO: Descomentar cuando se implementen
# import numpy as np
# from PIL import Image
# import cv2


# ============================================================================
# Validación de imágenes
# ============================================================================


def validate_image(file: UploadFile, max_size_mb: int = 10) -> None:
    """
    Valida formato y tamaño de imagen cargada.

    Args:
        file: Archivo subido via FastAPI
        max_size_mb: Tamaño máximo permitido en MB

    Raises:
        HTTPException 400: Si el formato es inválido
        HTTPException 413: Si el archivo es muy grande

    TODO: Verificar magic bytes para validar formato real (no solo extensión)
    TODO: Validar dimensiones mínimas/máximas
    TODO: Verificar que la imagen no esté corrupta
    TODO: Agregar soporte para más formatos (WebP, TIFF, etc.)
    """
    # Validar extensión
    allowed_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    file_ext = Path(file.filename).suffix.lower()

    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Formato de imagen no soportado: {file_ext}. "
                f"Use: {', '.join(allowed_extensions)}"
            ),
        )

    # TODO: Validar tamaño del archivo
    # if hasattr(file, 'size'):
    #     max_size_bytes = max_size_mb * 1024 * 1024
    #     if file.size > max_size_bytes:
    #         raise HTTPException(
    #             status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
    #             detail=f"Imagen muy grande. Máximo: {max_size_mb}MB"
    #         )

    # TODO: Validar que sea una imagen válida
    # try:
    #     img = Image.open(io.BytesIO(await file.read()))
    #     img.verify()
    #     await file.seek(0)  # Reset file pointer
    # except Exception as e:
    #     raise HTTPException(
    #         status_code=status.HTTP_400_BAD_REQUEST,
    #         detail=f"Archivo corrupto o no es una imagen válida: {str(e)}"
    #     )


def validate_image_dimensions(
    width: int,
    height: int,
    min_size: int = 320,
    max_size: int = 4096,
) -> None:
    """
    Valida dimensiones de una imagen.

    Args:
        width: Ancho en píxeles
        height: Alto en píxeles
        min_size: Tamaño mínimo permitido
        max_size: Tamaño máximo permitido

    Raises:
        HTTPException 400: Si las dimensiones están fuera de rango

    TODO: Implementar validación de aspect ratio
    TODO: Agregar warnings para imágenes muy pequeñas (baja precisión)
    """
    if width < min_size or height < min_size:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Imagen muy pequeña ({width}x{height}). "
                f"Mínimo: {min_size}x{min_size} píxeles"
            ),
        )

    if width > max_size or height > max_size:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Imagen muy grande ({width}x{height}). "
                f"Máximo: {max_size}x{max_size} píxeles"
            ),
        )


# ============================================================================
# Preprocessing de imágenes
# ============================================================================


def preprocess_image(
    image_bytes: bytes,
    target_size: int = 640,
    normalize: bool = True,
) -> Any:
    """
    Preprocesa imagen para inferencia con YOLOv8.

    Args:
        image_bytes: Bytes de la imagen
        target_size: Tamaño objetivo (default: 640 para YOLOv8)
        normalize: Si se debe normalizar píxeles a [0, 1]

    Returns:
        Imagen procesada (formato depende de implementación)

    TODO: Convertir bytes a numpy array o PIL Image
    TODO: Implementar letterbox resize (mantener aspect ratio)
    TODO: Normalizar píxeles según necesidades del modelo
    TODO: Convertir a formato RGB si es necesario (algunos formatos usan BGR)
    TODO: Agregar padding para mantener dimensiones cuadradas
    TODO: Retornar también las dimensiones originales (para deshacer transformaciones)
    """
    # TODO: Implementar
    # image = Image.open(io.BytesIO(image_bytes))
    # image = letterbox_resize(image, target_size)
    # if normalize:
    #     image = np.array(image) / 255.0
    # return image

    raise NotImplementedError("Preprocessing pendiente de implementar")


def letterbox_resize(
    image: Any,
    target_size: int,
    color: Tuple[int, int, int] = (114, 114, 114),
) -> Any:
    """
    Redimensiona imagen manteniendo aspect ratio (letterbox).

    Este método es el estándar de YOLO para evitar distorsión.
    Agrega padding gris en los bordes cuando es necesario.

    Args:
        image: Imagen PIL o numpy array
        target_size: Tamaño objetivo (cuadrado)
        color: Color del padding (default: gris)

    Returns:
        Imagen redimensionada con padding

    TODO: Implementar resize manteniendo aspect ratio
    TODO: Calcular padding necesario en cada lado
    TODO: Agregar padding con color especificado
    TODO: Retornar metadata de transformación (scale, padding) para revertir
    """
    raise NotImplementedError("Letterbox resize pendiente de implementar")


# ============================================================================
# Postprocessing y formateo de resultados
# ============================================================================


def format_detections(
    results: Any,
    class_names: Dict[int, str],
    image_width: int,
    image_height: int,
) -> List[Dict[str, Any]]:
    """
    Formatea resultados del modelo a estructura JSON estándar con traducción al español.

    Args:
        results: Output del modelo YOLOv8/ONNX
        class_names: Mapeo de IDs a nombres de clases (inglés)
        image_width: Ancho original de la imagen
        image_height: Alto original de la imagen

    Returns:
        Lista de detecciones formateadas:
        [
            {
                'class_id': 0,
                'class_name': 'hardhat',
                'class_name_es': 'Casco de seguridad',
                'confidence': 0.92,
                'bbox': [x_min, y_min, x_max, y_max],
                'bbox_normalized': [x_center, y_center, width, height]
            },
            ...
        ]

    TODO: Parsear formato específico de YOLOv8 results
    TODO: Convertir coordenadas a formato absoluto
    TODO: Filtrar detecciones por confianza
    TODO: Ordenar por confianza (descendente)
    TODO: Agregar metadata útil (área, centro, etc.)
    """
    # TODO: Implementar parsing real
    # detections = []
    # for result in results:
    #     boxes = result.boxes
    #     for box in boxes:
    #         class_name = class_names[int(box.cls)]
    #         detection = {
    #             'class_id': int(box.cls),
    #             'class_name': class_name,
    #             'class_name_es': EPP_CLASSES_ES.get(class_name, class_name),
    #             'confidence': float(box.conf),
    #             'bbox': box.xyxy[0].tolist(),
    #             'bbox_normalized': box.xywhn[0].tolist()
    #         }
    #         detections.append(detection)
    # return detections

    raise NotImplementedError("Format detections pendiente de implementar")


def apply_nms(
    boxes: List[List[float]],
    scores: List[float],
    class_ids: List[int],
    iou_threshold: float = 0.45,
) -> List[int]:
    """
    Aplica Non-Maximum Suppression manualmente.

    Útil cuando el modelo no aplica NMS automáticamente
    (algunos exports de ONNX).

    Args:
        boxes: Lista de bounding boxes [x_min, y_min, x_max, y_max]
        scores: Lista de scores de confianza
        class_ids: Lista de IDs de clase
        iou_threshold: Umbral IoU para considerar overlap

    Returns:
        Índices de las detecciones a mantener

    TODO: Implementar algoritmo NMS estándar
    TODO: Aplicar NMS por clase (no suprimir entre clases diferentes)
    TODO: Optimizar con operaciones vectorizadas (numpy)
    """
    raise NotImplementedError("NMS manual pendiente de implementar")


def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """
    Calcula Intersection over Union entre dos bounding boxes.

    Args:
        box1: [x_min, y_min, x_max, y_max]
        box2: [x_min, y_min, x_max, y_max]

    Returns:
        IoU score entre 0 y 1

    TODO: Implementar cálculo de IoU
    TODO: Manejar edge cases (boxes sin overlap, boxes inválidos)
    """
    # TODO: Implementar
    # x1 = max(box1[0], box2[0])
    # y1 = max(box1[1], box2[1])
    # x2 = min(box1[2], box2[2])
    # y2 = min(box1[3], box2[3])

    # intersection = max(0, x2 - x1) * max(0, y2 - y1)
    # area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    # area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    # union = area1 + area2 - intersection

    # return intersection / union if union > 0 else 0.0

    raise NotImplementedError("IoU calculation pendiente de implementar")


# ============================================================================
# Análisis de cumplimiento de EPP
# ============================================================================


def check_epp_compliance(detections: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Evalúa si una persona cumple con EPP obligatorio según DS 132.

    Lógica de cumplimiento basada en dataset actual (hardhat, head, person):
    - COMPLIANT: Si se detecta "hardhat" y NO se detecta "head"
    - NON_COMPLIANT: Si se detecta "head" (persona sin casco - violación)
    - UNKNOWN: Si solo se detecta "person" sin información de casco

    Args:
        detections: Lista de detecciones de EPP con class_name en inglés

    Returns:
        {
            'compliant': bool,
            'violations': list,  # Violaciones detectadas en español
            'summary': str,      # Resumen en español
            'confidence_avg': float
        }

    TODO: Cuando se agreguen más clases (vest, boots), actualizar lógica
    TODO: Considerar overlapping de bounding boxes (persona-EPP)
    TODO: Detectar múltiples personas y evaluar individualmente
    TODO: Agregar severidad de violaciones (crítico vs warning)
    TODO: Integrar con tracking para evaluar en secuencias de video
    """
    detected_classes = {det["class_name"] for det in detections}

    violations = []
    compliant = True

    # Regla 1: Si hay cabeza sin casco, es violación crítica
    if "head" in detected_classes:
        compliant = False
        violations.append("Persona sin casco detectada (violación crítica)")

    # Regla 2: Si hay persona pero no se detectó ni casco ni cabeza, incierto
    if "person" in detected_classes and "hardhat" not in detected_classes and "head" not in detected_classes:
        compliant = False
        violations.append("No se puede verificar uso de casco")

    # Generar resumen en español
    if compliant and "hardhat" in detected_classes:
        summary = "Cumplimiento de EPP verificado: Casco de seguridad detectado"
    elif not violations:
        summary = "No se detectaron personas en la imagen"
    else:
        summary = f"Violación de seguridad: {'; '.join(violations)}"

    return {
        "compliant": compliant,
        "violations": violations,
        "summary": summary,
        "confidence_avg": (
            sum(d["confidence"] for d in detections) / len(detections)
            if detections
            else 0.0
        ),
    }


# ============================================================================
# Utilidades de visualización
# ============================================================================


def draw_detections(
    image: Any,
    detections: List[Dict[str, Any]],
    show_confidence: bool = True,
    thickness: int = 2,
) -> Any:
    """
    Dibuja bounding boxes sobre la imagen.

    Args:
        image: Imagen PIL o numpy array
        detections: Lista de detecciones
        show_confidence: Si se debe mostrar score en el label
        thickness: Grosor de las líneas

    Returns:
        Imagen con detecciones dibujadas

    TODO: Implementar drawing con OpenCV o PIL
    TODO: Usar colores distintos por clase
    TODO: Agregar labels con nombre de clase y confianza
    TODO: Destacar violaciones con color rojo
    TODO: Agregar leyenda en esquina de la imagen
    """
    raise NotImplementedError("Visualización pendiente de implementar")


def get_class_color(class_name: str) -> Tuple[int, int, int]:
    """
    Retorna color RGB para cada clase de EPP.

    Args:
        class_name: Nombre de la clase (inglés)

    Returns:
        Tupla RGB (0-255)
        - Verde: EPP presente (compliant)
        - Rojo: Violación (head sin hardhat)
        - Azul: Persona (neutral)
    """
    return CLASS_COLORS.get(class_name, (128, 128, 128))  # Gris por defecto
