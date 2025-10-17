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

import numpy as np
from PIL import Image
import cv2


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
                f"Imagen muy pequeña ({width}x{height}). " f"Mínimo: {min_size}x{min_size} píxeles"
            ),
        )

    if width > max_size or height > max_size:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Imagen muy grande ({width}x{height}). " f"Máximo: {max_size}x{max_size} píxeles"
            ),
        )


# ============================================================================
# Preprocessing de imágenes
# ============================================================================


def preprocess_image(
    image_bytes: bytes,
    target_size: int = 640,
    normalize: bool = True,
) -> np.ndarray:
    """
    Preprocesa imagen para inferencia con YOLOv8.

    Convierte imagen a formato esperado por YOLO:
    - Letterbox resize manteniendo aspect ratio (evita distorsión)
    - Normaliza píxeles a [0, 1] para mejor convergencia del modelo
    - Convierte a RGB si es necesario (OpenCV usa BGR por defecto)

    Args:
        image_bytes: Bytes de la imagen
        target_size: Tamaño objetivo (default: 640 para YOLOv8)
        normalize: Si se debe normalizar píxeles a [0, 1]

    Returns:
        Imagen procesada como numpy array de shape (target_size, target_size, 3)
        con valores en [0, 1] si normalize=True, o [0, 255] si normalize=False

    Raises:
        HTTPException 400: Si la imagen no puede ser procesada
    """
    try:
        # Convertir bytes a PIL Image
        image = Image.open(io.BytesIO(image_bytes))

        # Asegurar que está en modo RGB (algunos formatos usan RGBA o escala de grises)
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Convertir a numpy array
        image_array = np.array(image)

        # Aplicar letterbox resize para mantener aspect ratio
        image_resized = letterbox_resize(image_array, target_size)

        # Normalizar si se solicita
        if normalize:
            image_resized = image_resized.astype(np.float32) / 255.0

        return image_resized

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=f"Error al preprocesar imagen: {str(e)}"
        )


def letterbox_resize(
    image: np.ndarray,
    target_size: int,
    color: Tuple[int, int, int] = (114, 114, 114),
) -> np.ndarray:
    """
    Redimensiona imagen manteniendo aspect ratio (letterbox).

    Este método es el estándar de YOLO para evitar distorsión.
    Agrega padding gris en los bordes cuando es necesario para crear
    una imagen cuadrada sin alterar las proporciones del contenido.

    Por qué es importante:
    - Mantiene aspect ratio original (evita que objetos se vean distorsionados)
    - Mejora precisión de detección para objetos elongados (chalecos, personas)
    - Método estándar usado en entrenamiento de YOLO

    Args:
        image: Imagen como numpy array de shape (H, W, C)
        target_size: Tamaño objetivo (cuadrado)
        color: Color del padding en RGB (default: gris 114,114,114)

    Returns:
        Imagen redimensionada con padding de shape (target_size, target_size, C)
    """
    h, w = image.shape[:2]

    # Calcular scale factor para que el lado más largo sea target_size
    scale = target_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)

    # Resize manteniendo aspect ratio
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Crear canvas cuadrado con color de padding
    canvas = np.full((target_size, target_size, 3), color, dtype=np.uint8)

    # Calcular offsets para centrar la imagen en el canvas
    offset_x = (target_size - new_w) // 2
    offset_y = (target_size - new_h) // 2

    # Colocar imagen redimensionada en el centro del canvas
    canvas[offset_y : offset_y + new_h, offset_x : offset_x + new_w] = resized

    return canvas


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

    NOTA: Esta función es legacy. La funcionalidad real de postprocessing
    está implementada en EPPDetector._postprocess() en api/model.py.

    Args:
        results: Output del modelo YOLOv8/ONNX
        class_names: Mapeo de IDs a nombres de clases (inglés)
        image_width: Ancho original de la imagen
        image_height: Alto original de la imagen

    Returns:
        Lista de detecciones formateadas con estructura estándar

    Raises:
        DeprecationWarning: Esta función está deprecada, usar EPPDetector directamente
    """
    import warnings

    warnings.warn(
        "format_detections() está deprecada. Usar EPPDetector._postprocess() directamente.",
        DeprecationWarning,
        stacklevel=2,
    )

    # Para compatibilidad, retornar lista vacía
    # La implementación real está en api/model.py
    return []


def apply_nms(
    boxes: List[List[float]],
    scores: List[float],
    class_ids: List[int],
    iou_threshold: float = 0.45,
) -> List[int]:
    """
    Aplica Non-Maximum Suppression manualmente.

    NMS elimina detecciones duplicadas del mismo objeto al suprimir
    cajas con alta superposición (IoU) manteniendo solo la de mayor confianza.

    Útil cuando el modelo no aplica NMS automáticamente (algunos exports de ONNX).
    Aplica NMS por clase para no suprimir entre clases diferentes.

    Args:
        boxes: Lista de bounding boxes [x_min, y_min, x_max, y_max]
        scores: Lista de scores de confianza
        class_ids: Lista de IDs de clase
        iou_threshold: Umbral IoU para considerar overlap (default: 0.45)

    Returns:
        Índices de las detecciones a mantener

    Raises:
        ValueError: Si las listas tienen longitudes diferentes
    """
    if not boxes:
        return []

    if not (len(boxes) == len(scores) == len(class_ids)):
        raise ValueError(
            f"Longitudes inconsistentes: boxes={len(boxes)}, "
            f"scores={len(scores)}, class_ids={len(class_ids)}"
        )

    # Convertir a numpy arrays para operaciones vectorizadas
    boxes_array = np.array(boxes)
    scores_array = np.array(scores)
    class_ids_array = np.array(class_ids)

    # Aplicar NMS por clase (no suprimir entre clases diferentes)
    keep_indices = []
    unique_classes = np.unique(class_ids_array)

    for class_id in unique_classes:
        # Filtrar detecciones de esta clase
        class_mask = class_ids_array == class_id
        class_boxes = boxes_array[class_mask]
        class_scores = scores_array[class_mask]
        class_indices = np.where(class_mask)[0]

        # Ordenar por score (descendente)
        sorted_indices = np.argsort(class_scores)[::-1]

        while len(sorted_indices) > 0:
            # Mantener el de mayor score
            best_idx = sorted_indices[0]
            keep_indices.append(class_indices[best_idx])

            if len(sorted_indices) == 1:
                break

            # Calcular IoU con el resto
            best_box = class_boxes[best_idx]
            remaining_boxes = class_boxes[sorted_indices[1:]]

            # Calcular IoU vectorizado
            ious = np.array(
                [calculate_iou(best_box.tolist(), box.tolist()) for box in remaining_boxes]
            )

            # Mantener solo boxes con IoU < threshold
            mask = ious < iou_threshold
            sorted_indices = sorted_indices[1:][mask]

    return sorted(keep_indices)


def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """
    Calcula Intersection over Union entre dos bounding boxes.

    IoU mide el overlap entre dos cajas como ratio:
    IoU = Area de Intersección / Area de Unión

    Valores cercanos a 1 indican alta superposición (probablemente mismo objeto).
    Valores cercanos a 0 indican poca o nula superposición.

    Args:
        box1: [x_min, y_min, x_max, y_max]
        box2: [x_min, y_min, x_max, y_max]

    Returns:
        IoU score entre 0 y 1

    Raises:
        ValueError: Si las cajas tienen coordenadas inválidas
    """
    # Validar que las coordenadas forman cajas válidas
    if box1[2] <= box1[0] or box1[3] <= box1[1]:
        raise ValueError(f"Caja 1 inválida (x_max <= x_min o y_max <= y_min): {box1}")
    if box2[2] <= box2[0] or box2[3] <= box2[1]:
        raise ValueError(f"Caja 2 inválida (x_max <= x_min o y_max <= y_min): {box2}")

    # Calcular coordenadas de la intersección
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    # Calcular área de intersección (0 si no hay overlap)
    inter_width = max(0.0, x2_inter - x1_inter)
    inter_height = max(0.0, y2_inter - y1_inter)
    intersection = inter_width * inter_height

    # Calcular áreas individuales
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Calcular unión (área total cubierta por ambas cajas)
    union = area1 + area2 - intersection

    # Retornar IoU (manejar caso de unión = 0 para evitar división por cero)
    if union <= 0.0:
        return 0.0

    return intersection / union


# ============================================================================
# Análisis de cumplimiento de EPP
# ============================================================================


def check_epp_compliance(detections: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Evalúa si una persona cumple con EPP obligatorio según DS 132.

    Lógica de cumplimiento basada en dataset actualizado (hardhat, safety_vest, no_hardhat, no_safety_vest, person):
    - COMPLIANT: Si se detectan "hardhat" Y "safety_vest", y NO se detectan "no_hardhat" ni "no_safety_vest"
    - NON_COMPLIANT: Si se detecta "no_hardhat" o "no_safety_vest" (violación crítica)
    - PARTIAL_COMPLIANT: Si se detecta solo hardhat o solo safety_vest
    - UNKNOWN: Si solo se detecta "person" sin información de EPP

    Regulación DS 132 Art. 42 requiere:
    - Casco de seguridad (hardhat)
    - Chaleco reflectante (safety_vest) para identificación y visibilidad

    Args:
        detections: Lista de detecciones de EPP con class_name en inglés

    Returns:
        {
            'compliant': bool,
            'violations': list,  # Violaciones detectadas en español
            'summary': str,      # Resumen en español
            'confidence_avg': float
        }

    TODO: Considerar overlapping de bounding boxes (persona-EPP)
    TODO: Detectar múltiples personas y evaluar individualmente
    TODO: Agregar severidad de violaciones (crítico vs warning)
    TODO: Integrar con tracking para evaluar en secuencias de video
    """
    detected_classes = {det["class_name"] for det in detections}

    violations = []
    compliant = True

    # Regla 1: Detectar violaciones críticas (no_hardhat, no_safety_vest)
    if "no_hardhat" in detected_classes:
        compliant = False
        violations.append("Persona sin casco detectada (violación crítica DS 132)")

    if "no_safety_vest" in detected_classes:
        compliant = False
        violations.append("Persona sin chaleco reflectante detectada (violación DS 132 Art. 42)")

    # Regla 2: Verificar cumplimiento completo (ambos EPP requeridos)
    has_hardhat = "hardhat" in detected_classes
    has_safety_vest = "safety_vest" in detected_classes
    has_person = "person" in detected_classes

    # Si se detectó persona pero falta EPP
    if has_person and not violations:
        if not has_hardhat and not has_safety_vest:
            compliant = False
            violations.append("No se puede verificar uso de EPP (casco y chaleco no detectados)")
        elif not has_hardhat:
            compliant = False
            violations.append("Falta casco de seguridad (solo chaleco detectado)")
        elif not has_safety_vest:
            compliant = False
            violations.append("Falta chaleco reflectante (solo casco detectado)")

    # Generar resumen en español
    if compliant and has_hardhat and has_safety_vest:
        summary = "Cumplimiento de EPP verificado: Casco y chaleco reflectante detectados (DS 132)"
    elif not violations:
        summary = "No se detectaron personas en la imagen"
    else:
        summary = f"Violación de seguridad: {'; '.join(violations)}"

    return {
        "compliant": compliant,
        "violations": violations,
        "summary": summary,
        "confidence_avg": (
            sum(d["confidence"] for d in detections) / len(detections) if detections else 0.0
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
