"""
Módulo de inferencia para detección de EPP con YOLOv8.

Este módulo encapsula la lógica de carga del modelo, inferencia,
y postprocesamiento de resultados. Soporta modelos en formato
PyTorch (.pt) y ONNX Runtime (.onnx) para deployment optimizado.
"""

import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from contextlib import contextmanager

import numpy as np

from api.config import Settings
from api.exceptions import (
    ModelNotLoadedError,
    ModelLoadError,
    InferenceError,
    ResourceCleanupError,
)
from api.model_loader import ModelLoader

logger = logging.getLogger(__name__)


class EPPDetector:
    """
    Detector de Equipos de Protección Personal basado en YOLOv8.

    Esta clase maneja la inferencia optimizada para detección de:
    - Cascos de seguridad
    - Chalecos reflectantes
    - Zapatos de seguridad
    - Ausencia de EPP (para alertas)

    Implements context manager protocol for proper resource cleanup.
    Follows SRP by delegating loading to ModelLoader.

    Attributes:
        model: Instancia del modelo YOLOv8 o ONNX Runtime session
        model_type: Tipo de modelo ('pytorch' o 'onnx')
        input_size: Tamaño de entrada esperado por el modelo (default: 640x640)
        class_names: Mapeo de IDs a nombres de clases
        is_loaded: Indica si el modelo está cargado correctamente
        settings: Configuration settings instance

    Example:
        >>> with EPPDetector(settings) as detector:
        ...     detections = detector.predict(image_array)
        ...     for det in detections:
        ...         print(f"{det['class_name']}: {det['confidence']:.2f}")
    """

    def __init__(
        self,
        settings: Settings,
        model_loader: Optional[ModelLoader] = None,
        auto_warmup: bool = True,
    ) -> None:
        """
        Inicializa el detector y carga el modelo.

        Args:
            settings: Settings instance with configuration
            model_loader: Optional ModelLoader instance (for dependency injection)
            auto_warmup: Whether to perform warmup after loading

        Raises:
            ModelLoadError: Si falla la carga del modelo

        Note:
            Model loading uses retry logic by default (3 attempts).
            Warmup is performed automatically unless auto_warmup=False.
        """
        self.settings = settings
        self.model_path = settings.get_model_path_absolute()
        self.input_size = settings.input_size
        self.confidence_threshold = settings.confidence_threshold
        self.iou_threshold = settings.iou_threshold

        self.model: Optional[Any] = None
        self.model_type: Optional[str] = None
        self.is_loaded: bool = False

        # Class mapping (must match Roboflow dataset)
        # NOTE: Using English names from Roboflow Hard Hat Workers dataset
        self.class_names: Dict[int, str] = {
            0: "hardhat",
            1: "head",  # head without hardhat (non-compliant)
            2: "person",
        }

        # Initialize model loader
        self._model_loader = model_loader or ModelLoader(settings)

        # Load model with retry logic
        self._load_model()

        # Perform warmup to reduce first inference latency
        if auto_warmup and self.is_loaded and settings.enable_warmup:
            self.warmup(num_iterations=settings.warmup_iterations)

    def _load_model(self) -> None:
        """
        Carga el modelo desde disco con retry logic.

        Uses ModelLoader with automatic retry for transient failures.
        Detects model format (PyTorch or ONNX) automatically.

        Raises:
            ModelLoadError: If model loading fails after all retries
        """
        logger.info(f"Loading model from {self.model_path}")

        try:
            # Load with retry (3 attempts by default)
            self.model = self._model_loader.load_with_retry(
                model_path=self.model_path, max_retries=3, retry_delay=1.0
            )

            # Detect model type from file extension
            suffix = self.model_path.suffix.lower()
            self.model_type = "pytorch" if suffix == ".pt" else "onnx"
            self.is_loaded = True

            logger.info(
                f"Model loaded successfully: {self.model_path.name} " f"(type: {self.model_type})"
            )

        except ModelLoadError as e:
            logger.error(f"Failed to load model: {e.message}")
            self.is_loaded = False
            raise

    def warmup(self, num_iterations: int = 5) -> None:
        """
        Realiza inferencias dummy para warmup del modelo.

        Warmup reduces cold-start latency by:
        - Initializing GPU kernels
        - Allocating memory buffers
        - JIT compilation of operations

        Args:
            num_iterations: Número de inferencias de warmup

        Raises:
            ModelNotLoadedError: If model is not loaded
        """
        if not self.is_loaded or self.model is None:
            raise ModelNotLoadedError(str(self.model_path))

        logger.info(f"Starting warmup with {num_iterations} iterations...")
        start_time = time.time()

        # Create dummy image of correct size
        dummy_image = np.random.randint(
            0, 255, (self.input_size, self.input_size, 3), dtype=np.uint8
        )

        # Run warmup iterations
        for i in range(num_iterations):
            try:
                _ = self._run_inference(dummy_image)
            except Exception as e:
                logger.warning(f"Warmup iteration {i+1} failed: {e}")

        elapsed = (time.time() - start_time) * 1000
        avg_time = elapsed / num_iterations

        logger.info(
            f"Warmup completed: {num_iterations} iterations in {elapsed:.2f}ms "
            f"(avg: {avg_time:.2f}ms per iteration)"
        )

    def predict(
        self,
        image: Any,
        return_format: str = "dict",
    ) -> List[Dict[str, Any]]:
        """
        Realiza detección de EPP en una imagen.

        Args:
            image: Imagen en formato PIL, numpy array, o ruta a archivo
            return_format: Formato de retorno ('dict' o 'yolo')

        Returns:
            Lista de detecciones con formato:
            [
                {
                    'class_id': 0,
                    'class_name': 'hardhat',
                    'confidence': 0.92,
                    'bbox': [x_min, y_min, x_max, y_max],
                    'bbox_norm': [x_center, y_center, width, height]
                },
                ...
            ]

        Raises:
            ModelNotLoadedError: Si el modelo no está cargado
            InferenceError: Si la inferencia falla
        """
        if not self.is_loaded or self.model is None:
            raise ModelNotLoadedError(str(self.model_path))

        try:
            # Preprocess image
            processed_image = self._preprocess(image)

            # Run inference
            results = self._run_inference(processed_image)

            # Postprocess results
            detections = self._postprocess(results)

            return detections

        except (ModelNotLoadedError, InferenceError):
            # Re-raise our custom exceptions
            raise
        except Exception as e:
            logger.error(f"Unexpected error during prediction: {e}", exc_info=True)
            raise InferenceError(reason=str(e), input_shape=getattr(image, "shape", None)) from e

    def predict_batch(
        self,
        images: List[Any],
        batch_size: int = 8,
    ) -> List[List[Dict[str, Any]]]:
        """
        Realiza detección en batch para múltiples imágenes.

        Batch processing improves throughput by:
        - Reducing per-image overhead
        - Better GPU utilization
        - Fewer kernel launches

        Args:
            images: Lista de imágenes
            batch_size: Tamaño del batch para inferencia

        Returns:
            Lista de listas de detecciones (una lista por imagen)

        Raises:
            ModelNotLoadedError: If model is not loaded
            InferenceError: If batch inference fails
        """
        if not self.is_loaded or self.model is None:
            raise ModelNotLoadedError(str(self.model_path))

        if not images:
            return []

        logger.debug(f"Processing {len(images)} images in batches of {batch_size}")

        all_results: List[List[Dict[str, Any]]] = []

        # Process images in batches
        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size]

            try:
                # Preprocess batch
                processed_batch = [self._preprocess(img) for img in batch]

                # Stack into batch tensor for ONNX, or keep as list for PyTorch
                if self.model_type == "onnx":
                    batch_array = np.stack(processed_batch)
                    results = self._run_inference(batch_array)
                else:
                    # PyTorch YOLO can handle list of images
                    results = self._run_inference(processed_batch)

                # Postprocess each result
                batch_detections = [self._postprocess(r) for r in results]
                all_results.extend(batch_detections)

            except Exception as e:
                logger.error(f"Batch inference failed at batch {i//batch_size}: {e}")
                raise InferenceError(reason=f"Batch processing failed: {str(e)}") from e

        logger.debug(f"Batch processing completed: {len(all_results)} results")
        return all_results

    def _preprocess(self, image: Any) -> np.ndarray:
        """
        Preprocesa imagen para inferencia.

        Converts image to format expected by model:
        - Resize to input_size with letterbox (maintains aspect ratio)
        - Normalize pixels to [0, 1]
        - Convert to NCHW format for ONNX

        Args:
            image: Imagen en formato PIL, numpy, o path

        Returns:
            Imagen procesada como numpy array

        Raises:
            InferenceError: If preprocessing fails
        """
        try:
            # Convert to numpy array if needed
            if isinstance(image, (str, Path)):
                from PIL import Image

                image = Image.open(image)
                image = np.array(image)
            elif hasattr(image, "convert"):  # PIL Image
                image = np.array(image)

            # Ensure RGB format
            if len(image.shape) == 2:
                # Grayscale to RGB
                image = np.stack([image] * 3, axis=-1)
            elif image.shape[2] == 4:
                # RGBA to RGB
                image = image[:, :, :3]

            # Letterbox resize (maintains aspect ratio)
            image = self._letterbox_resize(image, self.input_size)

            # Normalize to [0, 1]
            image = image.astype(np.float32) / 255.0

            # Convert to NCHW format for ONNX
            if self.model_type == "onnx":
                image = np.transpose(image, (2, 0, 1))  # HWC -> CHW
                image = np.expand_dims(image, axis=0)  # Add batch dimension

            return image

        except Exception as e:
            raise InferenceError(f"Preprocessing failed: {str(e)}") from e

    def _letterbox_resize(
        self, image: np.ndarray, target_size: int, color: tuple = (114, 114, 114)
    ) -> np.ndarray:
        """
        Resize image with letterboxing to maintain aspect ratio.

        This is the standard YOLO preprocessing approach that adds
        gray padding to maintain aspect ratio without distortion.

        Args:
            image: Input image array (H, W, C)
            target_size: Target size (square)
            color: Padding color (default: gray)

        Returns:
            Resized image with padding (target_size, target_size, C)
        """
        import cv2

        h, w = image.shape[:2]
        scale = target_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)

        # Resize image
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Create canvas with padding
        canvas = np.full((target_size, target_size, 3), color, dtype=np.uint8)

        # Calculate padding offsets (center the image)
        offset_x = (target_size - new_w) // 2
        offset_y = (target_size - new_h) // 2

        # Place resized image on canvas
        canvas[offset_y : offset_y + new_h, offset_x : offset_x + new_w] = resized

        return canvas

    def _run_inference(self, image: np.ndarray) -> Any:
        """
        Ejecuta inferencia con el modelo.

        Args:
            image: Imagen preprocesada

        Returns:
            Output crudo del modelo (formato depende de PyTorch/ONNX)

        Raises:
            InferenceError: If inference fails
        """
        try:
            if self.model_type == "pytorch":
                # YOLOv8 inference
                results = self.model(
                    image, conf=self.confidence_threshold, iou=self.iou_threshold, verbose=False
                )
                return results

            elif self.model_type == "onnx":
                # ONNX Runtime inference
                input_name = self.model.get_inputs()[0].name
                outputs = self.model.run(None, {input_name: image})
                return outputs

            else:
                raise InferenceError(f"Unknown model type: {self.model_type}")

        except Exception as e:
            logger.error(f"Inference failed: {e}", exc_info=True)
            raise InferenceError(f"Model execution failed: {str(e)}") from e

    def _postprocess(self, results: Any) -> List[Dict[str, Any]]:
        """
        Postprocesa resultados del modelo.

        Args:
            results: Output crudo del modelo

        Returns:
            Lista de detecciones formateadas

        Raises:
            InferenceError: If postprocessing fails
        """
        try:
            detections: List[Dict[str, Any]] = []

            if self.model_type == "pytorch":
                # YOLOv8 results format
                for result in results:
                    boxes = result.boxes
                    for box in boxes:
                        detection = {
                            "class_id": int(box.cls[0]),
                            "class_name": self.class_names[int(box.cls[0])],
                            "confidence": float(box.conf[0]),
                            "bbox": box.xyxy[0].cpu().numpy().tolist(),
                            "bbox_norm": box.xywhn[0].cpu().numpy().tolist(),
                        }
                        detections.append(detection)

            elif self.model_type == "onnx":
                # ONNX YOLOv8 output format: [batch, num_boxes, 5+num_classes]
                # Each detection: [x_center, y_center, width, height, *class_probs]
                # NOTE: YOLOv8 ONNX export concatenates bbox coords with class probs
                output = results[0]  # Shape: (batch, num_boxes, 5+num_classes)

                # Process each image in batch
                batch_size = output.shape[0]
                for batch_idx in range(batch_size):
                    predictions = output[batch_idx]  # Shape: (num_boxes, 5+num_classes)

                    # Extract bbox coordinates (center format) and class probabilities
                    boxes_center = predictions[:, :4]  # [x_center, y_center, w, h]
                    class_probs = predictions[:, 4:]  # [prob_class_0, prob_class_1, ...]

                    # Get class with highest probability for each box
                    class_ids = np.argmax(class_probs, axis=1)
                    confidences = np.max(class_probs, axis=1)

                    # Filter by confidence threshold
                    mask = confidences >= self.confidence_threshold
                    boxes_center = boxes_center[mask]
                    class_ids = class_ids[mask]
                    confidences = confidences[mask]

                    # Convert from center format to xyxy format
                    # center format: [x_center, y_center, width, height]
                    # xyxy format: [x_min, y_min, x_max, y_max]
                    boxes_xyxy = np.zeros_like(boxes_center)
                    boxes_xyxy[:, 0] = boxes_center[:, 0] - boxes_center[:, 2] / 2  # x_min
                    boxes_xyxy[:, 1] = boxes_center[:, 1] - boxes_center[:, 3] / 2  # y_min
                    boxes_xyxy[:, 2] = boxes_center[:, 0] + boxes_center[:, 2] / 2  # x_max
                    boxes_xyxy[:, 3] = boxes_center[:, 1] + boxes_center[:, 3] / 2  # y_max

                    # Scale coordinates from normalized [0,1] to pixel values
                    boxes_xyxy *= self.input_size

                    # Normalize bbox for bbox_norm (center format with normalized coords)
                    # Format: [x_center_norm, y_center_norm, width_norm, height_norm]
                    boxes_norm = boxes_center.copy()  # Already in [0,1] range

                    # Create detection dictionaries
                    for box_xyxy, box_norm, class_id, confidence in zip(
                        boxes_xyxy, boxes_norm, class_ids, confidences
                    ):
                        # Skip invalid class IDs
                        if int(class_id) not in self.class_names:
                            logger.warning(f"Skipping detection with invalid class_id: {class_id}")
                            continue

                        detection = {
                            "class_id": int(class_id),
                            "class_name": self.class_names[int(class_id)],
                            "confidence": float(confidence),
                            "bbox": box_xyxy.tolist(),
                            "bbox_norm": box_norm.tolist(),
                        }
                        detections.append(detection)

            return detections

        except Exception as e:
            logger.error(f"Postprocessing failed: {e}", exc_info=True)
            raise InferenceError(f"Result parsing failed: {str(e)}") from e

    def get_model_info(self) -> Dict[str, Any]:
        """
        Retorna información sobre el modelo cargado.

        Returns:
            Diccionario con metadata del modelo
        """
        return {
            "model_path": str(self.model_path),
            "model_type": self.model_type,
            "input_size": self.input_size,
            "classes": self.class_names,
            "confidence_threshold": self.confidence_threshold,
            "iou_threshold": self.iou_threshold,
            "is_loaded": self.is_loaded,
            "version": "0.1.0",
        }

    def __enter__(self) -> "EPPDetector":
        """Context manager entry.

        Returns:
            Self for use in with statement
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit with resource cleanup.

        Args:
            exc_type: Exception type if an error occurred
            exc_val: Exception value
            exc_tb: Exception traceback

        Note:
            Cleanup is performed regardless of whether an exception occurred.
        """
        self.cleanup()

    def cleanup(self) -> None:
        """
        Cleanup GPU memory and release model resources.

        Should be called when detector is no longer needed to free resources.
        Automatically called when using context manager.
        """
        if not self.is_loaded:
            return

        logger.info("Cleaning up detector resources...")

        try:
            if self.model_type == "pytorch":
                # Cleanup PyTorch resources
                del self.model
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    logger.debug("GPU cache cleared")

            elif self.model_type == "onnx":
                # Cleanup ONNX session
                # ONNX sessions are automatically cleaned up, but we can
                # explicitly delete the reference
                del self.model

            self.model = None
            self.is_loaded = False
            logger.info("Detector cleanup completed successfully")

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            raise ResourceCleanupError(resource="model", reason=str(e)) from e

    def __del__(self) -> None:
        """Destructor to ensure cleanup on garbage collection."""
        try:
            if self.is_loaded:
                self.cleanup()
        except Exception as e:
            # Suppress exceptions during destruction
            logger.error(f"Error in destructor: {e}")

    def __repr__(self) -> str:
        """Representación string del detector."""
        return (
            f"EPPDetector(model={self.model_path.name}, "
            f"type={self.model_type}, loaded={self.is_loaded})"
        )
