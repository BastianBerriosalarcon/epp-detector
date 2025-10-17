"""
Protocol definitions for dependency inversion.

This module defines abstract interfaces (protocols) that allow
the API layer to depend on abstractions rather than concrete implementations,
following the Dependency Inversion Principle (DIP).
"""

from typing import Protocol, List, Dict, Any, Optional
import numpy as np
from pathlib import Path


class DetectorProtocol(Protocol):
    """Protocol for PPE detection models.

    This abstract interface allows the API to work with any detector
    implementation (YOLOv8, ONNX, TensorRT, etc.) without tight coupling.

    The protocol defines the contract that any detector must fulfill,
    enabling easy testing with mocks and future model swaps.

    Example:
        >>> def process_image(detector: DetectorProtocol, image: np.ndarray):
        ...     results = detector.predict(image)
        ...     return results
    """

    @property
    def is_loaded(self) -> bool:
        """Indicates whether the model is loaded and ready for inference.

        Returns:
            True if model is loaded, False otherwise
        """
        ...

    @property
    def model_type(self) -> Optional[str]:
        """Type of model backend being used.

        Returns:
            String like 'pytorch', 'onnx', or None if not loaded
        """
        ...

    @property
    def input_size(self) -> int:
        """Expected input size for the model.

        Returns:
            Input size in pixels (e.g., 640 for YOLOv8)
        """
        ...

    @property
    def confidence_threshold(self) -> float:
        """Minimum confidence score for detections.

        Returns:
            Confidence threshold between 0.0 and 1.0
        """
        ...

    @property
    def class_names(self) -> Dict[int, str]:
        """Mapping of class IDs to class names.

        Returns:
            Dictionary mapping integer IDs to string names
        """
        ...

    def predict(self, image: Any, return_format: str = "dict") -> List[Dict[str, Any]]:
        """Perform detection on a single image.

        Args:
            image: Image in PIL, numpy array, or path format
            return_format: Format for results ('dict' or 'yolo')

        Returns:
            List of detections with bboxes, confidence, and class info

        Raises:
            ModelNotLoadedError: If model is not loaded
            InferenceError: If prediction fails
        """
        ...

    def predict_batch(self, images: List[Any], batch_size: int = 8) -> List[List[Dict[str, Any]]]:
        """Perform detection on multiple images in batches.

        Batching improves throughput by processing multiple images
        in a single forward pass through the model.

        Args:
            images: List of images
            batch_size: Number of images to process per batch

        Returns:
            List of detection lists (one per image)

        Raises:
            ModelNotLoadedError: If model is not loaded
            InferenceError: If prediction fails
        """
        ...

    def get_model_info(self) -> Dict[str, Any]:
        """Get metadata about the loaded model.

        Returns:
            Dictionary with model information:
            - model_path: Path to model file
            - model_type: Backend type (pytorch/onnx)
            - input_size: Expected input dimensions
            - classes: Available detection classes
            - thresholds: Confidence and IoU thresholds
        """
        ...

    def warmup(self, num_iterations: int = 5) -> None:
        """Perform warmup inference to reduce first-call latency.

        Executes dummy predictions to initialize GPU kernels and
        allocate memory, significantly reducing cold-start delays.

        Args:
            num_iterations: Number of warmup iterations
        """
        ...


class ImageValidatorProtocol(Protocol):
    """Protocol for image validation.

    Defines the interface for validating uploaded images before
    processing, ensuring they meet size, format, and dimension requirements.
    """

    def validate_format(self, filename: str) -> None:
        """Validate that image format is supported.

        Args:
            filename: Name or path of the file

        Raises:
            InvalidImageFormatError: If format is not supported
        """
        ...

    def validate_size(self, size_bytes: int, filename: str = None) -> None:
        """Validate that image size is within limits.

        Args:
            size_bytes: Size of the image in bytes
            filename: Optional filename for error messages

        Raises:
            ImageTooLargeError: If image exceeds maximum size
        """
        ...

    def validate_dimensions(self, width: int, height: int, filename: str = None) -> None:
        """Validate that image dimensions are acceptable.

        Args:
            width: Image width in pixels
            height: Image height in pixels
            filename: Optional filename for error messages

        Raises:
            InvalidImageDimensionsError: If dimensions are invalid
        """
        ...

    def validate_image_data(self, image_bytes: bytes, filename: str = None) -> None:
        """Validate that image data is not corrupted.

        Args:
            image_bytes: Raw image bytes
            filename: Optional filename for error messages

        Raises:
            InvalidImageError: If image is corrupted or invalid
        """
        ...


class ModelLoaderProtocol(Protocol):
    """Protocol for model loading strategies.

    Allows different model loading implementations with retry logic,
    remote loading from cloud storage, etc.
    """

    def load_model(self, model_path: Path) -> Any:
        """Load model from specified path.

        Args:
            model_path: Path to model file

        Returns:
            Loaded model instance

        Raises:
            ModelLoadError: If loading fails
        """
        ...

    def load_with_retry(
        self, model_path: Path, max_retries: int = 3, retry_delay: float = 1.0
    ) -> Any:
        """Load model with retry logic for transient failures.

        Args:
            model_path: Path to model file
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds

        Returns:
            Loaded model instance

        Raises:
            ModelLoadError: If all retries fail
        """
        ...
