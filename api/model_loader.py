"""
Model loading utilities with retry logic and resource management.

This module provides robust model loading functionality with:
- Retry logic for transient failures
- Proper error handling and logging
- Support for multiple model formats (PyTorch, ONNX)
"""

import time
import logging
from pathlib import Path
from typing import Any, Optional, Union

from api.config import Settings
from api.exceptions import ModelLoadError

logger = logging.getLogger(__name__)


class ModelLoader:
    """Loads ML models with retry logic and error handling.

    This class handles the complexity of model loading, including:
    - Automatic format detection
    - Retry logic for transient failures
    - GPU/CPU device selection
    - Detailed error logging

    Follows SRP by focusing solely on model loading, separate from
    inference logic.

    Attributes:
        settings: Configuration settings
    """

    def __init__(self, settings: Settings) -> None:
        """Initialize model loader with configuration.

        Args:
            settings: Settings instance with model configuration
        """
        self.settings = settings

    def load_model(self, model_path: Path) -> Any:
        """Load model from specified path.

        Automatically detects model format and uses appropriate loader.

        Args:
            model_path: Path to model file (.pt or .onnx)

        Returns:
            Loaded model instance (YOLO or ONNX session)

        Raises:
            ModelLoadError: If loading fails
        """
        if not model_path.exists():
            raise ModelLoadError(
                model_path=str(model_path), reason=f"Model file not found at {model_path}"
            )

        suffix = model_path.suffix.lower()
        logger.info(f"Loading model from {model_path} (format: {suffix})")

        try:
            if suffix == ".pt":
                return self._load_pytorch_model(model_path)
            elif suffix == ".onnx":
                return self._load_onnx_model(model_path)
            else:
                raise ModelLoadError(
                    model_path=str(model_path),
                    reason=f"Unsupported model format: {suffix}. Use .pt or .onnx",
                )

        except ModelLoadError:
            # Re-raise our custom exceptions
            raise
        except Exception as e:
            logger.error(f"Unexpected error loading model: {e}", exc_info=True)
            raise ModelLoadError(
                model_path=str(model_path), reason=f"Unexpected error: {str(e)}"
            ) from e

    def load_with_retry(
        self, model_path: Path, max_retries: int = 3, retry_delay: float = 1.0
    ) -> Any:
        """Load model with retry logic for transient failures.

        This is useful for handling temporary issues like:
        - Network file system delays
        - Temporary GPU memory issues
        - Race conditions in multi-process environments

        Args:
            model_path: Path to model file
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds

        Returns:
            Loaded model instance

        Raises:
            ModelLoadError: If all retries fail
        """
        last_error: Optional[Exception] = None

        for attempt in range(1, max_retries + 1):
            try:
                logger.info(
                    f"Model loading attempt {attempt}/{max_retries} " f"for {model_path.name}"
                )
                model = self.load_model(model_path)
                logger.info(f"Model loaded successfully on attempt {attempt}")
                return model

            except ModelLoadError as e:
                last_error = e
                logger.warning(f"Model loading attempt {attempt}/{max_retries} failed: {e.message}")

                if attempt < max_retries:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    # Exponential backoff
                    retry_delay *= 2
                else:
                    logger.error(f"All {max_retries} attempts failed")

        # All retries exhausted
        raise ModelLoadError(
            model_path=str(model_path),
            reason=f"Failed after {max_retries} attempts. Last error: {last_error.message}",
        ) from last_error

    def _load_pytorch_model(self, model_path: Path) -> Any:
        """Load PyTorch YOLOv8 model.

        Args:
            model_path: Path to .pt file

        Returns:
            YOLO model instance

        Raises:
            ModelLoadError: If loading fails
        """
        try:
            from ultralytics import YOLO

            logger.debug(f"Loading PyTorch model from {model_path}")

            # Load model
            model = YOLO(str(model_path))

            # Optionally move to GPU
            if self.settings.enable_gpu:
                import torch

                if torch.cuda.is_available():
                    logger.info("Moving model to GPU (CUDA)")
                    # YOLOv8 handles device automatically via model.to()
                    model.to("cuda")
                else:
                    logger.warning("GPU requested but CUDA not available, using CPU")

            logger.info(f"PyTorch model loaded successfully from {model_path}")
            return model

        except ImportError as e:
            raise ModelLoadError(
                model_path=str(model_path),
                reason="ultralytics package not installed. Run: pip install ultralytics",
            ) from e
        except Exception as e:
            raise ModelLoadError(
                model_path=str(model_path), reason=f"Failed to load PyTorch model: {str(e)}"
            ) from e

    def _load_onnx_model(self, model_path: Path) -> Any:
        """Load ONNX Runtime model.

        ONNX provides optimized inference with support for various
        execution providers (CPU, CUDA, TensorRT, etc.).

        Args:
            model_path: Path to .onnx file

        Returns:
            ONNX InferenceSession

        Raises:
            ModelLoadError: If loading fails
        """
        try:
            import onnxruntime as ort

            logger.debug(f"Loading ONNX model from {model_path}")

            # Configure execution providers
            providers = ["CPUExecutionProvider"]
            if self.settings.enable_gpu:
                import torch

                if torch.cuda.is_available():
                    # Prefer CUDA over CPU
                    providers.insert(0, "CUDAExecutionProvider")
                    logger.info("ONNX will use CUDA execution provider")
                else:
                    logger.warning("GPU requested but CUDA not available, using CPU")

            # Create inference session
            session = ort.InferenceSession(str(model_path), providers=providers)

            # Log session info
            logger.info(
                f"ONNX model loaded successfully from {model_path}. "
                f"Providers: {session.get_providers()}"
            )

            # Log input/output info for debugging
            input_info = session.get_inputs()[0]
            output_info = session.get_outputs()[0]
            logger.debug(
                f"ONNX model expects input '{input_info.name}' " f"with shape {input_info.shape}"
            )
            logger.debug(
                f"ONNX model produces output '{output_info.name}' "
                f"with shape {output_info.shape}"
            )

            return session

        except ImportError as e:
            raise ModelLoadError(
                model_path=str(model_path),
                reason="onnxruntime package not installed. Run: pip install onnxruntime or onnxruntime-gpu",
            ) from e
        except Exception as e:
            raise ModelLoadError(
                model_path=str(model_path), reason=f"Failed to load ONNX model: {str(e)}"
            ) from e


def create_model_loader(settings: Settings) -> ModelLoader:
    """Factory function to create a ModelLoader instance.

    Useful for dependency injection in tests and production code.

    Args:
        settings: Settings instance

    Returns:
        Configured ModelLoader instance
    """
    return ModelLoader(settings)
