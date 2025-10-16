"""
Custom exception classes for the EPP Detector API.

This module defines a hierarchy of exceptions for better error handling
and context-specific error messages throughout the application.
"""


class EPPDetectorError(Exception):
    """Base exception for all EPP detector errors.

    All custom exceptions in this module inherit from this base class,
    allowing for catch-all exception handling when needed.
    """

    def __init__(self, message: str, details: str = None) -> None:
        """Initialize exception with message and optional details.

        Args:
            message: Human-readable error message
            details: Additional context or technical details
        """
        self.message = message
        self.details = details
        super().__init__(self.message)


class ModelNotLoadedError(EPPDetectorError):
    """Raised when attempting inference without a loaded model.

    This typically occurs when the model fails to load during initialization
    or when trying to use a detector instance before model loading completes.
    """

    def __init__(self, model_path: str = None) -> None:
        """Initialize with optional model path for context.

        Args:
            model_path: Path where model was expected to be found
        """
        message = "Model not loaded. Cannot perform inference."
        details = f"Expected model at: {model_path}" if model_path else None
        super().__init__(message, details)


class ModelLoadError(EPPDetectorError):
    """Raised when model loading fails.

    Can occur due to:
    - Missing model file
    - Corrupted model weights
    - Incompatible model format
    - Insufficient GPU memory
    """

    def __init__(self, model_path: str, reason: str = None) -> None:
        """Initialize with model path and failure reason.

        Args:
            model_path: Path to the model that failed to load
            reason: Specific reason for the failure
        """
        message = f"Failed to load model from {model_path}"
        details = reason if reason else "Unknown error during model loading"
        super().__init__(message, details)


class InferenceError(EPPDetectorError):
    """Raised when model inference fails.

    Can occur due to:
    - Invalid input shape
    - Out of memory during inference
    - CUDA errors
    - Model runtime errors
    """

    def __init__(self, reason: str, input_shape: tuple = None) -> None:
        """Initialize with failure reason and optional input shape.

        Args:
            reason: Specific reason for inference failure
            input_shape: Shape of the input that caused the error
        """
        message = f"Inference failed: {reason}"
        details = f"Input shape: {input_shape}" if input_shape else None
        super().__init__(message, details)


class InvalidImageError(EPPDetectorError):
    """Raised when image validation fails.

    Can occur due to:
    - Unsupported image format
    - Corrupted image file
    - Image too large
    - Invalid dimensions
    """

    def __init__(self, reason: str, filename: str = None) -> None:
        """Initialize with failure reason and optional filename.

        Args:
            reason: Specific validation failure reason
            filename: Name of the file that failed validation
        """
        message = f"Invalid image: {reason}"
        details = f"File: {filename}" if filename else None
        super().__init__(message, details)


class InvalidImageFormatError(InvalidImageError):
    """Raised when image format is not supported.

    The API only accepts specific image formats (JPEG, PNG, BMP).
    """

    def __init__(self, format_detected: str, filename: str = None) -> None:
        """Initialize with detected format.

        Args:
            format_detected: The format that was detected
            filename: Name of the file
        """
        reason = f"Unsupported format: {format_detected}. Use JPEG, PNG, or BMP."
        super().__init__(reason, filename)


class ImageTooLargeError(InvalidImageError):
    """Raised when image exceeds maximum allowed size.

    This prevents memory exhaustion from extremely large images.
    """

    def __init__(self, size_mb: float, max_mb: float, filename: str = None) -> None:
        """Initialize with size information.

        Args:
            size_mb: Actual size of the image in MB
            max_mb: Maximum allowed size in MB
            filename: Name of the file
        """
        reason = f"Image size {size_mb:.2f}MB exceeds maximum {max_mb}MB"
        super().__init__(reason, filename)


class InvalidImageDimensionsError(InvalidImageError):
    """Raised when image dimensions are invalid.

    Images must be within minimum and maximum dimension constraints
    for reliable detection performance.
    """

    def __init__(
        self,
        width: int,
        height: int,
        min_dim: int = None,
        max_dim: int = None,
        filename: str = None
    ) -> None:
        """Initialize with dimension information.

        Args:
            width: Image width in pixels
            height: Image height in pixels
            min_dim: Minimum allowed dimension
            max_dim: Maximum allowed dimension
            filename: Name of the file
        """
        if min_dim and (width < min_dim or height < min_dim):
            reason = f"Image dimensions {width}x{height} below minimum {min_dim}x{min_dim}"
        elif max_dim and (width > max_dim or height > max_dim):
            reason = f"Image dimensions {width}x{height} exceed maximum {max_dim}x{max_dim}"
        else:
            reason = f"Invalid dimensions: {width}x{height}"
        super().__init__(reason, filename)


class ConfigurationError(EPPDetectorError):
    """Raised when configuration is invalid or missing.

    Can occur due to:
    - Missing required environment variables
    - Invalid configuration values
    - Conflicting configuration settings
    """

    def __init__(self, setting: str, reason: str) -> None:
        """Initialize with setting name and reason.

        Args:
            setting: Name of the configuration setting
            reason: Why the configuration is invalid
        """
        message = f"Configuration error for '{setting}': {reason}"
        super().__init__(message)


class ResourceCleanupError(EPPDetectorError):
    """Raised when resource cleanup fails.

    This is a non-critical error but should be logged for monitoring
    potential resource leaks.
    """

    def __init__(self, resource: str, reason: str) -> None:
        """Initialize with resource type and failure reason.

        Args:
            resource: Type of resource that failed to cleanup
            reason: Why cleanup failed
        """
        message = f"Failed to cleanup {resource}: {reason}"
        super().__init__(message)
