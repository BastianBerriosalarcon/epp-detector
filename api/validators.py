"""
Image validation utilities for the EPP Detector API.

This module provides comprehensive validation for uploaded images,
including format, size, dimensions, and data integrity checks.
"""

import io
import logging
from pathlib import Path
from typing import Optional

from PIL import Image

from api.config import Settings
from api.exceptions import (
    ImageTooLargeError,
    InvalidImageDimensionsError,
    InvalidImageError,
    InvalidImageFormatError,
)

logger = logging.getLogger(__name__)


class ImageValidator:
    """Validator for uploaded images.

    Performs comprehensive validation of image files to ensure they
    meet format, size, and dimension requirements before processing.

    This class follows the Single Responsibility Principle by focusing
    solely on image validation, separate from detection logic.

    Attributes:
        settings: Configuration settings for validation limits
    """

    def __init__(self, settings: Settings) -> None:
        """Initialize validator with configuration.

        Args:
            settings: Settings instance with validation limits
        """
        self.settings = settings
        self._allowed_extensions = set(settings.allowed_formats)
        self._max_size_bytes = settings.max_image_size_mb * 1024 * 1024
        self._min_dimension = settings.min_image_dimension
        self._max_dimension = settings.max_image_dimension

    def validate_format(self, filename: str) -> None:
        """Validate that image format is supported.

        Checks file extension against allowed formats. This is a fast
        preliminary check before attempting to load the image.

        Args:
            filename: Name or path of the file

        Raises:
            InvalidImageFormatError: If format is not supported
        """
        file_ext = Path(filename).suffix.lower()

        if file_ext not in self._allowed_extensions:
            logger.warning(
                f"Rejected image with unsupported format: {file_ext} " f"(file: {filename})"
            )
            raise InvalidImageFormatError(format_detected=file_ext, filename=filename)

        logger.debug(f"Image format validation passed: {file_ext}")

    def validate_size(self, size_bytes: int, filename: Optional[str] = None) -> None:
        """Validate that image size is within limits.

        Prevents memory exhaustion from extremely large images.
        This check happens before loading the full image into memory.

        Args:
            size_bytes: Size of the image in bytes
            filename: Optional filename for error messages

        Raises:
            ImageTooLargeError: If image exceeds maximum size
        """
        size_mb = size_bytes / (1024 * 1024)

        if size_bytes > self._max_size_bytes:
            logger.warning(
                f"Rejected oversized image: {size_mb:.2f}MB "
                f"(max: {self.settings.max_image_size_mb}MB, file: {filename})"
            )
            raise ImageTooLargeError(
                size_mb=size_mb, max_mb=self.settings.max_image_size_mb, filename=filename
            )

        logger.debug(f"Image size validation passed: {size_mb:.2f}MB")

    def validate_dimensions(self, width: int, height: int, filename: Optional[str] = None) -> None:
        """Validate that image dimensions are acceptable.

        Images must be within minimum and maximum constraints for:
        - Minimum: Ensure sufficient resolution for detection
        - Maximum: Prevent excessive memory usage

        Args:
            width: Image width in pixels
            height: Image height in pixels
            filename: Optional filename for error messages

        Raises:
            InvalidImageDimensionsError: If dimensions are invalid
        """
        if width < self._min_dimension or height < self._min_dimension:
            logger.warning(
                f"Rejected image with dimensions too small: {width}x{height} "
                f"(min: {self._min_dimension}x{self._min_dimension}, file: {filename})"
            )
            raise InvalidImageDimensionsError(
                width=width, height=height, min_dim=self._min_dimension, filename=filename
            )

        if width > self._max_dimension or height > self._max_dimension:
            logger.warning(
                f"Rejected image with dimensions too large: {width}x{height} "
                f"(max: {self._max_dimension}x{self._max_dimension}, file: {filename})"
            )
            raise InvalidImageDimensionsError(
                width=width, height=height, max_dim=self._max_dimension, filename=filename
            )

        logger.debug(f"Image dimensions validation passed: {width}x{height}")

    def validate_image_data(self, image_bytes: bytes, filename: Optional[str] = None) -> None:
        """Validate that image data is not corrupted.

        Attempts to open the image with PIL to verify it's a valid
        image file and not corrupted data. This validates the actual
        image content, not just the file extension.

        Args:
            image_bytes: Raw image bytes
            filename: Optional filename for error messages

        Raises:
            InvalidImageError: If image is corrupted or invalid
        """
        try:
            # Attempt to open and verify the image
            image = Image.open(io.BytesIO(image_bytes))
            image.verify()

            logger.debug(
                f"Image data validation passed: format={image.format}, "
                f"mode={image.mode}, size={image.size}"
            )

        except Exception as e:
            logger.error(f"Image data validation failed: {str(e)} (file: {filename})")
            raise InvalidImageError(
                reason=f"Corrupted or invalid image data: {str(e)}", filename=filename
            ) from e

    def validate_all(self, image_bytes: bytes, filename: str) -> tuple[int, int]:
        """Perform all validation checks on an image.

        This is a convenience method that runs all validation checks
        in the correct order:
        1. Format check (fast, based on filename)
        2. Size check (fast, based on byte length)
        3. Data integrity check (opens image to verify)
        4. Dimension check (after image is loaded)

        Args:
            image_bytes: Raw image bytes
            filename: Name of the file

        Returns:
            Tuple of (width, height) if all validations pass

        Raises:
            InvalidImageFormatError: If format is not supported
            ImageTooLargeError: If image exceeds maximum size
            InvalidImageError: If image data is corrupted
            InvalidImageDimensionsError: If dimensions are invalid
        """
        logger.debug(f"Starting comprehensive validation for: {filename}")

        # 1. Validate format (fast check)
        self.validate_format(filename)

        # 2. Validate size (fast check)
        self.validate_size(len(image_bytes), filename)

        # 3. Validate image data and get dimensions
        # We need to open the image again after verify() since verify() doesn't
        # load the full image and leaves it in an unusable state
        try:
            self.validate_image_data(image_bytes, filename)

            # Re-open to get dimensions
            image = Image.open(io.BytesIO(image_bytes))
            width, height = image.size

        except InvalidImageError:
            # Re-raise validation errors as-is
            raise
        except Exception as e:
            logger.error(f"Unexpected error during validation: {str(e)}")
            raise InvalidImageError(
                reason=f"Failed to process image: {str(e)}", filename=filename
            ) from e

        # 4. Validate dimensions
        self.validate_dimensions(width, height, filename)

        logger.info(
            f"All validations passed for {filename}: "
            f"{width}x{height}, {len(image_bytes) / 1024:.2f}KB"
        )

        return width, height


def create_validator(settings: Settings) -> ImageValidator:
    """Factory function to create an ImageValidator instance.

    This function is useful for dependency injection in FastAPI endpoints,
    allowing easy testing with mocked validators.

    Args:
        settings: Settings instance

    Returns:
        Configured ImageValidator instance

    Example:
        >>> from api.config import settings
        >>> validator = create_validator(settings)
        >>> validator.validate_all(image_bytes, "photo.jpg")
    """
    return ImageValidator(settings)
