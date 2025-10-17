#!/usr/bin/env python3
"""
Model Export Script for EPP Detection.

This script exports trained YOLOv8 models to various formats for deployment:
- ONNX: For cross-platform inference (primary format for epp-detector API)
- TorchScript: For optimized PyTorch inference
- TensorRT: For NVIDIA GPU acceleration
- CoreML: For iOS deployment
- TFLite: For mobile/edge devices

The epp-detector API uses ONNX format for optimal performance and compatibility.

Author: Bastián Berríos
Project: epp-detector
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import onnx
import onnxruntime as ort
import torch
from ultralytics import YOLO

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class ExportError(Exception):
    """Base exception for export errors."""
    pass


class ModelValidationError(ExportError):
    """Raised when exported model validation fails."""
    pass


def export_to_onnx(
    model: YOLO,
    output_path: Path,
    imgsz: int = 640,
    half: bool = False,
    simplify: bool = True,
    dynamic: bool = True,
    opset: int = 12
) -> Path:
    """Export YOLOv8 model to ONNX format.

    ONNX (Open Neural Network Exchange) provides:
    - Cross-platform compatibility (CPU, GPU, different frameworks)
    - Optimized inference performance
    - Smaller model size than PyTorch
    - Support for quantization and pruning

    Args:
        model: Trained YOLO model
        output_path: Path to save ONNX model
        imgsz: Input image size for export
        half: Export with FP16 precision (faster, slight accuracy loss)
        simplify: Simplify ONNX graph (remove training-only ops)
        dynamic: Allow dynamic batch sizes
        opset: ONNX opset version

    Returns:
        Path to exported ONNX model

    Raises:
        ExportError: If export fails
    """
    try:
        logger.info("Exporting model to ONNX format...")
        logger.info(f"  Input size: {imgsz}")
        logger.info(f"  Half precision: {half}")
        logger.info(f"  Simplify: {simplify}")
        logger.info(f"  Dynamic batching: {dynamic}")
        logger.info(f"  Opset version: {opset}")

        # Export using Ultralytics API
        export_path = model.export(
            format='onnx',
            imgsz=imgsz,
            half=half,
            simplify=simplify,
            dynamic=dynamic,
            opset=opset
        )

        # Move to desired output location if different
        export_path = Path(export_path)
        if export_path != output_path:
            import shutil
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(export_path), str(output_path))
            logger.info(f"Moved model to {output_path}")

        logger.info(f"ONNX export successful: {output_path}")
        return output_path

    except Exception as e:
        logger.error(f"ONNX export failed: {e}", exc_info=True)
        raise ExportError(f"ONNX export failed: {e}") from e


def validate_onnx_model(onnx_path: Path) -> Dict:
    """Validate exported ONNX model.

    Checks:
    1. ONNX model is well-formed
    2. Model can be loaded by onnxruntime
    3. Input/output shapes are correct
    4. Model runs a dummy inference

    Args:
        onnx_path: Path to ONNX model file

    Returns:
        Dictionary with model information:
        - input_shape: Input tensor shape
        - output_shape: Output tensor shape
        - input_name: Name of input tensor
        - output_name: Name of output tensor
        - opset_version: ONNX opset version
        - providers: Available execution providers

    Raises:
        ModelValidationError: If validation fails
    """
    try:
        logger.info(f"Validating ONNX model: {onnx_path}")

        # Check ONNX model structure
        onnx_model = onnx.load(str(onnx_path))
        onnx.checker.check_model(onnx_model)
        logger.info("ONNX model structure is valid")

        # Get model metadata
        opset_version = onnx_model.opset_import[0].version
        logger.info(f"ONNX opset version: {opset_version}")

        # Load with ONNX Runtime
        session = ort.InferenceSession(
            str(onnx_path),
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )

        # Get input/output info
        input_info = session.get_inputs()[0]
        output_info = session.get_outputs()[0]

        input_name = input_info.name
        input_shape = input_info.shape
        output_name = output_info.name
        output_shape = output_info.shape

        logger.info(f"Input: {input_name} {input_shape}")
        logger.info(f"Output: {output_name} {output_shape}")

        # Check available providers
        available_providers = ort.get_available_providers()
        logger.info(f"Available providers: {', '.join(available_providers)}")

        # Run dummy inference to verify model works
        import numpy as np

        # Create dummy input (batch=1, channels=3, height=640, width=640)
        # Handle dynamic batch dimension
        if isinstance(input_shape[0], str) or input_shape[0] == -1:
            batch_size = 1
        else:
            batch_size = input_shape[0]

        dummy_input = np.random.randn(
            batch_size, 3, input_shape[2], input_shape[3]
        ).astype(np.float32)

        # Run inference
        outputs = session.run([output_name], {input_name: dummy_input})

        logger.info(f"Dummy inference successful, output shape: {outputs[0].shape}")
        logger.info("ONNX model validation passed")

        return {
            'input_shape': input_shape,
            'output_shape': output_shape,
            'input_name': input_name,
            'output_name': output_name,
            'opset_version': opset_version,
            'providers': available_providers,
        }

    except onnx.checker.ValidationError as e:
        raise ModelValidationError(f"ONNX model structure invalid: {e}") from e

    except Exception as e:
        raise ModelValidationError(f"Model validation failed: {e}") from e


def get_model_size_mb(model_path: Path) -> float:
    """Get model file size in megabytes.

    Args:
        model_path: Path to model file

    Returns:
        File size in MB
    """
    size_bytes = model_path.stat().st_size
    return size_bytes / (1024 * 1024)


def compare_model_sizes(pt_path: Path, onnx_path: Path) -> None:
    """Compare PyTorch and ONNX model sizes.

    Args:
        pt_path: Path to PyTorch .pt model
        onnx_path: Path to ONNX model
    """
    if pt_path.exists() and onnx_path.exists():
        pt_size = get_model_size_mb(pt_path)
        onnx_size = get_model_size_mb(onnx_path)
        reduction = ((pt_size - onnx_size) / pt_size) * 100

        logger.info("\nModel Size Comparison:")
        logger.info(f"  PyTorch (.pt):  {pt_size:.2f} MB")
        logger.info(f"  ONNX:           {onnx_size:.2f} MB")
        logger.info(f"  Size reduction: {reduction:.1f}%")


def export_to_torchscript(
    model: YOLO,
    output_path: Path,
    imgsz: int = 640
) -> Path:
    """Export model to TorchScript format.

    TorchScript provides:
    - Optimized PyTorch inference
    - Can run without Python interpreter
    - Good for PyTorch-based production systems

    Args:
        model: Trained YOLO model
        output_path: Path to save TorchScript model
        imgsz: Input image size

    Returns:
        Path to exported model

    Raises:
        ExportError: If export fails
    """
    try:
        logger.info("Exporting to TorchScript...")
        export_path = model.export(format='torchscript', imgsz=imgsz)

        export_path = Path(export_path)
        if export_path != output_path:
            import shutil
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(export_path), str(output_path))

        logger.info(f"TorchScript export successful: {output_path}")
        return output_path

    except Exception as e:
        logger.error(f"TorchScript export failed: {e}")
        raise ExportError(f"TorchScript export failed: {e}") from e


def main():
    """Main entry point for export script."""
    parser = argparse.ArgumentParser(
        description="Export YOLOv8 model to various formats",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export to ONNX (default, recommended for API)
  python scripts/export_model.py --weights models/yolov8n_epp.pt

  # Export with FP16 precision for faster inference
  python scripts/export_model.py --weights models/yolov8n_epp.pt --half

  # Export to multiple formats
  python scripts/export_model.py --weights models/yolov8n_epp.pt --formats onnx torchscript

  # Custom output path
  python scripts/export_model.py --weights runs/train/epp_detector/weights/best.pt --output models/custom_epp.onnx
        """
    )

    parser.add_argument(
        '--weights',
        type=Path,
        required=True,
        help='Path to trained model weights (.pt file)'
    )

    parser.add_argument(
        '--output',
        type=Path,
        help='Output path for exported model (default: same as input with new extension)'
    )

    parser.add_argument(
        '--formats',
        nargs='+',
        choices=['onnx', 'torchscript', 'tensorrt', 'coreml', 'tflite'],
        default=['onnx'],
        help='Export formats (default: onnx)'
    )

    parser.add_argument(
        '--imgsz',
        type=int,
        default=640,
        help='Input image size for export (default: 640)'
    )

    parser.add_argument(
        '--half',
        action='store_true',
        help='Export with FP16 precision (faster, slight accuracy loss)'
    )

    parser.add_argument(
        '--no-simplify',
        action='store_true',
        help='Do not simplify ONNX model'
    )

    parser.add_argument(
        '--no-dynamic',
        action='store_true',
        help='Disable dynamic batch size (fixed batch=1)'
    )

    parser.add_argument(
        '--opset',
        type=int,
        default=12,
        help='ONNX opset version (default: 12)'
    )

    parser.add_argument(
        '--skip-validation',
        action='store_true',
        help='Skip ONNX validation after export'
    )

    args = parser.parse_args()

    try:
        # Validate input weights
        if not args.weights.exists():
            logger.error(f"Model weights not found: {args.weights}")
            sys.exit(1)

        # Load model
        logger.info(f"Loading model from {args.weights}")
        model = YOLO(str(args.weights))

        # Determine output path
        if args.output:
            output_base = args.output
        else:
            output_base = args.weights.parent / args.weights.stem

        exported_models = []

        # Export to requested formats
        for fmt in args.formats:
            if fmt == 'onnx':
                output_path = output_base.with_suffix('.onnx')

                onnx_path = export_to_onnx(
                    model,
                    output_path,
                    imgsz=args.imgsz,
                    half=args.half,
                    simplify=not args.no_simplify,
                    dynamic=not args.no_dynamic,
                    opset=args.opset
                )

                exported_models.append(('ONNX', onnx_path))

                # Validate ONNX model
                if not args.skip_validation:
                    try:
                        model_info = validate_onnx_model(onnx_path)
                        logger.info("\nModel Information:")
                        for key, value in model_info.items():
                            logger.info(f"  {key}: {value}")

                    except ModelValidationError as e:
                        logger.error(f"Validation failed: {e}")
                        logger.warning("Model exported but validation failed - use with caution")

                # Compare sizes
                compare_model_sizes(args.weights, onnx_path)

            elif fmt == 'torchscript':
                output_path = output_base.with_suffix('.torchscript')
                ts_path = export_to_torchscript(model, output_path, args.imgsz)
                exported_models.append(('TorchScript', ts_path))

            else:
                logger.warning(f"Export format '{fmt}' not yet implemented in this script")
                logger.info(f"Use: model.export(format='{fmt}') directly")

        # Summary
        logger.info("\n" + "=" * 70)
        logger.info("EXPORT COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Source model: {args.weights}")
        for fmt, path in exported_models:
            logger.info(f"{fmt}: {path} ({get_model_size_mb(path):.2f} MB)")

        if 'onnx' in args.formats:
            logger.info("\nModel ready for epp-detector API!")
            logger.info("Update .env file:")
            logger.info(f'  MODEL_PATH="{exported_models[0][1]}"')
            logger.info("  MODEL_TYPE=onnx")

    except KeyboardInterrupt:
        logger.info("\nExport interrupted by user")
        sys.exit(130)

    except Exception as e:
        logger.error(f"Export failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
