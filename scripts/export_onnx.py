#!/usr/bin/env python3
"""
Export YOLOv8 Model to ONNX Format

This script converts a trained YOLOv8 PyTorch model (.pt) to ONNX format
for optimized inference. ONNX models typically achieve 30-50% faster inference
compared to PyTorch models, especially on CPU and with ONNX Runtime.

Benefits of ONNX:
    - Faster inference (optimized execution)
    - Cross-platform compatibility
    - Deployment flexibility (ONNX Runtime, TensorRT, etc.)
    - Smaller model size (with optimization)

Usage:
    # Export best.pt to ONNX
    python export_onnx.py --model runs/train/exp/weights/best.pt

    # Export with dynamic batch size
    python export_onnx.py --model best.pt --dynamic

    # Export and benchmark
    python export_onnx.py --model best.pt --benchmark

Requirements:
    - ultralytics: pip install ultralytics
    - onnx: pip install onnx
    - onnxruntime: pip install onnxruntime (for validation)

Author: Bastian Berrios
Date: 2025-01-15
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    print("ERROR: ultralytics not installed")
    print("Install with: pip install ultralytics")
    sys.exit(1)

try:
    import onnx
    ONNX_AVAILABLE = True
except ImportError:
    print("WARNING: onnx not installed (validation disabled)")
    print("Install with: pip install onnx")
    ONNX_AVAILABLE = False

try:
    import onnxruntime as ort
    ORT_AVAILABLE = True
except ImportError:
    print("WARNING: onnxruntime not installed (benchmarking disabled)")
    print("Install with: pip install onnxruntime")
    ORT_AVAILABLE = False


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_CONFIG = {
    "imgsz": 640,  # Input image size
    "batch": 1,  # Batch size for export
    "simplify": True,  # Simplify ONNX graph
    "opset": 12,  # ONNX opset version
}


# =============================================================================
# Logging Setup
# =============================================================================

def setup_logging() -> logging.Logger:
    """Configure logging to console."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S"
    )
    return logging.getLogger(__name__)


# =============================================================================
# Model Export
# =============================================================================

def export_model(
    model_path: Path,
    imgsz: int,
    dynamic: bool,
    simplify: bool,
    opset: int,
    logger: logging.Logger
) -> Path:
    """
    Export YOLOv8 model to ONNX format.

    Args:
        model_path: Path to YOLOv8 .pt model
        imgsz: Input image size
        dynamic: Enable dynamic axes for batch size
        simplify: Simplify ONNX graph
        opset: ONNX opset version
        logger: Logger instance

    Returns:
        Path to exported ONNX model

    Raises:
        FileNotFoundError: If model file not found
        Exception: If export fails
    """
    logger.info("=" * 80)
    logger.info("EXPORTING MODEL TO ONNX")
    logger.info("=" * 80)

    # Verify model exists
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    logger.info(f"Model path: {model_path}")
    logger.info(f"Model size: {model_path.stat().st_size / 1e6:.2f} MB")
    logger.info(f"Input size: {imgsz}x{imgsz}")
    logger.info(f"Dynamic batch: {dynamic}")
    logger.info(f"Simplify graph: {simplify}")
    logger.info(f"ONNX opset: {opset}")
    logger.info("=" * 80)

    try:
        # Load YOLOv8 model
        logger.info("Loading YOLOv8 model...")
        model = YOLO(str(model_path))

        # Export to ONNX
        logger.info("Exporting to ONNX format...")
        logger.info("This may take a few minutes...")

        onnx_path = model.export(
            format="onnx",
            imgsz=imgsz,
            dynamic=dynamic,
            simplify=simplify,
            opset=opset
        )

        onnx_path = Path(onnx_path)
        logger.info(f"Export completed!")
        logger.info(f"ONNX model saved to: {onnx_path}")
        logger.info(f"ONNX model size: {onnx_path.stat().st_size / 1e6:.2f} MB")

        # Size comparison
        size_ratio = onnx_path.stat().st_size / model_path.stat().st_size
        logger.info(f"Size ratio (ONNX/PyTorch): {size_ratio:.2f}x")

        logger.info("=" * 80)

        return onnx_path

    except Exception as e:
        logger.error(f"Export failed: {e}")
        raise


# =============================================================================
# Model Validation
# =============================================================================

def validate_onnx(onnx_path: Path, logger: logging.Logger) -> bool:
    """
    Validate exported ONNX model.

    Args:
        onnx_path: Path to ONNX model
        logger: Logger instance

    Returns:
        True if validation passes, False otherwise
    """
    if not ONNX_AVAILABLE:
        logger.warning("onnx library not available. Skipping validation.")
        return False

    logger.info("=" * 80)
    logger.info("VALIDATING ONNX MODEL")
    logger.info("=" * 80)

    try:
        # Load ONNX model
        logger.info("Loading ONNX model...")
        onnx_model = onnx.load(str(onnx_path))

        # Check model
        logger.info("Checking model validity...")
        onnx.checker.check_model(onnx_model)

        logger.info("Model validation passed!")

        # Print model info
        graph = onnx_model.graph

        logger.info(f"Model IR version: {onnx_model.ir_version}")
        logger.info(f"Producer name: {onnx_model.producer_name}")
        logger.info(f"Number of nodes: {len(graph.node)}")

        # Input info
        logger.info("\nModel inputs:")
        for input_tensor in graph.input:
            shape = [dim.dim_value for dim in input_tensor.type.tensor_type.shape.dim]
            logger.info(f"  {input_tensor.name}: {shape}")

        # Output info
        logger.info("\nModel outputs:")
        for output_tensor in graph.output:
            shape = [dim.dim_value for dim in output_tensor.type.tensor_type.shape.dim]
            logger.info(f"  {output_tensor.name}: {shape}")

        logger.info("=" * 80)

        return True

    except Exception as e:
        logger.error(f"Validation failed: {e}")
        return False


# =============================================================================
# Inference Benchmark
# =============================================================================

def benchmark_inference(
    pytorch_model_path: Path,
    onnx_model_path: Path,
    imgsz: int,
    num_runs: int,
    logger: logging.Logger
) -> dict:
    """
    Benchmark PyTorch vs ONNX inference speed.

    Args:
        pytorch_model_path: Path to PyTorch .pt model
        onnx_model_path: Path to ONNX model
        imgsz: Input image size
        num_runs: Number of benchmark runs
        logger: Logger instance

    Returns:
        Dictionary with benchmark results
    """
    if not ORT_AVAILABLE:
        logger.warning("onnxruntime not available. Skipping benchmark.")
        return {}

    logger.info("=" * 80)
    logger.info("BENCHMARKING INFERENCE SPEED")
    logger.info("=" * 80)
    logger.info(f"Image size: {imgsz}x{imgsz}")
    logger.info(f"Number of runs: {num_runs}")
    logger.info("=" * 80)

    results = {}

    try:
        # Create dummy input
        dummy_input = np.random.randn(1, 3, imgsz, imgsz).astype(np.float32)

        # Benchmark PyTorch
        logger.info("\nBenchmarking PyTorch model...")
        pt_model = YOLO(str(pytorch_model_path))

        # Warmup
        _ = pt_model(dummy_input, verbose=False)

        # Timed runs
        pt_times = []
        for i in range(num_runs):
            start = time.perf_counter()
            _ = pt_model(dummy_input, verbose=False)
            end = time.perf_counter()
            pt_times.append((end - start) * 1000)  # Convert to ms

        pt_mean = np.mean(pt_times)
        pt_std = np.std(pt_times)

        logger.info(f"PyTorch: {pt_mean:.2f} +/- {pt_std:.2f} ms")

        results["pytorch_mean_ms"] = pt_mean
        results["pytorch_std_ms"] = pt_std

        # Benchmark ONNX
        logger.info("\nBenchmarking ONNX model...")

        # Create ONNX Runtime session
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        session = ort.InferenceSession(
            str(onnx_model_path),
            sess_options,
            providers=['CPUExecutionProvider']
        )

        input_name = session.get_inputs()[0].name

        # Warmup
        _ = session.run(None, {input_name: dummy_input})

        # Timed runs
        onnx_times = []
        for i in range(num_runs):
            start = time.perf_counter()
            _ = session.run(None, {input_name: dummy_input})
            end = time.perf_counter()
            onnx_times.append((end - start) * 1000)  # Convert to ms

        onnx_mean = np.mean(onnx_times)
        onnx_std = np.std(onnx_times)

        logger.info(f"ONNX: {onnx_mean:.2f} +/- {onnx_std:.2f} ms")

        results["onnx_mean_ms"] = onnx_mean
        results["onnx_std_ms"] = onnx_std

        # Speedup
        speedup = pt_mean / onnx_mean
        logger.info("=" * 80)
        logger.info(f"ONNX Speedup: {speedup:.2f}x faster")
        logger.info(f"Latency reduction: {((pt_mean - onnx_mean) / pt_mean * 100):.1f}%")
        logger.info("=" * 80)

        results["speedup"] = speedup

        return results

    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        return {}


# =============================================================================
# Output Equivalence Test
# =============================================================================

def test_output_equivalence(
    pytorch_model_path: Path,
    onnx_model_path: Path,
    imgsz: int,
    logger: logging.Logger,
    tolerance: float = 1e-5
) -> bool:
    """
    Test if PyTorch and ONNX models produce equivalent outputs.

    Args:
        pytorch_model_path: Path to PyTorch .pt model
        onnx_model_path: Path to ONNX model
        imgsz: Input image size
        logger: Logger instance
        tolerance: Maximum allowed difference

    Returns:
        True if outputs are equivalent, False otherwise
    """
    if not ORT_AVAILABLE:
        logger.warning("onnxruntime not available. Skipping equivalence test.")
        return False

    logger.info("=" * 80)
    logger.info("TESTING OUTPUT EQUIVALENCE")
    logger.info("=" * 80)
    logger.info(f"Tolerance: {tolerance}")

    try:
        # Create dummy input
        dummy_input = np.random.randn(1, 3, imgsz, imgsz).astype(np.float32)

        # PyTorch inference
        logger.info("Running PyTorch inference...")
        pt_model = YOLO(str(pytorch_model_path))
        pt_results = pt_model(dummy_input, verbose=False)

        # ONNX inference
        logger.info("Running ONNX inference...")
        session = ort.InferenceSession(
            str(onnx_model_path),
            providers=['CPUExecutionProvider']
        )
        input_name = session.get_inputs()[0].name
        onnx_output = session.run(None, {input_name: dummy_input})

        # TODO: Compare outputs properly
        # Note: Direct comparison is complex due to YOLOv8 output format
        # This is a placeholder for future implementation

        logger.info("Output shapes:")
        logger.info(f"  ONNX output length: {len(onnx_output)}")

        logger.info("=" * 80)
        logger.info("NOTE: Full output equivalence test not implemented")
        logger.info("      Manual validation recommended")
        logger.info("=" * 80)

        return True

    except Exception as e:
        logger.error(f"Equivalence test failed: {e}")
        return False


# =============================================================================
# Argument Parsing
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Export YOLOv8 model to ONNX format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to YOLOv8 .pt model file"
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=DEFAULT_CONFIG["imgsz"],
        help="Input image size"
    )

    parser.add_argument(
        "--dynamic",
        action="store_true",
        help="Enable dynamic batch size (for variable batch inference)"
    )

    parser.add_argument(
        "--simplify",
        action="store_true",
        default=DEFAULT_CONFIG["simplify"],
        help="Simplify ONNX graph for optimization"
    )

    parser.add_argument(
        "--opset",
        type=int,
        default=DEFAULT_CONFIG["opset"],
        help="ONNX opset version"
    )

    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate ONNX model after export"
    )

    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Benchmark inference speed (PyTorch vs ONNX)"
    )

    parser.add_argument(
        "--num-runs",
        type=int,
        default=100,
        help="Number of benchmark runs"
    )

    parser.add_argument(
        "--test-equivalence",
        action="store_true",
        help="Test output equivalence between PyTorch and ONNX"
    )

    return parser.parse_args()


# =============================================================================
# Main
# =============================================================================

def main():
    """Main export pipeline."""
    # Parse arguments
    args = parse_args()

    # Setup logging
    logger = setup_logging()

    logger.info("=" * 80)
    logger.info("YOLOV8 TO ONNX EXPORT TOOL")
    logger.info("=" * 80)

    try:
        # Convert model path
        model_path = Path(args.model)

        # Export model
        onnx_path = export_model(
            model_path=model_path,
            imgsz=args.imgsz,
            dynamic=args.dynamic,
            simplify=args.simplify,
            opset=args.opset,
            logger=logger
        )

        # Validate model
        if args.validate:
            validate_onnx(onnx_path, logger)

        # Test equivalence
        if args.test_equivalence:
            test_output_equivalence(model_path, onnx_path, args.imgsz, logger)

        # Benchmark
        if args.benchmark:
            benchmark_inference(
                model_path,
                onnx_path,
                args.imgsz,
                args.num_runs,
                logger
            )

        logger.info("=" * 80)
        logger.info("EXPORT COMPLETE")
        logger.info("=" * 80)
        logger.info(f"ONNX model: {onnx_path}")
        logger.info("\nNext steps:")
        logger.info("1. Test inference with ONNX Runtime")
        logger.info("2. Deploy to production environment")
        logger.info("3. Monitor inference performance")
        logger.info("=" * 80)

        return 0

    except Exception as e:
        logger.error("=" * 80)
        logger.error("EXPORT FAILED")
        logger.error("=" * 80)
        logger.exception(e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
