#!/usr/bin/env python3
"""
GCP GPU Training Script for YOLOv8 Hard Hat Detection

This script is designed to run on GCP VM instances with GPU support.
It handles the complete training pipeline from dataset loading to model export.

Usage:
    python train_gcp.py --epochs 50 --batch-size 16 --img-size 640
    python train_gcp.py --dataset-path ./data/roboflow --model yolov8n.pt

Requirements:
    - GPU-enabled GCP VM (e.g., n1-standard-4 with T4 GPU)
    - CUDA and cuDNN installed
    - Python 3.9+
    - See requirements.txt for dependencies

Author: Bastian Berrios
Date: 2025-01-15
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logging.warning("MLflow not available. Experiment tracking disabled.")


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_CONFIG = {
    "epochs": 50,
    "batch_size": 16,
    "img_size": 640,
    "model": "yolov8n.pt",
    "dataset_path": "./data/roboflow",
    "output_dir": "./runs/train",
    "device": "0",  # GPU 0, use "cpu" for CPU training
    "patience": 10,  # Early stopping patience
    "save_period": 5,  # Save checkpoint every N epochs
    "workers": 8,
    "optimizer": "SGD",
    "lr0": 0.01,
    "weight_decay": 0.0005,
}


# =============================================================================
# Logging Setup
# =============================================================================

def setup_logging(output_dir: Path) -> logging.Logger:
    """
    Configure detailed logging to both file and console.

    Args:
        output_dir: Directory to save log files

    Returns:
        Configured logger instance
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = output_dir / f"training_{timestamp}.log"

    # Create logger
    logger = logging.getLogger("train_gcp")
    logger.setLevel(logging.DEBUG)

    # Console handler (INFO and above)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S"
    )
    console_handler.setFormatter(console_format)

    # File handler (DEBUG and above)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)-8s | %(funcName)s:%(lineno)d | %(message)s"
    )
    file_handler.setFormatter(file_format)

    # Add handlers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


# =============================================================================
# GPU Verification
# =============================================================================

def verify_gpu_setup(logger: logging.Logger) -> dict:
    """
    Verify GPU availability and CUDA setup.

    Args:
        logger: Logger instance

    Returns:
        Dictionary with GPU information

    Raises:
        RuntimeError: If GPU is not available when expected
    """
    logger.info("=" * 80)
    logger.info("GPU VERIFICATION")
    logger.info("=" * 80)

    gpu_info = {
        "cuda_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "pytorch_version": torch.__version__,
    }

    logger.info(f"PyTorch version: {gpu_info['pytorch_version']}")
    logger.info(f"CUDA available: {gpu_info['cuda_available']}")

    if gpu_info["cuda_available"]:
        logger.info(f"CUDA version: {gpu_info['cuda_version']}")
        logger.info(f"Number of GPUs: {gpu_info['gpu_count']}")

        for i in range(gpu_info["gpu_count"]):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
            logger.info(f"  GPU {i}: {gpu_name} ({gpu_memory:.2f} GB)")
            gpu_info[f"gpu_{i}_name"] = gpu_name
            gpu_info[f"gpu_{i}_memory_gb"] = gpu_memory
    else:
        logger.warning("[ADVERTENCIA]  No GPU detected! Training will be VERY slow on CPU.")
        logger.warning("    Make sure you're running on a GPU-enabled VM.")

    logger.info("=" * 80)

    return gpu_info


# =============================================================================
# Dataset Verification
# =============================================================================

def verify_dataset(dataset_path: Path, logger: logging.Logger) -> dict:
    """
    Verify dataset structure and content.

    Args:
        dataset_path: Path to dataset directory
        logger: Logger instance

    Returns:
        Dictionary with dataset information

    Raises:
        FileNotFoundError: If dataset.yaml is not found
        ValueError: If dataset structure is invalid
    """
    logger.info("=" * 80)
    logger.info("DATASET VERIFICATION")
    logger.info("=" * 80)

    # Check for data.yaml or dataset.yaml
    yaml_path = None
    for yaml_name in ["data.yaml", "dataset.yaml"]:
        candidate = dataset_path / yaml_name
        if candidate.exists():
            yaml_path = candidate
            break

    if yaml_path is None:
        raise FileNotFoundError(
            f"No data.yaml or dataset.yaml found in {dataset_path}. "
            "Please run download_roboflow.py first."
        )

    logger.info(f"Dataset config: {yaml_path}")

    # Load dataset configuration
    with open(yaml_path, "r") as f:
        dataset_config = yaml.safe_load(f)

    logger.info(f"Dataset name: {dataset_config.get('name', 'Unknown')}")
    logger.info(f"Number of classes: {dataset_config.get('nc', 'Unknown')}")
    logger.info(f"Classes: {dataset_config.get('names', [])}")

    # Verify splits exist
    for split in ["train", "val", "test"]:
        split_path = dataset_path / split / "images"
        if split_path.exists():
            num_images = len(list(split_path.glob("*.jpg"))) + len(list(split_path.glob("*.png")))
            logger.info(f"  {split.capitalize()}: {num_images} images")
        else:
            logger.warning(f"  {split.capitalize()}: directory not found")

    logger.info("=" * 80)

    return dataset_config


# =============================================================================
# MLflow Integration
# =============================================================================

def setup_mlflow(logger: logging.Logger, args: argparse.Namespace) -> bool:
    """
    Setup MLflow experiment tracking.

    Args:
        logger: Logger instance
        args: Parsed command-line arguments

    Returns:
        True if MLflow is configured successfully, False otherwise
    """
    if not MLFLOW_AVAILABLE:
        logger.warning("MLflow not installed. Skipping experiment tracking.")
        return False

    try:
        # Set tracking URI (can be configured via env var)
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "./mlruns")
        mlflow.set_tracking_uri(tracking_uri)

        # Set experiment name
        experiment_name = "hardhat-yolov8-training"
        mlflow.set_experiment(experiment_name)

        # Start run
        mlflow.start_run()

        # Log parameters
        mlflow.log_params({
            "model": args.model,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "img_size": args.img_size,
            "optimizer": args.optimizer,
            "lr0": args.lr0,
            "weight_decay": args.weight_decay,
            "device": args.device,
            "dataset": args.dataset_path,
        })

        logger.info(f"MLflow tracking enabled: {tracking_uri}")
        logger.info(f"Experiment: {experiment_name}")

        return True

    except Exception as e:
        logger.error(f"Failed to setup MLflow: {e}")
        return False


# =============================================================================
# Training
# =============================================================================

def train_model(
    args: argparse.Namespace,
    dataset_yaml: Path,
    logger: logging.Logger,
    use_mlflow: bool
) -> Path:
    """
    Train YOLOv8 model.

    Args:
        args: Parsed command-line arguments
        dataset_yaml: Path to dataset configuration YAML
        logger: Logger instance
        use_mlflow: Whether MLflow tracking is enabled

    Returns:
        Path to best trained model weights
    """
    logger.info("=" * 80)
    logger.info("STARTING TRAINING")
    logger.info("=" * 80)

    # Initialize model
    logger.info(f"Loading model: {args.model}")
    model = YOLO(args.model)

    # Training configuration
    train_config = {
        "data": str(dataset_yaml),
        "epochs": args.epochs,
        "batch": args.batch_size,
        "imgsz": args.img_size,
        "device": args.device,
        "workers": args.workers,
        "patience": args.patience,
        "save_period": args.save_period,
        "optimizer": args.optimizer,
        "lr0": args.lr0,
        "weight_decay": args.weight_decay,
        "project": args.output_dir,
        "name": f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "exist_ok": True,
        "verbose": True,
        # Data augmentation
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "degrees": 0.0,
        "translate": 0.1,
        "scale": 0.5,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.5,
        "mosaic": 1.0,
        "mixup": 0.0,
    }

    logger.info("Training configuration:")
    for key, value in train_config.items():
        logger.info(f"  {key}: {value}")

    # Start training
    logger.info("\n[INICIO] Starting training...")
    results = model.train(**train_config)

    logger.info("[OK] Training completed!")
    logger.info("=" * 80)

    # Log metrics to MLflow
    if use_mlflow and MLFLOW_AVAILABLE:
        try:
            # TODO: Log final metrics from results
            # mlflow.log_metrics({
            #     "final_map50": results.maps[0],
            #     "final_map50-95": results.maps[1],
            # })
            pass
        except Exception as e:
            logger.error(f"Failed to log metrics to MLflow: {e}")

    # Find best model weights
    best_weights = Path(args.output_dir) / train_config["name"] / "weights" / "best.pt"

    if not best_weights.exists():
        raise FileNotFoundError(f"Best weights not found at {best_weights}")

    logger.info(f"Best model saved to: {best_weights}")

    return best_weights


# =============================================================================
# Model Export
# =============================================================================

def export_to_onnx(model_path: Path, logger: logging.Logger) -> Path:
    """
    Export trained model to ONNX format for optimized inference.

    Args:
        model_path: Path to trained .pt model
        logger: Logger instance

    Returns:
        Path to exported ONNX model
    """
    logger.info("=" * 80)
    logger.info("EXPORTING TO ONNX")
    logger.info("=" * 80)

    logger.info(f"Loading model from: {model_path}")
    model = YOLO(model_path)

    # Export to ONNX
    logger.info("Exporting to ONNX format...")
    onnx_path = model.export(format="onnx", simplify=True)

    logger.info(f"[OK] ONNX model exported to: {onnx_path}")
    logger.info("=" * 80)

    return Path(onnx_path)


# =============================================================================
# GCS Upload (Optional)
# =============================================================================

def upload_to_gcs(model_path: Path, logger: logging.Logger) -> bool:
    """
    Upload trained model to Google Cloud Storage.

    Args:
        model_path: Path to model file
        logger: Logger instance

    Returns:
        True if upload successful, False otherwise

    Note:
        Requires GOOGLE_CLOUD_BUCKET environment variable to be set.
    """
    bucket_name = os.getenv("GOOGLE_CLOUD_BUCKET")

    if not bucket_name:
        logger.info("GOOGLE_CLOUD_BUCKET not set. Skipping GCS upload.")
        return False

    try:
        from google.cloud import storage

        logger.info("=" * 80)
        logger.info("UPLOADING TO GOOGLE CLOUD STORAGE")
        logger.info("=" * 80)

        client = storage.Client()
        bucket = client.bucket(bucket_name)

        blob_name = f"models/{model_path.name}"
        blob = bucket.blob(blob_name)

        logger.info(f"Uploading {model_path} to gs://{bucket_name}/{blob_name}")
        blob.upload_from_filename(str(model_path))

        logger.info(f"[OK] Upload successful!")
        logger.info(f"   gs://{bucket_name}/{blob_name}")
        logger.info("=" * 80)

        return True

    except ImportError:
        logger.warning("google-cloud-storage not installed. Skipping GCS upload.")
        return False
    except Exception as e:
        logger.error(f"Failed to upload to GCS: {e}")
        return False


# =============================================================================
# Argument Parsing
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train YOLOv8 model on GCP GPU instance",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Training parameters
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_CONFIG["epochs"],
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_CONFIG["batch_size"],
        help="Batch size for training"
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=DEFAULT_CONFIG["img_size"],
        help="Input image size"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_CONFIG["model"],
        help="YOLOv8 model variant (yolov8n/s/m/l/x.pt)"
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=DEFAULT_CONFIG["dataset_path"],
        help="Path to dataset directory containing data.yaml"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_CONFIG["output_dir"],
        help="Directory to save training outputs"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=DEFAULT_CONFIG["device"],
        help="Device to use (0 for GPU 0, cpu for CPU)"
    )

    # Optimizer parameters
    parser.add_argument(
        "--optimizer",
        type=str,
        default=DEFAULT_CONFIG["optimizer"],
        choices=["SGD", "Adam", "AdamW"],
        help="Optimizer to use"
    )
    parser.add_argument(
        "--lr0",
        type=float,
        default=DEFAULT_CONFIG["lr0"],
        help="Initial learning rate"
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=DEFAULT_CONFIG["weight_decay"],
        help="Weight decay for optimizer"
    )

    # Training behavior
    parser.add_argument(
        "--patience",
        type=int,
        default=DEFAULT_CONFIG["patience"],
        help="Early stopping patience (epochs without improvement)"
    )
    parser.add_argument(
        "--save-period",
        type=int,
        default=DEFAULT_CONFIG["save_period"],
        help="Save checkpoint every N epochs"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_CONFIG["workers"],
        help="Number of dataloader workers"
    )

    # Export options
    parser.add_argument(
        "--export-onnx",
        action="store_true",
        help="Export best model to ONNX format after training"
    )
    parser.add_argument(
        "--upload-gcs",
        action="store_true",
        help="Upload best model to Google Cloud Storage"
    )

    # MLflow
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Disable MLflow experiment tracking"
    )

    return parser.parse_args()


# =============================================================================
# Main
# =============================================================================

def main():
    """Main training pipeline."""
    # Parse arguments
    args = parse_args()

    # Setup logging
    output_dir = Path(args.output_dir)
    logger = setup_logging(output_dir)

    logger.info("=" * 80)
    logger.info("YOLOv8 HARD HAT DETECTION - GCP TRAINING SCRIPT")
    logger.info("=" * 80)
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80)

    try:
        # Verify GPU setup
        gpu_info = verify_gpu_setup(logger)

        # Verify dataset
        dataset_path = Path(args.dataset_path)
        dataset_config = verify_dataset(dataset_path, logger)

        # Find dataset YAML
        dataset_yaml = None
        for yaml_name in ["data.yaml", "dataset.yaml"]:
            candidate = dataset_path / yaml_name
            if candidate.exists():
                dataset_yaml = candidate
                break

        # Setup MLflow
        use_mlflow = not args.no_mlflow
        if use_mlflow:
            use_mlflow = setup_mlflow(logger, args)

        # Train model
        best_model_path = train_model(args, dataset_yaml, logger, use_mlflow)

        # Export to ONNX
        if args.export_onnx:
            onnx_path = export_to_onnx(best_model_path, logger)

            # Upload ONNX to GCS if requested
            if args.upload_gcs:
                upload_to_gcs(onnx_path, logger)

        # Upload PyTorch model to GCS
        if args.upload_gcs:
            upload_to_gcs(best_model_path, logger)

        # End MLflow run
        if use_mlflow and MLFLOW_AVAILABLE:
            mlflow.end_run()

        logger.info("=" * 80)
        logger.info("[OK] TRAINING PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        logger.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Best model: {best_model_path}")
        logger.info("=" * 80)
        logger.info("\n[COMPLETADO] All done! Don't forget to STOP your GCP VM to avoid charges!")
        logger.info("   Command: gcloud compute instances stop <instance-name>")

        return 0

    except Exception as e:
        logger.error("=" * 80)
        logger.error("[ERROR] TRAINING FAILED")
        logger.error("=" * 80)
        logger.exception(e)

        if use_mlflow and MLFLOW_AVAILABLE:
            mlflow.end_run(status="FAILED")

        return 1


if __name__ == "__main__":
    sys.exit(main())
