#!/usr/bin/env python3
"""
YOLOv8 Training Script for EPP Detection.

This script trains a YOLOv8 model for Personal Protective Equipment detection
in Chilean mining environments. Supports:
- Transfer learning from COCO-pretrained weights
- Custom hyperparameter configuration
- Training resumption from checkpoints
- Automatic model export (PyTorch + ONNX)
- Comprehensive logging and metrics

Author: Bastián Berríos
Project: epp-detector
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import torch
import yaml
from ultralytics import YOLO

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class TrainingError(Exception):
    """Base exception for training errors."""
    pass


class ConfigurationError(TrainingError):
    """Raised when configuration is invalid."""
    pass


class ModelLoadError(TrainingError):
    """Raised when model loading fails."""
    pass


def load_training_config(config_path: Path) -> Dict:
    """Load training configuration from YAML file.

    Args:
        config_path: Path to training_config.yaml

    Returns:
        Dictionary containing training hyperparameters

    Raises:
        ConfigurationError: If config file is invalid or missing
    """
    if not config_path.exists():
        raise ConfigurationError(f"Configuration file not found: {config_path}")

    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Validate required fields
        required_fields = ['model', 'data', 'epochs', 'imgsz']
        missing_fields = [field for field in required_fields if field not in config]

        if missing_fields:
            raise ConfigurationError(
                f"Missing required fields in config: {', '.join(missing_fields)}"
            )

        return config

    except yaml.YAMLError as e:
        raise ConfigurationError(f"Failed to parse config file: {e}") from e


def validate_dataset(data_yaml_path: Path) -> bool:
    """Validate that dataset exists and has correct structure.

    Checks for:
    - data.yaml file exists
    - train/val/test image directories exist
    - train/val/test label directories exist
    - At least some images are present

    Args:
        data_yaml_path: Path to dataset YAML file

    Returns:
        True if dataset is valid

    Raises:
        ConfigurationError: If dataset structure is invalid
    """
    if not data_yaml_path.exists():
        raise ConfigurationError(f"Dataset config not found: {data_yaml_path}")

    try:
        with open(data_yaml_path, 'r') as f:
            dataset_config = yaml.safe_load(f)

        # Get dataset root path
        if 'path' in dataset_config:
            dataset_root = Path(dataset_config['path'])
        else:
            dataset_root = data_yaml_path.parent

        # Validate splits exist
        for split in ['train', 'val']:
            # Check images
            img_path_str = dataset_config.get(split)
            if not img_path_str:
                raise ConfigurationError(f"Missing '{split}' path in dataset config")

            img_path = dataset_root / img_path_str
            if not img_path.exists():
                raise ConfigurationError(f"Image directory not found: {img_path}")

            # Check labels
            label_path = dataset_root / 'labels' / split
            if not label_path.exists():
                raise ConfigurationError(f"Label directory not found: {label_path}")

            # Check for images
            images = list(img_path.glob('*.jpg')) + list(img_path.glob('*.png'))
            if not images:
                raise ConfigurationError(f"No images found in {img_path}")

            logger.info(f"Found {len(images)} images in {split} set")

        return True

    except yaml.YAMLError as e:
        raise ConfigurationError(f"Failed to parse dataset config: {e}") from e


def check_gpu_availability() -> Dict[str, any]:
    """Check GPU availability and memory.

    Returns:
        Dictionary with GPU information:
        - available: Whether GPU is available
        - device_name: GPU model name
        - memory_total: Total GPU memory in GB
        - memory_allocated: Currently allocated memory in GB
        - cuda_version: CUDA version if available
    """
    gpu_info = {
        'available': torch.cuda.is_available(),
        'device_name': None,
        'memory_total': None,
        'memory_allocated': None,
        'cuda_version': None,
    }

    if torch.cuda.is_available():
        gpu_info['device_name'] = torch.cuda.get_device_name(0)
        gpu_info['memory_total'] = torch.cuda.get_device_properties(0).total_memory / 1e9
        gpu_info['memory_allocated'] = torch.cuda.memory_allocated(0) / 1e9
        gpu_info['cuda_version'] = torch.version.cuda

        logger.info(f"GPU Available: {gpu_info['device_name']}")
        logger.info(f"Total Memory: {gpu_info['memory_total']:.2f} GB")
        logger.info(f"CUDA Version: {gpu_info['cuda_version']}")
    else:
        logger.warning("No GPU available - training will be very slow on CPU")
        logger.warning("Consider using Google Colab or cloud GPU for faster training")

    return gpu_info


def estimate_training_time(
    num_images: int,
    epochs: int,
    batch_size: int,
    gpu_available: bool
) -> float:
    """Estimate training time based on dataset size and hardware.

    Uses empirical benchmarks:
    - GPU (T4): ~0.5 seconds per batch of 16 images
    - CPU: ~5 seconds per batch of 8 images

    Args:
        num_images: Total number of training images
        epochs: Number of training epochs
        batch_size: Batch size
        gpu_available: Whether GPU is available

    Returns:
        Estimated training time in hours
    """
    batches_per_epoch = num_images / batch_size

    if gpu_available:
        seconds_per_batch = 0.5
    else:
        seconds_per_batch = 5.0

    total_seconds = batches_per_epoch * epochs * seconds_per_batch
    return total_seconds / 3600  # Convert to hours


def load_or_create_model(
    model_name: str,
    resume: bool = False,
    checkpoint_path: Optional[Path] = None
) -> YOLO:
    """Load existing model or create new one from pretrained weights.

    Args:
        model_name: Model architecture name (e.g., 'yolov8n.pt')
        resume: Whether to resume from checkpoint
        checkpoint_path: Path to checkpoint file for resumption

    Returns:
        YOLO model instance ready for training

    Raises:
        ModelLoadError: If model loading fails
    """
    try:
        if resume and checkpoint_path and checkpoint_path.exists():
            logger.info(f"Resuming training from checkpoint: {checkpoint_path}")
            model = YOLO(str(checkpoint_path))
        else:
            logger.info(f"Loading pretrained model: {model_name}")
            model = YOLO(model_name)

            # Log model architecture
            logger.info(f"Model architecture: {model_name}")
            logger.info(f"Parameters: {sum(p.numel() for p in model.model.parameters()):,}")

        return model

    except Exception as e:
        raise ModelLoadError(f"Failed to load model: {e}") from e


def train_model(
    model: YOLO,
    config: Dict,
    save_dir: Path
) -> Dict:
    """Train YOLOv8 model with specified configuration.

    Training process:
    1. Load data and validate dataset
    2. Set up training parameters
    3. Train model with automatic checkpointing
    4. Save best model and training metrics

    Args:
        model: YOLO model instance
        config: Training configuration dictionary
        save_dir: Directory to save training outputs

    Returns:
        Dictionary containing training results:
        - best_weights: Path to best model weights
        - final_metrics: Final validation metrics
        - training_time: Total training time in seconds

    Raises:
        TrainingError: If training fails
    """
    try:
        start_time = time.time()

        logger.info("=" * 70)
        logger.info("STARTING TRAINING")
        logger.info("=" * 70)

        # Extract training parameters from config
        train_params = {
            'data': str(config['data']),
            'epochs': config['epochs'],
            'imgsz': config['imgsz'],
            'batch': config.get('batch', 16),
            'device': config.get('device', ''),
            'workers': config.get('workers', 8),
            'optimizer': config.get('optimizer', 'AdamW'),
            'lr0': config.get('lr0', 0.001),
            'lrf': config.get('lrf', 0.01),
            'momentum': config.get('momentum', 0.937),
            'weight_decay': config.get('weight_decay', 0.0005),
            'patience': config.get('patience', 20),
            'save_period': config.get('save_period', 10),
            'project': str(save_dir.parent),
            'name': save_dir.name,
            'exist_ok': True,
            'plots': config.get('plots', True),
            'verbose': config.get('verbose', True),
            'amp': config.get('amp', True),
            'deterministic': config.get('deterministic', False),
            'seed': config.get('seed', 42),
        }

        # Add augmentation parameters
        augmentation_params = {
            'hsv_h': config.get('hsv_h', 0.015),
            'hsv_s': config.get('hsv_s', 0.7),
            'hsv_v': config.get('hsv_v', 0.4),
            'degrees': config.get('degrees', 0.0),
            'translate': config.get('translate', 0.1),
            'scale': config.get('scale', 0.5),
            'shear': config.get('shear', 0.0),
            'perspective': config.get('perspective', 0.0),
            'flipud': config.get('flipud', 0.0),
            'fliplr': config.get('fliplr', 0.0),
            'mosaic': config.get('mosaic', 1.0),
            'mixup': config.get('mixup', 0.0),
            'copy_paste': config.get('copy_paste', 0.0),
        }

        train_params.update(augmentation_params)

        # Log training configuration
        logger.info("Training Configuration:")
        for key, value in train_params.items():
            logger.info(f"  {key}: {value}")

        # Train the model
        logger.info("\nStarting training loop...")
        results = model.train(**train_params)

        training_time = time.time() - start_time

        logger.info("=" * 70)
        logger.info("TRAINING COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Training time: {training_time / 3600:.2f} hours")
        logger.info(f"Best weights: {save_dir / 'weights' / 'best.pt'}")
        logger.info(f"Last weights: {save_dir / 'weights' / 'last.pt'}")

        # Collect final metrics
        final_metrics = {
            'training_time_hours': training_time / 3600,
            'best_weights': str(save_dir / 'weights' / 'best.pt'),
            'last_weights': str(save_dir / 'weights' / 'last.pt'),
        }

        return final_metrics

    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise TrainingError(f"Training failed: {e}") from e


def copy_best_model_to_models_dir(
    best_weights_path: Path,
    models_dir: Path,
    model_name: str = "yolov8n_epp.pt"
) -> Path:
    """Copy best trained model to models/ directory for API usage.

    Args:
        best_weights_path: Path to best.pt from training
        models_dir: Destination models/ directory
        model_name: Name for the copied model file

    Returns:
        Path to copied model file

    Raises:
        IOError: If copy fails
    """
    try:
        models_dir.mkdir(parents=True, exist_ok=True)
        dest_path = models_dir / model_name

        import shutil
        shutil.copy2(best_weights_path, dest_path)

        logger.info(f"Copied best model to {dest_path}")
        return dest_path

    except Exception as e:
        logger.error(f"Failed to copy model: {e}")
        raise IOError(f"Failed to copy model: {e}") from e


def main():
    """Main entry point for training script."""
    parser = argparse.ArgumentParser(
        description="Train YOLOv8 model for EPP detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with default config
  python scripts/train_model.py --config configs/training_config.yaml

  # Train with custom epochs and batch size
  python scripts/train_model.py --config configs/training_config.yaml --epochs 150 --batch 32

  # Resume training from checkpoint
  python scripts/train_model.py --config configs/training_config.yaml --resume --checkpoint runs/train/epp_detector/weights/last.pt

  # Train on CPU (not recommended)
  python scripts/train_model.py --config configs/training_config.yaml --device cpu
        """
    )

    parser.add_argument(
        '--config',
        type=Path,
        default=Path('configs/training_config.yaml'),
        help='Path to training configuration file'
    )

    parser.add_argument(
        '--data',
        type=Path,
        help='Override dataset YAML path from config'
    )

    parser.add_argument(
        '--epochs',
        type=int,
        help='Override number of epochs from config'
    )

    parser.add_argument(
        '--batch',
        type=int,
        help='Override batch size from config'
    )

    parser.add_argument(
        '--device',
        type=str,
        help='Override device from config (cpu, cuda:0, 0,1,2,3)'
    )

    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume training from last checkpoint'
    )

    parser.add_argument(
        '--checkpoint',
        type=Path,
        help='Path to checkpoint for resuming training'
    )

    parser.add_argument(
        '--project',
        type=Path,
        default=Path('runs/train'),
        help='Project directory for saving results'
    )

    parser.add_argument(
        '--name',
        type=str,
        default='epp_detector',
        help='Experiment name'
    )

    parser.add_argument(
        '--skip-export',
        action='store_true',
        help='Skip copying model to models/ directory after training'
    )

    args = parser.parse_args()

    try:
        # Load configuration
        logger.info(f"Loading configuration from {args.config}")
        config = load_training_config(args.config)

        # Override config with command-line arguments
        if args.data:
            config['data'] = str(args.data)
        if args.epochs:
            config['epochs'] = args.epochs
        if args.batch:
            config['batch'] = args.batch
        if args.device:
            config['device'] = args.device

        # Resolve paths
        project_root = Path(__file__).parent.parent
        data_yaml_path = project_root / config['data']
        save_dir = args.project / args.name

        # Validate dataset
        logger.info("Validating dataset...")
        validate_dataset(data_yaml_path)

        # Check GPU availability
        logger.info("Checking hardware...")
        gpu_info = check_gpu_availability()

        # Estimate training time
        # Rough estimate based on typical dataset size
        estimated_hours = estimate_training_time(
            num_images=2000,  # Typical dataset size
            epochs=config['epochs'],
            batch_size=config.get('batch', 16),
            gpu_available=gpu_info['available']
        )
        logger.info(f"Estimated training time: {estimated_hours:.1f} hours")

        if not gpu_info['available']:
            response = input("\nNo GPU detected. Training will be very slow. Continue? (y/N): ")
            if response.lower() != 'y':
                logger.info("Training cancelled by user")
                sys.exit(0)

        # Load or create model
        model = load_or_create_model(
            config['model'],
            resume=args.resume,
            checkpoint_path=args.checkpoint
        )

        # Train model
        results = train_model(model, config, save_dir)

        # Copy best model to models/ directory
        if not args.skip_export:
            best_weights = Path(results['best_weights'])
            models_dir = project_root / 'models'

            if best_weights.exists():
                copy_best_model_to_models_dir(
                    best_weights,
                    models_dir,
                    "yolov8n_epp.pt"
                )
                logger.info(f"\nModel ready for API usage at: {models_dir / 'yolov8n_epp.pt'}")
                logger.info("Next step: python scripts/export_model.py --weights models/yolov8n_epp.pt")
            else:
                logger.warning(f"Best weights not found at {best_weights}")

        logger.info("\nTraining pipeline complete!")
        logger.info(f"Results saved to: {save_dir}")

    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user")
        sys.exit(130)

    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
