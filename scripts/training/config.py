"""
Training configuration management.

Separates configuration concerns from training logic (SRP).
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any


@dataclass
class TrainingConfig:
    """Configuration for YOLOv8 training.

    Uses dataclass for type safety and immutability.
    All configuration is centralized here, separate from logic.

    Attributes:
        epochs: Number of training epochs
        batch_size: Batch size for training
        img_size: Input image size
        model: YOLOv8 model variant to use
        dataset_path: Path to dataset directory
        output_dir: Directory for training outputs
        device: Device to use ('0' for GPU, 'cpu' for CPU)
        patience: Early stopping patience
        save_period: Save checkpoint every N epochs
        workers: Number of dataloader workers
        optimizer: Optimizer type
        lr0: Initial learning rate
        weight_decay: Weight decay for regularization
    """

    # Training parameters
    epochs: int = 50
    batch_size: int = 16
    img_size: int = 640
    model: str = "yolov8n.pt"

    # Paths
    dataset_path: Path = field(default_factory=lambda: Path("./data/roboflow"))
    output_dir: Path = field(default_factory=lambda: Path("./runs/train"))

    # Hardware
    device: str = "0"  # GPU 0 or 'cpu'

    # Training behavior
    patience: int = 10
    save_period: int = 5
    workers: int = 8

    # Optimizer
    optimizer: str = "SGD"
    lr0: float = 0.01
    weight_decay: float = 0.0005

    # Data augmentation
    augmentation_config: Dict[str, Any] = field(default_factory=lambda: {
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
    })

    def to_yolo_config(self, dataset_yaml: Path, run_name: str) -> Dict[str, Any]:
        """Convert to YOLOv8 training configuration dictionary.

        Args:
            dataset_yaml: Path to dataset YAML file
            run_name: Name for this training run

        Returns:
            Configuration dictionary for YOLO.train()
        """
        config = {
            "data": str(dataset_yaml),
            "epochs": self.epochs,
            "batch": self.batch_size,
            "imgsz": self.img_size,
            "device": self.device,
            "workers": self.workers,
            "patience": self.patience,
            "save_period": self.save_period,
            "optimizer": self.optimizer,
            "lr0": self.lr0,
            "weight_decay": self.weight_decay,
            "project": str(self.output_dir),
            "name": run_name,
            "exist_ok": True,
            "verbose": True,
        }

        # Add augmentation parameters
        config.update(self.augmentation_config)

        return config

    def validate(self) -> None:
        """Validate configuration values.

        Raises:
            ValueError: If configuration is invalid
        """
        if self.epochs < 1:
            raise ValueError(f"Epochs must be >= 1, got {self.epochs}")

        if self.batch_size < 1:
            raise ValueError(f"Batch size must be >= 1, got {self.batch_size}")

        if self.img_size < 320:
            raise ValueError(f"Image size must be >= 320, got {self.img_size}")

        if self.lr0 <= 0:
            raise ValueError(f"Learning rate must be > 0, got {self.lr0}")

        if not self.dataset_path.exists():
            raise ValueError(f"Dataset path does not exist: {self.dataset_path}")
