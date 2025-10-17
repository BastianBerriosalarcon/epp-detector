#!/usr/bin/env python3
"""
Download Construction Site Safety Dataset from Roboflow

This script downloads the "Construction Site Safety" dataset from Roboflow Universe
in YOLOv8 format. The dataset contains images of workers with/without hardhats and
safety vests for PPE detection in construction and mining environments.

Dataset: https://universe.roboflow.com/roboflow-universe-projects/construction-site-safety
Classes: hardhat, safety_vest, no_hardhat, no_safety_vest, person

Usage:
    # Download using API key from .env
    python download_roboflow.py

    # Download to specific location
    python download_roboflow.py --output-dir ./data/roboflow

    # Download specific version
    python download_roboflow.py --version 27

Requirements:
    - roboflow library: pip install roboflow
    - ROBOFLOW_API_KEY in .env file or environment variable

Setup:
    1. Create account at https://roboflow.com
    2. Get API key from https://app.roboflow.com/settings/api
    3. Add to .env file:
       ROBOFLOW_API_KEY=your_api_key_here

Author: Bastian Berrios
Date: 2025-01-15
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Third-party imports
try:
    from roboflow import Roboflow
except ImportError:
    print("ERROR: roboflow library not installed")
    print("Install with: pip install roboflow")
    sys.exit(1)

try:
    from dotenv import load_dotenv
except ImportError:
    print("WARNING: python-dotenv not installed")
    print("Install with: pip install python-dotenv")
    load_dotenv = None


# =============================================================================
# Configuration
# =============================================================================

DATASET_CONFIG = {
    "workspace": "roboflow-universe-projects",
    "project": "construction-site-safety",
    "version": 27,  # Default version
    "format": "yolov8",  # YOLOv8 format
}

EXPECTED_CLASSES = ["hardhat", "safety_vest", "no_hardhat", "no_safety_vest", "person"]


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
# Environment Setup
# =============================================================================

def load_api_key(logger: logging.Logger) -> str:
    """
    Load Roboflow API key from environment.

    Args:
        logger: Logger instance

    Returns:
        API key string

    Raises:
        ValueError: If API key is not found
    """
    # Try to load from .env file
    if load_dotenv is not None:
        env_path = Path(__file__).parent.parent.parent / ".env"
        if env_path.exists():
            load_dotenv(env_path)
            logger.info(f"Loaded environment from: {env_path}")

    # Get API key from environment
    api_key = os.getenv("ROBOFLOW_API_KEY")

    if not api_key:
        raise ValueError(
            "ROBOFLOW_API_KEY not found in environment.\n\n"
            "Setup instructions:\n"
            "1. Create account at https://roboflow.com\n"
            "2. Get API key from https://app.roboflow.com/settings/api\n"
            "3. Add to .env file:\n"
            "   ROBOFLOW_API_KEY=your_api_key_here\n"
            "   OR set environment variable:\n"
            "   export ROBOFLOW_API_KEY=your_api_key_here"
        )

    logger.info("API key loaded successfully")
    return api_key


# =============================================================================
# Dataset Download
# =============================================================================

def download_dataset(
    api_key: str,
    output_dir: Path,
    version: int,
    logger: logging.Logger
) -> Path:
    """
    Download Hard Hat Workers dataset from Roboflow.

    Args:
        api_key: Roboflow API key
        output_dir: Directory to save dataset
        version: Dataset version to download
        logger: Logger instance

    Returns:
        Path to downloaded dataset directory

    Raises:
        Exception: If download fails
    """
    logger.info("=" * 80)
    logger.info("DOWNLOADING ROBOFLOW DATASET")
    logger.info("=" * 80)
    logger.info(f"Workspace: {DATASET_CONFIG['workspace']}")
    logger.info(f"Project: {DATASET_CONFIG['project']}")
    logger.info(f"Version: {version}")
    logger.info(f"Format: {DATASET_CONFIG['format']}")
    logger.info(f"Output directory: {output_dir}")
    logger.info("=" * 80)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Initialize Roboflow
        logger.info("Connecting to Roboflow...")
        rf = Roboflow(api_key=api_key)

        # Get workspace and project
        logger.info(f"Accessing workspace: {DATASET_CONFIG['workspace']}")
        workspace = rf.workspace(DATASET_CONFIG['workspace'])

        logger.info(f"Accessing project: {DATASET_CONFIG['project']}")
        project = workspace.project(DATASET_CONFIG['project'])

        # Get specific version
        logger.info(f"Getting version {version}...")
        dataset = project.version(version)

        # Download dataset
        logger.info("Downloading dataset (this may take a few minutes)...")
        logger.info("Downloading...")

        # Download returns Dataset object with location attribute
        # Note: overwrite=True ensures fresh download even if partial download exists
        downloaded_dataset = dataset.download(
            model_format=DATASET_CONFIG['format'],
            location=str(output_dir),
            overwrite=True
        )

        # Get actual location from Dataset object
        dataset_location = Path(downloaded_dataset.location)

        logger.info(f"Download complete!")
        logger.info(f"Dataset saved to: {dataset_location}")
        logger.info("=" * 80)

        return dataset_location

    except Exception as e:
        logger.error(f"Download failed: {e}")
        raise


# =============================================================================
# Dataset Verification
# =============================================================================

def verify_dataset(dataset_path: Path, logger: logging.Logger) -> dict:
    """
    Verify downloaded dataset structure and content.

    Args:
        dataset_path: Path to downloaded dataset
        logger: Logger instance

    Returns:
        Dictionary with dataset statistics

    Raises:
        ValueError: If dataset structure is invalid
    """
    logger.info("=" * 80)
    logger.info("VERIFYING DATASET")
    logger.info("=" * 80)

    stats = {
        "train": {"images": 0, "labels": 0},
        "valid": {"images": 0, "labels": 0},
        "test": {"images": 0, "labels": 0},
    }

    # Check for data.yaml
    yaml_path = dataset_path / "data.yaml"
    if not yaml_path.exists():
        raise ValueError(f"data.yaml not found in {dataset_path}")

    logger.info(f"Found data.yaml")

    # Verify splits
    for split in ["train", "valid", "test"]:
        images_dir = dataset_path / split / "images"
        labels_dir = dataset_path / split / "labels"

        if not images_dir.exists():
            logger.warning(f"{split}/images directory not found")
            continue

        # Count images
        image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
        stats[split]["images"] = len(image_files)

        # Count labels
        if labels_dir.exists():
            label_files = list(labels_dir.glob("*.txt"))
            stats[split]["labels"] = len(label_files)

        logger.info(f"{split.capitalize():5s}: {stats[split]['images']:4d} images, "
                   f"{stats[split]['labels']:4d} labels")

    # Summary
    total_images = sum(s["images"] for s in stats.values())
    total_labels = sum(s["labels"] for s in stats.values())

    logger.info("-" * 80)
    logger.info(f"Total: {total_images:4d} images, {total_labels:4d} labels")
    logger.info("=" * 80)

    # Sanity checks
    if total_images == 0:
        raise ValueError("No images found in dataset!")

    if total_labels == 0:
        logger.warning("No label files found. Check dataset integrity.")

    # Check for class imbalance (optional)
    for split in ["train", "valid", "test"]:
        if stats[split]["images"] > 0:
            label_ratio = stats[split]["labels"] / stats[split]["images"]
            if label_ratio < 0.9:
                logger.warning(
                    f"{split} split has {label_ratio:.1%} images with labels. "
                    "Some images may be unlabeled."
                )

    return stats


# =============================================================================
# Post-Download Cleanup
# =============================================================================

def create_readme(dataset_path: Path, stats: dict, logger: logging.Logger):
    """
    Create a README with dataset information.

    Args:
        dataset_path: Path to dataset directory
        stats: Dataset statistics
        logger: Logger instance
    """
    readme_path = dataset_path / "DATASET_INFO.txt"

    total_images = sum(s["images"] for s in stats.values())
    total_labels = sum(s["labels"] for s in stats.values())

    readme_content = f"""
Construction Site Safety Dataset - Download Information
========================================================

Dataset: {DATASET_CONFIG['project']}
Workspace: {DATASET_CONFIG['workspace']}
Version: {DATASET_CONFIG['version']}
Format: {DATASET_CONFIG['format']}

Classes:
--------
{', '.join(EXPECTED_CLASSES)}

Dataset Statistics:
-------------------
Train:      {stats['train']['images']:4d} images, {stats['train']['labels']:4d} labels
Validation: {stats['valid']['images']:4d} images, {stats['valid']['labels']:4d} labels
Test:       {stats['test']['images']:4d} images, {stats['test']['labels']:4d} labels
-------------------
Total:      {total_images:4d} images, {total_labels:4d} labels

Directory Structure:
--------------------
{dataset_path}/
├── data.yaml          # Dataset configuration for YOLOv8
├── train/
│   ├── images/        # Training images
│   └── labels/        # Training labels (YOLO format)
├── valid/
│   ├── images/        # Validation images
│   └── labels/        # Validation labels
└── test/
    ├── images/        # Test images
    └── labels/        # Test labels

Label Format (YOLO):
--------------------
Each .txt file contains one line per object:
<class_id> <x_center> <y_center> <width> <height>
All values are normalized to [0, 1]

Next Steps:
-----------
1. Inspect data.yaml to verify class names and paths
2. Visualize some samples with scripts/visualize_data.py
3. Train model with scripts/train_gcp.py

Dataset Source:
---------------
https://universe.roboflow.com/roboflow-universe-projects/construction-site-safety
"""

    with open(readme_path, "w") as f:
        f.write(readme_content.strip())

    logger.info(f"Created dataset info: {readme_path}")


# =============================================================================
# Argument Parsing
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Download Construction Site Safety dataset from Roboflow",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data/roboflow",
        help="Directory to save downloaded dataset"
    )

    parser.add_argument(
        "--version",
        type=int,
        default=DATASET_CONFIG["version"],
        help="Dataset version to download"
    )

    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip dataset verification after download"
    )

    return parser.parse_args()


# =============================================================================
# Main
# =============================================================================

def main():
    """Main download pipeline."""
    # Parse arguments
    args = parse_args()

    # Setup logging
    logger = setup_logging()

    logger.info("=" * 80)
    logger.info("ROBOFLOW DATASET DOWNLOADER")
    logger.info("=" * 80)

    try:
        # Load API key
        api_key = load_api_key(logger)

        # Download dataset
        output_dir = Path(args.output_dir)
        dataset_path = download_dataset(api_key, output_dir, args.version, logger)

        # Verify dataset
        if not args.skip_verify:
            stats = verify_dataset(dataset_path, logger)

            # Create README
            create_readme(dataset_path, stats, logger)

        logger.info("=" * 80)
        logger.info("DATASET DOWNLOAD COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Dataset location: {dataset_path}")
        logger.info(f"Configuration file: {dataset_path / 'data.yaml'}")
        logger.info("\nNext steps:")
        logger.info("1. Review data.yaml to verify class names and paths")
        logger.info("2. Train model: python scripts/train_gcp.py --dataset-path " + str(dataset_path))
        logger.info("=" * 80)

        return 0

    except Exception as e:
        logger.error("=" * 80)
        logger.error("DOWNLOAD FAILED")
        logger.error("=" * 80)
        logger.exception(e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
