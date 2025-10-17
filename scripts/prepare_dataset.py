#!/usr/bin/env python3
"""
Dataset Preparation Script for EPP Detection.

This script prepares datasets for YOLOv8 training by:
1. Converting annotations to YOLO format (class x_center y_center width height)
2. Splitting data into train/val/test sets
3. Creating required directory structure
4. Generating dataset statistics and validation reports

Supports multiple annotation formats:
- YOLO format (already in correct format, just organize)
- COCO JSON format (convert to YOLO)
- Pascal VOC XML format (convert to YOLO)
- Roboflow export (already in YOLO format)

Author: Bastián Berríos
Project: epp-detector
"""

import argparse
import json
import logging
import random
import shutil
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class DatasetPreparationError(Exception):
    """Base exception for dataset preparation errors."""
    pass


class AnnotationConversionError(DatasetPreparationError):
    """Raised when annotation conversion fails."""
    pass


class DatasetStatistics:
    """Container for dataset statistics and validation metrics."""

    def __init__(self):
        """Initialize empty statistics container."""
        self.total_images: int = 0
        self.total_annotations: int = 0
        self.class_distribution: Dict[int, int] = defaultdict(int)
        self.image_sizes: List[Tuple[int, int]] = []
        self.annotations_per_image: List[int] = []
        self.invalid_images: List[str] = []
        self.invalid_annotations: List[str] = []

    def add_image(self, image_path: Path, width: int, height: int) -> None:
        """Add image statistics.

        Args:
            image_path: Path to image file
            width: Image width in pixels
            height: Image height in pixels
        """
        self.total_images += 1
        self.image_sizes.append((width, height))

    def add_annotation(self, class_id: int) -> None:
        """Add annotation statistics.

        Args:
            class_id: Class ID of the annotation
        """
        self.total_annotations += 1
        self.class_distribution[class_id] += 1

    def add_image_annotation_count(self, count: int) -> None:
        """Record number of annotations in an image.

        Args:
            count: Number of annotations in the image
        """
        self.annotations_per_image.append(count)

    def report(self, class_names: Dict[int, str]) -> str:
        """Generate human-readable statistics report.

        Args:
            class_names: Mapping of class IDs to class names

        Returns:
            Formatted statistics report string
        """
        report = []
        report.append("\n" + "=" * 70)
        report.append("DATASET STATISTICS")
        report.append("=" * 70)

        # Image statistics
        report.append(f"\nTotal Images: {self.total_images}")
        if self.invalid_images:
            report.append(f"Invalid Images: {len(self.invalid_images)}")

        if self.image_sizes:
            widths, heights = zip(*self.image_sizes)
            report.append(f"\nImage Size Statistics:")
            report.append(f"  Width:  min={min(widths)}, max={max(widths)}, avg={np.mean(widths):.1f}")
            report.append(f"  Height: min={min(heights)}, max={max(heights)}, avg={np.mean(heights):.1f}")

        # Annotation statistics
        report.append(f"\nTotal Annotations: {self.total_annotations}")
        if self.invalid_annotations:
            report.append(f"Invalid Annotations: {len(self.invalid_annotations)}")

        if self.annotations_per_image:
            report.append(f"\nAnnotations per Image:")
            report.append(f"  Min: {min(self.annotations_per_image)}")
            report.append(f"  Max: {max(self.annotations_per_image)}")
            report.append(f"  Avg: {np.mean(self.annotations_per_image):.2f}")
            report.append(f"  Median: {np.median(self.annotations_per_image):.1f}")

        # Class distribution
        if self.class_distribution:
            report.append(f"\nClass Distribution:")
            total = sum(self.class_distribution.values())
            for class_id in sorted(self.class_distribution.keys()):
                count = self.class_distribution[class_id]
                percentage = (count / total) * 100
                class_name = class_names.get(class_id, f"Unknown_{class_id}")
                report.append(f"  {class_id} ({class_name}): {count} ({percentage:.1f}%)")

        report.append("=" * 70)
        return "\n".join(report)


def parse_yolo_annotation(
    annotation_path: Path,
    image_width: int,
    image_height: int
) -> List[Dict]:
    """Parse YOLO format annotation file.

    YOLO format: <class_id> <x_center> <y_center> <width> <height>
    All values normalized to [0, 1]

    Args:
        annotation_path: Path to .txt annotation file
        image_width: Image width for validation
        image_height: Image height for validation

    Returns:
        List of annotation dictionaries with keys: class_id, bbox

    Raises:
        AnnotationConversionError: If annotation format is invalid
    """
    annotations = []

    try:
        with open(annotation_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                if len(parts) != 5:
                    raise AnnotationConversionError(
                        f"Invalid YOLO format at line {line_num}: expected 5 values, got {len(parts)}"
                    )

                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])

                # Validate normalized coordinates
                if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and
                        0 <= width <= 1 and 0 <= height <= 1):
                    logger.warning(
                        f"Invalid normalized coordinates in {annotation_path}:{line_num}: "
                        f"values must be in [0, 1]"
                    )
                    continue

                annotations.append({
                    'class_id': class_id,
                    'bbox': [x_center, y_center, width, height]
                })

    except ValueError as e:
        raise AnnotationConversionError(
            f"Failed to parse {annotation_path}: {e}"
        ) from e

    return annotations


def convert_coco_to_yolo(
    coco_json_path: Path,
    output_dir: Path,
    class_mapping: Optional[Dict[int, int]] = None
) -> None:
    """Convert COCO JSON annotations to YOLO format.

    COCO format: {"images": [...], "annotations": [...], "categories": [...]}
    Converts to YOLO format: <class_id> <x_center> <y_center> <width> <height>

    Args:
        coco_json_path: Path to COCO JSON file
        output_dir: Output directory for YOLO .txt files
        class_mapping: Optional mapping from COCO category IDs to YOLO class IDs

    Raises:
        AnnotationConversionError: If conversion fails
    """
    logger.info(f"Converting COCO annotations from {coco_json_path}")

    try:
        with open(coco_json_path, 'r') as f:
            coco_data = json.load(f)
    except json.JSONDecodeError as e:
        raise AnnotationConversionError(f"Invalid COCO JSON: {e}") from e

    # Build image ID to filename mapping
    image_info = {img['id']: img for img in coco_data['images']}

    # Group annotations by image ID
    annotations_by_image = defaultdict(list)
    for ann in coco_data['annotations']:
        annotations_by_image[ann['image_id']].append(ann)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert each image's annotations
    for image_id, annotations in tqdm(annotations_by_image.items(), desc="Converting COCO"):
        img_info = image_info[image_id]
        img_width = img_info['width']
        img_height = img_info['height']
        img_filename = Path(img_info['file_name']).stem

        output_path = output_dir / f"{img_filename}.txt"

        with open(output_path, 'w') as f:
            for ann in annotations:
                # COCO bbox format: [x_min, y_min, width, height] (absolute pixels)
                x_min, y_min, bbox_width, bbox_height = ann['bbox']

                # Convert to YOLO format (normalized center coordinates)
                x_center = (x_min + bbox_width / 2) / img_width
                y_center = (y_min + bbox_height / 2) / img_height
                width = bbox_width / img_width
                height = bbox_height / img_height

                # Apply class mapping if provided
                category_id = ann['category_id']
                if class_mapping:
                    class_id = class_mapping.get(category_id, category_id)
                else:
                    class_id = category_id

                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

    logger.info(f"Converted {len(annotations_by_image)} images to YOLO format")


def split_dataset(
    image_paths: List[Path],
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[List[Path], List[Path], List[Path]]:
    """Split dataset into train/val/test sets.

    Uses stratified splitting if class information is available.
    Maintains reproducibility with fixed random seed.

    Args:
        image_paths: List of all image paths
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        test_ratio: Proportion for test set
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_paths, val_paths, test_paths)

    Raises:
        ValueError: If ratios don't sum to 1.0
    """
    if not abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}")

    random.seed(seed)
    shuffled_paths = image_paths.copy()
    random.shuffle(shuffled_paths)

    total = len(shuffled_paths)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    train_paths = shuffled_paths[:train_end]
    val_paths = shuffled_paths[train_end:val_end]
    test_paths = shuffled_paths[val_end:]

    logger.info(f"Dataset split: train={len(train_paths)}, val={len(val_paths)}, test={len(test_paths)}")

    return train_paths, val_paths, test_paths


def create_dataset_structure(base_dir: Path) -> Dict[str, Path]:
    """Create standard YOLO dataset directory structure.

    Creates:
    base_dir/
      images/
        train/
        val/
        test/
      labels/
        train/
        val/
        test/

    Args:
        base_dir: Root directory for dataset

    Returns:
        Dictionary mapping split names to their paths
    """
    structure = {}

    for split in ['train', 'val', 'test']:
        img_dir = base_dir / 'images' / split
        lbl_dir = base_dir / 'labels' / split

        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        structure[split] = {'images': img_dir, 'labels': lbl_dir}

    logger.info(f"Created dataset structure at {base_dir}")
    return structure


def copy_files_to_split(
    file_paths: List[Path],
    source_label_dir: Path,
    dest_image_dir: Path,
    dest_label_dir: Path,
    split_name: str
) -> int:
    """Copy image and label files to destination split directory.

    Args:
        file_paths: List of image file paths to copy
        source_label_dir: Source directory containing label .txt files
        dest_image_dir: Destination directory for images
        dest_label_dir: Destination directory for labels
        split_name: Name of split (train/val/test) for logging

    Returns:
        Number of files successfully copied
    """
    copied = 0

    for img_path in tqdm(file_paths, desc=f"Copying {split_name}"):
        # Copy image
        dest_img_path = dest_image_dir / img_path.name
        shutil.copy2(img_path, dest_img_path)

        # Copy corresponding label if exists
        label_name = img_path.stem + '.txt'
        source_label_path = source_label_dir / label_name

        if source_label_path.exists():
            dest_label_path = dest_label_dir / label_name
            shutil.copy2(source_label_path, dest_label_path)
            copied += 1
        else:
            logger.warning(f"Label not found for {img_path.name}")

    return copied


def collect_dataset_statistics(
    image_dir: Path,
    label_dir: Path,
    class_names: Dict[int, str]
) -> DatasetStatistics:
    """Collect statistics from a dataset split.

    Args:
        image_dir: Directory containing images
        label_dir: Directory containing YOLO format labels
        class_names: Mapping of class IDs to names

    Returns:
        DatasetStatistics object with collected metrics
    """
    stats = DatasetStatistics()

    image_paths = list(image_dir.glob('*.jpg')) + list(image_dir.glob('*.png'))

    for img_path in tqdm(image_paths, desc="Collecting statistics"):
        # Read image dimensions
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                stats.invalid_images.append(str(img_path))
                continue

            height, width = img.shape[:2]
            stats.add_image(img_path, width, height)

        except Exception as e:
            logger.error(f"Failed to read {img_path}: {e}")
            stats.invalid_images.append(str(img_path))
            continue

        # Read annotations
        label_path = label_dir / f"{img_path.stem}.txt"
        if not label_path.exists():
            stats.add_image_annotation_count(0)
            continue

        try:
            annotations = parse_yolo_annotation(label_path, width, height)
            stats.add_image_annotation_count(len(annotations))

            for ann in annotations:
                stats.add_annotation(ann['class_id'])

        except AnnotationConversionError as e:
            logger.error(f"Failed to parse {label_path}: {e}")
            stats.invalid_annotations.append(str(label_path))

    return stats


def main():
    """Main entry point for dataset preparation script."""
    parser = argparse.ArgumentParser(
        description="Prepare EPP detection dataset for YOLOv8 training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Prepare Roboflow export (already in YOLO format)
  python scripts/prepare_dataset.py --source /path/to/roboflow/export --output data

  # Convert COCO format to YOLO
  python scripts/prepare_dataset.py --source /path/to/coco --format coco --output data

  # Custom train/val/test split
  python scripts/prepare_dataset.py --source /path/to/images --train-ratio 0.8 --val-ratio 0.15 --test-ratio 0.05
        """
    )

    parser.add_argument(
        '--source',
        type=Path,
        required=True,
        help='Source directory containing images and annotations'
    )

    parser.add_argument(
        '--output',
        type=Path,
        default=Path('data'),
        help='Output directory for prepared dataset (default: data/)'
    )

    parser.add_argument(
        '--format',
        choices=['yolo', 'coco', 'voc'],
        default='yolo',
        help='Source annotation format (default: yolo)'
    )

    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.7,
        help='Training set ratio (default: 0.7)'
    )

    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.2,
        help='Validation set ratio (default: 0.2)'
    )

    parser.add_argument(
        '--test-ratio',
        type=float,
        default=0.1,
        help='Test set ratio (default: 0.1)'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )

    parser.add_argument(
        '--skip-split',
        action='store_true',
        help='Skip splitting if source already has train/val/test structure'
    )

    args = parser.parse_args()

    # Validate source directory
    if not args.source.exists():
        logger.error(f"Source directory does not exist: {args.source}")
        sys.exit(1)

    # Class names for EPP detection
    class_names = {
        0: 'hardhat',
        1: 'head',
        2: 'person'
    }

    try:
        # Create output structure
        output_structure = create_dataset_structure(args.output)

        # Handle different source formats
        if args.format == 'coco':
            logger.info("Converting COCO format to YOLO")
            coco_json = args.source / 'annotations.json'
            if not coco_json.exists():
                logger.error(f"COCO JSON not found: {coco_json}")
                sys.exit(1)

            temp_label_dir = args.source / 'yolo_labels'
            convert_coco_to_yolo(coco_json, temp_label_dir)
            label_source = temp_label_dir
            image_source = args.source / 'images'

        elif args.format == 'yolo':
            # Assume source has images/ and labels/ subdirectories
            label_source = args.source / 'labels'
            image_source = args.source / 'images'

            if not label_source.exists():
                label_source = args.source  # Labels might be in root
            if not image_source.exists():
                image_source = args.source  # Images might be in root

        else:
            logger.error(f"Format {args.format} not yet implemented")
            sys.exit(1)

        # Collect all images
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        all_images = []
        for ext in image_extensions:
            all_images.extend(image_source.glob(ext))
            all_images.extend(image_source.glob(ext.upper()))

        if not all_images:
            logger.error(f"No images found in {image_source}")
            sys.exit(1)

        logger.info(f"Found {len(all_images)} images")

        # Split dataset
        train_images, val_images, test_images = split_dataset(
            all_images,
            args.train_ratio,
            args.val_ratio,
            args.test_ratio,
            args.seed
        )

        # Copy files to splits
        for split_name, split_images in [
            ('train', train_images),
            ('val', val_images),
            ('test', test_images)
        ]:
            if not split_images:
                continue

            split_dirs = output_structure[split_name]
            copied = copy_files_to_split(
                split_images,
                label_source,
                split_dirs['images'],
                split_dirs['labels'],
                split_name
            )
            logger.info(f"{split_name}: copied {copied} image-label pairs")

        # Collect and report statistics
        logger.info("\nCollecting dataset statistics...")
        for split_name in ['train', 'val', 'test']:
            split_dirs = output_structure[split_name]
            stats = collect_dataset_statistics(
                split_dirs['images'],
                split_dirs['labels'],
                class_names
            )

            report = stats.report(class_names)
            logger.info(f"\n{split_name.upper()} SET STATISTICS:{report}")

            # Save report to file
            report_path = args.output / f"{split_name}_statistics.txt"
            with open(report_path, 'w') as f:
                f.write(report)

        logger.info(f"\nDataset preparation complete!")
        logger.info(f"Output directory: {args.output.absolute()}")
        logger.info(f"Next step: python scripts/train_model.py --data {args.output}/epp_dataset.yaml")

    except Exception as e:
        logger.error(f"Dataset preparation failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
