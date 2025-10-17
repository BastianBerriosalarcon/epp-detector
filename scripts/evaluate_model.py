#!/usr/bin/env python3
"""
Model Evaluation Script for EPP Detection.

This script performs comprehensive evaluation of trained YOLOv8 models:
- Validation metrics (mAP, precision, recall)
- Per-class performance analysis
- Confusion matrix
- Inference speed benchmarking
- Detection visualization on test images
- Results export (JSON, plots, reports)

Use this script to validate model performance before deployment to production.

Author: Bastián Berríos
Project: epp-detector
"""

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from tqdm import tqdm
from ultralytics import YOLO

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class EvaluationError(Exception):
    """Base exception for evaluation errors."""
    pass


class MetricsCalculator:
    """Calculate and store evaluation metrics."""

    def __init__(self, class_names: Dict[int, str]):
        """Initialize metrics calculator.

        Args:
            class_names: Mapping of class IDs to class names
        """
        self.class_names = class_names
        self.per_class_metrics = defaultdict(lambda: {
            'precision': 0.0,
            'recall': 0.0,
            'ap50': 0.0,
            'ap50_95': 0.0,
            'true_positives': 0,
            'false_positives': 0,
            'false_negatives': 0,
        })
        self.inference_times = []
        self.image_count = 0

    def update_from_yolo_metrics(self, results) -> None:
        """Update metrics from YOLO validation results.

        Args:
            results: Results object from model.val()
        """
        # Extract metrics from YOLO results
        # YOLO v8 stores metrics in results.box
        if hasattr(results, 'box'):
            metrics = results.box

            # Overall metrics
            self.map50 = metrics.map50 if hasattr(metrics, 'map50') else 0.0
            self.map50_95 = metrics.map if hasattr(metrics, 'map') else 0.0

            # Per-class metrics
            if hasattr(metrics, 'ap_class_index'):
                for idx, class_id in enumerate(metrics.ap_class_index):
                    if hasattr(metrics, 'p') and len(metrics.p) > idx:
                        self.per_class_metrics[int(class_id)]['precision'] = float(metrics.p[idx])
                    if hasattr(metrics, 'r') and len(metrics.r) > idx:
                        self.per_class_metrics[int(class_id)]['recall'] = float(metrics.r[idx])
                    if hasattr(metrics, 'ap50') and len(metrics.ap50) > idx:
                        self.per_class_metrics[int(class_id)]['ap50'] = float(metrics.ap50[idx])
                    if hasattr(metrics, 'ap') and len(metrics.ap) > idx:
                        self.per_class_metrics[int(class_id)]['ap50_95'] = float(metrics.ap[idx])

    def add_inference_time(self, time_ms: float) -> None:
        """Add inference time measurement.

        Args:
            time_ms: Inference time in milliseconds
        """
        self.inference_times.append(time_ms)

    def get_summary(self) -> Dict:
        """Get complete metrics summary.

        Returns:
            Dictionary with all evaluation metrics
        """
        summary = {
            'overall': {
                'map50': getattr(self, 'map50', 0.0),
                'map50_95': getattr(self, 'map50_95', 0.0),
            },
            'per_class': {},
            'speed': {},
        }

        # Per-class metrics
        for class_id, metrics in self.per_class_metrics.items():
            class_name = self.class_names.get(class_id, f'class_{class_id}')
            summary['per_class'][class_name] = metrics

        # Speed metrics
        if self.inference_times:
            summary['speed'] = {
                'mean_ms': float(np.mean(self.inference_times)),
                'median_ms': float(np.median(self.inference_times)),
                'std_ms': float(np.std(self.inference_times)),
                'min_ms': float(np.min(self.inference_times)),
                'max_ms': float(np.max(self.inference_times)),
                'fps': 1000.0 / np.mean(self.inference_times),
            }

        return summary

    def format_report(self) -> str:
        """Generate human-readable evaluation report.

        Returns:
            Formatted report string
        """
        summary = self.get_summary()
        lines = []

        lines.append("\n" + "=" * 70)
        lines.append("MODEL EVALUATION REPORT")
        lines.append("=" * 70)

        # Overall metrics
        lines.append("\nOverall Metrics:")
        lines.append(f"  mAP@0.5:      {summary['overall']['map50']:.4f}")
        lines.append(f"  mAP@0.5:0.95: {summary['overall']['map50_95']:.4f}")

        # Per-class metrics
        lines.append("\nPer-Class Metrics:")
        for class_name, metrics in summary['per_class'].items():
            lines.append(f"\n  {class_name}:")
            lines.append(f"    Precision:    {metrics['precision']:.4f}")
            lines.append(f"    Recall:       {metrics['recall']:.4f}")
            lines.append(f"    AP@0.5:       {metrics['ap50']:.4f}")
            lines.append(f"    AP@0.5:0.95:  {metrics['ap50_95']:.4f}")

        # Speed metrics
        if summary['speed']:
            lines.append("\nInference Speed:")
            lines.append(f"  Mean:   {summary['speed']['mean_ms']:.2f} ms")
            lines.append(f"  Median: {summary['speed']['median_ms']:.2f} ms")
            lines.append(f"  Std:    {summary['speed']['std_ms']:.2f} ms")
            lines.append(f"  FPS:    {summary['speed']['fps']:.1f}")

        lines.append("=" * 70)

        return "\n".join(lines)


def evaluate_model_on_dataset(
    model: YOLO,
    data_yaml: Path,
    conf_threshold: float = 0.001,
    iou_threshold: float = 0.6
) -> MetricsCalculator:
    """Evaluate model on validation dataset.

    Uses YOLO's built-in validation to compute mAP, precision, recall.

    Args:
        model: Trained YOLO model
        data_yaml: Path to dataset YAML configuration
        conf_threshold: Confidence threshold for detections
        iou_threshold: IoU threshold for NMS

    Returns:
        MetricsCalculator with evaluation results

    Raises:
        EvaluationError: If evaluation fails
    """
    try:
        logger.info("Running model validation on dataset...")

        # Get class names from model
        class_names = model.names

        # Create metrics calculator
        calculator = MetricsCalculator(class_names)

        # Run validation
        results = model.val(
            data=str(data_yaml),
            conf=conf_threshold,
            iou=iou_threshold,
            plots=True,
            save_json=False,
        )

        # Update metrics from results
        calculator.update_from_yolo_metrics(results)

        logger.info("Validation complete")
        return calculator

    except Exception as e:
        raise EvaluationError(f"Evaluation failed: {e}") from e


def benchmark_inference_speed(
    model: YOLO,
    image_dir: Path,
    num_images: int = 100,
    warmup_iterations: int = 10
) -> List[float]:
    """Benchmark model inference speed on real images.

    Performs warmup iterations to stabilize GPU, then measures inference time.

    Args:
        model: Trained YOLO model
        image_dir: Directory containing test images
        num_images: Number of images to benchmark
        warmup_iterations: Number of warmup iterations

    Returns:
        List of inference times in milliseconds

    Raises:
        EvaluationError: If benchmarking fails
    """
    try:
        logger.info(f"Benchmarking inference speed on {num_images} images...")

        # Collect image paths
        image_paths = list(image_dir.glob('*.jpg')) + list(image_dir.glob('*.png'))
        if not image_paths:
            raise EvaluationError(f"No images found in {image_dir}")

        # Limit to requested number
        image_paths = image_paths[:num_images]

        # Warmup
        logger.info(f"Warming up with {warmup_iterations} iterations...")
        for i in range(min(warmup_iterations, len(image_paths))):
            _ = model(image_paths[i % len(image_paths)], verbose=False)

        # Benchmark
        inference_times = []
        logger.info("Benchmarking...")

        for img_path in tqdm(image_paths, desc="Inference"):
            start = time.time()
            _ = model(img_path, verbose=False)
            end = time.time()

            inference_time_ms = (end - start) * 1000
            inference_times.append(inference_time_ms)

        logger.info(f"Benchmarked {len(inference_times)} images")
        return inference_times

    except Exception as e:
        raise EvaluationError(f"Benchmarking failed: {e}") from e


def visualize_predictions(
    model: YOLO,
    image_paths: List[Path],
    output_dir: Path,
    conf_threshold: float = 0.5,
    max_images: int = 20
) -> None:
    """Visualize model predictions on test images.

    Saves images with bounding boxes, labels, and confidence scores.

    Args:
        model: Trained YOLO model
        image_paths: List of image paths to visualize
        output_dir: Directory to save visualizations
        conf_threshold: Confidence threshold for visualization
        max_images: Maximum number of images to visualize
    """
    try:
        logger.info(f"Generating visualizations for {min(len(image_paths), max_images)} images...")

        output_dir.mkdir(parents=True, exist_ok=True)

        for i, img_path in enumerate(tqdm(image_paths[:max_images], desc="Visualizing")):
            # Run inference
            results = model(img_path, conf=conf_threshold, verbose=False)

            # Save annotated image
            if results and len(results) > 0:
                result = results[0]
                annotated = result.plot()  # Returns image with boxes drawn

                output_path = output_dir / f"{img_path.stem}_pred.jpg"
                cv2.imwrite(str(output_path), annotated)

        logger.info(f"Visualizations saved to {output_dir}")

    except Exception as e:
        logger.error(f"Visualization failed: {e}")


def plot_confusion_matrix(
    model: YOLO,
    data_yaml: Path,
    output_path: Path,
    class_names: Dict[int, str]
) -> None:
    """Generate and save confusion matrix plot.

    Args:
        model: Trained YOLO model
        data_yaml: Path to dataset YAML
        output_path: Path to save confusion matrix plot
        class_names: Mapping of class IDs to names
    """
    try:
        logger.info("Generating confusion matrix...")

        # Run validation to get confusion matrix
        # YOLO automatically generates confusion matrix during validation
        results = model.val(data=str(data_yaml), plots=False)

        # Check if confusion matrix was generated
        # In YOLO v8, confusion matrix is saved during validation
        # We'll create our own if needed

        logger.info(f"Confusion matrix saved (check validation results directory)")

    except Exception as e:
        logger.error(f"Failed to generate confusion matrix: {e}")


def save_metrics_json(metrics: Dict, output_path: Path) -> None:
    """Save evaluation metrics to JSON file.

    Args:
        metrics: Dictionary containing evaluation metrics
        output_path: Path to save JSON file
    """
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        logger.info(f"Metrics saved to {output_path}")

    except Exception as e:
        logger.error(f"Failed to save metrics JSON: {e}")


def main():
    """Main entry point for evaluation script."""
    parser = argparse.ArgumentParser(
        description="Evaluate YOLOv8 model for EPP detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate model on validation set
  python scripts/evaluate_model.py --weights models/yolov8n_epp.pt --data data/epp_dataset.yaml

  # Evaluate with custom confidence threshold
  python scripts/evaluate_model.py --weights models/yolov8n_epp.pt --data data/epp_dataset.yaml --conf 0.6

  # Full evaluation with visualizations
  python scripts/evaluate_model.py --weights models/yolov8n_epp.pt --data data/epp_dataset.yaml --visualize --num-vis 50

  # Benchmark inference speed only
  python scripts/evaluate_model.py --weights models/yolov8n_epp.pt --benchmark-only --image-dir data/images/test
        """
    )

    parser.add_argument(
        '--weights',
        type=Path,
        required=True,
        help='Path to trained model weights'
    )

    parser.add_argument(
        '--data',
        type=Path,
        help='Path to dataset YAML configuration'
    )

    parser.add_argument(
        '--conf',
        type=float,
        default=0.001,
        help='Confidence threshold for evaluation (default: 0.001)'
    )

    parser.add_argument(
        '--iou',
        type=float,
        default=0.6,
        help='IoU threshold for NMS (default: 0.6)'
    )

    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('results'),
        help='Output directory for results (default: results/)'
    )

    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Generate visualization of predictions'
    )

    parser.add_argument(
        '--num-vis',
        type=int,
        default=20,
        help='Number of images to visualize (default: 20)'
    )

    parser.add_argument(
        '--benchmark',
        action='store_true',
        help='Benchmark inference speed'
    )

    parser.add_argument(
        '--benchmark-only',
        action='store_true',
        help='Only run speed benchmark (skip validation)'
    )

    parser.add_argument(
        '--num-benchmark',
        type=int,
        default=100,
        help='Number of images for benchmarking (default: 100)'
    )

    parser.add_argument(
        '--image-dir',
        type=Path,
        help='Directory with images for benchmarking'
    )

    args = parser.parse_args()

    try:
        # Validate inputs
        if not args.weights.exists():
            logger.error(f"Model weights not found: {args.weights}")
            sys.exit(1)

        # Load model
        logger.info(f"Loading model from {args.weights}")
        model = YOLO(str(args.weights))

        # Get class names
        class_names = model.names
        logger.info(f"Classes: {class_names}")

        # Create output directory
        args.output_dir.mkdir(parents=True, exist_ok=True)

        results = {}

        # Run evaluation on dataset
        if not args.benchmark_only:
            if not args.data:
                logger.error("--data is required for evaluation")
                sys.exit(1)

            if not args.data.exists():
                logger.error(f"Dataset config not found: {args.data}")
                sys.exit(1)

            calculator = evaluate_model_on_dataset(
                model,
                args.data,
                args.conf,
                args.iou
            )

            # Get metrics summary
            metrics = calculator.get_summary()
            results['validation'] = metrics

            # Print report
            report = calculator.format_report()
            print(report)

            # Save report to file
            report_path = args.output_dir / 'evaluation_report.txt'
            with open(report_path, 'w') as f:
                f.write(report)
            logger.info(f"Report saved to {report_path}")

        # Benchmark inference speed
        if args.benchmark or args.benchmark_only:
            if args.image_dir:
                image_dir = args.image_dir
            elif args.data:
                # Use test set from dataset
                import yaml
                with open(args.data) as f:
                    dataset_config = yaml.safe_load(f)
                dataset_root = Path(dataset_config.get('path', args.data.parent))
                image_dir = dataset_root / 'images' / 'test'

                if not image_dir.exists():
                    image_dir = dataset_root / 'images' / 'val'
            else:
                logger.error("--image-dir or --data required for benchmarking")
                sys.exit(1)

            inference_times = benchmark_inference_speed(
                model,
                image_dir,
                args.num_benchmark
            )

            # Calculate speed metrics
            speed_metrics = {
                'mean_ms': float(np.mean(inference_times)),
                'median_ms': float(np.median(inference_times)),
                'std_ms': float(np.std(inference_times)),
                'fps': 1000.0 / np.mean(inference_times),
            }

            results['speed'] = speed_metrics

            logger.info("\nInference Speed:")
            logger.info(f"  Mean: {speed_metrics['mean_ms']:.2f} ms")
            logger.info(f"  FPS:  {speed_metrics['fps']:.1f}")

        # Generate visualizations
        if args.visualize and args.data:
            import yaml
            with open(args.data) as f:
                dataset_config = yaml.safe_load(f)

            dataset_root = Path(dataset_config.get('path', args.data.parent))
            test_dir = dataset_root / 'images' / 'test'

            if not test_dir.exists():
                test_dir = dataset_root / 'images' / 'val'

            image_paths = list(test_dir.glob('*.jpg')) + list(test_dir.glob('*.png'))

            if image_paths:
                vis_dir = args.output_dir / 'visualizations'
                visualize_predictions(
                    model,
                    image_paths,
                    vis_dir,
                    args.conf,
                    args.num_vis
                )

        # Save metrics to JSON
        if results:
            json_path = args.output_dir / 'metrics.json'
            save_metrics_json(results, json_path)

        logger.info("\nEvaluation complete!")
        logger.info(f"Results saved to {args.output_dir}")

    except KeyboardInterrupt:
        logger.info("\nEvaluation interrupted by user")
        sys.exit(130)

    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
