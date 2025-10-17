"""
Script de aumento de datos (augmentation) para dataset de EPP.

Aplica transformaciones específicas para minería chilena:
- Variaciones de iluminación (subterráneo vs superficie)
- Simulación de polvo/niebla
- Transformaciones geométricas
- Sobremuestreo de clases minoritarias

Uso:
    python scripts/augment_dataset.py --source data/annotations/ --output data/augmented/ --factor 3

Autor: Equipo EPP Detector
Fecha: 2025-10-16
"""

import argparse
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import albumentations as A
import cv2
import numpy as np
import yaml
from tqdm import tqdm


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class EPPDataAugmenter:
    """Aumentador de datos específico para EPP en minería."""

    def __init__(self, config: Dict):
        """Inicializa aumentador con configuración."""
        self.config = config
        self.transform = self._build_transform()
        self.stats = {"images_processed": 0, "images_generated": 0}

    def _build_transform(self) -> A.Compose:
        """Construye pipeline de transformaciones."""
        transforms_list = []

        # Transformaciones fotométricas (iluminación/color)
        photo_config = self.config.get("photometric", {})

        if photo_config.get("random_brightness_contrast", {}).get("enable", True):
            transforms_list.append(A.RandomBrightnessContrast(
                brightness_limit=photo_config["random_brightness_contrast"].get("brightness_limit", 0.3),
                contrast_limit=photo_config["random_brightness_contrast"].get("contrast_limit", 0.3),
                p=photo_config["random_brightness_contrast"].get("probability", 0.7)
            ))

        if photo_config.get("hue_saturation_value", {}).get("enable", False):
            transforms_list.append(A.HueSaturationValue(
                hue_shift_limit=photo_config["hue_saturation_value"].get("hue_shift_limit", 20),
                sat_shift_limit=photo_config["hue_saturation_value"].get("sat_shift_limit", 30),
                val_shift_limit=photo_config["hue_saturation_value"].get("val_shift_limit", 20),
                p=photo_config["hue_saturation_value"].get("probability", 0.5)
            ))

        if photo_config.get("gaussian_noise", {}).get("enable", True):
            transforms_list.append(A.GaussNoise(
                var_limit=tuple(photo_config["gaussian_noise"].get("var_limit", [10, 50])),
                p=photo_config["gaussian_noise"].get("probability", 0.2)
            ))

        if photo_config.get("motion_blur", {}).get("enable", True):
            transforms_list.append(A.MotionBlur(
                blur_limit=photo_config["motion_blur"].get("blur_limit", 7),
                p=photo_config["motion_blur"].get("probability", 0.2)
            ))

        # Transformaciones geométricas
        geom_config = self.config.get("geometric", {})

        if geom_config.get("horizontal_flip", {}).get("enable", True):
            transforms_list.append(A.HorizontalFlip(
                p=geom_config["horizontal_flip"].get("probability", 0.5)
            ))

        if geom_config.get("rotate", {}).get("enable", True):
            transforms_list.append(A.Rotate(
                limit=geom_config["rotate"].get("limit", 15),
                p=geom_config["rotate"].get("probability", 0.5)
            ))

        if geom_config.get("shift_scale_rotate", {}).get("enable", True):
            transforms_list.append(A.ShiftScaleRotate(
                shift_limit=geom_config["shift_scale_rotate"].get("shift_limit", 0.1),
                scale_limit=geom_config["shift_scale_rotate"].get("scale_limit", 0.2),
                rotate_limit=geom_config["shift_scale_rotate"].get("rotate_limit", 10),
                p=geom_config["shift_scale_rotate"].get("probability", 0.5)
            ))

        # Oclusiones
        occl_config = self.config.get("occlusion", {})

        if occl_config.get("coarse_dropout", {}).get("enable", True):
            transforms_list.append(A.CoarseDropout(
                max_holes=occl_config["coarse_dropout"].get("max_holes", 3),
                max_height=occl_config["coarse_dropout"].get("max_height", 32),
                max_width=occl_config["coarse_dropout"].get("max_width", 32),
                min_holes=occl_config["coarse_dropout"].get("min_holes", 1),
                min_height=occl_config["coarse_dropout"].get("min_height", 8),
                min_width=occl_config["coarse_dropout"].get("min_width", 8),
                p=occl_config["coarse_dropout"].get("probability", 0.2)
            ))

        # Configurar para YOLO
        return A.Compose(
            transforms_list,
            bbox_params=A.BboxParams(
                format='yolo',
                min_area=self.config.get("min_area", 100),
                min_visibility=self.config.get("min_visibility", 0.3),
                label_fields=['class_labels']
            )
        )

    def augment_dataset(self, source_dir: str, output_dir: str, factor: int = 3) -> None:
        """Aumenta dataset completo."""
        images_dir = os.path.join(source_dir, "images")
        labels_dir = os.path.join(source_dir, "labels")

        output_images_dir = os.path.join(output_dir, "images")
        output_labels_dir = os.path.join(output_dir, "labels")

        os.makedirs(output_images_dir, exist_ok=True)
        os.makedirs(output_labels_dir, exist_ok=True)

        # Obtener imágenes
        image_files = list(Path(images_dir).glob("*.jpg")) + list(Path(images_dir).glob("*.png"))

        logger.info(f"Aumentando {len(image_files)} imágenes con factor {factor}...")

        for image_file in tqdm(image_files, desc="Aumentando imágenes"):
            label_file = Path(labels_dir) / f"{image_file.stem}.txt"

            if not label_file.exists():
                logger.warning(f"Sin anotación para {image_file.name}, copiando sin aumentar")
                shutil.copy(image_file, output_images_dir)
                continue

            # Copiar original
            shutil.copy(image_file, output_images_dir)
            shutil.copy(label_file, output_labels_dir)

            # Generar versiones aumentadas
            for i in range(factor):
                self._augment_image(image_file, label_file, output_images_dir, output_labels_dir, i)

            self.stats["images_processed"] += 1

        logger.info(f"Aumento completado: {self.stats['images_processed']} procesadas, {self.stats['images_generated']} generadas")

    def _augment_image(self, image_file: Path, label_file: Path, out_img_dir: str, out_lbl_dir: str, idx: int) -> None:
        """Aumenta una imagen individual."""
        # Leer imagen
        image = cv2.imread(str(image_file))
        if image is None:
            return

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Leer anotaciones
        bboxes, class_labels = self._read_yolo_labels(label_file, image.shape[:2])

        if not bboxes:
            return

        # Aplicar transformación
        try:
            transformed = self.transform(image=image, bboxes=bboxes, class_labels=class_labels)

            # Guardar imagen aumentada
            aug_image = cv2.cvtColor(transformed['image'], cv2.COLOR_RGB2BGR)
            aug_image_name = f"{image_file.stem}_aug{idx+1}{image_file.suffix}"
            aug_image_path = os.path.join(out_img_dir, aug_image_name)
            cv2.imwrite(aug_image_path, aug_image)

            # Guardar anotaciones aumentadas
            aug_label_name = f"{image_file.stem}_aug{idx+1}.txt"
            aug_label_path = os.path.join(out_lbl_dir, aug_label_name)
            self._write_yolo_labels(aug_label_path, transformed['bboxes'], transformed['class_labels'])

            self.stats["images_generated"] += 1

        except Exception as e:
            logger.warning(f"Error aumentando {image_file.name}: {e}")

    def _read_yolo_labels(self, label_file: Path, image_shape: Tuple[int, int]) -> Tuple[List, List]:
        """Lee anotaciones en formato YOLO."""
        bboxes = []
        class_labels = []

        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue

                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])

                bboxes.append([x_center, y_center, width, height])
                class_labels.append(class_id)

        return bboxes, class_labels

    def _write_yolo_labels(self, label_file: str, bboxes: List, class_labels: List) -> None:
        """Escribe anotaciones en formato YOLO."""
        with open(label_file, 'w') as f:
            for bbox, class_id in zip(bboxes, class_labels):
                f.write(f"{class_id} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")


def main():
    """Función principal."""
    parser = argparse.ArgumentParser(description="Aumento de datos para dataset de EPP")

    parser.add_argument("--source", type=str, required=True, help="Directorio de dataset original")
    parser.add_argument("--output", type=str, required=True, help="Directorio para dataset aumentado")
    parser.add_argument("--factor", type=int, default=3, help="Factor de aumento (imágenes adicionales por imagen)")
    parser.add_argument("--config", type=str, default="configs/dataset_config.yaml", help="Archivo de configuración")

    args = parser.parse_args()

    # Cargar configuración
    config = {}
    if os.path.exists(args.config):
        with open(args.config, 'r', encoding='utf-8') as f:
            full_config = yaml.safe_load(f)
            config = full_config.get("augmentation", {}).get("transforms", {})

    # Crear aumentador
    augmenter = EPPDataAugmenter(config)

    # Ejecutar aumento
    augmenter.augment_dataset(args.source, args.output, args.factor)


if __name__ == "__main__":
    main()
