"""
Generador de estadísticas y visualizaciones para dataset de EPP.

Genera reporte HTML completo con:
- Distribución de clases
- Estadísticas de bounding boxes
- Heatmaps espaciales
- Imágenes de ejemplo

Uso:
    python scripts/dataset_stats.py --data data/final/ --output results/dataset_report.html

Autor: Equipo EPP Detector
Fecha: 2025-10-16
"""

import argparse
import base64
import io
import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import cv2
import matplotlib
matplotlib.use('Agg')  # Backend sin GUI
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import tqdm


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class DatasetStatisticsGenerator:
    """Generador de estadísticas y visualizaciones de dataset."""

    def __init__(self):
        """Inicializa generador."""
        self.stats = {
            "total_images": 0,
            "total_annotations": 0,
            "class_counts": defaultdict(int),
            "bbox_widths": [],
            "bbox_heights": [],
            "bbox_areas": [],
            "bbox_aspect_ratios": [],
            "spatial_centers_x": [],
            "spatial_centers_y": [],
            "annotations_per_image": [],
            "class_cooccurrence": defaultdict(lambda: defaultdict(int))
        }
        self.class_names = ["hardhat", "safety_vest", "no_hardhat", "no_safety_vest", "person"]
        self.example_images = defaultdict(list)

    def analyze_dataset(self, data_dir: str) -> None:
        """Analiza dataset completo."""
        for split in ["train", "val", "test"]:
            images_dir = os.path.join(data_dir, "images", split)
            labels_dir = os.path.join(data_dir, "labels", split)

            if not os.path.exists(images_dir):
                continue

            logger.info(f"Analizando split: {split}")
            self._analyze_split(images_dir, labels_dir, split)

    def _analyze_split(self, images_dir: str, labels_dir: str, split: str) -> None:
        """Analiza un split del dataset."""
        image_files = list(Path(images_dir).glob("*.jpg")) + list(Path(images_dir).glob("*.png"))

        for img_file in tqdm(image_files, desc=f"Analizando {split}"):
            self.stats["total_images"] += 1

            label_file = Path(labels_dir) / f"{img_file.stem}.txt"
            if not label_file.exists():
                self.stats["annotations_per_image"].append(0)
                continue

            # Leer anotaciones
            annotations = self._read_annotations(label_file)
            self.stats["annotations_per_image"].append(len(annotations))

            # Clases presentes en esta imagen
            classes_in_image = set()

            for ann in annotations:
                self.stats["total_annotations"] += 1

                class_id = ann["class_id"]
                self.stats["class_counts"][class_id] += 1
                classes_in_image.add(class_id)

                # Estadísticas de bbox
                x, y, w, h = ann["bbox"]
                self.stats["bbox_widths"].append(w)
                self.stats["bbox_heights"].append(h)
                self.stats["bbox_areas"].append(w * h)
                self.stats["bbox_aspect_ratios"].append(w / h if h > 0 else 0)

                # Distribución espacial
                self.stats["spatial_centers_x"].append(x)
                self.stats["spatial_centers_y"].append(y)

                # Guardar ejemplos (máximo 3 por clase)
                if len(self.example_images[class_id]) < 3:
                    self.example_images[class_id].append(str(img_file))

            # Co-ocurrencia de clases
            for c1 in classes_in_image:
                for c2 in classes_in_image:
                    self.stats["class_cooccurrence"][c1][c2] += 1

    def _read_annotations(self, label_file: Path) -> List[Dict]:
        """Lee archivo de anotaciones YOLO."""
        annotations = []
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                annotations.append({
                    "class_id": int(parts[0]),
                    "bbox": [float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])]
                })
        return annotations

    def generate_html_report(self, output_path: str) -> None:
        """Genera reporte HTML completo."""
        logger.info(f"Generando reporte HTML: {output_path}")

        # Generar gráficos
        plots = {
            "class_dist": self._plot_class_distribution(),
            "bbox_sizes": self._plot_bbox_size_distribution(),
            "spatial_heatmap": self._plot_spatial_heatmap(),
            "annotations_per_image": self._plot_annotations_per_image()
        }

        # Construir HTML
        html = self._build_html(plots)

        # Guardar
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)

        logger.info(f"Reporte generado: {output_path}")

    def _plot_class_distribution(self) -> str:
        """Genera gráfico de distribución de clases."""
        fig, ax = plt.subplots(figsize=(10, 6))

        classes = list(range(5))
        counts = [self.stats["class_counts"][i] for i in classes]
        names = [self.class_names[i] for i in classes]

        colors = ['#FFD700', '#FFA500', '#FF0000', '#FF6464', '#00FF00']
        ax.bar(names, counts, color=colors)
        ax.set_xlabel('Clase', fontsize=12)
        ax.set_ylabel('Cantidad', fontsize=12)
        ax.set_title('Distribución de Clases en el Dataset', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        # Agregar porcentajes
        total = sum(counts)
        for i, (name, count) in enumerate(zip(names, counts)):
            percentage = (count / total * 100) if total > 0 else 0
            ax.text(i, count, f'{percentage:.1f}%\n({count})', ha='center', va='bottom')

        plt.tight_layout()
        return self._fig_to_base64(fig)

    def _plot_bbox_size_distribution(self) -> str:
        """Genera histogramas de tamaños de bbox."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Width distribution
        axes[0, 0].hist(self.stats["bbox_widths"], bins=50, color='steelblue', alpha=0.7)
        axes[0, 0].set_title('Distribución de Anchura de Bounding Boxes')
        axes[0, 0].set_xlabel('Ancho Normalizado')
        axes[0, 0].set_ylabel('Frecuencia')

        # Height distribution
        axes[0, 1].hist(self.stats["bbox_heights"], bins=50, color='green', alpha=0.7)
        axes[0, 1].set_title('Distribución de Altura de Bounding Boxes')
        axes[0, 1].set_xlabel('Alto Normalizado')
        axes[0, 1].set_ylabel('Frecuencia')

        # Area distribution
        axes[1, 0].hist(self.stats["bbox_areas"], bins=50, color='orange', alpha=0.7)
        axes[1, 0].set_title('Distribución de Área de Bounding Boxes')
        axes[1, 0].set_xlabel('Área Normalizada')
        axes[1, 0].set_ylabel('Frecuencia')

        # Aspect ratio distribution
        axes[1, 1].hist(self.stats["bbox_aspect_ratios"], bins=50, color='red', alpha=0.7)
        axes[1, 1].set_title('Distribución de Aspect Ratio (Ancho/Alto)')
        axes[1, 1].set_xlabel('Aspect Ratio')
        axes[1, 1].set_ylabel('Frecuencia')

        plt.tight_layout()
        return self._fig_to_base64(fig)

    def _plot_spatial_heatmap(self) -> str:
        """Genera heatmap de distribución espacial."""
        fig, ax = plt.subplots(figsize=(10, 8))

        if len(self.stats["spatial_centers_x"]) > 0:
            heatmap, xedges, yedges = np.histogram2d(
                self.stats["spatial_centers_x"],
                self.stats["spatial_centers_y"],
                bins=20,
                range=[[0, 1], [0, 1]]
            )

            extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
            im = ax.imshow(heatmap.T, extent=extent, origin='lower', cmap='hot', aspect='auto')
            ax.set_title('Heatmap de Distribución Espacial de Anotaciones', fontsize=14, fontweight='bold')
            ax.set_xlabel('X (Normalizado)')
            ax.set_ylabel('Y (Normalizado)')
            plt.colorbar(im, ax=ax, label='Densidad de Anotaciones')

        plt.tight_layout()
        return self._fig_to_base64(fig)

    def _plot_annotations_per_image(self) -> str:
        """Genera histograma de anotaciones por imagen."""
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.hist(self.stats["annotations_per_image"], bins=range(0, max(self.stats["annotations_per_image"]) + 2), color='purple', alpha=0.7)
        ax.set_title('Distribución de Anotaciones por Imagen', fontsize=14, fontweight='bold')
        ax.set_xlabel('Número de Anotaciones')
        ax.set_ylabel('Frecuencia')
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        return self._fig_to_base64(fig)

    def _fig_to_base64(self, fig) -> str:
        """Convierte figura matplotlib a base64 para HTML."""
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        return f"data:image/png;base64,{img_base64}"

    def _build_html(self, plots: Dict[str, str]) -> str:
        """Construye HTML del reporte."""
        total = self.stats["total_annotations"]
        avg_per_image = np.mean(self.stats["annotations_per_image"]) if self.stats["annotations_per_image"] else 0

        html = f"""
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <title>Estadísticas del Dataset - EPP Minería Chile</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        h1 {{ color: #333; text-align: center; }}
        .summary {{ background: white; padding: 20px; border-radius: 8px; margin: 20px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; margin: 20px 0; }}
        .card {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); text-align: center; }}
        .card h3 {{ margin: 0 0 10px 0; color: #555; }}
        .card .value {{ font-size: 2.5em; font-weight: bold; color: #4CAF50; }}
        .plot {{ background: white; padding: 20px; border-radius: 8px; margin: 20px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .plot img {{ width: 100%; height: auto; }}
    </style>
</head>
<body>
    <h1>Estadísticas del Dataset de EPP - Minería Chile</h1>

    <div class="summary">
        <h2>Resumen Ejecutivo</h2>
        <div class="grid">
            <div class="card">
                <h3>Total Imágenes</h3>
                <div class="value">{self.stats["total_images"]}</div>
            </div>
            <div class="card">
                <h3>Total Anotaciones</h3>
                <div class="value">{self.stats["total_annotations"]}</div>
            </div>
            <div class="card">
                <h3>Promedio por Imagen</h3>
                <div class="value">{avg_per_image:.1f}</div>
            </div>
        </div>
    </div>

    <div class="plot">
        <h2>Distribución de Clases</h2>
        <img src="{plots["class_dist"]}" alt="Distribución de Clases">
    </div>

    <div class="plot">
        <h2>Distribución de Tamaños de Bounding Boxes</h2>
        <img src="{plots["bbox_sizes"]}" alt="Tamaños de Bounding Boxes">
    </div>

    <div class="plot">
        <h2>Distribución Espacial de Anotaciones</h2>
        <img src="{plots["spatial_heatmap"]}" alt="Heatmap Espacial">
    </div>

    <div class="plot">
        <h2>Anotaciones por Imagen</h2>
        <img src="{plots["annotations_per_image"]}" alt="Anotaciones por Imagen">
    </div>
</body>
</html>
"""
        return html


def main():
    """Función principal."""
    parser = argparse.ArgumentParser(description="Generador de estadísticas de dataset")
    parser.add_argument("--data", type=str, required=True, help="Directorio del dataset")
    parser.add_argument("--output", type=str, default="results/dataset_statistics.html", help="Archivo de salida")
    args = parser.parse_args()

    if not os.path.exists(args.data):
        logger.error(f"Directorio no existe: {args.data}")
        sys.exit(1)

    generator = DatasetStatisticsGenerator()
    generator.analyze_dataset(args.data)
    generator.generate_html_report(args.output)

    logger.info(f"[OK] Estadísticas generadas exitosamente")


if __name__ == "__main__":
    main()
