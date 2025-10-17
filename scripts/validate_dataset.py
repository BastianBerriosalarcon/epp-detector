"""
Validador de calidad de dataset de EPP.

Este script verifica la calidad y consistencia de las anotaciones del dataset,
detectando errores comunes y generando reportes detallados.

Funcionalidades:
- Validación de formato YOLO
- Verificación de coordenadas y geometría de bounding boxes
- Detección de errores lógicos (clases contradictorias)
- Análisis de distribución de clases
- Generación de reportes HTML y JSON

Uso:
    python scripts/validate_dataset.py --data data/annotations/ --report results/validation_report.html

Autor: Equipo EPP Detector
Fecha: 2025-10-16
"""

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import yaml
from tqdm import tqdm


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DatasetValidator:
    """Validador de dataset de EPP con múltiples niveles de verificación."""

    def __init__(self, config: Dict):
        """
        Inicializa validador.

        Args:
            config: Configuración de validación
        """
        self.config = config
        self.errors = []
        self.warnings = []
        self.stats = {
            "total_images": 0,
            "total_annotations": 0,
            "images_without_labels": 0,
            "labels_without_images": 0,
            "class_distribution": defaultdict(int),
            "errors_by_type": defaultdict(int),
            "warnings_by_type": defaultdict(int)
        }

    def validate_dataset(self, images_dir: str, labels_dir: str) -> bool:
        """
        Valida dataset completo.

        Args:
            images_dir: Directorio con imágenes
            labels_dir: Directorio con anotaciones

        Returns:
            True si no hay errores críticos
        """
        logger.info("Iniciando validación de dataset...")

        # Obtener archivos
        image_files = self._get_image_files(images_dir)
        label_files = self._get_label_files(labels_dir)

        self.stats["total_images"] = len(image_files)

        # Verificar paridad de archivos
        self._check_file_pairing(image_files, label_files, images_dir, labels_dir)

        # Validar cada archivo de anotación
        for label_file in tqdm(label_files, desc="Validando anotaciones"):
            self._validate_annotation_file(label_file, images_dir)

        # Análisis de distribución
        self._analyze_class_distribution()

        # Resumen
        has_errors = len(self.errors) > 0
        return not has_errors

    def _get_image_files(self, images_dir: str) -> List[Path]:
        """Obtiene lista de archivos de imagen."""
        extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        files = []
        for ext in extensions:
            files.extend(Path(images_dir).rglob(f'*{ext}'))
        return sorted(files)

    def _get_label_files(self, labels_dir: str) -> List[Path]:
        """Obtiene lista de archivos de anotación."""
        return sorted(Path(labels_dir).rglob('*.txt'))

    def _check_file_pairing(
        self,
        image_files: List[Path],
        label_files: List[Path],
        images_dir: str,
        labels_dir: str
    ) -> None:
        """Verifica que cada imagen tenga su archivo de anotación."""
        image_stems = {f.stem for f in image_files}
        label_stems = {f.stem for f in label_files}

        # Imágenes sin anotación
        missing_labels = image_stems - label_stems
        if missing_labels:
            count = len(missing_labels)
            self.stats["images_without_labels"] = count
            self.warnings.append({
                "tipo": "archivo_faltante",
                "mensaje": f"{count} imágenes sin archivo de anotación",
                "ejemplos": list(missing_labels)[:5]
            })
            self.stats["warnings_by_type"]["archivo_faltante"] += count

        # Anotaciones sin imagen
        missing_images = label_stems - image_stems
        if missing_images:
            count = len(missing_images)
            self.stats["labels_without_images"] = count
            self.errors.append({
                "tipo": "imagen_faltante",
                "mensaje": f"{count} archivos de anotación sin imagen correspondiente",
                "ejemplos": list(missing_images)[:5]
            })
            self.stats["errors_by_type"]["imagen_faltante"] += count

    def _validate_annotation_file(self, label_file: Path, images_dir: str) -> None:
        """
        Valida archivo de anotación individual.

        Args:
            label_file: Ruta al archivo .txt
            images_dir: Directorio con imágenes (para verificar dimensiones)
        """
        try:
            with open(label_file, 'r') as f:
                lines = f.readlines()

            # Archivo vacío es válido (imagen sin objetos)
            if not lines:
                return

            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                if not line:
                    continue

                # Parsear anotación YOLO
                parts = line.split()

                if len(parts) != 5:
                    self._add_error(
                        "formato_invalido",
                        f"{label_file.name}:{line_num} - Formato inválido (esperado 5 valores, encontrado {len(parts)})",
                        {"archivo": str(label_file), "linea": line_num}
                    )
                    continue

                try:
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                except ValueError as e:
                    self._add_error(
                        "valor_invalido",
                        f"{label_file.name}:{line_num} - Valores no numéricos",
                        {"archivo": str(label_file), "linea": line_num, "error": str(e)}
                    )
                    continue

                # Validar class_id
                if class_id < 0 or class_id > 4:  # 5 clases (0-4)
                    self._add_error(
                        "class_id_invalido",
                        f"{label_file.name}:{line_num} - class_id fuera de rango: {class_id}",
                        {"archivo": str(label_file), "linea": line_num, "class_id": class_id}
                    )

                # Validar coordenadas (deben estar en rango 0-1)
                if not (0 <= x_center <= 1 and 0 <= y_center <= 1):
                    self._add_error(
                        "coordenadas_fuera_rango",
                        f"{label_file.name}:{line_num} - Coordenadas fuera de rango [0, 1]",
                        {"archivo": str(label_file), "linea": line_num, "x": x_center, "y": y_center}
                    )

                # Validar dimensiones
                if not (0 < width <= 1 and 0 < height <= 1):
                    self._add_error(
                        "dimensiones_invalidas",
                        f"{label_file.name}:{line_num} - Dimensiones inválidas",
                        {"archivo": str(label_file), "linea": line_num, "w": width, "h": height}
                    )

                # Validar geometría
                min_box_width = self.config.get("min_box_width", 0.01)
                min_box_height = self.config.get("min_box_height", 0.01)

                if width < min_box_width or height < min_box_height:
                    self._add_warning(
                        "bounding_box_pequeno",
                        f"{label_file.name}:{line_num} - Bounding box muy pequeño",
                        {"archivo": str(label_file), "linea": line_num, "w": width, "h": height}
                    )

                max_box_width = self.config.get("max_box_width", 0.95)
                max_box_height = self.config.get("max_box_height", 0.95)

                if width > max_box_width or height > max_box_height:
                    self._add_warning(
                        "bounding_box_grande",
                        f"{label_file.name}:{line_num} - Bounding box muy grande",
                        {"archivo": str(label_file), "linea": line_num, "w": width, "h": height}
                    )

                # Aspect ratio
                if self.config.get("check_aspect_ratio", True):
                    aspect_ratio = width / height if height > 0 else 0
                    min_ratio = self.config.get("min_aspect_ratio", 0.2)
                    max_ratio = self.config.get("max_aspect_ratio", 5.0)

                    if aspect_ratio < min_ratio or aspect_ratio > max_ratio:
                        self._add_warning(
                            "aspect_ratio_inusual",
                            f"{label_file.name}:{line_num} - Aspect ratio inusual: {aspect_ratio:.2f}",
                            {"archivo": str(label_file), "linea": line_num, "ratio": aspect_ratio}
                        )

                # Actualizar estadísticas
                self.stats["total_annotations"] += 1
                self.stats["class_distribution"][class_id] += 1

        except Exception as e:
            self._add_error(
                "error_lectura",
                f"Error leyendo {label_file.name}: {str(e)}",
                {"archivo": str(label_file)}
            )

    def _analyze_class_distribution(self) -> None:
        """Analiza distribución de clases y detecta desbalances."""
        total = self.stats["total_annotations"]
        if total == 0:
            return

        for class_id, count in self.stats["class_distribution"].items():
            ratio = count / total

            min_ratio = self.config.get("min_class_ratio", 0.05)
            if ratio < min_ratio:
                self._add_warning(
                    "clase_desbalanceada",
                    f"Clase {class_id} tiene muy pocas anotaciones ({ratio*100:.1f}%)",
                    {"class_id": class_id, "count": count, "ratio": ratio}
                )

    def _add_error(self, tipo: str, mensaje: str, detalles: Dict = None) -> None:
        """Agrega un error a la lista."""
        self.errors.append({
            "tipo": tipo,
            "mensaje": mensaje,
            "detalles": detalles or {}
        })
        self.stats["errors_by_type"][tipo] += 1

    def _add_warning(self, tipo: str, mensaje: str, detalles: Dict = None) -> None:
        """Agrega un warning a la lista."""
        self.warnings.append({
            "tipo": tipo,
            "mensaje": mensaje,
            "detalles": detalles or {}
        })
        self.stats["warnings_by_type"][tipo] += 1

    def generate_report(self, output_path: str, format: str = "html") -> None:
        """
        Genera reporte de validación.

        Args:
            output_path: Ruta de salida para el reporte
            format: Formato del reporte (html o json)
        """
        if format == "json":
            self._generate_json_report(output_path)
        else:
            self._generate_html_report(output_path)

    def _generate_json_report(self, output_path: str) -> None:
        """Genera reporte en formato JSON."""
        report = {
            "estadisticas": self.stats,
            "errores": self.errors[:50],  # Limitar a 50 errores
            "warnings": self.warnings[:50],
            "resumen": {
                "total_errores": len(self.errors),
                "total_warnings": len(self.warnings),
                "dataset_valido": len(self.errors) == 0
            }
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logger.info(f"Reporte JSON generado: {output_path}")

    def _generate_html_report(self, output_path: str) -> None:
        """Genera reporte en formato HTML."""
        html = f"""
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <title>Reporte de Validación - Dataset EPP</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        .summary {{ background: #f0f0f0; padding: 15px; border-radius: 5px; margin: 20px 0; }}
        .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; }}
        .stat-card {{ background: white; padding: 15px; border-left: 4px solid #4CAF50; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .error {{ border-left-color: #f44336; }}
        .warning {{ border-left-color: #ff9800; }}
        .success {{ border-left-color: #4CAF50; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #4CAF50; color: white; }}
        .error-list, .warning-list {{ max-height: 400px; overflow-y: auto; }}
    </style>
</head>
<body>
    <h1>Reporte de Validación - Dataset EPP Minería Chile</h1>

    <div class="summary">
        <h2>Resumen Ejecutivo</h2>
        <p><strong>Estado:</strong> {'[OK] Dataset válido' if len(self.errors) == 0 else '[ERROR] Errores detectados'}</p>
        <p><strong>Total de imágenes:</strong> {self.stats['total_images']}</p>
        <p><strong>Total de anotaciones:</strong> {self.stats['total_annotations']}</p>
        <p><strong>Errores:</strong> {len(self.errors)}</p>
        <p><strong>Warnings:</strong> {len(self.warnings)}</p>
    </div>

    <div class="stats">
        <div class="stat-card success">
            <h3>Imágenes</h3>
            <p style="font-size: 2em; margin: 0;">{self.stats['total_images']}</p>
        </div>
        <div class="stat-card success">
            <h3>Anotaciones</h3>
            <p style="font-size: 2em; margin: 0;">{self.stats['total_annotations']}</p>
        </div>
        <div class="stat-card error">
            <h3>Errores</h3>
            <p style="font-size: 2em; margin: 0;">{len(self.errors)}</p>
        </div>
        <div class="stat-card warning">
            <h3>Warnings</h3>
            <p style="font-size: 2em; margin: 0;">{len(self.warnings)}</p>
        </div>
    </div>

    <h2>Distribución de Clases</h2>
    <table>
        <tr>
            <th>Clase ID</th>
            <th>Nombre</th>
            <th>Cantidad</th>
            <th>Porcentaje</th>
        </tr>
"""
        class_names = ["hardhat", "safety_vest", "no_hardhat", "no_safety_vest", "person"]
        total = self.stats["total_annotations"]

        for class_id in range(5):
            count = self.stats["class_distribution"].get(class_id, 0)
            percentage = (count / total * 100) if total > 0 else 0
            html += f"""
        <tr>
            <td>{class_id}</td>
            <td>{class_names[class_id]}</td>
            <td>{count}</td>
            <td>{percentage:.1f}%</td>
        </tr>
"""

        html += """
    </table>

    <h2>Errores Detectados</h2>
    <div class="error-list">
"""
        if self.errors:
            html += "<table><tr><th>Tipo</th><th>Mensaje</th></tr>"
            for error in self.errors[:50]:  # Mostrar primeros 50
                html += f"<tr><td>{error['tipo']}</td><td>{error['mensaje']}</td></tr>"
            html += "</table>"
        else:
            html += "<p>[OK] No se detectaron errores</p>"

        html += """
    </div>

    <h2>Warnings</h2>
    <div class="warning-list">
"""
        if self.warnings:
            html += "<table><tr><th>Tipo</th><th>Mensaje</th></tr>"
            for warning in self.warnings[:50]:
                html += f"<tr><td>{warning['tipo']}</td><td>{warning['mensaje']}</td></tr>"
            html += "</table>"
        else:
            html += "<p>[OK] No se detectaron warnings</p>"

        html += """
    </div>
</body>
</html>
"""

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)

        logger.info(f"Reporte HTML generado: {output_path}")

    def print_summary(self) -> None:
        """Imprime resumen en consola."""
        logger.info("=" * 60)
        logger.info("RESUMEN DE VALIDACIÓN")
        logger.info("=" * 60)
        logger.info(f"Total de imágenes:      {self.stats['total_images']}")
        logger.info(f"Total de anotaciones:   {self.stats['total_annotations']}")
        logger.info(f"Errores detectados:     {len(self.errors)}")
        logger.info(f"Warnings:               {len(self.warnings)}")

        if len(self.errors) == 0:
            logger.info("\n[OK] Dataset válido - No se detectaron errores críticos")
        else:
            logger.error("\n[ERROR] Dataset tiene errores que deben corregirse")

        logger.info("=" * 60)


def main():
    """Función principal."""
    parser = argparse.ArgumentParser(
        description="Validador de calidad de dataset de EPP"
    )

    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Directorio raíz del dataset (debe contener subdirectorios images/ y labels/)"
    )

    parser.add_argument(
        "--images",
        type=str,
        help="Directorio de imágenes (si difiere de data/images/)"
    )

    parser.add_argument(
        "--labels",
        type=str,
        help="Directorio de labels (si difiere de data/labels/)"
    )

    parser.add_argument(
        "--report",
        type=str,
        default="results/validation_report.html",
        help="Ruta de salida para el reporte"
    )

    parser.add_argument(
        "--format",
        type=str,
        choices=["html", "json"],
        default="html",
        help="Formato del reporte"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/dataset_config.yaml",
        help="Archivo de configuración"
    )

    args = parser.parse_args()

    # Determinar directorios
    if args.images:
        images_dir = args.images
    else:
        images_dir = os.path.join(args.data, "images")

    if args.labels:
        labels_dir = args.labels
    else:
        labels_dir = os.path.join(args.data, "labels")

    # Verificar existencia
    if not os.path.exists(images_dir):
        logger.error(f"Directorio de imágenes no existe: {images_dir}")
        sys.exit(1)

    if not os.path.exists(labels_dir):
        logger.error(f"Directorio de labels no existe: {labels_dir}")
        sys.exit(1)

    # Cargar configuración
    config = {}
    if os.path.exists(args.config):
        with open(args.config, 'r', encoding='utf-8') as f:
            full_config = yaml.safe_load(f)
            config = full_config.get("validation", {}).get("geometry_checks", {})

    # Crear validador
    validator = DatasetValidator(config)

    # Ejecutar validación
    is_valid = validator.validate_dataset(images_dir, labels_dir)

    # Generar reporte
    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    validator.generate_report(args.report, args.format)

    # Imprimir resumen
    validator.print_summary()

    # Exit code
    sys.exit(0 if is_valid else 1)


if __name__ == "__main__":
    main()
