"""
Pipeline completo de construcción de dataset de EPP.

Orquesta todo el proceso end-to-end:
1. Recolección de imágenes desde videos
2. Filtrado de calidad
3. Lanzamiento de herramienta de anotación
4. Validación de anotaciones
5. Aumento de datos
6. División en train/val/test
7. Generación de estadísticas

Uso:
    python scripts/build_dataset_pipeline.py --source data/raw/videos/ --output data/final/

Autor: Equipo EPP Detector
Fecha: 2025-10-16
"""

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

import yaml


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class DatasetPipeline:
    """Orquestador del pipeline de construcción de dataset."""

    def __init__(self, config_path: str):
        """Inicializa pipeline con configuración."""
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        self.project_root = Path(__file__).parent.parent
        self.checkpoints = {}

    def run_step(self, step_name: str, command: list, skip_if_exists: str = None) -> bool:
        """
        Ejecuta un paso del pipeline.

        Args:
            step_name: Nombre del paso
            command: Comando a ejecutar
            skip_if_exists: Path que si existe, permite saltar este paso

        Returns:
            True si exitoso
        """
        if skip_if_exists and os.path.exists(skip_if_exists):
            response = input(f"\n{skip_if_exists} ya existe. ¿Saltar paso '{step_name}'? (s/n): ")
            if response.lower() in ['s', 'si', 'sí']:
                logger.info(f"[SALTAR]  Saltando paso: {step_name}")
                return True

        logger.info("=" * 70)
        logger.info(f">  Ejecutando: {step_name}")
        logger.info("=" * 70)

        try:
            result = subprocess.run(command, cwd=self.project_root, check=True)
            self.checkpoints[step_name] = "completado"
            logger.info(f"[OK] {step_name} completado")
            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"[ERROR] Error en {step_name}: {e}")
            self.checkpoints[step_name] = "error"
            return False

        except KeyboardInterrupt:
            logger.warning(f"\n[ADVERTENCIA]  {step_name} interrumpido por usuario")
            self.checkpoints[step_name] = "interrumpido"
            return False

    def run_full_pipeline(
        self,
        source_dir: str,
        output_dir: str,
        annotation_tool: str = "labelimg",
        augmentation_factor: int = 3
    ) -> None:
        """
        Ejecuta pipeline completo.

        Args:
            source_dir: Directorio con videos o imágenes raw
            output_dir: Directorio de salida final
            annotation_tool: Herramienta de anotación (labelimg o cvat)
            augmentation_factor: Factor de aumento de datos
        """
        logger.info("[INICIO] Iniciando pipeline de construcción de dataset de EPP")
        logger.info(f"Fuente: {source_dir}")
        logger.info(f"Salida: {output_dir}")

        # Directorios intermedios
        filtered_dir = "data/filtered/images"
        annotations_dir = "data/annotations"
        augmented_dir = "data/augmented"

        # PASO 1: Recolección de imágenes
        step1_success = self.run_step(
            "1. Recolección y filtrado de imágenes",
            [
                sys.executable,
                "scripts/collect_images.py",
                "--source", source_dir,
                "--output", filtered_dir,
                "--mode", "auto"
            ],
            skip_if_exists=filtered_dir
        )

        if not step1_success:
            logger.error("Pipeline detenido en paso 1")
            return

        # PASO 2: Anotación (interactivo)
        logger.info("\n" + "=" * 70)
        logger.info(">  Paso 2: Anotación de dataset")
        logger.info("=" * 70)
        logger.info("\nEste paso es interactivo. Anote las imágenes usando la herramienta.")
        logger.info("Presione ENTER cuando haya terminado de anotar...")

        step2_success = self.run_step(
            "2. Lanzamiento de herramienta de anotación",
            [
                sys.executable,
                "scripts/annotate_dataset.py",
                "--images", filtered_dir,
                "--labels", f"{annotations_dir}/labels",
                "--tool", annotation_tool
            ]
        )

        if not step2_success:
            logger.warning("[ADVERTENCIA]  Anotación incompleta, pero continuando...")

        # PASO 3: Validación
        step3_success = self.run_step(
            "3. Validación de anotaciones",
            [
                sys.executable,
                "scripts/validate_dataset.py",
                "--images", f"{annotations_dir}/images",
                "--labels", f"{annotations_dir}/labels",
                "--report", "results/validation_report.html"
            ]
        )

        if not step3_success:
            response = input("\n[ADVERTENCIA]  Validación falló. ¿Continuar de todos modos? (s/n): ")
            if response.lower() not in ['s', 'si', 'sí']:
                logger.error("Pipeline detenido por usuario")
                return

        # PASO 4: Aumento de datos
        step4_success = self.run_step(
            "4. Aumento de datos (augmentation)",
            [
                sys.executable,
                "scripts/augment_dataset.py",
                "--source", annotations_dir,
                "--output", augmented_dir,
                "--factor", str(augmentation_factor)
            ],
            skip_if_exists=augmented_dir
        )

        if not step4_success:
            logger.warning("[ADVERTENCIA]  Aumento de datos falló, usando dataset sin aumentar")
            augmented_dir = annotations_dir

        # PASO 5: Preparación final (train/val/test split)
        step5_success = self.run_step(
            "5. Preparación de dataset final (train/val/test)",
            [
                sys.executable,
                "scripts/prepare_dataset.py",
                "--source", augmented_dir,
                "--output", output_dir
            ]
        )

        if not step5_success:
            logger.error("Pipeline detenido en paso 5")
            return

        # PASO 6: Generación de estadísticas
        step6_success = self.run_step(
            "6. Generación de estadísticas",
            [
                sys.executable,
                "scripts/dataset_stats.py",
                "--data", output_dir,
                "--output", "results/dataset_statistics.html"
            ]
        )

        # Resumen final
        logger.info("\n" + "=" * 70)
        logger.info("[COMPLETADO] PIPELINE COMPLETADO")
        logger.info("=" * 70)
        logger.info(f"\n[OK] Dataset final: {output_dir}")
        logger.info(f"[OK] Reporte de validación: results/validation_report.html")
        logger.info(f"[OK] Estadísticas: results/dataset_statistics.html")
        logger.info(f"\nPróximo paso: Entrenar modelo")
        logger.info(f"  python scripts/train_model.py --data {output_dir}/epp_dataset.yaml")
        logger.info("=" * 70)


def main():
    """Función principal."""
    parser = argparse.ArgumentParser(
        description="Pipeline completo de construcción de dataset de EPP",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Este script ejecuta todo el pipeline de construcción de dataset:
  1. Recolección y filtrado de imágenes desde videos
  2. Anotación interactiva con LabelImg o CVAT
  3. Validación de calidad de anotaciones
  4. Aumento de datos con transformaciones específicas para minería
  5. División en train/val/test
  6. Generación de estadísticas y reportes

El pipeline puede pausarse y reanudarse en cualquier momento.
        """
    )

    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Directorio con videos o imágenes raw"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="data/final",
        help="Directorio para dataset final (default: data/final)"
    )

    parser.add_argument(
        "--annotation-tool",
        type=str,
        choices=["labelimg", "cvat"],
        default="labelimg",
        help="Herramienta de anotación (default: labelimg)"
    )

    parser.add_argument(
        "--augmentation-factor",
        type=int,
        default=3,
        help="Factor de aumento de datos (default: 3)"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/dataset_config.yaml",
        help="Archivo de configuración"
    )

    args = parser.parse_args()

    # Verificar existencia de directorio fuente
    if not os.path.exists(args.source):
        logger.error(f"Directorio fuente no existe: {args.source}")
        sys.exit(1)

    # Verificar configuración
    if not os.path.exists(args.config):
        logger.warning(f"Configuración no encontrada: {args.config}, usando defaults")
        args.config = None

    # Inicializar pipeline
    pipeline = DatasetPipeline(args.config if args.config else "configs/dataset_config.yaml")

    # Ejecutar pipeline
    try:
        pipeline.run_full_pipeline(
            args.source,
            args.output,
            args.annotation_tool,
            args.augmentation_factor
        )
    except KeyboardInterrupt:
        logger.info("\n\n[ADVERTENCIA]  Pipeline interrumpido por usuario")
        logger.info("El pipeline puede reanudarse ejecutando este script nuevamente.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error en pipeline: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
