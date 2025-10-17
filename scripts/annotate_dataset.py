"""
Wrapper para herramientas de anotación de dataset de EPP.

Este script facilita el lanzamiento y configuración de herramientas
de anotación (LabelImg, CVAT) con las clases y configuraciones
correctas para el proyecto EPP Detector.

Funcionalidades:
- Verificación de instalación de herramientas
- Lanzamiento de LabelImg con configuración pre-cargada
- Instrucciones para setup de CVAT
- Seguimiento de progreso de anotación
- Validación básica post-anotación

Uso:
    python scripts/annotate_dataset.py --images data/filtered/images/ --tool labelimg

Autor: Equipo EPP Detector
Fecha: 2025-10-16
"""

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

import yaml


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class AnnotationToolManager:
    """
    Gestor de herramientas de anotación para dataset de EPP.

    Proporciona interfaz unificada para configurar y lanzar
    diferentes herramientas de anotación.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Inicializa gestor de herramientas.

        Args:
            config_path: Ruta a archivo de configuración (opcional)
        """
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                full_config = yaml.safe_load(f)
                self.config = full_config.get("annotation", {})
                self.classes = list(full_config.get("classes", {}).get("names", {}).values())
        else:
            self.classes = ["hardhat", "safety_vest", "no_hardhat", "no_safety_vest", "person"]
            self.config = {"tool": "labelimg"}

    def check_labelimg_installed(self) -> bool:
        """
        Verifica si LabelImg está instalado.

        Returns:
            True si está instalado, False en caso contrario
        """
        try:
            result = subprocess.run(
                ["labelImg", "--help"],
                capture_output=True,
                timeout=5
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    def install_labelimg(self) -> bool:
        """
        Instala LabelImg usando pip.

        Returns:
            True si instalación exitosa, False en caso contrario
        """
        logger.info("Instalando LabelImg...")

        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "labelImg"],
                capture_output=True,
                text=True
            )

            if result.returncode == 0:
                logger.info("LabelImg instalado exitosamente")
                return True
            else:
                logger.error(f"Error instalando LabelImg: {result.stderr}")
                return False

        except Exception as e:
            logger.error(f"Excepción durante instalación: {e}")
            return False

    def create_classes_file(self, output_path: str) -> None:
        """
        Crea archivo de clases para LabelImg.

        Args:
            output_path: Ruta donde crear el archivo
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            for cls in self.classes:
                f.write(f"{cls}\n")

        logger.info(f"Archivo de clases creado: {output_path}")

    def launch_labelimg(
        self,
        images_dir: str,
        labels_dir: Optional[str] = None,
        classes_file: Optional[str] = None
    ) -> None:
        """
        Lanza LabelImg con configuración del proyecto.

        Args:
            images_dir: Directorio con imágenes para anotar
            labels_dir: Directorio para guardar anotaciones (opcional)
            classes_file: Archivo con lista de clases (opcional)
        """
        # Verificar instalación
        if not self.check_labelimg_installed():
            logger.warning("LabelImg no está instalado")

            response = input("¿Desea instalar LabelImg ahora? (s/n): ")
            if response.lower() in ['s', 'si', 'sí', 'y', 'yes']:
                if not self.install_labelimg():
                    logger.error("No se pudo instalar LabelImg")
                    return
            else:
                logger.info("Para instalar manualmente: pip install labelImg")
                return

        # Crear directorio de labels si no existe
        if labels_dir is None:
            labels_dir = os.path.join(os.path.dirname(images_dir), "labels")

        os.makedirs(labels_dir, exist_ok=True)

        # Crear archivo de clases si no se proporcionó
        if classes_file is None:
            classes_file = os.path.join(labels_dir, "classes.txt")
            self.create_classes_file(classes_file)

        logger.info("=" * 60)
        logger.info("LANZANDO LABELIMG")
        logger.info("=" * 60)
        logger.info(f"Imágenes:  {images_dir}")
        logger.info(f"Labels:    {labels_dir}")
        logger.info(f"Clases:    {classes_file}")
        logger.info("=" * 60)
        logger.info("")
        logger.info("INSTRUCCIONES DE USO:")
        logger.info("  - Tecla W: Dibujar bounding box")
        logger.info("  - Tecla D: Siguiente imagen")
        logger.info("  - Tecla A: Imagen anterior")
        logger.info("  - Ctrl+S:  Guardar anotación")
        logger.info("  - Ctrl+D:  Copiar bounding box anterior")
        logger.info("")
        logger.info("RECORDATORIO:")
        logger.info("  - Seleccionar 'YOLO' como formato de salida")
        logger.info("  - Cargar archivo de clases desde menú 'View'")
        logger.info("  - Guardar frecuentemente (auto-save recomendado)")
        logger.info("=" * 60)

        # Lanzar LabelImg
        try:
            subprocess.run([
                "labelImg",
                images_dir,
                classes_file,
                labels_dir
            ])

        except KeyboardInterrupt:
            logger.info("\nCerrando LabelImg...")
        except Exception as e:
            logger.error(f"Error lanzando LabelImg: {e}")

    def show_cvat_instructions(self, images_dir: str) -> None:
        """
        Muestra instrucciones para usar CVAT.

        Args:
            images_dir: Directorio con imágenes para anotar
        """
        logger.info("=" * 60)
        logger.info("INSTRUCCIONES PARA CVAT")
        logger.info("=" * 60)
        logger.info("")
        logger.info("CVAT es una herramienta web para anotación colaborativa.")
        logger.info("Requiere instalación de servidor (Docker recomendado).")
        logger.info("")
        logger.info("OPCIÓN 1: Usar CVAT en la nube (cvat.org)")
        logger.info("  1. Crear cuenta en https://app.cvat.ai")
        logger.info("  2. Crear nuevo proyecto: 'EPP Minería Chile'")
        logger.info("  3. Configurar labels:")

        for i, cls in enumerate(self.classes):
            logger.info(f"     - {cls}")

        logger.info("  4. Subir imágenes (crear tarea)")
        logger.info(f"     Directorio: {images_dir}")
        logger.info("  5. Anotar imágenes")
        logger.info("  6. Exportar en formato YOLO 1.1")
        logger.info("")
        logger.info("OPCIÓN 2: Instalar CVAT localmente")
        logger.info("  Ver: https://github.com/opencv/cvat")
        logger.info("  O usar script de setup:")
        logger.info("    python scripts/setup_cvat_project.py")
        logger.info("")
        logger.info("=" * 60)

    def check_annotation_progress(self, images_dir: str, labels_dir: str) -> None:
        """
        Verifica progreso de anotación.

        Args:
            images_dir: Directorio con imágenes
            labels_dir: Directorio con anotaciones
        """
        # Contar imágenes
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        images = []

        for ext in image_extensions:
            images.extend(Path(images_dir).glob(f'*{ext}'))

        total_images = len(images)

        # Contar archivos de anotación
        labels = list(Path(labels_dir).glob('*.txt'))
        total_labels = len(labels)

        # Calcular progreso
        if total_images > 0:
            progress = (total_labels / total_images) * 100
        else:
            progress = 0

        logger.info("=" * 60)
        logger.info("PROGRESO DE ANOTACIÓN")
        logger.info("=" * 60)
        logger.info(f"Total de imágenes:      {total_images}")
        logger.info(f"Imágenes anotadas:      {total_labels}")
        logger.info(f"Imágenes pendientes:    {total_images - total_labels}")
        logger.info(f"Progreso:               {progress:.1f}%")
        logger.info("=" * 60)

        # Mostrar barra de progreso simple
        bar_length = 40
        filled = int(bar_length * progress / 100)
        bar = '█' * filled + '░' * (bar_length - filled)
        logger.info(f"[{bar}] {progress:.1f}%")
        logger.info("=" * 60)


def main():
    """Función principal del script."""
    parser = argparse.ArgumentParser(
        description="Wrapper para herramientas de anotación de dataset de EPP",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:

  # Lanzar LabelImg para anotar imágenes
  python scripts/annotate_dataset.py --images data/filtered/images/ --tool labelimg

  # Especificar directorio de salida para anotaciones
  python scripts/annotate_dataset.py --images data/filtered/images/ --labels data/annotations/labels/ --tool labelimg

  # Ver instrucciones para CVAT
  python scripts/annotate_dataset.py --images data/filtered/images/ --tool cvat

  # Verificar progreso de anotación
  python scripts/annotate_dataset.py --images data/filtered/images/ --labels data/annotations/labels/ --check-progress
        """
    )

    parser.add_argument(
        "--images",
        type=str,
        required=True,
        help="Directorio con imágenes para anotar"
    )

    parser.add_argument(
        "--labels",
        type=str,
        help="Directorio para guardar anotaciones (default: images/../labels/)"
    )

    parser.add_argument(
        "--tool",
        type=str,
        choices=["labelimg", "cvat"],
        default="labelimg",
        help="Herramienta de anotación a usar"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/dataset_config.yaml",
        help="Ruta a archivo de configuración"
    )

    parser.add_argument(
        "--classes-file",
        type=str,
        help="Archivo con lista de clases (default: configs/classes.txt)"
    )

    parser.add_argument(
        "--check-progress",
        action="store_true",
        help="Solo verificar progreso de anotación (no lanzar herramienta)"
    )

    parser.add_argument(
        "--install",
        action="store_true",
        help="Instalar herramienta de anotación y salir"
    )

    args = parser.parse_args()

    # Verificar que directorio de imágenes existe
    if not os.path.exists(args.images):
        logger.error(f"Directorio de imágenes no existe: {args.images}")
        sys.exit(1)

    # Inicializar gestor
    config_path = args.config if os.path.exists(args.config) else None
    manager = AnnotationToolManager(config_path)

    # Determinar directorio de labels
    labels_dir = args.labels
    if labels_dir is None:
        parent = Path(args.images).parent
        labels_dir = str(parent / "labels")

    # Modo instalación
    if args.install:
        if args.tool == "labelimg":
            if manager.check_labelimg_installed():
                logger.info("LabelImg ya está instalado")
            else:
                manager.install_labelimg()
        else:
            manager.show_cvat_instructions(args.images)
        return

    # Modo verificación de progreso
    if args.check_progress:
        manager.check_annotation_progress(args.images, labels_dir)
        return

    # Lanzar herramienta
    if args.tool == "labelimg":
        manager.launch_labelimg(
            args.images,
            labels_dir,
            args.classes_file
        )
    elif args.tool == "cvat":
        manager.show_cvat_instructions(args.images)

    # Verificar progreso después de cerrar
    logger.info("")
    manager.check_annotation_progress(args.images, labels_dir)


if __name__ == "__main__":
    main()
