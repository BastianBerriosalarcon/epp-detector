#!/usr/bin/env python3
"""
Script para descargar modelo base YOLOv8 para EPP Detection

Este script descarga un modelo YOLOv8 pre-entrenado desde Ultralytics
y lo guarda en el directorio models/ del proyecto. Este modelo base
puede ser usado para:
- Inferencia directa (detección genérica de objetos)
- Fine-tuning con dataset de EPP chileno
- Baseline para comparación de performance

Uso:
    python scripts/download_base_model.py
    python scripts/download_base_model.py --model-size s
    python scripts/download_base_model.py --model-size m --output-dir ./custom_models

Tamaños disponibles:
    - n (nano): 3.2M params, más rápido, menor precisión
    - s (small): 11.2M params, balance velocidad/precisión
    - m (medium): 25.9M params, mejor precisión
    - l (large): 43.7M params, alta precisión
    - x (xlarge): 68.2M params, máxima precisión

Autor: Bastian Berrios
Fecha: 2025-10-16
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

try:
    from ultralytics import YOLO
except ImportError:
    print("Error: ultralytics no está instalado")
    print("Instalar con: pip install ultralytics")
    sys.exit(1)


# =============================================================================
# Configuración
# =============================================================================

VALID_MODEL_SIZES = ["n", "s", "m", "l", "x"]
DEFAULT_MODEL_SIZE = "n"
DEFAULT_OUTPUT_DIR = "models"
DEFAULT_MODEL_NAME = "yolov8{size}_epp.pt"

# Información de tamaños de modelo
MODEL_INFO = {
    "n": {"params": "3.2M", "speed": "Más rápido", "accuracy": "Base"},
    "s": {"params": "11.2M", "speed": "Rápido", "accuracy": "Buena"},
    "m": {"params": "25.9M", "speed": "Medio", "accuracy": "Mejor"},
    "l": {"params": "43.7M", "speed": "Lento", "accuracy": "Alta"},
    "x": {"params": "68.2M", "speed": "Más lento", "accuracy": "Máxima"},
}


# =============================================================================
# Logging Setup
# =============================================================================

def setup_logging(verbose: bool = False) -> logging.Logger:
    """
    Configura logging para el script.

    Args:
        verbose: Si True, muestra mensajes DEBUG

    Returns:
        Logger configurado
    """
    logger = logging.getLogger("download_base_model")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    console_format = logging.Formatter(
        "%(levelname)-8s | %(message)s"
    )
    console_handler.setFormatter(console_format)

    logger.addHandler(console_handler)

    return logger


# =============================================================================
# Funciones Auxiliares
# =============================================================================

def validate_model_size(size: str) -> str:
    """
    Valida que el tamaño del modelo sea válido.

    Args:
        size: Tamaño del modelo (n, s, m, l, x)

    Returns:
        Tamaño validado en minúsculas

    Raises:
        ValueError: Si el tamaño no es válido
    """
    size = size.lower()
    if size not in VALID_MODEL_SIZES:
        raise ValueError(
            f"Tamaño de modelo inválido: {size}. "
            f"Opciones válidas: {', '.join(VALID_MODEL_SIZES)}"
        )
    return size


def get_model_path(size: str, output_dir: Path, custom_name: Optional[str] = None) -> Path:
    """
    Genera el path completo para guardar el modelo.

    Args:
        size: Tamaño del modelo (n, s, m, l, x)
        output_dir: Directorio de salida
        custom_name: Nombre personalizado (opcional)

    Returns:
        Path completo para el archivo del modelo
    """
    if custom_name:
        filename = custom_name if custom_name.endswith(".pt") else f"{custom_name}.pt"
    else:
        filename = DEFAULT_MODEL_NAME.format(size=size)

    return output_dir / filename


def print_model_info(size: str, logger: logging.Logger) -> None:
    """
    Imprime información sobre el modelo a descargar.

    Args:
        size: Tamaño del modelo
        logger: Logger instance
    """
    info = MODEL_INFO.get(size, {})

    logger.info("=" * 70)
    logger.info("INFORMACIÓN DEL MODELO")
    logger.info("=" * 70)
    logger.info(f"Modelo: YOLOv8{size}")
    logger.info(f"Parámetros: {info.get('params', 'N/A')}")
    logger.info(f"Velocidad: {info.get('speed', 'N/A')}")
    logger.info(f"Precisión: {info.get('accuracy', 'N/A')}")
    logger.info("=" * 70)


# =============================================================================
# Función Principal de Descarga
# =============================================================================

def download_yolo_model(
    model_size: str,
    output_dir: Path,
    custom_name: Optional[str] = None,
    logger: Optional[logging.Logger] = None
) -> Path:
    """
    Descarga modelo YOLOv8 base desde Ultralytics.

    Esta función descarga automáticamente el modelo desde el repositorio
    oficial de Ultralytics. La primera vez descarga el modelo desde internet,
    las siguientes veces usa la caché local de Ultralytics.

    Args:
        model_size: Tamaño del modelo (n, s, m, l, x)
        output_dir: Directorio donde guardar el modelo
        custom_name: Nombre personalizado para el archivo (opcional)
        logger: Logger instance (opcional)

    Returns:
        Path al modelo descargado

    Raises:
        ValueError: Si el tamaño del modelo es inválido
        RuntimeError: Si la descarga falla
        FileNotFoundError: Si no se puede crear el directorio de salida
    """
    if logger is None:
        logger = setup_logging()

    # Validar tamaño del modelo
    model_size = validate_model_size(model_size)

    # Crear directorio de salida si no existe
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Directorio de salida: {output_dir.absolute()}")
    except Exception as e:
        raise FileNotFoundError(
            f"No se pudo crear el directorio {output_dir}: {e}"
        ) from e

    # Determinar path de salida
    output_path = get_model_path(model_size, output_dir, custom_name)

    # Mostrar información del modelo
    print_model_info(model_size, logger)

    # Descargar modelo
    logger.info(f"Descargando YOLOv8{model_size}...")
    logger.info("Esto puede tomar unos minutos la primera vez...")

    try:
        # Ultralytics YOLO() descarga automáticamente el modelo si no existe
        model_name = f"yolov8{model_size}.pt"
        logger.debug(f"Cargando modelo desde Ultralytics: {model_name}")

        model = YOLO(model_name)

        # Verificar que el modelo se cargó correctamente
        if model.model is None:
            raise RuntimeError("El modelo no se cargó correctamente")

        logger.info(f"Modelo descargado exitosamente")

    except Exception as e:
        logger.error(f"Error al descargar el modelo: {e}")
        raise RuntimeError(f"Descarga del modelo falló: {e}") from e

    # Guardar modelo en la ubicación deseada
    logger.info(f"Guardando modelo en: {output_path}")

    try:
        # Obtener el path del modelo descargado por Ultralytics
        # Ultralytics guarda modelos en ~/.cache/ultralytics/ o similar
        import shutil
        from ultralytics.utils import WEIGHTS_DIR

        source_path = WEIGHTS_DIR / model_name

        if not source_path.exists():
            raise FileNotFoundError(f"Modelo descargado no encontrado en {source_path}")

        # Copiar a la ubicación deseada
        shutil.copy2(source_path, output_path)

        logger.info(f"Modelo guardado exitosamente")

    except Exception as e:
        logger.error(f"Error al guardar el modelo: {e}")
        raise RuntimeError(f"No se pudo guardar el modelo: {e}") from e

    # Verificar que el archivo existe y tiene tamaño > 0
    if not output_path.exists():
        raise FileNotFoundError(f"El modelo no se guardó correctamente en {output_path}")

    file_size_mb = output_path.stat().st_size / (1024 * 1024)

    if file_size_mb < 1:
        raise RuntimeError(f"El modelo descargado parece corrupto (tamaño: {file_size_mb:.2f} MB)")

    logger.info("=" * 70)
    logger.info("DESCARGA COMPLETADA")
    logger.info("=" * 70)
    logger.info(f"Ubicación: {output_path.absolute()}")
    logger.info(f"Tamaño: {file_size_mb:.2f} MB")
    logger.info("=" * 70)
    logger.info("")
    logger.info("Próximos pasos:")
    logger.info("1. Configurar MODEL_PATH en .env:")
    logger.info(f"   MODEL_PATH={output_path}")
    logger.info("2. Iniciar la API:")
    logger.info("   make run")
    logger.info("3. Para fine-tuning con dataset de EPP:")
    logger.info("   python scripts/train_gcp.py --model " + str(output_path))
    logger.info("=" * 70)

    return output_path


# =============================================================================
# Argument Parsing
# =============================================================================

def parse_args() -> argparse.Namespace:
    """
    Parsea argumentos de línea de comandos.

    Returns:
        Namespace con argumentos parseados
    """
    parser = argparse.ArgumentParser(
        description="Descargar modelo base YOLOv8 para EPP Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:

  Descargar modelo nano (más rápido):
    python scripts/download_base_model.py

  Descargar modelo small (recomendado para producción):
    python scripts/download_base_model.py --model-size s

  Descargar modelo medium con nombre personalizado:
    python scripts/download_base_model.py --model-size m --name yolov8m_custom.pt

  Descargar a directorio personalizado:
    python scripts/download_base_model.py --output-dir ./my_models

Tamaños disponibles y características:
  n (nano):   3.2M params  | Más rápido  | Precisión base
  s (small):  11.2M params | Rápido      | Buena precisión  (recomendado)
  m (medium): 25.9M params | Medio       | Mejor precisión
  l (large):  43.7M params | Lento       | Alta precisión
  x (xlarge): 68.2M params | Más lento   | Máxima precisión
        """
    )

    parser.add_argument(
        "--model-size",
        type=str,
        default=DEFAULT_MODEL_SIZE,
        choices=VALID_MODEL_SIZES,
        help=f"Tamaño del modelo YOLOv8 (default: {DEFAULT_MODEL_SIZE})"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directorio donde guardar el modelo (default: {DEFAULT_OUTPUT_DIR})"
    )

    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Nombre personalizado para el archivo (opcional)"
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Mostrar mensajes de debug"
    )

    return parser.parse_args()


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    """
    Función principal del script.

    Returns:
        Código de salida (0 = éxito, 1 = error)
    """
    # Parsear argumentos
    args = parse_args()

    # Setup logging
    logger = setup_logging(verbose=args.verbose)

    logger.info("=" * 70)
    logger.info("DESCARGA DE MODELO BASE YOLOV8")
    logger.info("=" * 70)

    try:
        # Descargar modelo
        output_dir = Path(args.output_dir)

        model_path = download_yolo_model(
            model_size=args.model_size,
            output_dir=output_dir,
            custom_name=args.name,
            logger=logger
        )

        logger.info("Descarga completada exitosamente")
        return 0

    except ValueError as e:
        logger.error(f"Error de validación: {e}")
        return 1
    except FileNotFoundError as e:
        logger.error(f"Error de archivo: {e}")
        return 1
    except RuntimeError as e:
        logger.error(f"Error en ejecución: {e}")
        return 1
    except Exception as e:
        logger.error(f"Error inesperado: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
