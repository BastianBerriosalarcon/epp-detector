"""
Script de recolección y filtrado de imágenes para dataset de EPP.

Este script extrae frames desde videos o filtra imágenes existentes,
aplicando controles de calidad automáticos para construir un dataset
robusto para detección de EPP en minería chilena.

Funcionalidades:
- Extracción de frames desde videos (con sampling inteligente)
- Detección y eliminación de imágenes borrosas
- Detección y eliminación de duplicados (hashing perceptual)
- Filtrado por brillo y contraste
- Organización automática por fecha/ubicación
- Generación de reporte de estadísticas

Uso:
    python scripts/collect_images.py --source data/raw/videos/ --output data/filtered/images/

Autor: Equipo EPP Detector
Fecha: 2025-10-16
"""

import argparse
import cv2
import hashlib
import imagehash
import json
import logging
import os
import shutil
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import yaml
from PIL import Image
from tqdm import tqdm


# Configuración de logging en español
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class ImageQualityFilter:
    """
    Filtro de calidad de imágenes para dataset de EPP.

    Detecta y rechaza imágenes de baja calidad basado en múltiples criterios:
    - Desenfoque (blur)
    - Brillo excesivo o insuficiente
    - Bajo contraste
    - Duplicados
    """

    def __init__(self, config: Dict):
        """
        Inicializa el filtro con configuración.

        Args:
            config: Diccionario con parámetros de filtrado
        """
        self.config = config
        self.seen_hashes: Set[str] = set()
        self.stats = {
            "total_processed": 0,
            "rejected_blur": 0,
            "rejected_brightness": 0,
            "rejected_contrast": 0,
            "rejected_duplicates": 0,
            "rejected_resolution": 0,
            "accepted": 0
        }

    def compute_blur_score(self, image: np.ndarray) -> float:
        """
        Calcula puntaje de desenfoque usando varianza de Laplaciano.

        Mayor puntaje = imagen más nítida.

        Args:
            image: Imagen en formato numpy array (BGR)

        Returns:
            Puntaje de desenfoque (0-infinito, típicamente 0-500)
        """
        # Convertir a escala de grises
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Calcular varianza del Laplaciano
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        score = laplacian.var()

        return score

    def compute_brightness(self, image: np.ndarray) -> float:
        """
        Calcula brillo promedio de la imagen.

        Args:
            image: Imagen en formato numpy array (BGR)

        Returns:
            Brillo promedio (0-255)
        """
        # Convertir a escala de grises
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Brillo promedio
        brightness = np.mean(gray)

        return brightness

    def compute_contrast(self, image: np.ndarray) -> float:
        """
        Calcula contraste de la imagen (desviación estándar).

        Args:
            image: Imagen en formato numpy array (BGR)

        Returns:
            Contraste (desviación estándar de píxeles)
        """
        # Convertir a escala de grises
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Contraste = desviación estándar
        contrast = np.std(gray)

        return contrast

    def compute_perceptual_hash(self, image_path: str) -> str:
        """
        Calcula hash perceptual de la imagen para detección de duplicados.

        Usa average hash (aHash) que es robusto a redimensionamiento
        y cambios menores de compresión.

        Args:
            image_path: Ruta a la imagen

        Returns:
            Hash perceptual como string hexadecimal
        """
        try:
            with Image.open(image_path) as img:
                # Calcular average hash
                hash_size = self.config.get("duplicate_hash_size", 16)
                img_hash = imagehash.average_hash(img, hash_size=hash_size)
                return str(img_hash)
        except Exception as e:
            logger.warning(f"Error calculando hash para {image_path}: {e}")
            return ""

    def is_duplicate(self, image_hash: str) -> bool:
        """
        Verifica si una imagen es duplicada basado en su hash.

        Args:
            image_hash: Hash perceptual de la imagen

        Returns:
            True si es duplicado, False en caso contrario
        """
        if not image_hash:
            return False

        threshold = self.config.get("duplicate_threshold", 5)

        # Comparar con hashes vistos
        for seen_hash in self.seen_hashes:
            # Calcular distancia de Hamming
            distance = bin(int(image_hash, 16) ^ int(seen_hash, 16)).count('1')

            if distance <= threshold:
                return True

        # No es duplicado - agregar a conjunto
        self.seen_hashes.add(image_hash)
        return False

    def check_image_quality(
        self,
        image_path: str,
        image: Optional[np.ndarray] = None
    ) -> Tuple[bool, str]:
        """
        Verifica calidad de imagen contra todos los criterios.

        Args:
            image_path: Ruta a la imagen
            image: Imagen cargada (opcional, se carga si no se proporciona)

        Returns:
            Tupla (es_válida, razón_rechazo)
        """
        self.stats["total_processed"] += 1

        # Cargar imagen si no se proporcionó
        if image is None:
            image = cv2.imread(image_path)
            if image is None:
                return False, "No se pudo cargar la imagen"

        # Verificar resolución
        height, width = image.shape[:2]
        min_width = self.config.get("min_width", 640)
        min_height = self.config.get("min_height", 480)
        max_width = self.config.get("max_width", 4096)
        max_height = self.config.get("max_height", 4096)

        if width < min_width or height < min_height:
            self.stats["rejected_resolution"] += 1
            return False, f"Resolución muy baja ({width}x{height})"

        if width > max_width or height > max_height:
            self.stats["rejected_resolution"] += 1
            return False, f"Resolución muy alta ({width}x{height})"

        # Verificar blur
        if self.config.get("enable_blur_detection", True):
            blur_score = self.compute_blur_score(image)
            min_blur = self.config.get("min_blur_score", 100)

            if blur_score < min_blur:
                self.stats["rejected_blur"] += 1
                return False, f"Imagen borrosa (score: {blur_score:.2f})"

        # Verificar brillo
        if self.config.get("enable_brightness_filter", True):
            brightness = self.compute_brightness(image)
            min_brightness = self.config.get("min_brightness", 30)
            max_brightness = self.config.get("max_brightness", 225)

            if brightness < min_brightness:
                self.stats["rejected_brightness"] += 1
                return False, f"Imagen muy oscura (brillo: {brightness:.1f})"

            if brightness > max_brightness:
                self.stats["rejected_brightness"] += 1
                return False, f"Imagen muy brillante (brillo: {brightness:.1f})"

        # Verificar contraste
        if self.config.get("enable_contrast_filter", True):
            contrast = self.compute_contrast(image)
            min_contrast = self.config.get("min_contrast", 20)

            if contrast < min_contrast:
                self.stats["rejected_contrast"] += 1
                return False, f"Contraste muy bajo (contraste: {contrast:.1f})"

        # Verificar duplicados
        if self.config.get("enable_duplicate_detection", True):
            img_hash = self.compute_perceptual_hash(image_path)

            if self.is_duplicate(img_hash):
                self.stats["rejected_duplicates"] += 1
                return False, "Imagen duplicada"

        # Imagen aceptada
        self.stats["accepted"] += 1
        return True, "OK"

    def get_statistics(self) -> Dict:
        """
        Retorna estadísticas de filtrado.

        Returns:
            Diccionario con estadísticas
        """
        return self.stats.copy()


class VideoFrameExtractor:
    """
    Extractor de frames desde videos para construcción de dataset.

    Implementa sampling inteligente para evitar frames redundantes
    y extraer solo frames con contenido relevante.
    """

    def __init__(self, config: Dict, quality_filter: ImageQualityFilter):
        """
        Inicializa extractor de frames.

        Args:
            config: Configuración de extracción
            quality_filter: Filtro de calidad para frames extraídos
        """
        self.config = config
        self.quality_filter = quality_filter
        self.stats = {
            "videos_processed": 0,
            "frames_extracted": 0,
            "frames_rejected": 0
        }

    def extract_frames(
        self,
        video_path: str,
        output_dir: str,
        progress_bar: Optional[tqdm] = None
    ) -> int:
        """
        Extrae frames desde un video.

        Args:
            video_path: Ruta al archivo de video
            output_dir: Directorio de salida para frames
            progress_bar: Barra de progreso opcional (tqdm)

        Returns:
            Número de frames extraídos exitosamente
        """
        # Abrir video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"No se pudo abrir el video: {video_path}")
            return 0

        # Obtener propiedades del video
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        logger.info(f"Video: {Path(video_path).name} - FPS: {fps}, Total frames: {total_frames}")

        # Calcular intervalo de extracción
        target_fps = self.config.get("fps", 0.33)  # Default: 1 frame cada 3 segundos
        frame_interval = int(fps / target_fps) if target_fps > 0 else 1

        # Variables de control
        frame_count = 0
        extracted_count = 0
        prev_frame = None

        # Nombre base para frames
        video_name = Path(video_path).stem

        # Extraer frames
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Verificar si debemos extraer este frame
            if frame_count % frame_interval == 0:
                # Verificar cambio de escena (si está habilitado)
                if self.config.get("skip_similar", True) and prev_frame is not None:
                    if not self._is_scene_change(prev_frame, frame):
                        frame_count += 1
                        continue

                # Guardar frame temporalmente para verificar calidad
                timestamp = frame_count / fps
                frame_filename = f"{video_name}_t{timestamp:.2f}_f{frame_count:06d}.jpg"
                frame_path = os.path.join(output_dir, frame_filename)

                # Guardar frame
                cv2.imwrite(frame_path, frame)

                # Verificar calidad
                is_valid, reason = self.quality_filter.check_image_quality(frame_path, frame)

                if is_valid:
                    extracted_count += 1
                    prev_frame = frame.copy()

                    if progress_bar:
                        progress_bar.set_postfix({"Extraídos": extracted_count})
                else:
                    # Eliminar frame rechazado
                    os.remove(frame_path)
                    self.stats["frames_rejected"] += 1
                    logger.debug(f"Frame rechazado: {frame_filename} - {reason}")

            frame_count += 1

            if progress_bar:
                progress_bar.update(1)

        # Liberar recursos
        cap.release()

        self.stats["videos_processed"] += 1
        self.stats["frames_extracted"] += extracted_count

        logger.info(f"Extraídos {extracted_count} frames de {video_path}")

        return extracted_count

    def _is_scene_change(self, frame1: np.ndarray, frame2: np.ndarray) -> bool:
        """
        Detecta si hay cambio significativo entre dos frames.

        Usa diferencia de histogramas para detectar cambios de escena.

        Args:
            frame1: Frame anterior
            frame2: Frame actual

        Returns:
            True si hay cambio de escena significativo
        """
        # Convertir a escala de grises
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        # Calcular histogramas
        hist1 = cv2.calcHist([gray1], [0], None, [256], [0, 256])
        hist2 = cv2.calcHist([gray2], [0], None, [256], [0, 256])

        # Normalizar
        hist1 = cv2.normalize(hist1, hist1).flatten()
        hist2 = cv2.normalize(hist2, hist2).flatten()

        # Comparar histogramas (correlación)
        correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

        # Umbral de similitud
        similarity_threshold = self.config.get("similarity_threshold", 0.95)

        # Si correlación es alta, frames son similares (no hay cambio de escena)
        return correlation < similarity_threshold

    def get_statistics(self) -> Dict:
        """
        Retorna estadísticas de extracción.

        Returns:
            Diccionario con estadísticas
        """
        return self.stats.copy()


class ImageCollector:
    """
    Recolector y organizador de imágenes para dataset de EPP.

    Coordina la extracción desde videos, filtrado de calidad,
    y organización de archivos en estructura de directorios.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Inicializa recolector de imágenes.

        Args:
            config_path: Ruta a archivo de configuración YAML (opcional)
        """
        # Cargar configuración
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                full_config = yaml.safe_load(f)
                self.config = full_config.get("collection", {})
        else:
            # Configuración por defecto
            self.config = {
                "video_extraction": {"fps": 0.33, "skip_similar": True},
                "quality_filters": {
                    "enable_blur_detection": True,
                    "min_blur_score": 100,
                    "enable_duplicate_detection": True
                }
            }

        # Inicializar componentes
        quality_config = {**self.config.get("quality_filters", {}),
                         **self.config.get("resolution", {})}
        self.quality_filter = ImageQualityFilter(quality_config)

        extraction_config = self.config.get("video_extraction", {})
        self.frame_extractor = VideoFrameExtractor(extraction_config, self.quality_filter)

    def collect_from_videos(
        self,
        source_dir: str,
        output_dir: str,
        organize_by_date: bool = True
    ) -> None:
        """
        Recolecta imágenes desde videos en directorio de origen.

        Args:
            source_dir: Directorio con videos
            output_dir: Directorio de salida para imágenes
            organize_by_date: Organizar en subdirectorios por fecha
        """
        # Crear directorio de salida
        os.makedirs(output_dir, exist_ok=True)

        # Buscar videos
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv']
        video_files = []

        for ext in video_extensions:
            video_files.extend(Path(source_dir).rglob(f'*{ext}'))

        if not video_files:
            logger.warning(f"No se encontraron videos en {source_dir}")
            return

        logger.info(f"Encontrados {len(video_files)} videos para procesar")

        # Procesar cada video
        for video_path in tqdm(video_files, desc="Procesando videos"):
            # Determinar directorio de salida
            if organize_by_date:
                date_str = datetime.now().strftime("%Y-%m-%d")
                video_output_dir = os.path.join(output_dir, date_str)
            else:
                video_output_dir = output_dir

            os.makedirs(video_output_dir, exist_ok=True)

            # Extraer frames
            self.frame_extractor.extract_frames(
                str(video_path),
                video_output_dir
            )

        # Generar reporte
        self._generate_report(output_dir)

    def collect_from_images(
        self,
        source_dir: str,
        output_dir: str,
        copy_mode: bool = True
    ) -> None:
        """
        Filtra imágenes existentes y las copia/mueve al directorio de salida.

        Args:
            source_dir: Directorio con imágenes existentes
            output_dir: Directorio de salida
            copy_mode: True para copiar, False para mover
        """
        # Crear directorio de salida
        os.makedirs(output_dir, exist_ok=True)

        # Buscar imágenes
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []

        for ext in image_extensions:
            image_files.extend(Path(source_dir).rglob(f'*{ext}'))

        if not image_files:
            logger.warning(f"No se encontraron imágenes en {source_dir}")
            return

        logger.info(f"Encontradas {len(image_files)} imágenes para procesar")

        # Procesar cada imagen
        for image_path in tqdm(image_files, desc="Filtrando imágenes"):
            # Verificar calidad
            is_valid, reason = self.quality_filter.check_image_quality(str(image_path))

            if is_valid:
                # Copiar o mover imagen
                dest_path = os.path.join(output_dir, image_path.name)

                if copy_mode:
                    shutil.copy2(str(image_path), dest_path)
                else:
                    shutil.move(str(image_path), dest_path)
            else:
                logger.debug(f"Imagen rechazada: {image_path.name} - {reason}")

        # Generar reporte
        self._generate_report(output_dir)

    def _generate_report(self, output_dir: str) -> None:
        """
        Genera reporte de estadísticas de recolección.

        Args:
            output_dir: Directorio donde guardar el reporte
        """
        # Combinar estadísticas
        stats = {
            "fecha_recoleccion": datetime.now().isoformat(),
            "filtrado_calidad": self.quality_filter.get_statistics(),
            "extraccion_video": self.frame_extractor.get_statistics()
        }

        # Guardar como JSON
        report_path = os.path.join(output_dir, "reporte_recoleccion.json")
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)

        logger.info(f"Reporte generado: {report_path}")

        # Imprimir resumen
        logger.info("=" * 60)
        logger.info("RESUMEN DE RECOLECCIÓN")
        logger.info("=" * 60)

        quality_stats = stats["filtrado_calidad"]
        logger.info(f"Total procesadas:     {quality_stats['total_processed']}")
        logger.info(f"Imágenes aceptadas:   {quality_stats['accepted']}")
        logger.info(f"Rechazadas (blur):    {quality_stats['rejected_blur']}")
        logger.info(f"Rechazadas (brillo):  {quality_stats['rejected_brightness']}")
        logger.info(f"Rechazadas (contraste): {quality_stats['rejected_contrast']}")
        logger.info(f"Rechazadas (duplicados): {quality_stats['rejected_duplicates']}")
        logger.info(f"Rechazadas (resolución): {quality_stats['rejected_resolution']}")

        if stats["extraccion_video"]["videos_processed"] > 0:
            video_stats = stats["extraccion_video"]
            logger.info(f"\nVideos procesados:    {video_stats['videos_processed']}")
            logger.info(f"Frames extraídos:     {video_stats['frames_extracted']}")

        logger.info("=" * 60)


def main():
    """Función principal del script."""
    parser = argparse.ArgumentParser(
        description="Recolección y filtrado de imágenes para dataset de EPP en minería chilena",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:

  # Extraer frames desde videos
  python scripts/collect_images.py --source data/raw/videos/ --output data/filtered/images/ --mode video

  # Filtrar imágenes existentes
  python scripts/collect_images.py --source data/raw/images/ --output data/filtered/images/ --mode images

  # Usar configuración personalizada
  python scripts/collect_images.py --source data/raw/ --output data/filtered/ --config configs/dataset_config.yaml

  # Ajustar parámetros de calidad
  python scripts/collect_images.py --source data/raw/ --output data/filtered/ --min-blur 150 --fps 0.5
        """
    )

    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Directorio de origen con videos o imágenes"
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Directorio de salida para imágenes filtradas"
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["video", "images", "auto"],
        default="auto",
        help="Modo de recolección: video (extraer frames), images (filtrar imágenes), auto (detectar automáticamente)"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/dataset_config.yaml",
        help="Ruta a archivo de configuración YAML"
    )

    parser.add_argument(
        "--fps",
        type=float,
        help="Frames por segundo a extraer de videos (ej: 0.33 = 1 frame cada 3 seg)"
    )

    parser.add_argument(
        "--min-blur",
        type=float,
        help="Umbral mínimo de blur score (mayor = más estricto)"
    )

    parser.add_argument(
        "--organize-by-date",
        action="store_true",
        help="Organizar imágenes en subdirectorios por fecha"
    )

    parser.add_argument(
        "--no-duplicates",
        action="store_true",
        help="Habilitar detección y eliminación de duplicados"
    )

    parser.add_argument(
        "--move",
        action="store_true",
        help="Mover archivos en lugar de copiar (modo images)"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Mostrar logs detallados (DEBUG level)"
    )

    args = parser.parse_args()

    # Configurar nivel de logging
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # Inicializar recolector
    collector = ImageCollector(args.config if os.path.exists(args.config) else None)

    # Aplicar overrides de CLI
    if args.fps is not None:
        collector.config["video_extraction"]["fps"] = args.fps

    if args.min_blur is not None:
        collector.config["quality_filters"]["min_blur_score"] = args.min_blur

    if args.no_duplicates:
        collector.config["quality_filters"]["enable_duplicate_detection"] = True

    # Determinar modo
    mode = args.mode
    if mode == "auto":
        # Detectar automáticamente basado en contenido del directorio
        source_path = Path(args.source)
        has_videos = any(source_path.rglob("*.mp4")) or any(source_path.rglob("*.avi"))
        has_images = any(source_path.rglob("*.jpg")) or any(source_path.rglob("*.png"))

        if has_videos:
            mode = "video"
        elif has_images:
            mode = "images"
        else:
            logger.error("No se encontraron videos ni imágenes en el directorio de origen")
            sys.exit(1)

    logger.info(f"Modo de recolección: {mode}")

    # Ejecutar recolección
    try:
        if mode == "video":
            collector.collect_from_videos(
                args.source,
                args.output,
                organize_by_date=args.organize_by_date
            )
        else:
            collector.collect_from_images(
                args.source,
                args.output,
                copy_mode=not args.move
            )

        logger.info("Recolección completada exitosamente")

    except Exception as e:
        logger.error(f"Error durante la recolección: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
