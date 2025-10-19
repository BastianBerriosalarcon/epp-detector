"""
Aplicación FastAPI principal para detección de EPP.

Este módulo define todos los endpoints REST y maneja el ciclo de vida
de la aplicación, incluyendo carga del modelo y tracking de métricas.
"""

import io
import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel, Field

from api import EPP_CLASSES, EPP_CLASSES_ES, __version__
from api.config import Settings, get_settings
from api.exceptions import (
    EPPDetectorError,
    InferenceError,
    InvalidImageError,
    ModelNotLoadedError,
)
from api.middleware import RateLimitMiddleware, RequestLoggingMiddleware
from api.model import EPPDetector
from api.validators import ImageValidator, create_validator

# ============================================================================
# Logging Setup
# ============================================================================

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ============================================================================
# Application State
# ============================================================================


class AppState:
    """Container for application state.

    Centralizes global state management following DI principles.
    """

    def __init__(self) -> None:
        """Initialize application state."""
        self.detector: Optional[EPPDetector] = None
        self.validator: Optional[ImageValidator] = None
        self.request_count: int = 0
        self.total_inference_time: float = 0.0
        self.startup_time: datetime = datetime.now()


# Global state instance
app_state = AppState()

# ============================================================================
# FastAPI Application
# ============================================================================

settings = get_settings()

app = FastAPI(
    title="EPP Detector API - Minería Chile",
    description=(
        "API REST para detección automática de Equipos de Protección Personal "
        "en faenas mineras chilenas. Utiliza YOLOv8 optimizado para condiciones "
        "de minería subterránea y rajo abierto."
    ),
    version=__version__,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ============================================================================
# Middleware Configuration
# ============================================================================

# CORS middleware
if settings.enable_cors:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    logger.info(f"CORS enabled for origins: {settings.cors_origins}")

# Rate limiting middleware
if settings.enable_rate_limit:
    app.add_middleware(RateLimitMiddleware, settings=settings)
    logger.info("Rate limiting middleware enabled")

# Request logging middleware
app.add_middleware(RequestLoggingMiddleware)

# ============================================================================
# Pydantic Models
# ============================================================================


class Detection(BaseModel):
    """Modelo para una detección individual de EPP."""

    class_id: int = Field(..., description="ID de la clase detectada")
    class_name: str = Field(..., description="Nombre técnico de la clase (ej: 'hardhat')")
    class_name_es: str = Field(..., description="Nombre en español (ej: 'Casco de seguridad')")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confianza del modelo")
    bbox: List[float] = Field(..., description="Bounding box [x_min, y_min, x_max, y_max]")


class PredictionResponse(BaseModel):
    """Respuesta del endpoint /predict."""

    success: bool = Field(..., description="Indica si la inferencia fue exitosa")
    detections: List[Detection] = Field(..., description="Lista de detecciones de EPP")
    inference_time_ms: float = Field(..., description="Tiempo de inferencia en ms")
    image_size: Dict[str, int] = Field(..., description="Dimensiones de la imagen procesada")
    total_detections: int = Field(..., description="Total de objetos detectados")
    epp_compliant: bool = Field(
        ..., description="True si se detectó EPP completo (casco + chaleco)"
    )


class HealthResponse(BaseModel):
    """Respuesta del endpoint /health."""

    status: str = Field(..., description="Estado del servicio")
    version: str = Field(..., description="Versión de la API")
    model_loaded: bool = Field(..., description="Indica si el modelo está cargado")
    timestamp: str = Field(..., description="Timestamp del health check")


class MetricsResponse(BaseModel):
    """Respuesta del endpoint /metrics."""

    requests_total: int = Field(..., description="Total de requests procesados")
    avg_latency_ms: float = Field(..., description="Latencia promedio en ms")
    uptime_seconds: float = Field(..., description="Tiempo activo del servicio")
    model_loaded: bool = Field(..., description="Estado del modelo")


class ModelInfo(BaseModel):
    """Respuesta del endpoint /info."""

    model_version: str = Field(..., description="Versión del modelo YOLOv8")
    model_type: str = Field(..., description="Tipo de modelo (ej: YOLOv8n)")
    classes: Dict[int, str] = Field(..., description="Clases detectables")
    confidence_threshold: float = Field(..., description="Umbral de confianza configurado")
    dataset_info: str = Field(..., description="Información del dataset de entrenamiento")


# ============================================================================
# Dependency Injection Functions
# ============================================================================


def get_detector() -> EPPDetector:
    """Dependency injection for detector instance.

    Returns:
        Loaded EPPDetector instance

    Raises:
        HTTPException: If detector is not loaded
    """
    if app_state.detector is None or not app_state.detector.is_loaded:
        logger.error("Detector not loaded when requested")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modelo no disponible. El servicio está iniciando o hay un error.",
        )
    return app_state.detector


def get_validator(settings: Settings = Depends(get_settings)) -> ImageValidator:
    """Dependency injection for image validator.

    Args:
        settings: Configuration settings

    Returns:
        Configured ImageValidator instance
    """
    if app_state.validator is None:
        app_state.validator = create_validator(settings)
    return app_state.validator


# ============================================================================
# Lifecycle Events
# ============================================================================


@app.on_event("startup")
async def startup_event() -> None:
    """
    Inicializa el modelo al arrancar la aplicación.

    Loads detector with dependency injection and proper error handling.
    Performs warmup if enabled in settings.
    """
    logger.info("=" * 60)
    logger.info("EPP DETECTOR API STARTUP")
    logger.info("=" * 60)
    logger.info(f"Version: {__version__}")
    logger.info(f"Environment: {settings.environment}")
    logger.info(f"Debug mode: {settings.debug}")

    app_state.startup_time = datetime.now()

    # Initialize validator
    try:
        app_state.validator = create_validator(settings)
        logger.info("Image validator initialized")
    except Exception as e:
        logger.error(f"Failed to initialize validator: {e}")

    # Load model
    try:
        logger.info(f"Loading model from: {settings.model_path}")

        app_state.detector = EPPDetector(settings=settings, auto_warmup=settings.enable_warmup)

        logger.info(
            f"Model loaded successfully: {app_state.detector.model_type} "
            f"({settings.model_path})"
        )

    except Exception as e:
        logger.error(f"Failed to load model during startup: {e}", exc_info=True)
        logger.warning("API will start in degraded mode without model")
        app_state.detector = None

    logger.info("=" * 60)
    logger.info("STARTUP COMPLETED")
    logger.info("=" * 60)


@app.on_event("shutdown")
async def shutdown_event() -> None:
    """
    Limpia recursos al apagar la aplicación.

    Properly cleans up model resources and GPU memory.
    """
    logger.info("Shutting down API...")

    if app_state.detector is not None:
        try:
            app_state.detector.cleanup()
            logger.info("Model resources cleaned up successfully")
        except Exception as e:
            logger.error(f"Error during model cleanup: {e}")

    logger.info("Shutdown complete")


# ============================================================================
# Exception Handlers
# ============================================================================


@app.exception_handler(EPPDetectorError)
async def epp_detector_error_handler(request, exc: EPPDetectorError) -> JSONResponse:
    """Handler for custom EPP detector exceptions.

    Args:
        request: HTTP request
        exc: EPPDetectorError instance

    Returns:
        JSON response with error details
    """
    # Map exception types to HTTP status codes
    status_code_map = {
        ModelNotLoadedError: status.HTTP_503_SERVICE_UNAVAILABLE,
        InvalidImageError: status.HTTP_400_BAD_REQUEST,
        InferenceError: status.HTTP_500_INTERNAL_SERVER_ERROR,
    }

    status_code = status_code_map.get(type(exc), status.HTTP_500_INTERNAL_SERVER_ERROR)

    logger.error(
        f"EPP Detector Error: {exc.message}",
        extra={"details": exc.details, "status_code": status_code},
    )

    return JSONResponse(
        status_code=status_code,
        content={
            "success": False,
            "error": exc.message,
            "details": exc.details,
            "timestamp": datetime.now().isoformat(),
        },
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException) -> JSONResponse:
    """Handler for HTTP exceptions.

    Args:
        request: HTTP request
        exc: HTTPException instance

    Returns:
        JSON response with error details
    """
    logger.warning(f"HTTP Exception: {exc.status_code} - {exc.detail}")

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": exc.detail,
            "timestamp": datetime.now().isoformat(),
        },
    )


# ============================================================================
# Endpoints
# ============================================================================


@app.get("/", tags=["Root"])
async def root() -> Dict[str, Any]:
    """
    Endpoint raíz - información básica de la API.

    Returns:
        Información básica de la API y enlaces a docs
    """
    return {
        "message": "EPP Detector API - Minería Chile",
        "version": __version__,
        "docs": "/docs",
        "health": "/health",
        "github": "https://github.com/tu-usuario/epp-detector",
    }


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check() -> HealthResponse:
    """
    Health check para Kubernetes liveness/readiness probes.

    Verifica que la API esté respondiendo y que el modelo esté cargado.
    Este endpoint debe responder rápido (<50ms) para no afectar orchestration.

    Returns:
        HealthResponse: Estado del servicio y modelo
    """
    model_loaded = app_state.detector is not None and app_state.detector.is_loaded

    return HealthResponse(
        status="healthy" if model_loaded else "degraded",
        version=__version__,
        model_loaded=model_loaded,
        timestamp=datetime.now().isoformat(),
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Detection"])
async def predict_epp(
    file: UploadFile = File(...),
    detector: EPPDetector = Depends(get_detector),
    validator: ImageValidator = Depends(get_validator),
) -> PredictionResponse:
    """
    Detecta EPP en una imagen cargada.

    Proceso:
    1. Valida formato y tamaño de imagen
    2. Preprocesa imagen (resize, normalización)
    3. Ejecuta inferencia con YOLOv8
    4. Postprocesa detecciones (NMS, filtrado por confianza)
    5. Evalúa cumplimiento de EPP

    Args:
        file: Imagen en formato JPG/PNG (max 10MB)
        detector: Injected detector instance
        validator: Injected validator instance

    Returns:
        PredictionResponse: Detecciones con bounding boxes y confianza

    Raises:
        HTTPException 400: Formato de imagen inválido
        HTTPException 500: Error en inferencia
        HTTPException 503: Modelo no disponible
    """
    # Read image bytes
    try:
        image_bytes = await file.read()
    except Exception as e:
        logger.error(f"Failed to read uploaded file: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=f"Error leyendo imagen: {str(e)}"
        )

    # Validate image
    try:
        width, height = validator.validate_all(image_bytes, file.filename)
        logger.debug(f"Image validation passed: {width}x{height}")
    except InvalidImageError as e:
        # Exception handler will catch this
        raise

    # Convert to PIL Image for inference
    try:
        image = Image.open(io.BytesIO(image_bytes))
    except Exception as e:
        logger.error(f"Failed to open image: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="No se pudo procesar la imagen"
        )

    # Run inference
    start_time = time.time()

    try:
        detections_raw = detector.predict(image)

        # Format detections with Spanish translations
        detections = [
            Detection(
                class_id=det["class_id"],
                class_name=det["class_name"],
                class_name_es=EPP_CLASSES_ES.get(det["class_name"], det["class_name"]),
                confidence=det["confidence"],
                bbox=det["bbox"],
            )
            for det in detections_raw
        ]

    except InferenceError as e:
        # Exception handler will catch this
        raise

    inference_time = (time.time() - start_time) * 1000  # ms

    # Update metrics
    app_state.request_count += 1
    app_state.total_inference_time += inference_time

    # Determine EPP compliance
    detected_classes = {det.class_name for det in detections}
    epp_compliant = "hardhat" in detected_classes and "head" not in detected_classes

    logger.info(
        f"Prediction completed: {len(detections)} detections in {inference_time:.2f}ms "
        f"(compliant: {epp_compliant})"
    )

    return PredictionResponse(
        success=True,
        detections=detections,
        inference_time_ms=round(inference_time, 2),
        image_size={"width": width, "height": height},
        total_detections=len(detections),
        epp_compliant=epp_compliant,
    )


@app.get("/metrics", response_model=MetricsResponse, tags=["System"])
async def get_metrics() -> MetricsResponse:
    """
    Retorna métricas de performance del sistema.

    Útil para monitoring con Prometheus/Grafana y análisis de latencia.

    Returns:
        MetricsResponse: Estadísticas de uso y performance
    """
    uptime = (datetime.now() - app_state.startup_time).total_seconds()
    avg_latency = (
        app_state.total_inference_time / app_state.request_count
        if app_state.request_count > 0
        else 0.0
    )

    return MetricsResponse(
        requests_total=app_state.request_count,
        avg_latency_ms=round(avg_latency, 2),
        uptime_seconds=round(uptime, 2),
        model_loaded=app_state.detector is not None and app_state.detector.is_loaded,
    )


@app.get("/info", response_model=ModelInfo, tags=["System"])
async def get_model_info(
    detector: Optional[EPPDetector] = Depends(lambda: app_state.detector),
    settings: Settings = Depends(get_settings),
) -> ModelInfo:
    """
    Retorna información sobre el modelo y configuración.

    Útil para debugging, auditoría y documentación automática.

    Args:
        detector: Optional detector instance
        settings: Configuration settings

    Returns:
        ModelInfo: Metadata del modelo y configuración
    """
    if detector is not None and detector.is_loaded:
        model_info = detector.get_model_info()
        model_version = model_info.get("version", "unknown")
        model_type = model_info.get("model_type", "unknown")
    else:
        model_version = "not-loaded"
        model_type = "unknown"

    return ModelInfo(
        model_version=model_version,
        model_type=model_type,
        classes=EPP_CLASSES,
        confidence_threshold=settings.confidence_threshold,
        dataset_info="Roboflow Hard Hat Workers (5k+ images) + Chilean Mining Dataset",
    )
