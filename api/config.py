"""
Configuración centralizada para la API de detección de EPP.

Este módulo centraliza todas las configuraciones del sistema,
permitiendo fácil ajuste de parámetros sin modificar código.
"""

from pathlib import Path
from typing import List, Union

try:
    from pydantic import Field, field_validator
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseSettings, Field
    from pydantic import validator as field_validator


class Settings(BaseSettings):
    """
    Configuración de la API cargada desde variables de entorno.

    Permite override de configuraciones via .env o variables de entorno,
    siguiendo el patrón 12-factor app.

    Attributes:
        app_name: Nombre de la aplicación
        app_version: Versión de la API
        debug: Modo debug (más logs, auto-reload)

        model_path: Ruta al modelo YOLOv8/ONNX
        model_type: Tipo de modelo ('pytorch' o 'onnx')
        input_size: Tamaño de entrada del modelo

        confidence_threshold: Umbral mínimo de confianza
        iou_threshold: Umbral IoU para NMS
        max_detections: Número máximo de detecciones por imagen

        max_image_size_mb: Tamaño máximo de imagen en MB
        allowed_formats: Formatos de imagen permitidos

        enable_gpu: Usar GPU si está disponible
        num_workers: Workers para inferencia paralela

    Example:
        >>> settings = Settings()
        >>> print(settings.model_path)
        'models/yolov8n_epp.onnx'

    TODO: Agregar validación de paths (verificar que modelo existe)
    TODO: Agregar configuración de logging (level, formato)
    TODO: Agregar configuración de CORS para frontend
    TODO: Agregar configuración de rate limiting
    TODO: Agregar configuración de autenticación (API keys, JWT)
    """

    # ========================================================================
    # Configuración de la aplicación
    # ========================================================================

    app_name: str = Field(
        default="EPP Detector API",
        description="Nombre de la aplicación",
    )

    app_version: str = Field(
        default="0.1.0",
        description="Versión de la API",
    )

    debug: bool = Field(
        default=False,
        env="DEBUG",
        description="Modo debug (más logs, auto-reload)",
    )

    environment: str = Field(
        default="development",
        env="ENVIRONMENT",
        description="Entorno de ejecución (development, staging, production)",
    )

    # ========================================================================
    # Configuración del modelo
    # ========================================================================

    model_path: str = Field(
        default="models/yolov8n_epp.onnx",
        env="MODEL_PATH",
        description="Ruta al archivo del modelo",
    )

    model_type: str = Field(
        default="onnx",
        env="MODEL_TYPE",
        description="Tipo de modelo (pytorch o onnx)",
    )

    input_size: int = Field(
        default=640,
        env="INPUT_SIZE",
        description="Tamaño de entrada del modelo (640 para YOLOv8)",
    )

    # ========================================================================
    # Configuración de inferencia
    # ========================================================================

    confidence_threshold: float = Field(
        default=0.5,
        env="CONFIDENCE_THRESHOLD",
        ge=0.0,
        le=1.0,
        description="Umbral mínimo de confianza para detecciones",
    )

    iou_threshold: float = Field(
        default=0.45,
        env="IOU_THRESHOLD",
        ge=0.0,
        le=1.0,
        description="Umbral IoU para Non-Maximum Suppression",
    )

    max_detections: int = Field(
        default=100,
        env="MAX_DETECTIONS",
        description="Número máximo de detecciones por imagen",
    )

    enable_warmup: bool = Field(
        default=True,
        env="ENABLE_WARMUP",
        description="Ejecutar warmup del modelo en startup",
    )

    warmup_iterations: int = Field(
        default=5,
        env="WARMUP_ITERATIONS",
        description="Número de iteraciones de warmup",
    )

    # ========================================================================
    # Configuración de imágenes
    # ========================================================================

    max_image_size_mb: int = Field(
        default=10,
        env="MAX_IMAGE_SIZE_MB",
        description="Tamaño máximo de imagen en MB",
    )

    allowed_formats: List[str] = Field(
        default=[".jpg", ".jpeg", ".png", ".bmp"],
        description="Formatos de imagen permitidos",
    )

    min_image_dimension: int = Field(
        default=320,
        env="MIN_IMAGE_DIMENSION",
        description="Dimensión mínima permitida (ancho o alto)",
    )

    max_image_dimension: int = Field(
        default=4096,
        env="MAX_IMAGE_DIMENSION",
        description="Dimensión máxima permitida (ancho o alto)",
    )

    # ========================================================================
    # Configuración de hardware
    # ========================================================================

    enable_gpu: bool = Field(
        default=True,
        env="ENABLE_GPU",
        description="Usar GPU si está disponible (CUDA)",
    )

    num_workers: int = Field(
        default=1,
        env="NUM_WORKERS",
        description="Workers para inferencia paralela",
    )

    # ========================================================================
    # Configuración de logging
    # ========================================================================

    log_level: str = Field(
        default="INFO",
        env="LOG_LEVEL",
        description="Nivel de logging (DEBUG, INFO, WARNING, ERROR)",
    )

    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Formato de logs",
    )

    # ========================================================================
    # Configuración de GCP (para deployment)
    # ========================================================================

    gcp_project_id: str = Field(
        default="",
        env="GCP_PROJECT_ID",
        description="ID del proyecto de Google Cloud",
    )

    gcp_bucket_name: str = Field(
        default="",
        env="GCP_BUCKET_NAME",
        description="Nombre del bucket de GCS para modelos",
    )

    # ========================================================================
    # Configuración de seguridad
    # ========================================================================

    enable_cors: bool = Field(
        default=True,
        env="ENABLE_CORS",
        description="Habilitar CORS para frontend",
    )

    cors_origins: Union[str, List[str]] = Field(
        default="*",
        description="Orígenes permitidos para CORS (string o lista)",
    )

    @field_validator("cors_origins", mode="after")
    @classmethod
    def parse_cors_origins(cls, v):
        """Parse CORS origins to list format for middleware.

        Handles formats from environment variables:
        - "*" -> ["*"]
        - "http://localhost:3000" -> ["http://localhost:3000"]
        - "http://localhost:3000,http://localhost:8080" ->
          ["http://localhost:3000", "http://localhost:8080"]
        """
        if isinstance(v, list):
            return v
        if isinstance(v, str):
            if v == "*" or v.strip() == "*":
                return ["*"]
            if "," in v:
                return [origin.strip() for origin in v.split(",") if origin.strip()]
            return [v.strip()] if v.strip() else ["*"]
        return ["*"]  # Fallback default

    enable_api_key: bool = Field(
        default=False,
        env="ENABLE_API_KEY",
        description="Requerir API key para endpoints",
    )

    api_key: str = Field(
        default="",
        env="API_KEY",
        description="API key para autenticación",
    )

    # ========================================================================
    # Configuración de rate limiting
    # ========================================================================

    enable_rate_limit: bool = Field(
        default=False,
        env="ENABLE_RATE_LIMIT",
        description="Habilitar rate limiting",
    )

    rate_limit_requests: int = Field(
        default=100,
        env="RATE_LIMIT_REQUESTS",
        description="Número de requests permitidos",
    )

    rate_limit_window_seconds: int = Field(
        default=60,
        env="RATE_LIMIT_WINDOW_SECONDS",
        description="Ventana de tiempo para rate limiting",
    )

    class Config:
        """Configuración de Pydantic BaseSettings."""

        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        # Allow extra fields in .env that aren't in the model (for flexibility)
        extra = "ignore"

    def get_model_path_absolute(self) -> Path:
        """
        Retorna path absoluto del modelo.

        TODO: Implementar descarga automática desde GCS si no existe
        """
        path = Path(self.model_path)
        if not path.is_absolute():
            # Asumir que es relativo al directorio raíz del proyecto
            project_root = Path(__file__).parent.parent
            path = project_root / path
        return path

    def validate_model_exists(self) -> bool:
        """
        Verifica que el modelo existe en el path configurado.

        Returns:
            True si el modelo existe, False en caso contrario

        TODO: Implementar descarga automática si no existe
        """
        return self.get_model_path_absolute().exists()

    def get_device(self) -> str:
        """
        Determina el device a usar (cuda, cpu).

        Returns:
            'cuda' si GPU está disponible y habilitada, 'cpu' en caso contrario

        TODO: Agregar soporte para MPS (Apple Silicon)
        TODO: Verificar disponibilidad real de CUDA
        """
        if self.enable_gpu:
            # TODO: Verificar con torch.cuda.is_available()
            return "cuda"
        return "cpu"


# ============================================================================
# Instancia global de settings
# ============================================================================

# Singleton de configuración cargado al importar el módulo
settings = Settings()


# ============================================================================
# Helper functions
# ============================================================================


def get_settings() -> Settings:
    """
    Retorna la instancia global de settings.

    Útil para dependency injection en FastAPI.

    Example:
        ```python
        from fastapi import Depends
        from api.config import get_settings, Settings

        @app.get("/")
        def root(settings: Settings = Depends(get_settings)):
            return {"version": settings.app_version}
        ```
    """
    return settings


def print_settings():
    """
    Imprime configuración actual (útil para debugging).

    TODO: Ocultar valores sensibles (API keys, secrets)
    """
    print("=" * 60)
    print("CONFIGURACIÓN DE LA API")
    print("=" * 60)
    print(f"App Name:           {settings.app_name}")
    print(f"Version:            {settings.app_version}")
    print(f"Environment:        {settings.environment}")
    print(f"Debug:              {settings.debug}")
    print("-" * 60)
    print(f"Model Path:         {settings.model_path}")
    print(f"Model Type:         {settings.model_type}")
    print(f"Input Size:         {settings.input_size}")
    print(f"Confidence Thresh:  {settings.confidence_threshold}")
    print(f"IoU Threshold:      {settings.iou_threshold}")
    print("-" * 60)
    print(f"Enable GPU:         {settings.enable_gpu}")
    print(f"Device:             {settings.get_device()}")
    print(f"Workers:            {settings.num_workers}")
    print("-" * 60)
    print(f"Max Image Size:     {settings.max_image_size_mb} MB")
    print(f"Allowed Formats:    {', '.join(settings.allowed_formats)}")
    print("=" * 60)


if __name__ == "__main__":
    # Para testing: python -m api.config
    print_settings()
