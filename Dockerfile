# =============================================================================
# Multi-stage Dockerfile para EPP Detector API
# Optimizado para producción con imagen final <500MB
# =============================================================================

# =============================================================================
# Stage 1: Builder - Compilar dependencias y crear wheels
# =============================================================================
FROM python:3.10-slim AS builder

LABEL maintainer="Bastián Berríos"
LABEL stage="builder"
LABEL description="Build stage para compilar dependencias Python"

# Variables de entorno para optimización de build
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Instalar dependencias del sistema necesarias para compilar
# gcc, g++: Para compilar extensiones C/C++ (numpy, opencv, etc.)
# libgomp1: OpenMP para operaciones paralelas
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Crear directorio de trabajo
WORKDIR /build

# Copiar solo requirements para aprovechar layer caching
# Si requirements no cambia, Docker reutiliza esta layer
COPY requirements.txt .

# Crear wheels de todas las dependencias
# Esto precompila todo y acelera instalación en stage runtime
# --wheel-dir: Directorio donde guardar wheels
# --no-deps: No instalar dependencias de dependencias (las resolvemos después)
RUN pip install --upgrade pip setuptools wheel && \
    pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt

# TODO: Si modelo ONNX necesita optimización, hacerlo aquí
# TODO: Compilar extensiones custom si las hubiera

# =============================================================================
# Stage 2: Runtime - Imagen final optimizada
# =============================================================================
FROM python:3.10-slim AS runtime

LABEL maintainer="Bastián Berríos"
LABEL version="0.1.0"
LABEL description="API REST para detección de EPP en minería chilena"
LABEL org.opencontainers.image.source="https://github.com/tu-usuario/epp-detector"

# Metadata para identificación en producción
ARG BUILD_DATE
ARG VCS_REF
LABEL org.opencontainers.image.created="${BUILD_DATE}"
LABEL org.opencontainers.image.revision="${VCS_REF}"

# Variables de entorno
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # Configuración de la API
    API_HOST=0.0.0.0 \
    API_PORT=8000 \
    # Path del modelo (se puede override con -e)
    MODEL_PATH=/app/models/yolov8n_epp.onnx \
    # Workers de Uvicorn (1 por defecto, ajustar según CPU)
    UVICORN_WORKERS=1 \
    # Nivel de log
    LOG_LEVEL=info \
    # Directorio de la app
    APP_DIR=/app

# Instalar solo runtime dependencies (mucho más ligero que builder)
# libgomp1: OpenMP runtime (no headers)
# libgl1-mesa-glx, libglib2.0-0: Para OpenCV headless
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Crear usuario no-root para seguridad
# No ejecutar como root en producción
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Crear directorio de la aplicación
WORKDIR ${APP_DIR}

# Copiar wheels desde builder stage
# Esto evita tener que recompilar en runtime
COPY --from=builder /wheels /wheels

# Instalar dependencias desde wheels precompilados
# Mucho más rápido que pip install -r requirements.txt
RUN pip install --no-cache-dir --no-index --find-links=/wheels /wheels/* && \
    rm -rf /wheels

# Copiar solo archivos necesarios para runtime
# IMPORTANTE: Copiar en orden de menos a más probable de cambiar
# para maximizar cache de Docker layers

# 1. Archivos de configuración (cambian raramente)
COPY requirements.txt .
COPY .env.example .env

# 2. Código de la API (cambia frecuentemente)
COPY api/ ./api/

# 3. Directorio para modelos (se puede montar como volumen)
RUN mkdir -p models && chown -R appuser:appuser models

# TODO: En producción, copiar modelo desde GCS o incluir en imagen
# COPY models/yolov8n_epp.onnx ./models/
# O descargar en startup si está en cloud storage

# Cambiar ownership de todos los archivos a usuario no-root
RUN chown -R appuser:appuser ${APP_DIR}

# Cambiar a usuario no-root
USER appuser

# Exponer puerto de la API
EXPOSE ${API_PORT}

# Healthcheck para Kubernetes/Docker Compose
# --interval: Cada cuánto ejecutar (30s)
# --timeout: Timeout del check (10s)
# --start-period: Tiempo de gracia al iniciar (40s para cargar modelo)
# --retries: Intentos antes de marcar unhealthy (3)
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:${API_PORT}/health').raise_for_status()" || exit 1

# Comando por defecto: ejecutar API con Uvicorn
# --host: Escuchar en todas las interfaces
# --port: Puerto configurado
# --workers: Número de workers (override con env var)
# --log-level: Nivel de logging
CMD ["sh", "-c", "uvicorn api.main:app --host ${API_HOST} --port ${API_PORT} --workers ${UVICORN_WORKERS} --log-level ${LOG_LEVEL}"]

# Alternativa para debugging (descomentar si se necesita)
# CMD ["sh", "-c", "uvicorn api.main:app --host ${API_HOST} --port ${API_PORT} --reload --log-level debug"]

# =============================================================================
# Uso:
# =============================================================================
# Build:
#   docker build -t epp-detector:latest .
#   docker build -t epp-detector:v0.1.0 --build-arg BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ') .
#
# Run:
#   docker run -p 8000:8000 epp-detector:latest
#   docker run -p 8000:8000 -v $(pwd)/models:/app/models epp-detector:latest
#   docker run -p 8000:8000 -e UVICORN_WORKERS=4 epp-detector:latest
#
# Shell:
#   docker run -it epp-detector:latest /bin/bash
#
# Healthcheck manual:
#   docker inspect --format='{{.State.Health.Status}}' <container_id>
# =============================================================================

# =============================================================================
# Optimizaciones aplicadas:
# =============================================================================
# 1. Multi-stage build: Reduce tamaño final (builder ~800MB → runtime ~400MB)
# 2. Wheels precompilados: Acelera deployments (no recompilar cada vez)
# 3. Layer caching: requirements.txt antes de código (cache más frecuente)
# 4. Usuario no-root: Seguridad (no ejecutar como root)
# 5. .dockerignore: Excluir archivos innecesarios
# 6. Healthcheck: Para orquestadores (K8s, Docker Compose)
# 7. Cleanup de apt: Reduce tamaño (~100MB menos)
# 8. PYTHONDONTWRITEBYTECODE: No generar .pyc (menor tamaño)
# 9. PIP_NO_CACHE_DIR: No cachear en imagen final
# 10. Labels OCI: Metadata estándar para registries
# =============================================================================

# =============================================================================
# TODOs para producción:
# =============================================================================
# TODO: Integrar descarga de modelo desde GCS en startup
# TODO: Agregar script de entrypoint para validaciones pre-start
# TODO: Configurar logging estructurado (JSON) para producción
# TODO: Agregar instrumentación (Prometheus metrics endpoint)
# TODO: Considerar imagen base alpine (más pequeña pero compleja con ML)
# TODO: Implementar multi-arch build (amd64, arm64 para Jetson)
# TODO: Configurar registry privado (GCR, ECR, Harbor)
# TODO: Implementar versionado semántico automático (git tags)
# TODO: Agregar scanning de vulnerabilidades en CI (Trivy, Snyk)
# TODO: Considerar distroless para máxima seguridad
# =============================================================================
