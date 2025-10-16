# API REST - EPP Detector

API REST para detección de Equipos de Protección Personal en faenas mineras chilenas.

## Arquitectura

```
api/
├── __init__.py       # Configuración de módulo y constantes
├── main.py           # Aplicación FastAPI y endpoints
├── model.py          # Clase EPPDetector (inferencia)
├── utils.py          # Funciones helper (validación, preprocessing)
├── config.py         # Configuración centralizada
└── README.md         # Este archivo
```

## Endpoints

### `GET /health`
Health check para Kubernetes probes.

**Response:**
```json
{
  "status": "healthy",
  "version": "0.1.0",
  "model_loaded": true,
  "timestamp": "2024-01-15T10:30:00"
}
```

### `POST /predict`
Detecta EPP en una imagen.

**Request:**
- `file`: Imagen JPG/PNG (max 10MB)

**Response:**
```json
{
  "success": true,
  "detections": [
    {
      "class_id": 0,
      "class_name": "casco",
      "confidence": 0.92,
      "bbox": [120.5, 50.2, 200.8, 150.3]
    }
  ],
  "inference_time_ms": 45.3,
  "image_size": {"width": 640, "height": 640},
  "total_detections": 2,
  "epp_compliant": true
}
```

### `GET /metrics`
Métricas de performance del sistema.

**Response:**
```json
{
  "requests_total": 1523,
  "avg_latency_ms": 52.3,
  "uptime_seconds": 86400,
  "model_loaded": true
}
```

### `GET /info`
Información del modelo y configuración.

**Response:**
```json
{
  "model_version": "v0.1.0-beta",
  "model_type": "YOLOv8n",
  "classes": {
    "0": "casco",
    "1": "chaleco_reflectante",
    "2": "zapatos_seguridad"
  },
  "confidence_threshold": 0.5,
  "dataset_info": "Roboflow + Dataset Chile"
}
```

## Instalación

```bash
# Instalar dependencias
pip install -r requirements.txt

# Configurar variables de entorno
cp .env.example .env
# Editar .env con tu configuración

# Ejecutar servidor de desarrollo
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

## Desarrollo

### Testing

```bash
# Tests unitarios
pytest tests/test_api.py

# Tests con coverage
pytest --cov=api tests/
```

### Linting

```bash
# Formatear código
black api/
isort api/

# Verificar estilo
flake8 api/
```

## Deployment

### Docker

```bash
# Build
docker build -t epp-detector:latest .

# Run
docker run -p 8000:8000 epp-detector:latest
```

### Google Cloud Run

```bash
# Deploy
gcloud run deploy epp-detector \
  --source . \
  --region us-central1 \
  --allow-unauthenticated
```

## TODOs

### Prioridad Alta
- [ ] Implementar carga real de modelo YOLOv8/ONNX (model.py)
- [ ] Implementar inferencia (model.py:predict)
- [ ] Implementar preprocessing de imágenes (utils.py)
- [ ] Implementar formateo de detecciones (utils.py)

### Prioridad Media
- [ ] Agregar tests unitarios para cada endpoint
- [ ] Implementar batch inference para múltiples imágenes
- [ ] Agregar soporte para video (frame por frame)
- [ ] Integrar con MLflow para tracking

### Prioridad Baja
- [ ] Agregar autenticación con API keys
- [ ] Implementar rate limiting
- [ ] Agregar cache de resultados (Redis)
- [ ] Integrar con Sentry para error tracking

## Configuración

Todas las configuraciones se manejan via `api/config.py` usando Pydantic BaseSettings.

Variables de entorno principales:

| Variable | Default | Descripción |
|----------|---------|-------------|
| `MODEL_PATH` | `models/yolov8n_epp.onnx` | Ruta al modelo |
| `CONFIDENCE_THRESHOLD` | `0.5` | Umbral de confianza |
| `ENABLE_GPU` | `true` | Usar GPU si disponible |
| `MAX_IMAGE_SIZE_MB` | `10` | Tamaño máximo de imagen |

Ver `.env.example` para lista completa.

## Notas de implementación

### Decisiones de diseño

1. **FastAPI sobre Flask**: Mejor performance, async nativo, auto-documentación
2. **ONNX Runtime**: Mayor portabilidad, menor overhead que PyTorch en producción
3. **Pydantic**: Validación de datos robusta, type safety
4. **Configuración centralizada**: Facilita deployment en diferentes entornos

### Performance

- **Target**: <100ms inference en CPU, <30ms en GPU
- **Optimizaciones**: ONNX Runtime, batch inference, warmup
- **Bottleneck**: Preprocessing de imagen (resize, normalización)

### Seguridad

- Validación de tamaño y formato de imágenes
- Rate limiting configurable
- CORS configurable por entorno
- API keys opcionales

---

**Última actualización**: Iteración 3
**Estado**: Estructura completa con TODOs
**Próximos pasos**: Implementar inferencia real (Iteración 4)
