# API REST - EPP Detector

API REST para detección de Equipos de Protección Personal en faenas mineras chilenas.

## Documentación Completa

Ver documentación detallada en:
- **[README.md principal](../README.md)** - Overview del proyecto y quick start
- **[CLAUDE.md](../CLAUDE.md)** - Guía completa de desarrollo y arquitectura

## Quick Start

```bash
# Instalar dependencias
pip install -r requirements.txt

# Configurar variables de entorno
cp .env.example .env

# Ejecutar servidor de desarrollo
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

## Endpoints Principales

- `GET /health` - Health check
- `POST /predict` - Detección de EPP en imagen
- `GET /metrics` - Métricas del sistema
- `GET /info` - Información del modelo
- `GET /docs` - Documentación interactiva (Swagger)

## Arquitectura

```
api/
├── __init__.py       # EPP_CLASSES, EPP_CLASSES_ES, constantes
├── main.py           # App FastAPI, endpoints, modelos Pydantic
├── model.py          # Clase EPPDetector (inferencia YOLOv8/ONNX)
├── utils.py          # format_detections(), check_epp_compliance()
└── config.py         # Settings (Pydantic BaseSettings con .env)
```

## Testing

```bash
# Tests de la API
pytest tests/test_api.py -v

# Tests con coverage
pytest --cov=api tests/
```

Más información en [CLAUDE.md](../CLAUDE.md#testing).
