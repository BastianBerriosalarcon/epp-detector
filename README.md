# Detector de EPP para Minería Chilena

[![CI/CD Pipeline](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-blue)](https://github.com/BastianBerriosalarcon/epp-detector/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Objetivo

Sistema de detección automática de Equipos de Protección Personal (EPP) mediante visión por computadora, diseñado específicamente para la industria minera chilena. El sistema identifica en tiempo real el uso correcto de:

- **Casco de seguridad** (hardhat detection)
- **Detección de violaciones** (personas sin casco)
- **Personas** (para análisis de cumplimiento)

**Problema:** Los accidentes laborales en minería chilena representan el 8.5% del total nacional (SUSESO 2023). El uso incorrecto o ausencia de EPP es una de las principales causas de lesiones graves y fatalidades. La supervisión manual es inconsistente y costosa.

**Solución:** Automatización de la detección de EPP mediante YOLOv8, desplegada en edge devices y sistemas de video vigilancia existentes.

---

## Diferenciador: Contexto Chile

Este proyecto NO es un detector genérico de EPP. Está diseñado para:

1. **Cumplimiento DS 132**: Reglamento de Seguridad Minera (actualizado 2024)
2. **Condiciones locales**: Polvo de cobre, iluminación subterránea, equipos amarillos/naranjos
3. **Dataset chileno**: Imágenes reales de faenas mineras chilenas (División Andina, El Teniente, minas del norte)
4. **Integración con normativa**: Alertas configurables según estándares SERNAGEOMIN

**Fine-tuning específico** en datos de:
- Minería subterránea (baja iluminación)
- Rajo abierto (alto contraste, polvo)
- Equipos y uniformes de empresas chilenas (Codelco, Antofagasta Minerals, etc.)

---

## Arquitectura

```
Input (Imagen) → Validación → YOLOv8 (ONNX/PyTorch) → Post-processing →
  ├─ API REST (FastAPI) con 5 endpoints
  ├─ Evaluación de cumplimiento DS 132
  ├─ Respuestas bilingües (EN/ES)
  └─ Métricas y monitoreo
```

**Componentes Implementados:**
- **API REST**: FastAPI con arquitectura SOLID y dependency injection
- **Validación**: Formato, tamaño y dimensiones de imágenes
- **Detector**: Clase EPPDetector con warmup, batch processing y context manager
- **Middleware**: Rate limiting y request logging
- **Excepciones**: Jerarquía completa de errores personalizados
- **Protocols**: Abstracciones para DetectorProtocol, ImageValidatorProtocol
- **Model Loader**: Carga de modelos con retry logic y soporte PyTorch/ONNX
- **Testing**: 107 tests con pytest, fixtures y mocks
- **CI/CD**: GitHub Actions con pipelines de test, lint y deploy
- **Docker**: Multi-stage builds optimizados
- **Documentación**: Completa en CLAUDE.md y docs/

---

## Estado del Proyecto

| Componente | Estado | Descripción |
|------------|--------|-------------|
| **Infraestructura** | Completo | API, testing, CI/CD, Docker |
| **Arquitectura** | Completo | SOLID, DI, protocols, excepciones |
| **Configuración** | Completo | Pydantic Settings, 50+ parámetros |
| **Validación** | Completo | Imágenes, formatos, dimensiones |
| **Modelo YOLOv8** | Pendiente | Implementación de inferencia |
| **Entrenamiento** | Pendiente | Scripts y pipelines |
| **Dataset Chile** | Pendiente | Recolección y anotación |

**Objetivos de Performance:**
- **mAP@0.5**: ≥ 0.85
- **Precision**: ≥ 0.90
- **Recall**: ≥ 0.88
- **Inference (CPU)**: < 100ms
- **Inference (GPU)**: < 30ms

---

## Quick Start

### 1. Clonar e Instalar

```bash
# Clonar repositorio
git clone https://github.com/BastianBerriosalarcon/epp-detector.git
cd epp-detector

# Setup completo (recomendado)
make setup

# O manualmente:
pip install -r requirements.txt
pip install -r requirements-dev.txt
cp .env.example .env
```

### 2. Ejecutar la API

```bash
# Modo desarrollo (con hot reload)
make run

# O directamente:
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

**Endpoints disponibles:**
- `http://localhost:8000` - Página principal
- `http://localhost:8000/docs` - Documentación interactiva (Swagger)
- `http://localhost:8000/redoc` - Documentación alternativa (ReDoc)
- `http://localhost:8000/health` - Health check
- `http://localhost:8000/predict` - Detección de EPP (POST)
- `http://localhost:8000/metrics` - Métricas del sistema
- `http://localhost:8000/info` - Información del modelo

### 3. Testing

```bash
# Ejecutar todos los tests
make test

# Tests específicos
pytest tests/test_api.py -v
pytest tests/test_model.py -v

# Con coverage
make test-coverage
```

### 4. Linting y Formateo

```bash
# Formatear código
make format

# Verificar linting
make lint

# Type checking
make typecheck
```

### 5. Docker

```bash
# Build desarrollo
make docker-build

# Build producción
make docker-build-prod

# Ejecutar con docker-compose
make docker-up

# Ver logs
make docker-logs
```

---

## Tech Stack

| Componente | Tecnología | Justificación |
|------------|------------|---------------|
| **Detección** | YOLOv8 (Ultralytics) | SOTA en real-time object detection, <30ms inference |
| **Runtime** | ONNX Runtime | Portabilidad multi-plataforma, optimización CPU/GPU |
| **API** | FastAPI | Async, auto-docs, validación con Pydantic |
| **Frontend** | Streamlit | Prototipado rápido, ideal para demos ML |
| **Cloud** | Google Cloud Platform | Cloud Storage + Compute Engine |
| **MLOps** | MLflow | Tracking de experimentos y registro de modelos |
| **CI/CD** | GitHub Actions | Testing automático, build Docker |
| **Containerización** | Docker | Reproducibilidad, deployment simplificado |

---

## Datasets

### 1. Dataset Base (Roboflow)
- **Fuente**: [Construction Safety Dataset](https://universe.roboflow.com/roboflow-universe-projects/construction-site-safety)
- **Clases**: Hardhat, Safety Vest, NO-Hardhat, NO-Safety Vest, Person
- **Imágenes**: ~5,000 anotadas (YOLO format)

### 2. Dataset Chile (Propio)
- **Fuente**: Imágenes de faenas mineras chilenas (colaboración empresas)
- **Objetivo**: 1,000+ imágenes adicionales
- **Clases específicas**:
  - Casco amarillo/blanco (estándar minería)
  - Chaleco naranja reflectante
  - Zapatos de seguridad (visible en planos medios)
- **Condiciones**: Subterránea, rajo abierto, diferentes iluminaciones

**Pendiente**: Recolección y anotación de datos locales

---

## Roadmap

### Fase 1: Infraestructura - COMPLETADO
- [x] Estructura de directorios y organización del proyecto
- [x] Configuración con Pydantic BaseSettings (50+ parámetros)
- [x] API REST con FastAPI (5 endpoints)
- [x] Sistema de testing completo (107 tests con pytest)
- [x] CI/CD con GitHub Actions (test, lint, build, deploy)
- [x] Docker multi-stage builds
- [x] Documentación completa (CLAUDE.md, README.md, docs/)
- [x] Arquitectura SOLID con dependency injection
- [x] Protocols para abstracciones
- [x] Jerarquía de excepciones personalizada
- [x] Middleware de rate limiting y logging
- [x] Validación completa de imágenes
- [x] Sistema bilingüe (inglés/español)

### Fase 2: Modelo YOLOv8 - EN PROGRESO
- [ ] Implementar métodos de inferencia en EPPDetector
- [ ] Integrar YOLOv8 (PyTorch y ONNX)
- [ ] Implementar preprocessing de imágenes
- [ ] Implementar postprocessing de detecciones
- [ ] Descargar o entrenar modelo pre-entrenado
- [ ] Tests de integración con modelo real
- [ ] Optimización de performance (warmup, batch)

### Fase 3: Entrenamiento
- [ ] Pipeline de descarga de dataset (Roboflow)
- [ ] Scripts de entrenamiento con configuración
- [ ] Integración con MLflow para tracking
- [ ] Validación y métricas (mAP, precision, recall)
- [ ] Export a ONNX para producción
- [ ] Entrenamiento en GCP con GPU

### Fase 4: Dataset Chileno
- [ ] Recolección de imágenes de faenas mineras
- [ ] Anotación con herramientas (Roboflow/LabelImg)
- [ ] Fine-tuning con datos locales
- [ ] Validación en condiciones chilenas específicas
- [ ] Mejora iterativa del modelo

### Fase 5: Producción
- [ ] Deployment en GCP Cloud Run
- [ ] Monitoreo y alertas
- [ ] Optimización de costos
- [ ] Documentación de API para usuarios finales
- [ ] Integración con sistemas de vigilancia existentes

---

## Estructura del Proyecto

```
epp-detector/
├── api/                      # FastAPI REST API
│   ├── __init__.py          # Constantes y traducciones bilingües
│   ├── main.py              # Aplicación FastAPI y endpoints
│   ├── model.py             # Clase EPPDetector (YOLOv8)
│   ├── config.py            # Configuración con Pydantic
│   ├── utils.py             # Funciones auxiliares
│   ├── validators.py        # Validación de imágenes
│   ├── protocols.py         # Abstracciones (DetectorProtocol, etc.)
│   ├── exceptions.py        # Jerarquía de excepciones
│   ├── middleware.py        # Rate limiting y logging
│   └── model_loader.py      # Carga de modelos con retry
├── tests/                   # Testing (107 tests)
│   ├── conftest.py          # Fixtures compartidos
│   ├── test_api.py          # Tests de endpoints
│   ├── test_model.py        # Tests del detector
│   └── test_utils.py        # Tests de utilidades
├── scripts/                 # Scripts de utilidad
│   ├── train_gcp.py         # Entrenamiento en GCP
│   ├── export_onnx.py       # Export de modelos
│   ├── docker-build.sh      # Build de imágenes Docker
│   └── training/            # Módulos de entrenamiento
├── data/                    # Datasets
│   ├── scripts/             # Download y preprocessing
│   └── README.md            # Documentación de datos
├── docs/                    # Documentación
│   ├── gcp_setup.md         # Guía de setup en GCP
│   └── chile_context.md     # Contexto de minería chilena
├── models/                  # Modelos entrenados (.pt, .onnx)
├── .github/workflows/       # CI/CD Pipelines
│   ├── ci.yml               # Testing y linting
│   ├── docker-build.yml     # Build de imágenes
│   └── deploy.yml           # Deployment a GCP
├── Dockerfile               # Multi-stage build
├── docker-compose.yml       # Orquestación de servicios
├── Makefile                 # Comandos de desarrollo
├── pytest.ini               # Configuración de pytest
├── requirements.txt         # Dependencias de producción
├── requirements-dev.txt     # Dependencias de desarrollo
├── .env.example             # Template de configuración
├── CLAUDE.md                # Guía completa para desarrollo
└── README.md                # Este archivo
```

---

## Contribución

Este proyecto está en desarrollo activo. Para contribuir:

1. Fork el repositorio
2. Crea una rama feature (`git checkout -b feature/nueva-funcionalidad`)
3. Commit cambios (`git commit -m 'Agregar detección de guantes'`)
4. Push a la rama (`git push origin feature/nueva-funcionalidad`)
5. Abre un Pull Request

**Código de estilo:**
- Black para formateo
- isort para imports
- flake8 para linting
- Type hints en funciones públicas

---

## License

Este proyecto está bajo la licencia MIT. Ver archivo `LICENSE` para más detalles.

---

## Contacto

**Autor**: Bastián Berríos
**Email**: bastianberrios.a@gmail.com
**GitHub**: [@BastianBerriosalarcon](https://github.com/BastianBerriosalarcon)
**Repositorio**: [github.com/BastianBerriosalarcon/epp-detector](https://github.com/BastianBerriosalarcon/epp-detector)

---

## Referencias

- [Ultralytics YOLOv8](https://docs.ultralytics.com/)
- [Decreto Supremo 132 - Reglamento de Seguridad Minera](https://www.sernageomin.cl/)
- [SUSESO - Estadísticas de Accidentabilidad](https://www.suseso.cl/)
- [Roboflow Universe](https://universe.roboflow.com/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Pydantic Documentation](https://docs.pydantic.dev/)

---

Desarrollado para mejorar la seguridad en la minería chilena
