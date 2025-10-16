# CLAUDE.md

Este archivo proporciona orientación a Claude Code (claude.ai/code) cuando trabaja con código en este repositorio.

## Resumen del Proyecto

**EPP Detector para Minería Chilena** es un sistema de visión por computadora para detectar Equipos de Protección Personal (EPP) en operaciones mineras chilenas. Utiliza YOLOv8 para detección en tiempo real de cascos, chalecos de seguridad, y cumplimiento con las regulaciones DS 132 de seguridad minera en Chile.

**Diferenciador Clave**: Este NO es un detector genérico de EPP. Está específicamente diseñado para el contexto minero chileno con dataset localizado, UX en español, y enfoque en cumplimiento regulatorio.

**Estado Actual**: Fase MVP con infraestructura completa. Implementación de entrenamiento del modelo e inferencia pendiente (marcados con TODOs).

**POR FAVOR GENERAR CODIGO SIN EMOJIS**
---

## Comandos de Desarrollo

### Configuración Inicial
```bash
make setup              # Setup completo: instalar deps + crear .env
make install            # Instalar todas las dependencias (prod + dev)
make env               # Crear .env desde .env.example
```

### Ejecutar la API
```bash
make run               # Modo desarrollo (hot reload, debug)
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

make run-prod          # Modo producción (4 workers)
```

Acceso:
- API: http://localhost:8000
- Docs Swagger: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### Testing
```bash
make test              # Ejecutar todos los tests
make test-unit         # Solo tests unitarios
make test-api          # Tests de endpoints de API
make test-coverage     # Con reporte de coverage (abre htmlcov/index.html)
make test-fast         # Sin tests lentos (útil para pre-commit)

# Ejecutar archivo de test específico
pytest tests/test_api.py -v

# Ejecutar función de test específica
pytest tests/test_api.py::test_health_endpoint -v

# Ejecutar con markers
pytest -m "not slow"   # Excluir tests lentos
pytest -m "unit"       # Solo tests unitarios
pytest -m "gpu"        # Solo tests que requieren GPU
```

### Calidad de Código
```bash
make lint              # Ejecutar flake8
make format            # Formatear con black + isort
make format-check      # Verificar formato sin modificar
make typecheck         # Ejecutar mypy type checking
make check             # Ejecutar todas las verificaciones (lint + format-check)
```

### Docker
```bash
make docker-build      # Build imagen de desarrollo
make docker-build-prod # Build imagen de producción
./scripts/docker-build.sh dev --no-cache  # Forzar rebuild

make docker-up         # Iniciar servicios (docker-compose)
make docker-down       # Detener servicios
make docker-logs       # Ver logs de API
make docker-shell      # Abrir shell en container de API
make docker-clean      # Eliminar todos los recursos Docker
```

### Entrenamiento en GCP
```bash
# Descargar dataset
python data/scripts/download_roboflow.py --output-dir ./data/roboflow

# Entrenar en VM de GCP con GPU
python scripts/train_gcp.py --epochs 50 --batch-size 16 --export-onnx

# Exportar modelo entrenado a ONNX
python scripts/export_onnx.py --model runs/train/.../best.pt --output models/
```

Ver `docs/gcp_setup.md` para guía completa de entrenamiento en GCP.

### Utilidades
```bash
make clean             # Limpiar archivos cache de Python
make clean-all         # Limpiar Python + Docker
make info              # Mostrar info del proyecto (versiones, etc.)
make ci                # Ejecutar pipeline CI completo localmente
```

---

## Arquitectura y Conceptos Clave

### Patrón de Diseño Bilingüe

**CRÍTICO**: El proyecto usa una arquitectura bilingüe para balancear requisitos técnicos con UX chilena:

1. **Capa del Modelo (Inglés)**: YOLOv8 entrenado con dataset Roboflow usa nombres de clase en inglés
   - Clases: `hardhat`, `head`, `person`
   - Ubicación: `api/__init__.py` → `EPP_CLASSES`

2. **Capa de Traducción (Español)**: Respuestas de API incluyen traducciones al español para usuarios finales
   - Diccionario de traducción: `api/__init__.py` → `EPP_CLASSES_ES`
   - Modelo Detection incluye ambos: `class_name` (inglés) y `class_name_es` (español)

3. **Por qué**: No se pueden cambiar nombres de clase del modelo sin re-entrenar, pero usuarios chilenos necesitan UX en español

**Ejemplo de Respuesta de API**:
```json
{
  "class_id": 0,
  "class_name": "hardhat",
  "class_name_es": "Casco de seguridad",
  "confidence": 0.92
}
```

Ver `CAMBIOS_TRADUCCION.md` para explicación detallada.

### Organización del Código

```
api/
├── __init__.py        # EPP_CLASSES, EPP_CLASSES_ES, constantes
├── main.py            # App FastAPI, endpoints, modelos Pydantic
├── model.py           # Clase EPPDetector (inferencia YOLOv8/ONNX)
├── utils.py           # format_detections(), check_epp_compliance()
└── config.py          # Settings (Pydantic BaseSettings con .env)

tests/
├── conftest.py        # Fixtures compartidos (client, mock_model, sample_image_bytes)
├── test_api.py        # Tests de endpoints de API
├── test_model.py      # Tests de inferencia del modelo
└── test_utils.py      # Tests de funciones utilitarias
```

**Patrón Clave**: Tests usan nombres de clase en inglés (consistente con modelo), pero respuestas de API son bilingües.

### Gestión de Configuración

Toda la configuración está centralizada en `api/config.py` usando Pydantic `BaseSettings`:

- Override vía variables de entorno o archivo `.env`
- Acceso: `from api.config import settings`
- Settings importantes:
  - `MODEL_PATH`: Ruta al modelo ONNX/PyTorch
  - `CONFIDENCE_THRESHOLD`: Umbral de detección (default 0.5)
  - `ENABLE_GPU`: Usar CUDA si está disponible
  - `MAX_IMAGE_SIZE_MB`: Tamaño máximo de subida

**No hardcodear valores de configuración**. Usar `settings` en su lugar:
```python
# ❌ Mal
confidence = 0.5

# ✅ Bien
from api.config import settings
confidence = settings.confidence_threshold
```

### Estado de Implementación del Modelo

**Estado Actual**: La infraestructura está completa pero la inferencia del modelo usa mocks.

Archivos con implementación pendiente (buscar "TODO: Implementar" o "NotImplementedError"):
- `api/model.py`: `_load_model()`, `predict()`, `_preprocess()`, `_run_inference()`, `_postprocess()`
- `api/utils.py`: `format_detections()`, `preprocess_image()`, `letterbox_resize()`
- `api/main.py`: Endpoint `/predict` usa datos mock

**Al implementar el modelo**:
1. Descargar modelo YOLOv8 pre-entrenado o entrenar con `scripts/train_gcp.py`
2. Colocar modelo en directorio `models/`
3. Implementar métodos de `EPPDetector` en `api/model.py`
4. Actualizar `api/utils.py` para procesar resultados reales de YOLOv8
5. Eliminar datos mock de `api/main.py`
6. Habilitar coverage en `pytest.ini` y `.github/workflows/ci.yml` (cambiar `--cov-fail-under=0` a 80)

### Arquitectura de Testing

**Fixtures** (en `tests/conftest.py`):
- `client`: FastAPI TestClient
- `mock_model`: EPPDetector mockeado con comportamiento configurable
- `sample_image_bytes`: Imagen dummy para tests de upload
- `sample_detections`: Resultados de detección pre-definidos

**Markers** (definidos en `pytest.ini`):
- `@pytest.mark.slow`: Para tests que toman >1s
- `@pytest.mark.unit`: Tests unitarios rápidos
- `@pytest.mark.integration`: Tests con dependencias externas
- `@pytest.mark.gpu`: Requieren hardware GPU
- `@pytest.mark.model`: Requieren modelo real cargado

**Coverage**: Target es 80% pero actualmente deshabilitado (código usa mocks). Re-habilitar cuando modelo esté implementado.

### Endpoints de FastAPI

Todos los endpoints en `api/main.py`:

1. **GET /health**: Probe de liveness/readiness de Kubernetes
   - Retorna: `{ "status": "healthy", "model_loaded": bool }`

2. **POST /predict**: Endpoint principal de detección
   - Input: Formulario multipart con archivo de imagen
   - Retorna: Detecciones bilingües con bboxes, confidence, estado de cumplimiento
   - Actualmente retorna datos mock

3. **GET /metrics**: Métricas compatibles con Prometheus
   - Retorna: request_count, avg_latency_ms, uptime_seconds

4. **GET /info**: Metadata del modelo
   - Retorna: model_version, classes (bilingüe), confidence_threshold

5. **GET /**: Endpoint raíz
   - Retorna: Info básica de API con enlaces a docs

### Lógica de Cumplimiento

Evaluación de cumplimiento de EPP en `api/utils.py` → `check_epp_compliance()`:

**Lógica actual** (basada en modelo de 3 clases):
- **Cumple**: `hardhat` detectado Y `head` NO detectado
- **No cumple**: `head` detectado (persona sin casco - violación crítica)
- **Desconocido**: Solo `person` detectado sin info de hardhat/head

**Expansión futura** (cuando se agreguen chaleco/botas):
- Cumplimiento requiere: hardhat + safety_vest
- Referencia: Regulación DS 132 de minería en Chile

Retorna mensajes en español: `{ "compliant": bool, "violations": [...], "summary": "..." }`

---

## Contexto de Minería Chilena

Este proyecto está diseñado para la industria minera chilena con requisitos específicos:

### Cumplimiento Regulatorio
- **DS 132**: Reglamento de seguridad minera (Ministerio de Minería)
- EPP requerido: Casco + Chaleco reflectante
- Multas: 1-2000 UTM (~$600K-$1.2B CLP) por no cumplimiento
- Ver `docs/chile_context.md` para contexto regulatorio completo

### Estrategia de Dataset
1. **Base**: Dataset Roboflow "Hard Hat Workers" (~5K imágenes, internacional)
2. **Fine-tuning**: Imágenes de minería chilena (objetivo: 1K+ imágenes)
   - Condiciones específicas: Minas subterráneas, rajo abierto, iluminación desértica
   - Marcas/colores de EPP chilenos (cascos amarillos, chalecos naranjos)
3. **Approach de transfer learning documentado en `docs/chile_context.md`**

### Localización
- **UI/Respuestas de API**: Español (ver patrón bilingüe arriba)
- **Documentación**: Español en `docs/` y README
- **Código/logs**: Inglés (estándar internacional)
- **Comentarios**: Español para lógica de negocio, inglés para técnico

---

## Patrones Comunes

### Agregar un Nuevo Endpoint
1. Definir modelos Pydantic en `api/main.py` (request + response)
2. Implementar endpoint con manejo de errores apropiado
3. Agregar tests en `tests/test_api.py` con fixture `client`
4. Actualizar este archivo si el endpoint es crítico

### Agregar una Nueva Clase de EPP
1. Actualizar `EPP_CLASSES` en `api/__init__.py` (nombre en inglés)
2. Agregar traducción a `EPP_CLASSES_ES` (nombre en español)
3. Actualizar lógica de `check_epp_compliance()` en `api/utils.py`
4. Actualizar `get_class_color()` para visualización
5. Re-entrenar modelo con nueva clase

### Agregar Configuración
1. Agregar campo a clase `Settings` en `api/config.py`
2. Establecer valor default y nombre de variable de entorno
3. Actualizar `.env.example` con nueva variable
4. Documentar en docstring de `api/config.py`

### Debugging de Problemas del Modelo
```python
# Verificar si el modelo está cargado
from api.main import detector
print(detector.is_loaded if detector else "No cargado")

# Ver configuración actual
from api.config import print_settings
print_settings()

# Probar detección en imagen de muestra
from api.model import EPPDetector
det = EPPDetector("models/yolov8n_epp.onnx")
results = det.predict(image_bytes)
```

---

## Convenciones de Nomenclatura de Archivos

- Archivos Python: `snake_case.py`
- Archivos de test: `test_*.py`
- Documentación: Español con nombres descriptivos (`chile_context.md`, `gcp_setup.md`)
- Scripts: Ejecutables con extensión `.sh` o `.py`
- Modelos: `yolov8{tamaño}_epp.{pt|onnx}` (ej: `yolov8n_epp.onnx`)

---

## Pipeline CI/CD

Workflows de GitHub Actions en `.github/workflows/`:

1. **ci.yml**: Pipeline principal (test, lint, security, build)
   - Se ejecuta en: push a main/develop, pull requests
   - Jobs: test, lint, security, typecheck (deshabilitado), build-test
   - Requerido para merge

2. **docker-build.yml**: Build y push de imágenes Docker
   - Trigger: tags, dispatch manual

3. **deploy.yml**: Deploy a GCP Cloud Run
   - Trigger: CI exitoso + push de tag

**Simulación local de CI**: `make ci` ejecuta checks + tests con coverage

---

## Documentación

- **README.md**: Resumen del proyecto, quick start (español)
- **docs/gcp_setup.md**: Guía completa de entrenamiento en GCP con setup de GPU (español)
- **docs/chile_context.md**: Contexto de negocio, regulaciones, estrategia de dataset (español)
- **CAMBIOS_TRADUCCION.md**: Explicación del patrón de diseño bilingüe (español)
- **Este archivo**: Guía de desarrollador para Claude Code (español)

---

## Notas Importantes

### Cuando el Modelo Aún No Está Implementado
- Endpoints de API retornan datos mock (detecciones hardcodeadas)
- Tests usan fixture `mock_model` en lugar de modelo real
- Coverage está deshabilitado (habilitar cuando implementación esté completa)
- Todos los TODOs relacionados con implementación del modelo deben abordarse juntos

### Estilo de Código
- **Formatter**: Black con line-length=100
- **Import sorting**: isort con perfil black
- **Linter**: flake8 (max-line-length=100, ignore E203, W503)
- **Type hints**: Requeridos para funciones públicas (enforced por mypy cuando esté habilitado)
- Ejecutar `make format` antes de hacer commit

### Mejores Prácticas de Testing
- Usar fixtures de `conftest.py` en lugar de crear datos de test inline
- Marcar tests lentos con `@pytest.mark.slow`
- Mantener tests unitarios rápidos (<100ms cada uno)
- Tests de integración deben limpiarse a sí mismos
- Objetivo de 80% de coverage cuando modelo esté implementado

### Seguridad
- Nunca hacer commit de archivos `.env` (está en `.gitignore`)
- API keys deben estar en variables de entorno
- Usar `settings.api_key` de config, nunca hardcodear
- Datos sensibles en logs deben ser redactados

---

## Problemas Comunes y Soluciones

### Error "Model not loaded"
- Verificar que `MODEL_PATH` en `.env` apunta a archivo existente
- Ejecutar `python -c "from api.config import settings; print(settings.get_model_path_absolute())"`
- Si el modelo no existe, descargarlo o entrenarlo primero

### Falla el build de Docker
- Verificar espacio en disco: `df -h`
- Limpiar cache de Docker: `make docker-clean`
- Rebuild sin cache: `./scripts/docker-build.sh prod --no-cache`

### Tests fallan con errores de import
- Instalar dependencias de dev: `make install`
- Verificar versión de Python: `python --version` (requiere 3.10+)
- Activar entorno virtual

### Coverage muy bajo
- Esto es esperado hasta que el modelo esté completamente implementado
- Umbral de coverage está en 0 en CI (ver `pytest.ini` y `.github/workflows/ci.yml`)
- Cambiar a 80 cuando esté listo para producción

---

## Referencia de Tech Stack

- **ML**: YOLOv8 (Ultralytics), ONNX Runtime
- **API**: FastAPI, Pydantic, Uvicorn
- **Testing**: pytest, pytest-cov, pytest-mock
- **Linting**: Black, isort, flake8, mypy
- **CI/CD**: GitHub Actions
- **Cloud**: Google Cloud Platform (Compute Engine, Cloud Storage, Cloud Run)
- **MLOps**: MLflow para tracking de experimentos
- **Container**: Docker multi-stage builds

---

## Información de Versión

- **Versión actual**: 0.1.0 (ver `api/__init__.py`)
- **Python**: 3.10+
- **Dataset**: Roboflow Hard Hat Workers + minería chilena (pendiente)
- **Modelo**: YOLOv8n (tamaño objetivo de deployment)
- **Estado**: Infraestructura MVP completa, implementación de modelo pendiente
