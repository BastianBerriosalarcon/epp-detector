# Tests - EPP Detector

Suite de tests para el proyecto EPP Detector.

## Documentación Completa

Ver guía completa de testing en **[CLAUDE.md - Testing](../CLAUDE.md#testing)**.

## Quick Start

```bash
# Ejecutar todos los tests
pytest tests/

# Tests específicos
pytest tests/test_api.py -v
pytest tests/test_model.py -v
pytest tests/test_utils.py -v

# Con coverage
pytest --cov=api --cov-report=html
```

## Estructura

```
tests/
├── conftest.py         # Fixtures compartidos
├── test_api.py         # Tests de endpoints FastAPI
├── test_model.py       # Tests de EPPDetector
└── test_utils.py       # Tests de funciones helper
```

## Tipos de Tests

```bash
# Tests unitarios rápidos
pytest -m unit

# Excluir tests lentos
pytest -m "not slow"

# Tests de integración
pytest -m integration

# Tests que requieren GPU
pytest -m gpu
```

## Fixtures Disponibles

- `client` - TestClient de FastAPI
- `mock_model` - Mock de EPPDetector
- `sample_image_bytes` - Imagen dummy para tests
- `sample_detections` - Detecciones de ejemplo

Ver más fixtures en `conftest.py`.

## Coverage

```bash
# Coverage en terminal
pytest --cov=api --cov-report=term-missing

# Coverage en HTML (abrir htmlcov/index.html)
pytest --cov=api --cov-report=html
```

## Comandos Útiles

```bash
# Verbose
pytest -v

# Stop en primer error
pytest -x

# Ejecutar últimos tests fallidos
pytest --lf

# Mostrar prints
pytest -s

# Tests en paralelo (requiere pytest-xdist)
pytest -n auto
```

Para más información sobre testing, arquitectura de tests, markers personalizados y mejores prácticas, consultar [CLAUDE.md](../CLAUDE.md#arquitectura-de-testing).
